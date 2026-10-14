#include "ninfer/engine.h"
#include "core/device.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

namespace {
using Json = nlohmann::json;

std::string read(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read input file");
    return {std::istreambuf_iterator<char>(f), {}};
}

Json devices() {
    Json out = Json::array();
    for (int d = 0; d < 3; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        std::size_t free, total;
        CUDA_CHECK(hipMemGetInfo(&free, &total));
        out.push_back({{"device", d},
                       {"total_bytes", total},
                       {"free_bytes", free},
                       {"used_bytes", total - free}});
    }
    return out;
}

struct MemorySampler {
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::array<std::size_t, 3> peaks{};
    std::thread worker;

    MemorySampler() {
        worker = std::thread([this] {
            try {
                while (!stop) {
                    auto snapshot = devices();
                    {
                        std::lock_guard lock(mutex);
                        for (int d = 0; d < 3; ++d)
                            peaks[d] =
                                std::max(peaks[d], snapshot[d].at("used_bytes").get<std::size_t>());
                    }
                    for (int i = 0; i < 25 && !stop; ++i)
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            } catch (...) { stop = true; }
        });
    }

    ~MemorySampler() {
        stop = true;
        if (worker.joinable()) worker.join();
    }

    Json peak() {
        std::lock_guard lock(mutex);
        return peaks;
    }
};

Json run(const std::filesystem::path& artifact, const std::string& input, const std::string& kind,
         int context, int output, int window, bool draft) {
    Json report = {{"backend", "native ninfer::Engine"},
                   {"speculative", draft ? "DFlash2 K7" : "ordinary"},
                   {"kvmem_window_pages", window},
                   {"max_context", context}};
    ninfer::EngineOptions options;
    options.artifact_path      = artifact;
    options.device             = 0;
    options.max_context        = context;
    options.kv_capacity        = ninfer::KvCapacityPolicy::explicit_capacity(context);
    options.max_concurrency    = 1;
    options.use_cuda_graph     = false;
    options.kvmem_window_pages = window;
    options.pending_timeout_ms = 0;
    if (draft) {
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
    }
    options.startup_observer.callback = [](const ninfer::StartupEvent& event) {
        if (event.status == ninfer::StartupStatus::Begin ||
            event.status == ninfer::StartupStatus::Complete)
            std::cerr << "startup phase=" << static_cast<int>(event.phase)
                      << " status=" << static_cast<int>(event.status) << '\n';
    };
    ninfer::Engine engine(options);
    report["devices_after_load"] = devices();
    MemorySampler sampler;
    auto load      = engine.load_summary();
    report["load"] = {
        {"architecture", load.architecture},      {"seconds", load.load_seconds},
        {"h2d_bytes", load.host_to_device_bytes}, {"device_objects", load.device_object_count},
        {"host_objects", load.host_object_count}, {"peak_staging_bytes", load.peak_staging_bytes}};
    ninfer::PreparedPrompt prompt;
    if (kind == "tokens")
        prompt = engine.prepare_tokens(Json::parse(input).get<std::vector<ninfer::TokenId>>());
    else if (kind == "raw")
        prompt = engine.prepare_tokens(engine.tokenize_text(input));
    else {
        ninfer::PromptInput chat;
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back({ninfer::MessagePartKind::Text, input, {}});
        chat.messages.push_back(std::move(message));
        chat.options.reasoning_effort = ninfer::ReasoningEffort::Low;
        prompt                        = engine.prepare(std::move(chat));
    }
    std::cerr << "native prefill tokens=" << prompt.summary().prompt_tokens
              << " backend=" << (draft ? "DFlash2 K7" : "ordinary") << '\n';
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = output;
    request.execution.sampling.temperature    = 0.f;
    auto result                        = engine.generate(std::move(prompt), std::move(request));
    report["devices_after_generation"] = devices();
    report["sampled_peak_bytes_500ms"] = sampler.peak();
    report["result"]                   = {
        {"prompt_tokens", result.prompt.prompt_tokens},
        {"generated_ids", result.generated_token_ids},
        {"content", result.content},
        {"reasoning", result.reasoning},
        {"finish_reason", static_cast<int>(result.finish_reason)},
        {"prefill_seconds", result.timings.prefill_seconds},
        {"decode_seconds", result.timings.decode_seconds},
        {"total_seconds", result.timings.total_seconds},
        {"decode_tokens_per_second",
         result.timings.decode_seconds > 0 && result.generated_token_ids.size() > 1
             ? (result.generated_token_ids.size() - 1) / result.timings.decode_seconds
             : 0},
        {"draft_rounds", result.speculative.rounds},
        {"drafted_tokens", result.speculative.drafted_tokens},
        {"accepted_draft_tokens", result.speculative.accepted_tokens},
        {"accepted_per_position", result.speculative.accepted_per_position},
        {"kvmem_selection_calls", result.timings.kvmem.selection_calls}};
    auto memory                  = engine.memory_summary();
    report["native_allocations"] = {{"encoded_weights", memory.weights.capacity_bytes},
                                    {"state_kv_features_snapshots", memory.sequence.capacity_bytes},
                                    {"workspace_capacity", memory.workspace.capacity_bytes},
                                    {"workspace_logical_peak", memory.workspace_logical_peak_bytes},
                                    {"host_kv_bytes", memory.host_kv_occupied_bytes},
                                    {"host_state_slots", memory.host_state_occupied_slots}};
    return report;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact, report_path;
        std::string input, kind;
        int context = 32768, output = 64, window = 6;
        bool draft = true, verify = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value    = [&] {
                if (i + 1 == argc) throw std::invalid_argument("missing argument value");
                return std::string(argv[++i]);
            };
            if (a == "--help") {
                std::cout << "Native GLM-5.3-Flash on three Hygon K100 devices; v3 .ninfer "
                             "only.\n--artifact FILE (--prompt TEXT | --prompt-file FILE | "
                             "--raw-text TEXT | --token-ids-file FILE)\n[--max-context 32768] "
                             "[--max-new-tokens 64] [--kvmem-window-pages 6]\n[--ordinary | "
                             "--verify-native-greedy] [--report FILE]\nGreedy Aggregate profile; "
                             "DFlash2 K7 by default. No weight/KV offload.\nVerification loads "
                             "ordinary then DFlash2 serially and compares every generated token.\n";
                return 0;
            }
            if (a == "--artifact")
                artifact = value();
            else if (a == "--report")
                report_path = value();
            else if (a == "--prompt" || a == "--raw-text" || a == "--prompt-file" ||
                     a == "--token-ids-file") {
                if (!kind.empty()) throw std::invalid_argument("choose one input route");
                input = value();
                if (a == "--prompt-file" || a == "--token-ids-file") input = read(input);
                kind = a == "--raw-text" ? "raw" : a == "--token-ids-file" ? "tokens" : "chat";
            } else if (a == "--max-context")
                context = std::stoi(value());
            else if (a == "--max-new-tokens")
                output = std::stoi(value());
            else if (a == "--kvmem-window-pages")
                window = std::stoi(value());
            else if (a == "--ordinary")
                draft = false;
            else if (a == "--verify-native-greedy")
                verify = true;
            else
                throw std::invalid_argument("unknown argument: " + a);
        }
        if (artifact.empty() || kind.empty() || context < 1 || context > 32768 || output < 1 ||
            window < 0 || (!draft && verify))
            throw std::invalid_argument("invalid or missing native inference arguments");
        Json result;
        if (verify) {
            auto ordinary    = run(artifact, input, kind, context, output, window, false);
            auto speculative = run(artifact, input, kind, context, output, window, true);
            bool equal =
                ordinary["result"]["generated_ids"] == speculative["result"]["generated_ids"];
            result = {{"ordinary", std::move(ordinary)},
                      {"dflash2", std::move(speculative)},
                      {"greedy_equal", equal}};
            if (!equal)
                throw std::runtime_error(
                    "native DFlash2 output differs from native ordinary generation");
            if (result["dflash2"]["result"]["accepted_draft_tokens"].get<std::uint64_t>() == 0)
                throw std::runtime_error("DFlash2 verification produced no positive acceptance");
        } else
            result = run(artifact, input, kind, context, output, window, draft);
        if (!report_path.empty()) {
            std::ofstream f(report_path);
            if (!f) throw std::runtime_error("cannot write native report");
            f << result.dump(2) << '\n';
        }
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
