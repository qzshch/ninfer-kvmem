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
         int context, int output, int window, bool draft, int repeat = 1, int lanes = 1,
         bool graph = false, bool mtp = false) {
    Json report = {{"backend", "native ninfer::Engine"},
                   {"speculative", mtp     ? "MTP K1"
                                   : draft ? "DFlash2 K7"
                                           : "ordinary"},
                   {"kvmem_window_pages", window},
                   {"max_context", context}};
    ninfer::EngineOptions options;
    options.artifact_path      = artifact;
    options.device             = 0;
    options.max_context        = context;
    options.kv_capacity        = ninfer::KvCapacityPolicy::explicit_capacity(context);
    options.max_concurrency    = lanes;
    options.use_cuda_graph     = graph;
    options.kvmem_window_pages = window;
    options.pending_timeout_ms = 0;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 1;
    } else if (draft) {
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
    }
    options.startup_observer.callback = [last = std::uint64_t{}](
                                            const ninfer::StartupEvent& event) mutable {
        if (event.status == ninfer::StartupStatus::Begin) last = 0;
        if (event.status == ninfer::StartupStatus::Begin ||
            event.status == ninfer::StartupStatus::Complete ||
            (event.status == ninfer::StartupStatus::Progress &&
             event.current >= last + (4ULL << 30))) {
            last = event.current;
            std::cerr << "startup phase=" << static_cast<int>(event.phase)
                      << " status=" << static_cast<int>(event.status) << " bytes=" << event.current
                      << '/' << event.total << " seconds=" << event.elapsed_ns / 1e9 << '\n';
        }
    };
    ninfer::Engine engine(options);
    report["devices_after_load"] = devices();
    MemorySampler sampler;
    auto load      = engine.load_summary();
    report["load"] = {
        {"architecture", load.architecture},      {"seconds", load.load_seconds},
        {"h2d_bytes", load.host_to_device_bytes}, {"device_objects", load.device_object_count},
        {"host_objects", load.host_object_count}, {"peak_staging_bytes", load.peak_staging_bytes}};
    auto iterations = Json::array();
    report["lanes"] = lanes;
    report["graph"] = graph;
    for (int iteration = 0; iteration < repeat; ++iteration) {
        auto prepare = [&] {
            ninfer::PreparedPrompt prompt;
            if (kind == "tokens")
                prompt =
                    engine.prepare_tokens(Json::parse(input).get<std::vector<ninfer::TokenId>>());
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
            return prompt;
        };
        auto prompt = prepare();
        std::cerr << "native prefill tokens=" << prompt.summary().prompt_tokens << " backend="
                  << (mtp     ? "MTP K1"
                      : draft ? "DFlash2 K7"
                              : "ordinary")
                  << '\n';
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = output;
        request.execution.sampling.temperature    = 0.f;
        const auto batch_start                    = std::chrono::steady_clock::now();
        std::vector<ninfer::GenerationHandle> handles;
        std::vector<double> submitted_offsets;
        auto mark_submit = [&] {
            submitted_offsets.push_back(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_start)
                    .count());
        };
        mark_submit();
        handles.push_back(engine.submit(std::move(prompt), request));
        for (int lane = 1; lane < lanes; ++lane) {
            auto prepared = prepare();
            mark_submit();
            handles.push_back(engine.submit(std::move(prepared), request));
        }
        Json requests             = Json::array();
        double first_token        = 1.e30;
        std::size_t decode_tokens = 0;
        std::size_t row           = 0;
        for (auto& handle : handles) {
            auto result = handle.wait();
            first_token = std::min(first_token, submitted_offsets[row++] +
                                                    result.engine_timing.queue_wait_seconds +
                                                    result.timings.first_token_seconds);
            decode_tokens +=
                result.generated_token_ids.empty() ? 0 : result.generated_token_ids.size() - 1;
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
                {"kvmem_selection_calls", result.timings.kvmem.selection_calls},
                {"kvmem_restricted_calls", result.timings.kvmem.selection_restricted_calls}};
            requests.push_back(report["result"]);
        }
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_start).count();
        report["batch_wall_seconds"] = wall;
        report["aggregate_decode_tokens_per_second"] =
            decode_tokens / std::max(1.e-9, wall - first_token);
        report["requests"] = std::move(requests);
        iterations.push_back(
            {{"requests", report["requests"]},
             {"batch_wall_seconds", wall},
             {"aggregate_decode_tokens_per_second", report["aggregate_decode_tokens_per_second"]}});
        std::cerr << "native iteration=" << iteration + 1 << '/' << repeat
                  << " prefill_seconds=" << report["result"]["prefill_seconds"]
                  << " decode_seconds=" << report["result"]["decode_seconds"] << '\n';
    }
    if (repeat > 1) {
        if (lanes == 1) {
            Json rows = Json::array();
            for (const auto& iteration : iterations) rows.push_back(iteration["requests"][0]);
            report["iterations"] = rows;
        } else
            report["iterations"] = iterations;
    }
    const auto runtime           = engine.runtime_stats();
    report["execution"]          = {{"graph_captures", runtime.device_graph_captures},
                                    {"graph_replays", runtime.device_graph_replays},
                                    {"packed_prefill_batches", runtime.packed_prefill_batches},
                                    {"packed_prefill_tokens", runtime.packed_prefill_tokens},
                                    {"decode_rounds", runtime.decode_rounds},
                                    {"decode_rows", runtime.decode_row_rounds}};
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
        int context = 32768, output = 64, window = 6, repeat = 1, lanes = 1;
        bool draft = true, verify = false, graph = false, mtp = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value    = [&] {
                if (i + 1 == argc) throw std::invalid_argument("missing argument value");
                return std::string(argv[++i]);
            };
            if (a == "--help") {
                std::cout
                    << "Native GLM-5.3-Flash on three Hygon K100 devices; v3 .ninfer "
                       "only.\n--artifact FILE (--prompt TEXT | --prompt-file FILE | "
                       "--raw-text TEXT | --token-ids-file FILE)\n[--max-context 32768] "
                       "[--max-new-tokens 64] [--kvmem-window-pages 6]\n[--ordinary | --mtp | "
                       "--verify-native-greedy] [--report FILE]\n[--graph] [--lanes 1..8] "
                       "[--repeat N (ordinary/MTP "
                       "only)]\nGreedy Aggregate profile; "
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
            else if (a == "--repeat")
                repeat = std::stoi(value());
            else if (a == "--graph")
                graph = true;
            else if (a == "--lanes")
                lanes = std::stoi(value());
            else if (a == "--mtp") {
                mtp   = true;
                draft = false;
            } else if (a == "--ordinary")
                draft = false;
            else if (a == "--verify-native-greedy")
                verify = true;
            else
                throw std::invalid_argument("unknown argument: " + a);
        }
        if (artifact.empty() || kind.empty() || context < 1 || context > 32768 || output < 1 ||
            window < 0 || repeat < 1 || lanes < 1 || lanes > 8 || (mtp && (lanes != 1 || verify)) ||
            ((lanes > 1 || graph) && (draft || verify)) || (repeat > 1 && (draft || verify)) ||
            (!draft && verify))
            throw std::invalid_argument("invalid or missing native inference arguments");
        Json result;
        std::string verification_failure;
        if (verify) {
            auto ordinary    = run(artifact, input, kind, context, output, window, false);
            auto speculative = run(artifact, input, kind, context, output, window, true);
            bool equal =
                ordinary["result"]["generated_ids"] == speculative["result"]["generated_ids"];
            result = {{"ordinary", std::move(ordinary)},
                      {"dflash2", std::move(speculative)},
                      {"greedy_equal", equal}};
            if (!equal)
                verification_failure =
                    "native DFlash2 output differs from native ordinary generation";
            else if (result["dflash2"]["result"]["accepted_draft_tokens"].get<std::uint64_t>() == 0)
                verification_failure = "DFlash2 verification produced no positive acceptance";
        } else
            result = run(artifact, input, kind, context, output, window, draft, repeat, lanes,
                         graph, mtp);
        if (!report_path.empty()) {
            std::ofstream f(report_path);
            if (!f) throw std::runtime_error("cannot write native report");
            f << result.dump(2) << '\n';
        }
        std::cout << result.dump(2) << '\n';
        if (!verification_failure.empty()) throw std::runtime_error(verification_failure);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
