
#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
using namespace ninfer;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char** argv) try {
    require(argc == 3 || (argc == 4 && std::string(argv[3]) == "--tp-ep"), "artifact and report required; optional --tp-ep");
    EngineOptions options;
    options.artifact_path      = argv[1];
    options.max_context        = 1024;
    options.kv_capacity        = KvCapacityPolicy::explicit_capacity(1024);
    options.max_concurrency    = 8;
    options.decode_frames      = 2;
    options.tensor_expert_parallel = argc == 4;
    options.use_cuda_graph     = true;
    options.kvmem_window_pages = 6;
    options.pending_timeout_ms = 0;
    options.prefill_pack_projections = true;
    Engine engine(options);
    std::vector<std::vector<TokenId>> prompts;
    for (const auto* text :
         {"The sky is", "2 + 3 =", "Write the next number: 7, 8, 9,", "The capital of France is"})
        prompts.push_back(engine.tokenize_text(text));
    RequestOptions request;
    request.execution.requested_output_tokens = 12;
    request.execution.sampling.temperature    = 0;
    request.stop.include_model_defaults       = false;
    std::vector<GenerationResult> reference;
    nlohmann::json serial_timings = nlohmann::json::array(),
                   batch_timings  = nlohmann::json::array();
    for (int repeat = 0; repeat < 3; ++repeat) {
        double decode = 0, wall = 0;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            auto result = engine.generate(engine.prepare_tokens(prompts[i]), request);
            decode += result.timings.decode_seconds;
            wall += result.timings.total_seconds;
            if (!repeat)
                reference.push_back(result);
            else
                require(result.generated_token_ids == reference[i].generated_token_ids,
                        "serial reference state reset");
        }
        serial_timings.push_back({{"decode_tokens_per_second", 44 / decode},
                                  {"service_tokens_per_second", 48 / wall},
                                  {"wall_seconds", wall}});
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto started = std::chrono::steady_clock::now();
        std::vector<GenerationHandle> batch;
        std::vector<double> submitted;
        for (int i = 0; i < 8; ++i) {
            submitted.push_back(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
            batch.push_back(engine.submit(engine.prepare_tokens(prompts[i % 4]), request));
        }
        double first             = 1.e30;
        nlohmann::json latencies = nlohmann::json::array();
        for (int i = 0; i < 8; ++i) {
            auto result = batch[i].wait();
            require(result.generated_token_ids == reference[i % 4].generated_token_ids,
                    "heterogeneous full-model batch vs independent references");
            first = std::min(first, submitted[i] + result.engine_timing.queue_wait_seconds +
                                        result.timings.first_token_seconds);
            latencies.push_back({{"prefill_seconds", result.timings.prefill_seconds},
                                 {"decode_seconds", result.timings.decode_seconds}});
        }
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        batch_timings.push_back({{"aggregate_decode_tokens_per_second", 88 / (wall - first)},
                                 {"service_tokens_per_second", 96 / wall},
                                 {"wall_seconds", wall},
                                 {"requests", latencies}});
    }
    std::vector<GenerationHandle> handles;
    for (int i = 0; i < 8; ++i) {
        auto r                              = request;
        r.execution.requested_output_tokens = 8 + i % 4;
        handles.push_back(engine.submit(engine.prepare_tokens(prompts[i % 4]), r));
    }
    auto expired =
        engine.submit(engine.prepare_tokens(prompts[0]), request, OutputConsumerMode::Aggregate, {},
                      std::chrono::steady_clock::now() - std::chrono::seconds(1));
    for (int i = 0; i < 8; ++i) {
        auto result          = handles[i].wait();
        const auto& expected = reference[i % 4].generated_token_ids;
        require(result.generated_token_ids ==
                    std::vector<TokenId>(expected.begin(), expected.begin() + 8 + i % 4),
                "ragged compact batch leaked state or changed token sequence");
        require(result.finish_reason == FinishReason::OutputLimit, "ragged request output limit");
    }
    bool deadline = false;
    try {
        expired.wait();
    } catch (const std::runtime_error&) { deadline = true; }
    require(deadline, "pending deadline not enforced");
    // Cancellation while execution is active, followed by physical lane reuse.
    std::atomic<bool> cancel                       = false;
    auto long_request                              = request;
    long_request.execution.requested_output_tokens = 128;
    auto cancelled = engine.submit(engine.prepare_tokens(prompts[0]), long_request);
    auto future =
        std::async(std::launch::async, [handle = std::move(cancelled), &cancel]() mutable {
            return handle.wait(nullptr, CancellationView([&] { return cancel.load(); }));
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    cancel = true;
    require(future.get().finish_reason == FinishReason::Cancelled, "active cancellation");
    { auto abandoned = engine.submit(engine.prepare_tokens(prompts[1]), long_request); }
    auto reused = engine.generate(engine.prepare_tokens(prompts[0]), request);
    require(reused.generated_token_ids == reference[0].generated_token_ids,
            "reused lane retained previous KDA/KV state");
    const auto& ids        = reference[0].generated_token_ids;
    auto stopped           = request;
    stopped.stop.token_ids = {ids[3]};
    auto stop              = engine.generate(engine.prepare_tokens(prompts[0]), stopped);
    auto first             = std::find(ids.begin(), ids.end(), ids[3]);
    require(stop.finish_reason == FinishReason::StopToken &&
                stop.generated_token_ids == std::vector<TokenId>(ids.begin(), first + 1),
            "batch custom token stop");
    require(engine.is_available(), "ordinary cancellation/deadline poisoned Engine");
    const auto stat   = engine.runtime_stats();
    const auto memory = engine.memory_summary();
    require(stat.device_graph_captures > 0 &&
                stat.device_graph_replays > stat.device_graph_captures,
            "Graph not actually replayed");
    require(stat.decode_row_rounds > stat.decode_rounds && stat.packed_prefill_batches > 0,
            "cross-request matrices not evaluated");
    require(!memory.host_kv_occupied_bytes && !memory.host_state_occupied_slots,
            "unexpected host KV/state offload");
    nlohmann::json report = {
        {"passed", true},
        {"cases",
         {"serial independent references", "eight mixed length requests", "varying output budgets",
          "pending deadline", "active cancellation", "abandoned handle", "lane reuse",
          "custom token stop", "graph replay", "true batch rows", "no offload"}},
        {"graph_captures", stat.device_graph_captures},
        {"graph_replays", stat.device_graph_replays},
        {"decode_rounds", stat.decode_rounds},
        {"decode_rows", stat.decode_row_rounds},
        {"packed_prefill_batches", stat.packed_prefill_batches}};
    report["heterogeneous_workload"] = {
        {"input_route", "raw tokens"},
        {"prompts",
         {"The sky is", "2 + 3 =", "Write the next number: 7, 8, 9,", "The capital of France is"}},
        {"context_capacity", 1024},
        {"output_budget_each", 12},
        {"model_default_stops", false},
        {"configured_lanes", 8},
        {"serial_four_requests", serial_timings},
        {"batched_eight_requests_two_per_prompt", batch_timings}};
    std::ofstream(argv[2]) << report.dump(2) << '\n';
    std::cout << report.dump() << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
