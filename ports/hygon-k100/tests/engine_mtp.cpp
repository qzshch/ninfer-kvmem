#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main(int argc, char** argv) try {
    require(argc >= 3, "explicit artifact and local report required; optional --graph --fp32-activations");
    EngineOptions options;
    options.artifact_path = argv[1];
    options.max_context = 1024;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(1024);
    options.kvmem_window_pages = 0;
    options.max_concurrency = 1;
    options.pending_timeout_ms = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--graph") options.use_cuda_graph = true;
        else if (flag == "--fp32-activations") options.bf16_activations = false;
        else throw std::invalid_argument("unknown MTP qualification argument");
    }
    std::array<std::vector<TokenId>, 3> prompts, expected;
    std::array<std::vector<TokenId>, 3> mtp_expected;
    const std::array<unsigned, 3> budgets{1, 2, 17};
    nlohmann::json modes = nlohmann::json::array();
    // Separate lifetimes avoid simultaneous baseline+MTP model residency.
    for (bool mtp : {false, true}) {
        options.speculative.backend = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
        options.speculative.draft_tokens = mtp ? 1 : 0;
        Engine engine(options);
        if (!mtp) {
            prompts[0] = engine.tokenize_text("The sky is");
            prompts[1] = engine.tokenize_text("Compute 19 + 23.");
            std::string text = "Remember marker RIVER 83. ";
            while (engine.tokenize_text(text).size() < 140) text += "alpha beta gamma. ";
            prompts[2] = engine.tokenize_text(text);
        }
        nlohmann::json runs = nlohmann::json::array();
        for (int repeat = 0; repeat < 2; ++repeat) for (unsigned r = 0; r < prompts.size(); ++r) {
            RequestOptions request;
            request.execution.requested_output_tokens = budgets[r];
            request.execution.sampling.temperature = 0;
            request.stop.include_model_defaults = false;
            const auto result = engine.generate(engine.prepare_tokens(prompts[r]), request);
            require(result.generated_token_ids.size() == budgets[r] && result.finish_reason == FinishReason::OutputLimit,
                    "MTP qualification output bound");
            require(!result.timings.kvmem.enabled && !result.timings.kvmem.selection_calls &&
                    !result.timings.kvmem.selection_restricted_calls, "MTP baseline enabled KVMem");
            if (!mtp && !repeat) expected[r] = result.generated_token_ids;
            else require(result.generated_token_ids == expected[r], "native MTP/ordinary greedy mismatch or reset leak");
            if (mtp) {
                require(result.speculative.enabled && result.speculative.backend == SpeculativeBackend::Mtp &&
                        result.speculative.draft_window == 1, "MTP K1 was not selected");
                require((budgets[r] <= 2) == (result.speculative.rounds == 0), "MTP final C1 boundary or proposal path");
                if (budgets[r] == 2) require(result.speculative.fallback_steps == 1, "MTP final C1 fallback missing");
                if (!repeat) mtp_expected[r] = result.generated_token_ids;
                else require(result.generated_token_ids == mtp_expected[r], "MTP repeated request state leak");
            }
            const auto decode_tokens = budgets[r] - 1;
            runs.push_back({{"case", r}, {"repeat", repeat}, {"prompt_tokens", prompts[r].size()},
                {"generated_ids", result.generated_token_ids}, {"prefill_seconds", result.timings.prefill_seconds},
                {"decode_seconds", result.timings.decode_seconds}, {"total_seconds", result.timings.total_seconds},
                {"decode_seconds_per_delivered_token", decode_tokens ? nlohmann::json(result.timings.decode_seconds / decode_tokens) : nlohmann::json(nullptr)},
                {"draft_rounds", result.speculative.rounds}, {"drafted_tokens", result.speculative.drafted_tokens},
                {"accepted_tokens", result.speculative.accepted_tokens}});
        }
        // A stop at the first verifier output must suppress the bonus and leave
        // the next request independent of the discarded target/MTP suffix.
        RequestOptions stop;
        stop.execution.requested_output_tokens = 17;
        stop.execution.sampling.temperature = 0;
        stop.stop.include_model_defaults = false;
        stop.stop.token_ids = {expected[2][1]};
        const auto stopped = engine.generate(engine.prepare_tokens(prompts[2]), stop);
        const auto at = std::find(expected[2].begin(), expected[2].end(), expected[2][1]);
        const std::vector<TokenId> stopped_ids(expected[2].begin(), at + 1);
        require(stopped.finish_reason == FinishReason::StopToken && stopped.generated_token_ids == stopped_ids,
                "MTP stop-token prefix published bonus output");
        auto tail = stop;
        tail.stop.token_ids.clear();
        tail.execution.requested_output_tokens = 1;
        const std::vector<TokenId> near_capacity(1023, prompts[0].front());
        const auto one = engine.generate(engine.prepare_tokens(near_capacity), tail);
        require(one.generated_token_ids.size() == 1 && one.finish_reason == FinishReason::OutputLimit,
                "MTP last prompt/one-output capacity boundary");
        tail.execution.requested_output_tokens = 4;
        bool oversized = false;
        try { engine.generate(engine.prepare_tokens(near_capacity), tail); }
        catch (const std::invalid_argument&) { oversized = true; }
        require(oversized, "MTP context admission accepted oversized output budget");
        auto cancelled = engine.submit(engine.prepare_tokens(prompts[2]), stop);
        // Destruction of an unconsumed handle requests cancellation. The worker
        // then drains or aborts before admitting the following request.
        cancelled = {};
        tail.execution.requested_output_tokens = 17;
        const auto reused = engine.generate(engine.prepare_tokens(prompts[2]), tail);
        require(reused.generated_token_ids == expected[2], "MTP post-stop/cancel/context-tail request leaked state");
        const auto memory = engine.memory_summary();
        require(!memory.host_kv_occupied_bytes && !memory.host_state_occupied_slots, "MTP unexpected offload");
        modes.push_back({{"backend", mtp ? "native MTP K1" : "native ordinary"}, {"runs", runs},
            {"weights_bytes", memory.weights.capacity_bytes}, {"state_bytes", memory.sequence.capacity_bytes},
            {"workspace_peak_bytes", memory.workspace_logical_peak_bytes}});
    }
    nlohmann::json report{{"passed", true}, {"graph", options.use_cuda_graph},
        {"bf16_activations", options.bf16_activations}, {"kvmem_window_pages", 0},
        {"parallelism", "PP"}, {"verification", "layer-major two-position K1 with accepted-prefix commit; speed requires measured A/B"},
        {"modes", modes}};
    std::ofstream output(argv[2]); require(bool(output), "cannot open local MTP report");
    output << report.dump(2) << '\n'; require(bool(output), "cannot save local MTP report");
    std::cout << "PASS: native ordinary/MTP greedy, output bounds, request reuse and no-KVMem; costs recorded without a speed assertion\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
