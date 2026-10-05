#include "ninfer/engine.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage out;
    out.role = role;
    out.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    return out;
}

ninfer::PromptInput prompt(std::string question, bool tool_tail = false) {
    ninfer::PromptInput out;
    out.options.enable_thinking = false;
    std::string history = "Use the user's instruction. Reference notes:\n";
    for (int i = 0; i < 1100; ++i) { history += "alpha beta gamma delta "; }
    out.messages.push_back(message(ninfer::ChatRole::System, std::move(history)));
    out.messages.push_back(message(ninfer::ChatRole::User, std::move(question)));
    // Protocol automatic writes are rounded to a complete pre-query chunk.
    // Storage-level tests cover partial-page/block snapshot ownership separately.
    out.context_cache.markers.push_back({.after_message_count = 2,
        .kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic});
    if (tool_tail) {
        auto assistant = message(ninfer::ChatRole::Assistant, "");
        assistant.tool_calls.push_back({.id = "call_reference", .name = "reference",
                                       .arguments_json = "{}"});
        out.messages.push_back(std::move(assistant));
        auto tool = message(ninfer::ChatRole::Tool,
            "Reference result: " + std::string(8000, 'x'));
        tool.tool_call_id = "call_reference";
        out.messages.push_back(std::move(tool));
        out.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"reference","parameters":{"type":"object","properties":{}}}})");
    }
    return out;
}

ninfer::RequestOptions request(bool reuse = true, std::uint32_t output_tokens = 24) {
    ninfer::RequestOptions out;
    out.execution.requested_output_tokens = output_tokens;
    out.execution.sampling.temperature = 0.0F;
    out.execution.sampling.seed = 42;
    out.execution.allow_prefix_reuse = reuse;
    out.stop.include_model_defaults = false;
    return out;
}

// Opt-in, real-model regression for a checkpoint offer produced while a
// disjoint lane's Host restore still owns the global context transaction.
void exercise_overlap_prefetch(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 262144;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(81920);
    options.max_concurrency                  = 2;
    options.prefill_chunk                    = 1024;
    options.prefill_token_budget             = 1024;
    options.kvmem_window_pages               = 576;
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens         = 7;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.enable_vision                    = true;
    options.cpu_gpu_overlap                  = true;
    options.cache_prefetch                   = true;
    options.cache_layerwise_restore = std::getenv("NINFER_TEST_LAYERWISE_RESTORE") != nullptr;
    if (const char* directory = std::getenv("NINFER_TEST_KV_FILE_DIR")) {
        options.context_cache.kv_file_directory = directory;
    }
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 4;
    options.context_cache.host_kv_capacity_bytes    = 8ULL << 30;
    options.context_cache.max_private_continuations = 2;
    options.context_cache.max_shared_prefixes       = 8;
    const bool file_mode  = !options.context_cache.kv_file_directory.empty();
    const bool layer_mode = options.cache_layerwise_restore;
    ninfer::Engine engine(std::move(options));
    const auto input = [](std::string tag, int repetitions, bool counting = false) {
        ninfer::PromptInput out;
        out.options.enable_thinking = false;
        std::string prefix = "Public archive " + tag + ". Follow the final instruction exactly.\n";
        for (int i = 0; i < repetitions; ++i) { prefix += " alpha beta gamma delta epsilon."; }
        out.messages.push_back(message(ninfer::ChatRole::System, std::move(prefix)));
        out.messages.push_back(
            message(ninfer::ChatRole::User,
                    counting ? "Print every integer from 1 to 100000 separated by spaces, in "
                               "order. Do not skip, summarize, or write code."
                             : "Reply with exactly READY-" + tag + "."));
        out.context_cache.markers.push_back(
            {.after_message_count = 2,
             .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
             .evidence            = ninfer::SharedCandidateEvidence::DefaultAutomatic});
        return out;
    };
    const auto generation = [](std::uint32_t tokens) {
        auto out                        = request(true, tokens);
        out.stop.include_model_defaults = true;
        return out;
    };
    auto original = input("A-17319", 8000);
    std::cerr << "phase overlap cold\n";
    const auto cold = engine.generate(engine.prepare(original), generation(32));
    if (layer_mode) {
        const auto before    = engine.runtime_stats().host_work.layerwise_restore_admissions;
        bool observed        = false;
        auto interrupted     = engine.submit(engine.prepare(original), generation(32));
        const auto cancelled = interrupted.wait(
            nullptr, ninfer::CancellationView([&] {
                observed = observed ||
                           engine.runtime_stats().host_work.layerwise_restore_admissions > before;
                return observed;
            }));
        require(observed && cancelled.finish_reason == ninfer::FinishReason::Cancelled,
                "cancellation did not exercise an admitted pending layer payload");
        require(engine.runtime_stats().running_requests == 0,
                "cancelled restoring owner retained an active lane");
    }
    if (file_mode) {
        const auto before = engine.runtime_stats().file_cache.read_bytes;
        bool observed     = false;
        auto survivor     = engine.submit(engine.prepare(input("FILE-CANCEL-SURVIVOR", 10, true)),
                                          generation(1024));
        auto interrupted  = engine.submit(engine.prepare(original), generation(32));
        const auto cancelled =
            interrupted.wait(nullptr, ninfer::CancellationView([&] {
                                 const auto stats = engine.runtime_stats();
                                 observed = observed || (stats.materializing_requests != 0 &&
                                                         stats.file_cache.read_bytes > before &&
                                                         stats.file_cache.pending_reads != 0);
                                 return observed;
                             }));
        require(observed && cancelled.finish_reason == ninfer::FinishReason::Cancelled,
                "file cancellation did not observe actual reads during materialization");
        require(survivor.wait().generated_token_ids.size() == 1024,
                "file restore cancellation damaged the disjoint decoder");
        const auto drained = engine.runtime_stats().file_cache;
        require(drained.pending_reads == 0 && drained.pending_writes == 0,
                "cancelled file materialization retained unfinished IO");
        std::cout << "file-cancel observed reads during materialization\n";
    }
    std::cerr << "phase overlap warm\n";
    const auto warm = engine.generate(engine.prepare(original), generation(32));
    require(cold.prompt.prompt_tokens > 36864, "prefetch fixture did not cross the window");
    require(warm.reused_prompt_tokens > 36864, "prefetch fixture did not retain a long prefix");
    require(warm.generated_token_ids == cold.generated_token_ids, "warm target output changed");
    std::cerr << "phase overlap pressure\n";
    auto pressure_a = engine.submit(engine.prepare(input("P-11", 8000)), generation(64));
    auto pressure_b = engine.submit(engine.prepare(input("P-12", 8000)), generation(64));
    require(!pressure_a.wait().generated_token_ids.empty(), "first pressure request failed");
    require(!pressure_b.wait().generated_token_ids.empty(), "second pressure request failed");
    for (int round = 0; round < 2; ++round) {
        std::cerr << "phase overlap revisit " << round << '\n';
        auto unrelated =
            engine.submit(engine.prepare(input("UNRELATED-" + std::to_string(round), 500, true)),
                          generation(4096));
        auto revisit        = engine.submit(engine.prepare(original), generation(32));
        const auto restored = revisit.wait();
        require(restored.reused_prompt_tokens > 36864, "pressure lost the long Host prefix");
        require(restored.generated_token_ids == cold.generated_token_ids,
                "Host restore changed target output");
        require(unrelated.wait().generated_token_ids.size() == 4096,
                "disjoint owner failed during Host restore");
    }
    const auto stats = engine.runtime_stats();
    if (layer_mode) {
        require(stats.host_work.layerwise_restore_admissions > 0,
                "aligned Host owner was never admitted before all layers completed");
        require(stats.host_work.layerwise_restore_admissions ==
                    stats.host_work.layerwise_restore_completions,
                "a layerwise payload lease was left unsettled");
    }
    require(stats.host_work.cache_prefetch_units > 0, "no execution began during outstanding H2D");
    require(stats.host_work.deferred_capture_offers == stats.host_work.deferred_capture_resumptions,
            "a checkpoint offer was lost or left deferred");
    if (std::getenv("NINFER_TEST_REQUIRE_CAPTURE_DEFERRAL") != nullptr) {
        require(stats.host_work.deferred_capture_offers > 0,
                "capture-deferral path was not exercised");
    }
    std::cout << "overlap-prefetch units=" << stats.host_work.cache_prefetch_units
              << " deferred=" << stats.host_work.deferred_capture_offers << '\n';
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) { std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n"; return 77; }
    try {
        if (std::getenv("NINFER_TEST_OVERLAP_PREFETCH") != nullptr) {
            exercise_overlap_prefetch(artifact);
            std::cout << "ok\n";
            return 0;
        }
        ninfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 8192;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        options.max_concurrency = 2;
        options.prefill_chunk = 256;
        options.kvmem_window_pages = 32;
        options.kv_cache = std::getenv("NINFER_TEST_FP8") != nullptr
                               ? ninfer::KvCacheStorage::Fp8E4M3Row256
                               : ninfer::KvCacheStorage::Int8Group64;
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        if (const char* spec = std::getenv("NINFER_TEST_SPEC")) {
            if (std::string(spec) == "mtp") {
                options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                options.speculative.draft_tokens = 3;
            } else if (std::string(spec) == "none") {
                options.speculative.backend = ninfer::SpeculativeBackend::None;
                options.speculative.draft_tokens = 0;
                options.speculative.proposal_head = ninfer::ProposalHead::Full;
            } else {
                throw std::invalid_argument("NINFER_TEST_SPEC must be mtp or none");
            }
        }
        options.context_cache.device_state_slots = 2;
        options.context_cache.host_state_slots = 4;
        options.context_cache.host_kv_capacity_bytes = 1ULL << 30;
        ninfer::Engine engine(std::move(options));
        if (std::getenv("NINFER_TEST_COLD_CONTROL") != nullptr) {
            const auto alone = engine.generate(engine.prepare(prompt("Print READY.")),
                                                request(false, 96));
            for (int round = 0; round < 3; ++round) {
                auto a = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 96));
                auto b = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 96));
                const auto ar = a.wait();
                const auto br = b.wait();
                std::cout << "cold pair " << round << " same-as-single="
                          << (ar.generated_token_ids == alone.generated_token_ids) << ','
                          << (br.generated_token_ids == alone.generated_token_ids) << ':';
                for (const auto token : ar.generated_token_ids) { std::cout << ' ' << token; }
                std::cout << " |";
                for (const auto token : br.generated_token_ids) { std::cout << ' ' << token; }
                std::cout << '\n';
            }
            return 0;
        }
        std::cerr << "phase cold long\n";
        const auto cold = engine.generate(engine.prepare(prompt("Print READY.")), request());
        std::cerr << "phase warm long\n";
        const auto warm = engine.generate(engine.prepare(prompt("Print READY.")), request());
        require(cold.prompt.prompt_tokens > 2048, "fixture is not longer than its window");
        require(warm.reused_prompt_tokens > 2048, "Host-backed long prefix was not reused");
        require(warm.generated_token_ids == cold.generated_token_ids,
                "warm checkpoint changed deterministic target output");
        // Compare prefill's target token before batched speculative decoding can
        // introduce the existing batch-shape-dependent floating-point differences.
        // Full single-lane output and tool-tail replay remain exact checks above/below.
        std::cerr << "phase cached parallel\n";
        auto first = engine.submit(engine.prepare(prompt("Print READY.")), request(true, 1));
        auto second = engine.submit(engine.prepare(prompt("Print READY.")), request(true, 1));
        const auto cached_first = first.wait();
        const auto cached_second = second.wait();
        std::cerr << "phase cold parallel\n";
        auto root_first = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 1));
        auto root_second = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 1));
        const auto uncached_first = root_first.wait();
        const auto uncached_second = root_second.wait();
        const auto show = [](const char* label, const auto& result) {
            std::cerr << label << " reused=" << result.reused_prompt_tokens << " tokens:";
            for (const auto token : result.generated_token_ids) { std::cerr << ' ' << token; }
            std::cerr << '\n';
        };
        show("single", cold);
        show("cached0", cached_first);
        show("cached1", cached_second);
        show("root0", uncached_first);
        show("root1", uncached_second);
        require(cached_first.generated_token_ids == uncached_first.generated_token_ids,
                "first cached lane disagrees with two-lane cold control");
        require(cached_second.generated_token_ids == uncached_second.generated_token_ids,
                "second cached lane disagrees with two-lane cold control");
        std::cerr << "phase cold tool replay\n";
        const auto replay_cold = engine.generate(
            engine.prepare(prompt("Print COMPLETE.", true)), request());
        require(replay_cold.reused_prompt_tokens == 0, "new tool schema reused an incompatible prefix");
        std::cerr << "phase warm tool replay\n";
        const auto replay_warm = engine.generate(
            engine.prepare(prompt("Print COMPLETE.", true)), request());
        require(replay_warm.reused_prompt_tokens > 2048,
                "tool-tail replay lost its safe pre-query prefix");
        require(replay_cold.generated_token_ids == replay_warm.generated_token_ids,
                "cached activation changed long tool-tail replay output");
        // Reusing one long checkpoint and then publishing a longer checkpoint must
        // account for borrowed pages evicted by the rolling prefill window. A saved
        // admission claim can be smaller than the newly private pages at publication.
        std::cerr << "phase growing cached conversation\n";
        auto growing = prompt("Print READY.", true);
        std::string continuation;
        for (int i = 0; i < 320; ++i) { continuation += "alpha beta gamma delta "; }
        growing.messages.push_back(message(ninfer::ChatRole::Assistant, std::move(continuation)));
        growing.messages.push_back(message(ninfer::ChatRole::User, "Print EXTENDED."));
        growing.context_cache.markers.push_back({.after_message_count = 6,
            .kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic});
        const auto growing_warm = engine.generate(engine.prepare(growing), request(true, 1));
        require(growing_warm.reused_prompt_tokens > 2048,
                "growing conversation did not reuse its Host-backed prefix");
        const auto growing_cold = engine.generate(engine.prepare(growing), request(false, 1));
        require(growing_warm.generated_token_ids == growing_cold.generated_token_ids,
                "growing cached conversation changed the target prefill token");

        // Destroy a live cold-prefill owner while another lane remains in flight.
        std::cerr << "phase cancellation\n";
        auto survivor = engine.submit(engine.prepare(prompt("Print SURVIVE.")), request(false));
        {
            auto abandoned = engine.submit(engine.prepare(prompt("Print CANCELLED.")), request(false));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (engine.runtime_stats().prefilling_requests < 2 &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            require(engine.runtime_stats().prefilling_requests == 2,
                    "two admitted requests never owned prefill concurrently");
        }
        require(survivor.wait().generated_token_ids.size() == 24,
                "cancelling one prefill broke the other lane");
        const auto after = engine.generate(engine.prepare(prompt("Print READY.")), request());
        require(after.generated_token_ids == cold.generated_token_ids,
                "cancellation polluted a retained shared checkpoint");
        // Root checkpoints can also leave a deferred GDN fork. Admission must
        // wait for that fork to settle instead of invalidating the next request.
        const auto short_prompt = [](int id) {
            ninfer::PromptInput out;
            out.options.enable_thinking = false;
            out.messages.push_back(message(ninfer::ChatRole::User,
                "Run identifier " + std::to_string(id) +
                "; ignore it. Print consecutive integers starting at 1. Continue to 10000."));
            out.context_cache.markers.push_back({.after_message_count = 1,
                .kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic});
            return out;
        };
        std::cerr << "phase short root forks\n";
        require(engine.generate(engine.prepare(short_prompt(0)), request(true, 128))
                    .generated_token_ids.size() == 128, "short retained warmup failed");
        for (int round = 0; round < 2; ++round) {
            auto a = engine.submit(engine.prepare(short_prompt(1 + round * 2)), request(true, 128));
            auto b = engine.submit(engine.prepare(short_prompt(2 + round * 2)), request(true, 128));
            require(a.wait().generated_token_ids.size() == 128,
                    "root checkpoint fork invalidated first concurrent admission");
            require(b.wait().generated_token_ids.size() == 128,
                    "root checkpoint fork invalidated second concurrent admission");
        }
        std::cout << "ok long-prefix=" << warm.reused_prompt_tokens
                  << " prompt=" << warm.prompt.prompt_tokens << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
    return 0;
}
