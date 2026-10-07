#include "ninfer/engine.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <thread>

namespace {
void require(bool v, const char* message) {
    if (!v) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(bool reuse = true) {
    ninfer::RequestOptions r;
    r.execution.requested_output_tokens = 24;
    r.execution.allow_prefix_reuse      = reuse;
    r.execution.sampling.temperature    = 0;
    r.execution.sampling.seed           = 42;
    r.stop.include_model_defaults       = false;
    return r;
}

ninfer::PromptInput media_prompt() {
    ninfer::PromptInput input;
    std::string records;
    for (unsigned i = 0; i < 720; ++i) {
        records += "Reference record: alpha beta gamma delta epsilon.\n";
    }
    input.messages.push_back(
        {.role  = ninfer::ChatRole::System,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = std::move(records)}}});
    ninfer::MessagePart image;
    image.kind               = ninfer::MessagePartKind::Media;
    image.media.kind         = ninfer::MediaKind::Image;
    image.media.media_type   = "image/x-portable-pixmap";
    image.media.source_name  = "native-pattern.ppm";
    const std::string header = "P6\n64 64\n255\n";
    image.media.bytes.assign(header.begin(), header.end());
    for (int i = 0; i < 64 * 64; ++i) {
        image.media.bytes.push_back(i & 255);
        image.media.bytes.push_back((i * 3) & 255);
        image.media.bytes.push_back((i * 7) & 255);
    }
    input.messages.push_back({.role  = ninfer::ChatRole::User,
                              .parts = {std::move(image),
                                        {.kind = ninfer::MessagePartKind::Text,
                                         .text = "Describe the pattern briefly."}}});
    input.options.enable_thinking   = false;
    input.context_cache.session_key = "native-media";
    return input;
}
} // namespace

int main(int argc, char** argv) {
    const auto* artifact = argc > 1 ? argv[1] : std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: explicit real model artifact required\n";
        return 77;
    }
    try {
        ninfer::EngineOptions o;
        o.artifact_path      = artifact;
        o.max_context        = 16384;
        o.prefill_chunk      = 512;
        o.kvmem_window_pages = 64;
        o.max_concurrency    = argc > 2 ? std::stoul(argv[2]) : 2;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(o.max_concurrency * 6144);
        o.kv_cache    = ninfer::KvCacheStorage::Fp8E4M3Row256;
        // Explicitly configurable quota for guarded local runs; the coverage/input gates stay
        // fixed. The default retains the four-GiB qualification profile.
        o.context_cache.host_capacity_bytes        = (argc > 5 ? std::stoull(argv[5]) : 4ULL) << 30;
        o.speculative.backend                        = ninfer::SpeculativeBackend::DFlash2;
        o.speculative.draft_tokens                   = 7;
        o.speculative.proposal_head                  = ninfer::ProposalHead::Optimized;
        o.enable_vision                              = argc > 3 && std::stoi(argv[3]);
        if (argc > 4) {
            const std::string backend = argv[4];
            require(backend == "none" || backend == "mtp" || backend == "dflash2",
                    "unsupported test backend");
            o.speculative.backend      = backend == "none"  ? ninfer::SpeculativeBackend::None
                                         : backend == "mtp" ? ninfer::SpeculativeBackend::Mtp
                                                            : ninfer::SpeculativeBackend::DFlash2;
            o.speculative.draft_tokens = backend == "none" ? 0 : backend == "mtp" ? 3 : 7;
            if (backend == "none") { o.speculative.proposal_head = ninfer::ProposalHead::Full; }
        }
        ninfer::Engine engine(o);
        const auto unit =
            engine.tokenize_text("Reference record: alpha beta gamma delta epsilon.\n");
        std::vector<ninfer::TokenId> tokens;
        while (tokens.size() < 10003) { tokens.insert(tokens.end(), unit.begin(), unit.end()); }
        tokens.resize(10003);
        const auto first = engine.generate(engine.prepare_tokens(tokens), request());
        require(first.generated_token_ids.size() == 24, "cold generation length");
        require(first.timings.kvmem.enabled, "sparse execution disabled");
        require(first.timings.kvmem.replay_tokens > 0, "cold query replay was not exercised");
        std::cout << "ok cold, replay=" << first.timings.kvmem.replay_tokens << std::endl;
        const auto warm = engine.generate(engine.prepare_tokens(tokens), request());
        require(warm.reused_prompt_tokens > 0, "warm cache did not reuse any prefix");
        require(warm.generated_token_ids == first.generated_token_ids,
                "warm sparse checkpoint changed greedy output");
        std::cout << "ok warm, reused=" << warm.reused_prompt_tokens << std::endl;
        std::vector<ninfer::GenerationHandle> handles;
        for (unsigned i = 0; i < o.max_concurrency; ++i) {
            auto continuation = tokens;
            const auto suffix = engine.tokenize_text("\nContinue record " + std::to_string(i));
            continuation.insert(continuation.end(), suffix.begin(), suffix.end());
            handles.push_back(engine.submit(engine.prepare_tokens(continuation), request()));
        }
        for (auto& handle : handles) {
            const auto result = handle.wait();
            require(result.generated_token_ids.size() == 24, "raw branch generation failed");
            // Raw tokens offer an anonymous private source. Native may consume it
            // for the first branch; shared retention is tested explicitly below.
            std::cout << "ok raw branch, reused=" << result.reused_prompt_tokens << std::endl;
        }
        auto shared = media_prompt();
        shared.messages.back().parts.erase(shared.messages.back().parts.begin());
        shared.messages.back().parts.back().text = "Continue record zero.";
        shared.context_cache.session_key.reset();
        shared.context_cache.allow_engine_automatic_shared_prefixes = false;
        shared.context_cache.markers.push_back({.after_message_count = 1});
        const auto shared_seed  = engine.generate(engine.prepare(shared), request());
        const auto shared_stats = engine.runtime_stats();
        std::cout << "shared seed captures=" << shared_stats.active_captures_completed
                  << " aborted=" << shared_stats.active_captures_aborted << std::endl;
        require(shared_seed.timings.kvmem.replay_tokens > 0,
                "shared fixture did not cross the sparse window");
        handles.clear();
        for (unsigned i = 0; i < o.max_concurrency; ++i) {
            auto branch                              = shared;
            branch.messages.back().parts.back().text = "Continue record " + std::to_string(i);
            handles.push_back(engine.submit(engine.prepare(branch), request()));
        }
        for (auto& handle : handles) {
            const auto result = handle.wait();
            std::cout << "shared fork reused=" << result.reused_prompt_tokens
                      << " generated=" << result.generated_token_ids.size() << std::endl;
            require(result.reused_prompt_tokens > o.kvmem_window_pages * 64 &&
                        result.generated_token_ids.size() == 24,
                    "multi-lane shared sparse prefix fork failed");
        }
        std::cout << "ok concurrent shared prefix forks" << std::endl;
        if (o.enable_vision) {
            auto input    = media_prompt();
            auto prepared = engine.prepare(input);
            require(prepared.summary().prompt_tokens > o.kvmem_window_pages * 64 &&
                        prepared.summary().has_media,
                    "vision fixture did not cross the sparse window");
            const auto cold_image = engine.generate(std::move(prepared), request());
            const auto warm_image = engine.generate(engine.prepare(input), request());
            std::cout << "vision warm reused=" << warm_image.reused_prompt_tokens
                      << " replay=" << warm_image.timings.kvmem.replay_tokens << std::endl;
            require(cold_image.generated_token_ids.size() == 24 &&
                        warm_image.generated_token_ids == cold_image.generated_token_ids &&
                        warm_image.reused_prompt_tokens > 0,
                    "vision sparse checkpoint changed greedy output");
            require(cold_image.timings.kvmem.replay_tokens > 0,
                    "vision query replay was not exercised");
            if (warm_image.reused_prompt_tokens > shared_seed.prompt.prompt_tokens) {
                // Only the image-containing checkpoint can carry this completed query.
                require(
                    warm_image.timings.kvmem.replay_tokens == 0 &&
                        warm_image.timings.kvmem.selection_calls == 0,
                    "identical input reselected history at its partial vision recovery boundary");
            } else {
                // Native may keep the proven shared system checkpoint instead of an optional
                // image checkpoint under pressure. That donor does not contain this query.
                require(warm_image.timings.kvmem.replay_tokens > 0 &&
                            warm_image.timings.kvmem.selection_calls > 0,
                        "fallback system checkpoint falsely inherited the image query selection");
                std::cout << "vision warm used system-only fallback checkpoint" << std::endl;
            }
            auto changed                              = input;
            changed.messages.back().parts.back().text = "Describe the geometry briefly.";
            const auto changed_question = engine.generate(engine.prepare(changed), request());
            require(changed_question.generated_token_ids.size() == 24 &&
                        changed_question.timings.kvmem.selection_calls > 0 &&
                        changed_question.timings.kvmem.replay_tokens > 0,
                    "a changed question falsely inherited the old canonical query selection");
            input.messages.push_back(
                {.role  = ninfer::ChatRole::Assistant,
                 .parts = {{.kind = ninfer::MessagePartKind::Text, .text = cold_image.content}}});
            input.messages.push_back({.role  = ninfer::ChatRole::User,
                                      .parts = {{.kind = ninfer::MessagePartKind::Text,
                                                 .text = "Which colors dominate?"}}});
            const auto appended = engine.generate(engine.prepare(input), request());
            require(appended.reused_prompt_tokens > 0 && appended.generated_token_ids.size() == 24,
                    "vision append did not preserve reusable state");
            require(appended.timings.kvmem.selection_calls > 0 &&
                        appended.timings.kvmem.replay_tokens > 0,
                    "a new conversation turn falsely inherited the old canonical query selection");
            std::cout << "ok vision cold/warm/append, reused=" << appended.reused_prompt_tokens
                      << std::endl;
        }
        // Cancel during the unpublished query probe, before a canonical Begin token. Poll
        // public completed-work boundaries instead of guessing a sleep duration.
        std::uint64_t retained_after_cancel = 0;
        for (unsigned repetition = 0; repetition < 2; ++repetition) {
            const auto before_cancel = engine.runtime_stats();
            bool cancelled_probe     = false;
            const ninfer::CancellationView cancellation([&] {
                const auto current = engine.runtime_stats();
                const bool at_probe =
                    current.prefilling_requests != 0 &&
                    current.computed_prefill_tokens >=
                        before_cancel.computed_prefill_tokens + tokens.size() - 512U &&
                    current.generated_tokens == before_cancel.generated_tokens;
                cancelled_probe |= at_probe;
                return cancelled_probe;
            });
            const auto cancelled = engine.generate(engine.prepare_tokens(tokens), request(false),
                                                   nullptr, cancellation);
            require(cancelled_probe && cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                        cancelled.generated_token_ids.empty(),
                    "unpublished query cancellation did not occur before output publication");
            const auto idle_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            for (;;) {
                const auto idle = engine.runtime_stats();
                if (idle.running_requests == 0 && idle.materializing_requests == 0) {
                    require(idle.host_context_reserved_bytes == 0 &&
                                idle.host_context_metadata_bytes <=
                                    before_cancel.host_context_metadata_bytes,
                            "cancelled probe retained pending allocations or private metadata");
                    // The first pressure event can create useful Host replicas of older retained
                    // checkpoints. With the same second probe that occupancy must settle, not grow.
                    if (repetition != 0) {
                        require(idle.host_context_occupied_bytes <= retained_after_cancel,
                                "repeated cancellation keeps growing retained Host occupancy");
                    }
                    retained_after_cancel = idle.host_context_occupied_bytes;
                    break;
                }
                if (std::chrono::steady_clock::now() >= idle_deadline) {
                    std::cerr << "cancel idle running=" << idle.running_requests
                              << " materializing=" << idle.materializing_requests
                              << '\n';
                }
                require(std::chrono::steady_clock::now() < idle_deadline,
                        "cancelled probe did not retire execution resources");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        const auto survivor = engine.generate(engine.prepare_tokens(tokens), request());
        require(survivor.generated_token_ids == first.generated_token_ids,
                "probe cancellation poisoned subsequent greedy inference");
        std::cout << "ok cancelled unpublished probe and surviving inference" << std::endl;
        const auto memory       = engine.memory_summary();
        const auto stats        = engine.runtime_stats();
        std::uint64_t generated = 0, computed = 0, decoded = 0, rounds = 0, replayed = 0;
        require(stats.lane_count == o.max_concurrency, "runtime lost configured lane visibility");
        for (unsigned lane = 0; lane < stats.lane_count; ++lane) {
            generated += stats.lanes[lane].generated_tokens;
            computed += stats.lanes[lane].computed_prefill_tokens;
            decoded += stats.lanes[lane].committed_decode_tokens;
            rounds += stats.lanes[lane].decode_rounds;
            replayed += stats.lanes[lane].replayed_tokens;
        }
        require(generated == stats.generated_tokens && computed == stats.computed_prefill_tokens &&
                    decoded == stats.committed_decode_tokens && rounds == stats.decode_row_rounds &&
                    replayed == stats.replayed_tokens,
                "lane counters duplicated or lost aggregate work across reuse and branching");
        require(memory.host_context_occupied_bytes <= *o.context_cache.host_capacity_bytes &&
                    memory.host_context_reserved_bytes == 0,
                "Native Host quota or pending reservations did not settle");
        std::cout << "ok cold/warm/fork/Native RAM, reused=" << warm.reused_prompt_tokens
                  << " host_bytes=" << memory.host_context_occupied_bytes << '\n';
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
    return 0;
}
