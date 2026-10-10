#include "ninfer/engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

ninfer::PromptInput prompt(const std::string& archive) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    for (const auto& [role, text] : {
        std::pair{ninfer::ChatRole::User, archive},
        std::pair{ninfer::ChatRole::Developer, std::string("Follow the requested output format.")},
        std::pair{ninfer::ChatRole::User, std::string("Return the route identifier.")}}) {
        input.messages.push_back({.role = role,
            .parts = {{.kind = ninfer::MessagePartKind::Text, .text = text}}});
    }
    input.context_cache.session_key = "sparse-constrained-regression";
    return input;
}

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) return 77;
    try {
        const std::string backend = argc > 1 ? argv[1] : "dflash2";
        ninfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 16384;
        options.max_concurrency = 2;
        options.prefill_chunk = 512;
        options.kvmem_window_pages = 64;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(12288);
        options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        options.context_cache.host_capacity_bytes = 4ULL << 30;
        options.context_cache.host_memory = ninfer::HostContextMemory::Pageable;
        if (backend == "dflash2") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens = 7;
            options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        } else if (backend == "mtp") {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = 3;
            options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        } else require(backend == "none", "unknown backend");
        ninfer::Engine engine(options);
        std::string archive;
        for (int i = 0; i < 800; ++i) archive += "Archived note: the earlier topic was scheduling.\n";
        const auto tokens = engine.count_tokens(prompt(archive));
        require(tokens > options.kvmem_window_pages * 64 && tokens < options.max_context - 128,
                "fixture must actually cross the sparse window without truncation");
        ninfer::RequestOptions first, second;
        std::string answer_a, answer_b;
        for (int i = 0; i < 8; ++i) {
            answer_a += "LANE_A_ALPHA_0123456789_";
            answer_b += "LANE_B_BETA_9876543210_";
        }
        first.constraint = ninfer::OutputConstraint::choice({answer_a});
        second.constraint = ninfer::OutputConstraint::regex(answer_b);
        for (auto* request : {&first, &second}) {
            request->execution.requested_output_tokens = 256;
            request->execution.sampling.temperature = 0;
            request->execution.sampling.seed = 1234;
        }
        for (int round = 0; round < 3; ++round) {
            std::cerr << backend << ": round=" << round << '\n';
            auto a = engine.submit(engine.prepare(prompt(archive)), first);
            auto b = engine.submit(engine.prepare(prompt(archive)), second);
            const auto ra = a.wait();
            const auto rb = b.wait();
            require(ra.content == answer_a && rb.content == answer_b, "lane masks or grammar state leaked");
            require(ra.finish_reason == ninfer::FinishReason::StopToken &&
                        rb.finish_reason == ninfer::FinishReason::StopToken, "grammar EOS changed");
            if (round > 0) {
                require(ra.reused_prompt_tokens > 0 && rb.reused_prompt_tokens > 0,
                        "constrained requests lost Native prefix reuse");
            }
        }
        const auto stats = engine.runtime_stats();
        require(stats.lane_count == 2 && engine.is_available(), "lane reactivation or Engine failed");
        require(stats.decode_row_rounds > stats.decode_rounds,
                "test must execute a real two-row decode batch, not just queue two requests");
        const auto memory = engine.memory_summary();
        require(memory.host_context_reserved_bytes == 0, "completed constrained rounds left reservations");
        std::cout << backend << ": sparse cold/warm masks, native reuse and reactivation passed; tokens="
                  << tokens << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
