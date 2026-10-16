#include "models/glm5_next/frame_ledger.h"
#include <array>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace models::glm5_next;
void require(bool condition) { if (!condition) throw std::runtime_error("frame ownership invariant"); }
template <class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::logic_error&) { rejected = true; }
    require(rejected);
}
int main() try {
    FrameLedger ledger(4, 2), other(4, 2);
    std::array<int, 1> token{12}, outputs{0};
    std::array<runtime::TokenSpan, 1> a{{{3, 0, 0, 1}}}, b{{{0, 0, 0, 1}}}, c{{{2, 0, 0, 1}}};
    auto batch = [&](const auto& spans) { return runtime::TokenBatch{runtime::TokenBatchKind::Decode, token, spans, outputs}; };
    const auto first = ledger.reserve(batch(a));
    require(ledger.frontier(3) == 0 && ledger.busy(3));
    rejects([&] { ledger.reserve(batch(a)); });
    rejects([&] { ledger.reset_lane(3); });
    rejects([&] { ledger.reset(); });
    const auto second = ledger.reserve(batch(b));
    rejects([&] { ledger.reserve(batch(c)); });
    rejects([&] { other.complete(first); });
    // Out-of-order completion cannot advance an unrelated request's frontier.
    ledger.complete(second);
    require(ledger.frontier(0) == 1 && ledger.frontier(3) == 0 && ledger.busy(3));
    const auto third = ledger.reserve(batch(c));
    rejects([&] { ledger.complete(second); });
    ledger.complete(first);
    require(ledger.frontier(3) == 1 && ledger.frontier(2) == 0);
    ledger.complete(third);
    rejects([&] { ledger.reserve(batch(a)); }); // stale committed frontier
    ledger.reset_lane(3);
    const auto reused = ledger.reserve(batch(a));
    rejects([&] { ledger.complete(first); }); // old generation after slot reuse
    ledger.complete(reused); // cancellation drains physical writes, then reset
    ledger.reset_lane(3);
    ledger.reset();
    // Only the primary frame accepts temporal work; no silent scratch overflow.
    auto prefill = batch(a); prefill.kind = runtime::TokenBatchKind::Prefill;
    const auto temporal = ledger.reserve(prefill);
    rejects([&] { auto x = batch(b); x.kind = runtime::TokenBatchKind::Prefill; ledger.reserve(x); });
    const auto decode = ledger.reserve(batch(b));
    ledger.complete(decode);
    auto next = b; next[0].input_frontier = 1;
    const auto continuation = ledger.reserve(batch(next));
    require(ledger.busy(3) && ledger.frontier(0) == 1);
    ledger.complete(continuation);
    ledger.complete(temporal);
    // A full set of ready rows fits one Decode owner, leaving the temporal
    // frame unreserved. Every logical lane commits exactly one frontier step.
    FrameLedger compact(8, 2);
    std::array<int, 8> ready_tokens{}, ready_outputs{};
    std::array<runtime::TokenSpan, 8> ready_spans{};
    const std::array<unsigned, 8> lane_order{7,0,6,1,5,2,4,3};
    for (unsigned r = 0; r < 8; ++r) {
        ready_tokens[r] = 12 + r; ready_outputs[r] = r;
        ready_spans[r] = {lane_order[r], 0, r, 1};
    }
    const auto packed = compact.reserve({runtime::TokenBatchKind::Decode, ready_tokens, ready_spans, ready_outputs});
    require(packed.slot == 1 && compact.active(1) && !compact.active(0));
    for (unsigned lane = 0; lane < 8; ++lane) require(compact.busy(lane) && compact.frontier(lane) == 0);
    compact.complete(packed);
    for (unsigned lane = 0; lane < 8; ++lane) require(!compact.busy(lane) && compact.frontier(lane) == 1);
    ledger.fail();
    rejects([&] { ledger.reserve(batch(c)); });
    rejects([&] { ledger.reset(); });
    FrameLedger speculative(1, 1);
    const std::array<int, 2> pair{12, 34}, rows{0, 1};
    for (unsigned count : {0u, 1u, 2u, 0u, 2u}) {
        const unsigned before = speculative.frontier(0);
        const runtime::TokenSpan span{0, before, 0, 2};
        const runtime::TokenBatch verify{runtime::TokenBatchKind::VerifyK1, pair, {&span, 1}, rows};
        verify.validate(1, 128, 2);
        const auto ticket = speculative.reserve(verify);
        require(speculative.busy(0) && speculative.frontier(0) == static_cast<int>(before));
        rejects([&] { speculative.reset(); });
        rejects([&] { speculative.complete(ticket); });
        rejects([&] { speculative.reserve(verify); });
        rejects([&] { other.complete_prefix(ticket, 1); });
        rejects([&] { speculative.complete_prefix(ticket, 3); });
        require(speculative.busy(0) && speculative.frontier(0) == static_cast<int>(before));
        speculative.complete_prefix(ticket, count);
        require(!speculative.busy(0) && speculative.frontier(0) == static_cast<int>(before + count));
        rejects([&] { speculative.complete_prefix(ticket, count); });
    }
    std::cout << "PASS: independent frames, write exclusion, completion order, cancellation drain, generations\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
