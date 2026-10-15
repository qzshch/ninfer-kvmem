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
    ledger.complete(decode); ledger.complete(temporal);
    ledger.fail();
    rejects([&] { ledger.reserve(batch(c)); });
    rejects([&] { ledger.reset(); });
    std::cout << "PASS: independent frames, write exclusion, completion order, cancellation drain, generations\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
