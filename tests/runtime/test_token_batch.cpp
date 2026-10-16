#include "runtime/contract/token_batch.h"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer::runtime;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F> void rejects(F&& f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid input unit accepted");
}

int main() try {
    const std::vector<std::int32_t> ids(137, 123);
    const std::array<TokenSpan, 3> spans{{{7, 126, 0, 1}, {1, 0, 1, 128}, {4, 120, 129, 8}}};
    const std::array<std::int32_t, 2> output{{0, 136}};
    const TokenBatch prefill{TokenBatchKind::Prefill, ids, spans, output};
    prefill.validate(8, 256, 1024);
    TokenBatch{TokenBatchKind::Prefill, ids, spans, {}}.validate(8, 256, 1024);
    rejects([&] { TokenBatch{TokenBatchKind::Decode, ids, spans, output}.validate(8, 256, 1024); });
    rejects([&] { prefill.validate(4, 256, 1024); });
    rejects([&] { prefill.validate(8, 127, 1024); });
    rejects([&] { prefill.validate(8, 256, 136); });
    auto bad = spans;
    bad[1].lane = bad[0].lane;
    rejects([&] { TokenBatch{TokenBatchKind::Prefill, ids, bad, output}.validate(8, 256, 1024); });
    bad = spans;
    bad[2].begin = 128;
    rejects([&] { TokenBatch{TokenBatchKind::Prefill, ids, bad, output}.validate(8, 256, 1024); });
    bad = spans;
    bad[2].input_frontier = std::numeric_limits<std::uint32_t>::max();
    rejects([&] { TokenBatch{TokenBatchKind::Prefill, ids, bad, output}.validate(8, 256, 1024); });
    for (auto rows : {std::array<std::int32_t, 2>{0, 0}, {136, 0}, {0, 7}, {0, -1}})
        rejects([&] { TokenBatch{TokenBatchKind::Prefill, ids, spans, rows}.validate(8, 256, 1024); });
    const std::array<std::int32_t, 3> next{{10, 20, 30}}, all_rows{{0, 1, 2}};
    const std::array<TokenSpan, 3> decode{{{7, 127, 0, 1}, {1, 128, 1, 1}, {4, 128, 2, 1}}};
    TokenBatch{TokenBatchKind::Decode, next, decode, all_rows}.validate(8, 256, 8);
    rejects([&] { TokenBatch{TokenBatchKind::Decode, next, decode, {}}.validate(8, 256, 8); });
    const std::array<int, 2> pair{10, 20}, both{0, 1};
    const TokenSpan temporal{0, 254, 0, 2};
    TokenBatch{TokenBatchKind::VerifyK1, pair, {&temporal, 1}, both}.validate(1, 256, 2);
    rejects([&] { TokenBatch{TokenBatchKind::Prefill, pair, {&temporal, 1}, both}.validate(1, 256, 2); });
    rejects([&] { TokenBatch{TokenBatchKind::VerifyK1, pair, {&temporal, 1}, {}}.validate(1, 256, 2); });
    rejects([&] { TokenBatch{TokenBatchKind::VerifyK1, pair, {&temporal, 1}, both}.validate(1, 255, 2); });
    rejects([&] { TokenBatch{TokenBatchKind::VerifyK1, pair, {&temporal, 1}, both}.validate(1, 256, 1); });
    const std::array<TokenSpan, 2> independent{{{0, 0, 0, 1}, {1, 0, 1, 1}}};
    rejects([&] { TokenBatch{TokenBatchKind::VerifyK1, pair, independent, both}.validate(2, 256, 2); });
    std::cout << "PASS token spans: ragged time, independent lanes, causal frontiers, output selection\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
