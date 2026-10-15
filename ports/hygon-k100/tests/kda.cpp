// Independent complete FP64 recurrence oracle, including the published state.
#include "ninfer/ops/kimi_delta_attention.h"
#include "core/device.h"
#include "ops/kda_ref.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using ninfer::test::kda_ref::Inputs;
constexpr float scale = 1.f / 11.313708498984761f;

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

std::uint16_t encode(float x) {
    auto u = std::bit_cast<std::uint32_t>(x);
    return (u + 0x7fffu + ((u >> 16) & 1)) >> 16;
}

float decode(std::uint16_t x) { return std::bit_cast<float>(static_cast<std::uint32_t>(x) << 16); }

struct Input {
    DeviceBuffer data;

    Input(const std::vector<float>& values, bool bf16) : data(values.size() * (bf16 ? 2 : 4)) {
        if (bf16) {
            std::vector<std::uint16_t> bits;
            for (auto x : values) bits.push_back(encode(x));
            data.copy_from_host(bits.data(), data.bytes);
        } else
            data.copy_from_host(values.data(), data.bytes);
    }
};

Inputs inputs(int qheads, int heads, int tokens, int mode) {
    Inputs a;
    a.qk_heads    = qheads;
    a.value_heads = heads;
    a.tokens      = tokens;
    a.q.resize(128 * qheads * tokens);
    a.k.resize(a.q.size());
    a.v.resize(128 * heads * tokens);
    a.g.resize(a.v.size());
    a.beta.resize(heads * tokens);
    a.a_log.resize(heads);
    a.dt_bias.resize(128 * heads);
    a.state.resize(128 * 128 * heads);
    std::mt19937 rng(12071 + mode + heads + tokens);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    for (auto* v : {&a.q, &a.k, &a.v, &a.g, &a.beta, &a.a_log, &a.dt_bias, &a.state})
        for (auto& x : *v) x = uniform(rng);
    for (auto& x : a.state) x *= .02f;
    if (mode == 1) {
        for (auto& x : a.q) x *= 1e-5f;
        for (auto& x : a.k) x *= 1e-5f;
    }
    if (mode == 2) {
        for (std::size_t i = 0; i < a.g.size(); ++i) a.g[i] = i % 2 ? 80.f : -80.f;
    }
    if (mode == 3) {
        std::fill(a.q.begin(), a.q.end(), 0.f);
        std::fill(a.state.begin(), a.state.end(), 0.f);
    }
    for (auto* v : {&a.q, &a.k, &a.v, &a.g, &a.beta})
        for (auto& x : *v) x = decode(encode(x));
    return a;
}

double check(const std::vector<double>& actual, const std::vector<double>& expected, bool state) {
    require(actual.size() == expected.size(), "KDA oracle shape mismatch");
    double error2 = 0, reference2 = 0, peak = 0, worst = 0;
    for (auto v : expected) peak = std::max(peak, std::abs(v));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "nonfinite KDA result");
        auto error = std::abs(actual[i] - expected[i]);
        error2 += error * error;
        reference2 += expected[i] * expected[i];
        const double allowance = state ? 5e-6 + 3e-5 * peak : 1e-5 + 4e-3 * peak;
        worst                  = std::max(worst, error / allowance);
    }
    const auto relative = reference2 ? std::sqrt(error2 / reference2) : std::sqrt(error2);
    if (relative > (state ? 1.5e-5 : 2.2e-3) || worst > 1) {
        std::fprintf(stderr, "state=%d relative_L2=%g gross_ratio=%g\n", state, relative, worst);
        throw std::runtime_error("KDA complete FP64 oracle criterion failed");
    }
    return relative;
}

void run(DeviceContext& ctx, int qheads, int heads, int tokens, int mode, float lower, bool alias,
         bool negative = false, bool packed = false) {
    auto a = inputs(qheads, heads, tokens, mode);
    if (negative) {
        for (auto& x : a.a_log) x = -std::exp(x);
        a.negative_rate = true;
    }
    auto oracle = ninfer::test::kda_ref::evaluate(a, lower, scale);
    Input q(a.q, true), k(a.k, true), v(a.v, true), g(a.g, true), beta(a.beta, true),
        al(a.a_log, false), dt(a.dt_bias, false), initial(a.state, false);
    DeviceBuffer result(a.v.size() * 2 + 32), published(a.state.size() * 4 + 32);
    result.fill(0xa5);
    published.fill(0xa5);
    Tensor qt(q.data.p, DType::BF16, {128, qheads, tokens}),
        kt(k.data.p, DType::BF16, {128, qheads, tokens});
    Tensor vt(v.data.p, DType::BF16, {128, heads, tokens}),
        gt(g.data.p, DType::BF16, {128, heads, tokens}),
        bt(beta.data.p, DType::BF16, {heads, tokens});
    Tensor at(al.data.p, DType::FP32, {heads}), dtb(dt.data.p, DType::FP32, {128, heads});
    Tensor si(initial.data.p, DType::FP32, {128, 128, heads});
    Tensor so(alias ? initial.data.p : static_cast<std::byte*>(published.p) + 16, DType::FP32,
              {128, 128, heads});
    Tensor out(static_cast<std::byte*>(result.p) + 16, DType::BF16, {128, heads, tokens});
    WorkspaceArena work(packed ? std::max<std::size_t>(256, ops::kimi_delta_attention_packed_workspace_capacity_bytes(qheads, heads, 1, tokens)) : 256);
    DeviceBuffer selectors, bounds;
    if (packed) {
        require(alias, "packed KDA publishes directly into the selected state");
        selectors = DeviceBuffer(4); bounds = DeviceBuffer(8);
        const int slot = 0, offsets[2]{0, tokens};
        selectors.copy_from_host(&slot, 4); bounds.copy_from_host(offsets, 8);
        Tensor pool(si.data, DType::FP32, {128, 128, heads, 1}), ids(selectors.p, DType::I32, {1}), ranges(bounds.p, DType::I32, {2});
        ops::kimi_delta_attention_packed_update(qt, kt, vt, gt, bt, at, dtb, lower, scale,
            work, pool, ids, ranges, out, ctx.stream,
            negative ? ops::KdaDecayParameter::NegativeRate : ops::KdaDecayParameter::LogRate);
    } else ops::kimi_delta_attention(
        qt, kt, vt, gt, bt, at, dtb, lower, scale, work, si, so, out, ctx.execution_view(),
        negative ? ops::KdaDecayParameter::NegativeRate : ops::KdaDecayParameter::LogRate);
    ctx.synchronize();
    std::vector<std::uint16_t> observed(a.v.size() + 16);
    result.copy_to_host(observed.data(), result.bytes);
    for (int i = 0; i < 8; ++i)
        require(observed[i] == 0xa5a5 && observed[observed.size() - 1 - i] == 0xa5a5,
                "KDA output guard changed");
    std::vector<double> y;
    for (std::size_t i = 0; i < a.v.size(); ++i) y.push_back(decode(observed[i + 8]));
    check(y, oracle.out, false);
    std::vector<float> state(a.state.size());
    if (alias)
        initial.data.copy_to_host(state.data(), state.size() * 4);
    else {
        published.copy_to_host(state.data(), state.size() * 4, 16);
        std::vector<unsigned char> guarded(published.bytes);
        published.copy_to_host(guarded.data(), guarded.size());
        for (int i = 0; i < 16; ++i)
            require(guarded[i] == 0xa5 && guarded[guarded.size() - 1 - i] == 0xa5,
                    "KDA state guard changed");
        std::vector<float> source(a.state.size());
        initial.data.copy_to_host(source.data(), source.size() * 4);
        require(source == a.state, "distinct KDA overwrote input state");
    }
    check(std::vector<double>(state.begin(), state.end()), oracle.final_state, true);
}

void run_batch(DeviceContext& ctx, int qheads, int heads, bool negative = false,
                bool packed = false, bool short_spans = false, const std::vector<int>& lengths = {}) {
    const int batch = lengths.empty() ? 3 : static_cast<int>(lengths.size());
    constexpr int slot_count = 10;
    const std::array<int, 8> mapping{8, 2, 6, 0, 9, 4, 1, 7};
    const std::vector<std::int32_t> slots(mapping.begin(), mapping.begin() + batch);
    const std::size_t state_size = 128 * 128 * heads, vector_size = 128 * heads;
    std::vector<float> pool(slot_count * state_size);
    for (std::size_t i = 0; i < pool.size(); ++i) pool[i] = .001f * (static_cast<int>(i % 31) - 15);
    std::vector<Inputs> rows;
    std::vector<float> q, k, v, g, b;
    std::vector<int> offsets{0};
    for (int row = 0; row < batch; ++row) {
        const int length = packed ? (!lengths.empty() ? lengths[row] : (short_spans ? std::array{1, 2, 5}[row] : std::array{1, 8, 33}[row])) : 1;
        auto a = inputs(qheads, heads, length, row);
        offsets.push_back(offsets.back() + length);
        a.negative_rate = negative;
        if (negative && !row)
            for (auto& x : a.a_log) x = -std::exp(x);
        if (row) {
            a.a_log   = rows[0].a_log;
            a.dt_bias = rows[0].dt_bias;
        }
        a.state.assign(pool.begin() + slots[row] * state_size,
                       pool.begin() + (slots[row] + 1) * state_size);
        const auto append = [](auto& dst, const auto& src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };
        append(q, a.q);
        append(k, a.k);
        append(v, a.v);
        append(g, a.g);
        append(b, a.beta);
        rows.push_back(std::move(a));
    }
    Input qi(q, true), ki(k, true), vi(v, true), gi(g, true), bi(b, true), al(rows[0].a_log, false),
        dt(rows[0].dt_bias, false), state(pool, false);
    const int total = offsets.back();
    DeviceBuffer selectors(slots.size() * 4), bounds(offsets.size() * 4), result(v.size() * 2);
    selectors.copy_from_host(slots.data(), selectors.bytes);
    bounds.copy_from_host(offsets.data(), bounds.bytes);
    Tensor qt(qi.data.p, DType::BF16, {128, qheads, packed ? total : 1, packed ? 1 : batch}),
        kt(ki.data.p, DType::BF16, {128, qheads, packed ? total : 1, packed ? 1 : batch});
    Tensor vt(vi.data.p, DType::BF16, {128, heads, packed ? total : 1, packed ? 1 : batch}),
        gt(gi.data.p, DType::BF16, {128, heads, packed ? total : 1, packed ? 1 : batch});
    Tensor bt(bi.data.p, DType::BF16, {heads, packed ? total : 1, packed ? 1 : batch}),
        at(al.data.p, DType::FP32, {heads}),
        dtb(dt.data.p, DType::FP32, {128, heads});
    Tensor states(state.data.p, DType::FP32, {128, 128, heads, slot_count}),
        ids(selectors.p, DType::I32, {batch});
    Tensor out(result.p, DType::BF16, {128, heads, packed ? total : 1, packed ? 1 : batch});
    const auto parameter = negative ? ops::KdaDecayParameter::NegativeRate
                                    : ops::KdaDecayParameter::LogRate;
    WorkspaceArena work(packed ? std::max<std::size_t>(256, ops::kimi_delta_attention_packed_workspace_capacity_bytes(qheads, heads, batch, total)) : 256);
    if (packed) {
        Tensor ranges(bounds.p, DType::I32, {batch + 1});
        ops::kimi_delta_attention_packed_update(qt, kt, vt, gt, bt, at, dtb, -5, scale,
                                                work, states, ids, ranges, out, ctx.stream, parameter);
    } else
        ops::kimi_delta_attention_batch_update(qt, kt, vt, gt, bt, at, dtb, -5, scale,
                                               states, ids, out, ctx.stream, parameter);
    ctx.synchronize();
    std::vector<float> after(pool.size());
    state.data.copy_to_host(after.data(), state.data.bytes);
    std::vector<std::uint16_t> y(v.size());
    result.copy_to_host(y.data(), result.bytes);
    for (int slot = 0; slot < slot_count; ++slot) {
        auto found = std::find(slots.begin(), slots.end(), slot);
        if (found == slots.end()) {
            require(std::equal(pool.begin() + slot * state_size,
                               pool.begin() + (slot + 1) * state_size,
                               after.begin() + slot * state_size),
                    "KDA overwrote inactive pooled state");
        } else {
            const auto row = found - slots.begin();
            auto oracle    = ninfer::test::kda_ref::evaluate(rows[row], -5, scale);
            std::vector<double> output;
            for (std::size_t i = 0; i < rows[row].v.size(); ++i)
                output.push_back(decode(y[offsets[row] * vector_size + i]));
            check(output, oracle.out, false);
            check(std::vector<double>(after.begin() + slot * state_size,
                                      after.begin() + (slot + 1) * state_size),
                  oracle.final_state, true);
        }
    }
}
} // namespace

int main() try {
    DeviceContext ctx;
    unsigned cases = 0;
    for (bool alias : {false, true}) {
        for (int tokens : {1, 8, 33}) {
            run(ctx, 64, 64, tokens, 0, -5, alias);
            ++cases;
        }
        for (int mode : {1, 2, 3}) {
            run(ctx, 1, 5, 8, mode, -5, alias);
            ++cases;
        }
        run(ctx, 1, 5, 65, 0, 0, alias);
        ++cases;
        run(ctx, 1, 5, 1024, 0, -5, alias, true);
        ++cases;
        for (int tokens : {1, 8, 33}) {
            run(ctx, 64, 64, tokens, 0, -5, alias, true);
            ++cases;
        }
    }
    run_batch(ctx, 2, 4);
    ++cases;
    run_batch(ctx, 64, 64);
    ++cases;
    run_batch(ctx, 2, 4, true);
    ++cases;
    run_batch(ctx, 64, 64, true);
    ++cases;
    for (bool negative : {false, true}) {
        run_batch(ctx, 2, 4, negative, true);
        run_batch(ctx, 64, 64, negative, true);
        cases += 2;
    }
    for (int heads : {20, 24}) for (int tokens : {1, 8, 15, 16, 17, 33}) {
        run(ctx, heads, heads, tokens, 0, -5, true, true, true);
        ++cases;
    }
    for (bool negative : {false, true}) {
        run_batch(ctx, 2, 4, negative, true, true);
        ++cases;
        for (int heads : {20, 24}) {
            run(ctx, heads, heads, 1, 2, -5, true, negative, true);
            ++cases;
        }
    }
    for (const auto& spans : {std::vector<int>(8, 16), std::vector<int>{1,2,3,4,8,16,17,65},
                               std::vector<int>{1,1,1,1,1,1,1,128}}) {
        run_batch(ctx, 2, 4, true, true, false, spans);
        ++cases;
    }
    for (int mode : {1, 2, 3}) {
        run(ctx, 1, 5, 65, mode, -5, true, true, true);
        ++cases;
    }
    for (float lower : {-5.f, 0.f}) {
        run(ctx, 1, 5, 1024, 0, lower, true, true, true);
        ++cases;
    }
    std::printf("PASS: %u KDA cases; independent full FP64 output/state oracle; H64; K7 width8; "
                "distinct/alias state; guards; ragged FP32 time tiles and recurrent routes\n",
                cases);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
