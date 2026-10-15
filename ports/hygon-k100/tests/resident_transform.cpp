#include "ninfer/ops/resident_transform.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;

void require(bool x, const char* m) {
    if (!x) throw std::runtime_error(m);
}

std::uint16_t word(double x) {
    auto b = std::bit_cast<unsigned>(static_cast<float>(x));
    return (b + 0x7fffu + ((b >> 16) & 1)) >> 16;
}

double unword(std::uint16_t x) { return std::bit_cast<float>(static_cast<unsigned>(x) << 16); }

double bf(double x) { return unword(word(x)); }

double sig(double x) { return 1 / (1 + std::exp(-x)); }

float value(int i) {
    unsigned b = static_cast<unsigned>(i) * 747796405u + 2891336453u;
    b          = ((b >> ((b >> 28) + 4)) ^ b) * 277803737u;
    b ^= b >> 22;
    return bf(std::bit_cast<float>((b & 0x807fffffu) | ((124 + b % 6) << 23)));
}

struct WeightData {
    DeviceBuffer bytes;
    WeightParent parent;
    Tensor view;

    WeightData(const std::vector<float>& x, std::vector<std::uint64_t> shape, bool bfloat)
        : bytes(x.size() * (bfloat ? 2 : 4)),
          parent{
              weight_geometry(bfloat ? QType::BF16 : QType::FP32, QuantLayout::Contiguous, shape),
              static_cast<const std::byte*>(bytes.p)} {
        if (bfloat) {
            std::vector<std::uint16_t> b;
            for (auto v : x) b.push_back(word(v));
            bytes.copy_from_host(b.data(), bytes.bytes);
        } else
            bytes.copy_from_host(x.data(), bytes.bytes);
        int dims[4] = {1, 1, 1, 1};
        for (unsigned j = 0; j < shape.size(); ++j) dims[j] = shape[shape.size() - 1 - j];
        view = weight_tensor(WeightView{shape, {{&parent, 0, x.size()}}},
                             {dims[0], dims[1], dims[2], dims[3]});
    }
};

DeviceBuffer upload(const std::vector<float>& x) {
    DeviceBuffer d(x.size() * 4);
    d.copy_from_host(x.data(), d.bytes);
    return d;
}

std::vector<float> download(const DeviceBuffer& d) {
    std::vector<float> x(d.bytes / 4);
    d.copy_to_host(x.data(), d.bytes);
    return x;
}

void check(double observed, double ideal, bool rounded, const char* what) {
    double floor = 0;
    if (rounded) {
        auto w = static_cast<int>(word(ideal));
        floor  = std::abs(unword(w) - ideal);
        for (int delta : {-1, 1})
            if (w + delta >= 0 && w + delta <= 65535)
                floor = std::min(floor, std::abs(unword(w + delta) - ideal));
    }
    if (!std::isfinite(observed) ||
        std::abs(observed - ideal) > floor + 8e-6 * (1 + std::abs(ideal))) {
        std::fprintf(stderr, "%s observed=%.12g ideal=%.12g floor=%.12g\n", what, observed, ideal,
                     floor);
        throw std::runtime_error("resident-transform FP64 per-element criterion");
    }
}

void norm(DeviceContext& ctx, int n, int rows, bool center, bool gate, bool rounded, bool bfloat,
          bool alias) {
    std::vector<float> x(n * rows), g(n), b(n), z(x.size());
    for (int i = 0; i < n * rows; ++i) {
        x[i] = value(i);
        z[i] = value(i + 35) * 4;
    }
    for (int c = 0; c < n; ++c) {
        g[c] = bfloat ? bf(.25 + std::sin(c) * .137) : static_cast<float>(.25 + std::sin(c) * .137);
        b[c] = bfloat ? bf(std::cos(c) * .173) : static_cast<float>(std::cos(c) * .173);
    }
    WeightData wg(g, {static_cast<unsigned>(n)}, bfloat), wb(b, {static_cast<unsigned>(n)}, bfloat);
    auto dx = upload(x), dz = upload(z);
    DeviceBuffer dy(x.size() * 4);
    auto* o = alias ? static_cast<float*>(dx.p) : static_cast<float*>(dy.p);
    ops::resident_norm(static_cast<float*>(dx.p), &wg.view, center ? &wb.view : nullptr,
                       gate ? static_cast<float*>(dz.p) : nullptr, o, n, rows, 1e-5f, center,
                       rounded, ctx.stream);
    ctx.synchronize();
    std::vector<float> y(x.size());
    CUDA_CHECK(cudaMemcpy(y.data(), o, y.size() * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) {
        double mean = 0, var = 0;
        if (center) {
            for (int c = 0; c < n; ++c) mean += x[r * n + c];
            mean /= n;
        }
        for (int c = 0; c < n; ++c) var += (x[r * n + c] - mean) * (x[r * n + c] - mean);
        double inv = 1 / std::sqrt(var / n + double(1e-5f));
        for (int c = 0; c < n; ++c) {
            double v = (x[r * n + c] - mean) * inv * g[c] + (center ? b[c] : 0);
            if (gate) v *= sig(z[r * n + c]);
            check(y[r * n + c], v, rounded, "norm");
        }
    }
}

void conv(DeviceContext& ctx, int channels, int t, bool alias) {
    std::vector<float> x(channels * t), s(channels * 3), w(channels * 4);
    for (int i = 0; i < channels * t; ++i) x[i] = value(i);
    for (int i = 0; i < channels * 3; ++i) s[i] = value(i + 91);
    for (int i = 0; i < channels * 4; ++i) w[i] = static_cast<float>(std::sin(i * 1.7) * .23);
    WeightData weight(w, {static_cast<unsigned>(channels), 1, 4}, false);
    auto dx = upload(x), ds = upload(s);
    DeviceBuffer dy(x.size() * 4);
    auto* o = alias ? static_cast<float*>(dx.p) : static_cast<float*>(dy.p);
    ops::resident_causal_conv(static_cast<float*>(dx.p), weight.view, static_cast<float*>(ds.p), o,
                              channels, t, ctx.stream);
    ctx.synchronize();
    std::vector<float> y(x.size());
    CUDA_CHECK(cudaMemcpy(y.data(), o, y.size() * 4, cudaMemcpyDeviceToHost));
    auto state = download(ds);
    for (int c = 0; c < channels; ++c) {
        double a = s[c * 3], b = s[c * 3 + 1], d = s[c * 3 + 2];
        for (int j = 0; j < t; ++j) {
            double v   = x[j * channels + c],
                   sum = a * w[c * 4] + b * w[c * 4 + 1] + d * w[c * 4 + 2] + v * w[c * 4 + 3],
                   pre = bf(sum);
            if (std::abs(y[j * channels + c] - bf(pre * sig(pre))) >
                8e-6 * (1 + std::abs(pre * sig(pre))))
                std::fprintf(
                    stderr,
                    "conv diagnostic channels=%d T=%d alias=%d c=%d token=%d raw=%.17g pre=%.17g "
                    "a=%.17g b=%.17g d=%.17g v=%.17g weights=%.17g,%.17g,%.17g,%.17g\n",
                    channels, t, alias, c, j, sum, pre, a, b, d, v, double(w[c * 4]),
                    double(w[c * 4 + 1]), double(w[c * 4 + 2]), double(w[c * 4 + 3]));
            check(y[j * channels + c], pre * sig(pre), true, "causal conv/SiLU");
            a = b;
            b = d;
            d = v;
        }
        require(state[c * 3] == a && state[c * 3 + 1] == b && state[c * 3 + 2] == d,
                "conv final state");
    }
}

void elementwise(DeviceContext& ctx, int n, float limit) {
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = value(i) * 8;
        b[i] = value(i + 99) * 8;
    }
    auto da = upload(a), db = upload(b);
    DeviceBuffer out(n * 4), packed(n * 2), unpacked(n * 4);
    ops::resident_swiglu(static_cast<float*>(da.p), static_cast<float*>(db.p),
                         static_cast<float*>(out.p), n, limit, ctx.stream);
    ops::pack_bf16(static_cast<float*>(da.p), static_cast<std::uint16_t*>(packed.p), n, ctx.stream);
    ops::unpack_bf16(static_cast<std::uint16_t*>(packed.p), static_cast<float*>(unpacked.p), n,
                     ctx.stream);
    ctx.synchronize();
    auto y = download(out);
    require(download(unpacked) == a, "pack/unpack BF16 exact values");
    for (int i = 0; i < n; ++i) {
        double g = limit > 0 ? std::min<double>(a[i], limit) : a[i],
               u = limit > 0 ? std::clamp<double>(b[i], -limit, limit) : b[i];
        check(y[i], bf(g * sig(g)) * u, true, "SwiGLU");
    }
    ops::resident_add(static_cast<float*>(da.p), static_cast<float*>(db.p),
                      static_cast<float*>(da.p), n, ctx.stream);
    ctx.synchronize();
    y = download(da);
    for (int i = 0; i < n; ++i) check(y[i], double(a[i]) + b[i], true, "residual add");
}

void router(DeviceContext& ctx, int experts, int active, int rows, int mode) {
    std::vector<float> l(experts * rows), bias(experts);
    for (int i = 0; i < experts * rows; ++i) l[i] = mode ? 0 : value(i) * 2;
    for (int i = 0; i < experts; ++i) bias[i] = mode ? 0 : static_cast<float>(std::sin(i) * .025);
    if (mode == 2)
        for (int i = 0; i < experts * rows; ++i) l[i] = i % 2 ? 128.f : -128.f;
    auto dl = upload(l), db = upload(bias);
    DeviceBuffer ids(rows * active * 4), weights(rows * active * 4);
    ops::resident_router(static_cast<float*>(dl.p), static_cast<float*>(db.p),
                         static_cast<std::int32_t*>(ids.p), static_cast<float*>(weights.p), experts,
                         active, rows, 2.5f, ctx.stream);
    ctx.synchronize();
    auto w = download(weights);
    std::vector<int> got(rows * active);
    ids.copy_to_host(got.data(), ids.bytes);
    for (int r = 0; r < rows; ++r) {
        std::vector<int> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return sig(l[r * experts + a]) + bias[a] > sig(l[r * experts + b]) + bias[b];
        });
        double sum = 1e-20;
        for (int j = 0; j < active; ++j) sum += sig(l[r * experts + order[j]]);
        for (int j = 0; j < active; ++j) {
            require(got[r * active + j] == order[j],
                    "router selected IDs / bias-only ranking / ties");
            check(w[r * active + j], 2.5 * sig(l[r * experts + order[j]]) / sum, false,
                  "router weights");
        }
    }
    constexpr int hidden = 4096;
    std::vector<float> values(rows * active * hidden), shared(rows * hidden);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = value(i);
    for (std::size_t i = 0; i < shared.size(); ++i) shared[i] = value(i + 98);
    auto dv = upload(values), ds = upload(shared);
    DeviceBuffer output(shared.size() * 4);
    ops::resident_moe_sum(static_cast<float*>(dv.p), static_cast<float*>(weights.p),
                          static_cast<float*>(ds.p), static_cast<float*>(output.p), hidden, active,
                          rows, ctx.stream);
    ctx.synchronize();
    auto y = download(output);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < hidden; ++c) {
            double sum = shared[r * hidden + c];
            for (int j = 0; j < active; ++j)
                sum += double(values[(r * active + j) * hidden + c]) * w[r * active + j];
            check(y[r * hidden + c], sum, true, "weighted expert sum");
        }
}

void argmax(DeviceContext& ctx) {
    constexpr int n = 154880;
    std::vector<float> v(n * 8, -3);
    for (int r = 0; r < 8; ++r) {
        v[r * n + 100 + r]  = 4;
        v[r * n + 1000 + r] = 4;
    }
    auto d = upload(v);
    DeviceBuffer ids(32);
    ops::resident_argmax(static_cast<float*>(d.p), static_cast<std::int32_t*>(ids.p), n, 8,
                         ctx.stream);
    ctx.synchronize();
    std::vector<int> y(8);
    ids.copy_to_host(y.data(), 32);
    for (int r = 0; r < 8; ++r) require(y[r] == 100 + r, "argmax exact lower-ID tie");
    v[71]         = NAN;
    v[n + 17]     = INFINITY;
    v[2 * n + 99] = -INFINITY;
    d.copy_from_host(v.data(), d.bytes);
    ops::resident_argmax(static_cast<float*>(d.p), static_cast<std::int32_t*>(ids.p), n, 8,
                         ctx.stream);
    ctx.synchronize();
    ids.copy_to_host(y.data(), 32);
    for (int r = 0; r < 8; ++r)
        require(y[r] == (r < 3 ? -1 : 100 + r), "argmax nonfinite rows must fail explicitly");
}

void metadata(DeviceContext& ctx, int rows) {
    constexpr int width = 128;
    std::vector<float> x(rows * width);
    std::vector<std::uint16_t> b(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        unsigned bits = static_cast<unsigned>(i) * 747796405u + 2891336453u;
        x[i]          = std::bit_cast<float>((bits & 0x807fffffu) | ((124 + bits % 6) << 23));
        b[i]          = word(x[i]);
    }
    auto dx = upload(x);
    DeviceBuffer db(b.size() * 2), sum(width * 4), mean(width * 4);
    db.copy_from_host(b.data(), db.bytes);
    ops::resident_sum_rows(static_cast<float*>(dx.p), static_cast<float*>(sum.p), width, rows,
                           ctx.stream);
    ops::resident_mean_bf16(static_cast<std::uint16_t*>(db.p), static_cast<float*>(mean.p), width,
                            rows, ctx.stream);
    ctx.synchronize();
    auto s = download(sum), m = download(mean);
    for (int c = 0; c < width; ++c) {
        double total = 0, btotal = 0;
        for (int r = 0; r < rows; ++r) {
            total += x[r * width + c];
            btotal += unword(b[r * width + c]);
        }
        check(s[c], total, false, "query retrieval sum");
        check(m[c], btotal / rows, false, "block retrieval mean");
    }
}
} // namespace

int main() {
    try {
        DeviceContext ctx;
        unsigned cases = 0;
        for (int n : {1, 128, 4096, 16384})
            for (bool center : {false, true})
                for (bool gate : {false, true})
                    for (bool rounded : {false, true})
                        for (bool bfloat : {false, true})
                            for (bool alias : {false, true}) {
                                norm(ctx, n, 8, center, gate, rounded, bfloat, alias);
                                ++cases;
                            }
        for (int c : {7, 8192})
            for (int t : {1, 8, 33})
                for (bool alias : {false, true}) {
                    conv(ctx, c, t, alias);
                    ++cases;
                }
        for (int n : {17, 4096, 131072})
            for (float limit : {0.f, 10.f}) {
                elementwise(ctx, n, limit);
                ++cases;
            }
        for (int rows : {1, 8})
            for (int mode : {0, 1, 2})
                for (auto [experts, active] :
                     {std::pair{288, 8}, std::pair{1, 1}, std::pair{17, 17}, std::pair{512, 32},
                      std::pair{513, 8}}) {
                    router(ctx, experts, active, rows, mode);
                    ++cases;
                }
        argmax(ctx);
        ++cases;
        for (int rows : {32, 128}) {
            metadata(ctx, rows);
            cases += 2;
        }
        std::printf("PASS: %u resident transform fixtures; complete scalar FP64 "
                    "norms/conv/SwiGLU/router/expert sums; represented inputs; direct FP32/BF16 "
                    "gains; alias; exact state/codecs/argmax\n",
                    cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
