#include "ninfer/ops/resident_transform.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
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

DeviceBuffer upload_bf16(const std::vector<float>& x) {
    std::vector<std::uint16_t> words; for (auto v : x) words.push_back(word(v));
    DeviceBuffer d(words.size() * 2); d.copy_from_host(words.data(), d.bytes); return d;
}
std::vector<float> download_bf16(const DeviceBuffer& d) {
    std::vector<std::uint16_t> words(d.bytes / 2); d.copy_to_host(words.data(), d.bytes);
    std::vector<float> out; for (auto v : words) out.push_back(unword(v)); return out;
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
          bool alias, int storage = 0) {
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
    const bool input_bf16 = storage == 1 || storage == 3;
    const bool output_bf16 = storage == 1 || storage == 2;
    auto dx = input_bf16 ? upload_bf16(x) : upload(x);
    auto dz = input_bf16 ? upload_bf16(z) : upload(z);
    DeviceBuffer dy(x.size() * (output_bf16 ? 2 : 4));
    auto* o = alias ? static_cast<float*>(dx.p) : static_cast<float*>(dy.p);
    if (storage) {
        Tensor input(dx.p, input_bf16 ? DType::BF16 : DType::FP32, {n, rows});
        Tensor gate_tensor(dz.p, input.dtype, {n, rows});
        Tensor output(o, output_bf16 ? DType::BF16 : DType::FP32, {n, rows});
        ops::resident_norm(input, &wg.view, center ? &wb.view : nullptr,
                           gate ? &gate_tensor : nullptr, output, 1e-5f, center, ctx.stream);
        rounded = output_bf16;
    } else ops::resident_norm(static_cast<float*>(dx.p), &wg.view, center ? &wb.view : nullptr,
                       gate ? static_cast<float*>(dz.p) : nullptr, o, n, rows, 1e-5f, center,
                       rounded, ctx.stream);
    ctx.synchronize();
    std::vector<float> y(x.size());
    if (output_bf16) {
        std::vector<std::uint16_t> words(x.size());
        CUDA_CHECK(cudaMemcpy(words.data(), o, words.size() * 2, cudaMemcpyDeviceToHost));
        for (unsigned j = 0; j < words.size(); ++j) y[j] = unword(words[j]);
    } else CUDA_CHECK(cudaMemcpy(y.data(), o, y.size() * 4, cudaMemcpyDeviceToHost));
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

void pooled_conv(DeviceContext& ctx, int channels, bool alias, bool packed = false, bool typed = false) {
    constexpr int rows = 3, slots = 5;
    const std::vector<int> ids{4, 0, 2};
    const std::vector<int> offsets = packed ? std::vector{0, 1, 4, 12} : std::vector{0, 1, 2, 3};
    std::vector<float> x(offsets.back() * channels), s(slots * channels * 3), w(channels * 4);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = value(i + 319);
    for (std::size_t i = 0; i < s.size(); ++i) s[i] = value(i + 91);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::sin(i * 1.7) * .23;
    WeightData weight(w, {static_cast<unsigned>(channels), 1, 4}, false);
    auto dx = typed ? upload_bf16(x) : upload(x); auto ds = upload(s);
    DeviceBuffer selectors(ids.size() * sizeof(int)), bounds(offsets.size() * sizeof(int)),
        dy(x.size() * (typed ? 2 : 4));
    selectors.copy_from_host(ids.data(), selectors.bytes);
    bounds.copy_from_host(offsets.data(), bounds.bytes);
    auto* output = static_cast<float*>(alias ? dx.p : dy.p);
    if (typed) {
        Tensor input(dx.p, DType::BF16, {channels, offsets.back()});
        Tensor result(output, DType::BF16, {channels, offsets.back()});
        ops::resident_causal_conv_packed(input, weight.view, static_cast<float*>(ds.p),
            static_cast<int*>(selectors.p), static_cast<int*>(bounds.p), result, rows, ctx.stream);
    } else ops::resident_causal_conv_packed(static_cast<float*>(dx.p), weight.view,
                                     static_cast<float*>(ds.p), static_cast<int*>(selectors.p),
                                     static_cast<int*>(bounds.p), output, channels, rows, ctx.stream);
    ctx.synchronize();
    auto after = download(ds);
    const auto y = typed ? download_bf16(alias ? dx : dy) : download(alias ? dx : dy);
    for (int slot = 0; slot < slots; ++slot) {
        const auto found = std::find(ids.begin(), ids.end(), slot);
        for (int c = 0; c < channels; ++c) {
            const auto base = (slot * channels + c) * 3;
            if (found == ids.end()) {
                for (int j = 0; j < 3; ++j)
                    require(after[base + j] == s[base + j], "conv changed inactive slot");
                continue;
            }
            const int row = found - ids.begin();
            double a = s[base], b = s[base + 1], d = s[base + 2];
            for (int t = offsets[row]; t < offsets[row + 1]; ++t) {
                const double input = x[t * channels + c];
                const double sum = a * w[c * 4] + b * w[c * 4 + 1] +
                                   d * w[c * 4 + 2] + input * w[c * 4 + 3];
                const auto pre = bf(sum);
                check(y[t * channels + c], pre * sig(pre), true, "pooled conv FP64");
                a = b;
                b = d;
                d = input;
            }
            require(after[base] == a && after[base + 1] == b && after[base + 2] == d,
                    "pooled conv final state");
        }
    }
}

void row_gather(DeviceContext& ctx) {
    constexpr int width = 16384, source_rows = 5;
    const std::vector<int> ids{4, 0, 2};
    std::vector<float> x(width * source_rows);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = value(i + 137);
    auto input = upload(x);
    DeviceBuffer selectors(ids.size() * sizeof(int)), out(ids.size() * width * 4 + 32);
    selectors.copy_from_host(ids.data(), selectors.bytes);
    out.fill(0xa5);
    auto* data = reinterpret_cast<float*>(static_cast<std::byte*>(out.p) + 16);
    ops::resident_gather_rows(static_cast<float*>(input.p), static_cast<int*>(selectors.p),
                              data, width, ids.size(), ctx.stream);
    ctx.synchronize();
    std::vector<unsigned char> bytes(out.bytes);
    out.copy_to_host(bytes.data(), bytes.size());
    for (int i = 0; i < 16; ++i)
        require(bytes[i] == 0xa5 && bytes[bytes.size() - 1 - i] == 0xa5,
                "row gather output guard");
    std::vector<float> y(ids.size() * width);
    out.copy_to_host(y.data(), y.size() * 4, 16);
    for (std::size_t r = 0; r < ids.size(); ++r)
        for (int c = 0; c < width; ++c)
            require(std::bit_cast<unsigned>(y[r * width + c]) ==
                        std::bit_cast<unsigned>(x[ids[r] * width + c]), "row gather exact value");
    require(download(input) == x, "row gather changed input");
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

void typed_swiglu(DeviceContext& ctx) {
    std::vector<float> a(1025), b(a.size());
    for (unsigned j = 0; j < a.size(); ++j) { a[j] = value(j); b[j] = value(j + 7); }
    auto da = upload_bf16(a), db = upload_bf16(b);
    Tensor x(da.p, DType::BF16, {1025}), y(db.p, DType::BF16, {1025});
    ops::resident_swiglu(x, y, x, 10.f, ctx.stream); ctx.synchronize();
    const auto result = download_bf16(da);
    for (unsigned j = 0; j < a.size(); ++j) {
        const auto g = std::min<double>(a[j], 10), u = std::clamp<double>(b[j], -10, 10);
        check(result[j], bf(g * sig(g)) * u, true, "typed SwiGLU FP64");
    }
}

void owned_swiglu(DeviceContext& ctx) {
    constexpr int width = 129, rows = 8;
    const std::array<int, rows> owner{95, 96, 191, 192, 287, 0, 120, -1};
    DeviceBuffer ids(sizeof(owner)); ids.copy_from_host(owner.data(), ids.bytes);
    for (bool bf16 : {false, true}) for (bool alias : {false, true}) {
        std::vector<float> g(width * rows), u(g.size());
        for (int row = 0; row < rows; ++row) for (int c = 0; c < width; ++c) {
            const auto i = row * width + c;
            g[i] = owner[row] >= 96 && owner[row] < 192 ? value(i) * 8 : NAN;
            u[i] = owner[row] >= 96 && owner[row] < 192 ? value(i + 7) * 8 : NAN;
        }
        auto dg = bf16 ? upload_bf16(g) : upload(g), du = bf16 ? upload_bf16(u) : upload(u);
        DeviceBuffer output(g.size() * (bf16 ? 2 : 4) + 32); output.fill(0xa5);
        Tensor a(dg.p, bf16 ? DType::BF16 : DType::FP32, {width, rows});
        Tensor b(du.p, a.dtype, {width, rows});
        Tensor y(alias ? a.data : static_cast<std::byte*>(output.p) + 16, a.dtype, {width, rows});
        ops::resident_swiglu_owned(a, b, static_cast<int*>(ids.p), 96, 96, y, 10, ctx.stream);
        ctx.synchronize();
        std::vector<unsigned char> raw(g.size() * (bf16 ? 2 : 4));
        if (alias) dg.copy_to_host(raw.data(), raw.size());
        else output.copy_to_host(raw.data(), raw.size(), 16);
        for (int row = 0; row < rows; ++row) for (int c = 0; c < width; ++c) {
            const auto i = row * width + c;
            float observed;
            if (bf16) { std::uint16_t bits; std::memcpy(&bits, raw.data() + i * 2, 2); observed = std::bit_cast<float>(static_cast<unsigned>(bits) << 16); }
            else std::memcpy(&observed, raw.data() + i * 4, 4);
            if (owner[row] < 96 || owner[row] >= 192) require(observed == 0, "unowned nonfinite SwiGLU input was read");
            else {
                const double gate = std::min<double>(g[i], 10), up = std::clamp<double>(u[i], -10, 10);
                check(observed, bf(gate * sig(gate)) * up, true, "owned SwiGLU FP64");
            }
        }
        if (!alias) {
            std::vector<unsigned char> guards(output.bytes); output.copy_to_host(guards.data(), guards.size());
            for (int i = 0; i < 16; ++i) require(guards[i] == 0xa5 && guards[guards.size()-1-i] == 0xa5, "owned SwiGLU guards");
        }
    }
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
void tensor_sums(DeviceContext& ctx) {
    for (int tokens : {1, 8, 33}) for (bool physical : {false, true}) {
        constexpr int hidden = 4096, active = 8;
        std::vector<float> experts(tokens * active * hidden), weights(tokens * active);
        for (unsigned i = 0; i < experts.size(); ++i) experts[i] = value(i + 71);
        for (unsigned i = 0; i < weights.size(); ++i)
            weights[i] = .05f + static_cast<float>(i % 13) / 97.f;
        auto dx = physical ? upload_bf16(experts) : upload(experts); auto dw = upload(weights);
        DeviceBuffer result(tokens * hidden * sizeof(float));
        Tensor expert_tensor(dx.p, DType::BF16, {hidden, active, tokens});
        if (physical) ops::resident_moe_partial(expert_tensor, static_cast<const float*>(dw.p),
            static_cast<float*>(result.p), hidden, active, tokens, ctx.stream);
        else ops::resident_moe_partial(static_cast<float*>(dx.p), static_cast<float*>(dw.p),
            static_cast<float*>(result.p), hidden, active, tokens, ctx.stream);
        ctx.synchronize();
        const auto observed = download(result);
        for (int t = 0; t < tokens; ++t) for (int c = 0; c < hidden; ++c) {
            double ideal = 0, absolute = 0;
            for (int j = 0; j < active; ++j) {
                const double product = static_cast<double>(experts[(t * active + j) * hidden + c]) * weights[t * active + j];
                ideal += product; absolute += std::abs(product);
            }
            require(std::isfinite(observed[t * hidden + c]) &&
                        std::abs(observed[t * hidden + c] - ideal) <= 4e-6 * absolute + 1e-30,
                    "local EP weighted sum complete FP64 oracle");
        }
    }
    constexpr int count = 4096 * 3;
    for (bool routed : {false, true}) {
        std::vector<float> a(count * (routed ? 2 : 1)), b(a.size()), c(a.size());
        for (unsigned i = 0; i < a.size(); ++i) {
            a[i] = value(i + 1); b[i] = value(i + 17001); c[i] = value(i + 37001);
        }
        auto da = upload(a), db = upload(b), dc = upload(c);
        DeviceBuffer out(count * 4 + 32);
        out.fill(0xa5);
        ops::resident_tp_sum(static_cast<float*>(da.p), static_cast<float*>(db.p), static_cast<float*>(dc.p),
            reinterpret_cast<float*>(static_cast<std::byte*>(out.p) + 16), count, routed, ctx.stream);
        ctx.synchronize();
        auto observed = download(out);
        for (int i = 0; i < count; ++i) {
            double ideal = static_cast<double>(a[i]) + b[i] + c[i];
            if (routed) ideal = bf(ideal) + static_cast<double>(a[count + i]) + b[count + i] + c[count + i];
            check(observed[i + 4], ideal, true, "three-rank reduction/semantic shared boundary");
        }
        std::vector<std::byte> guards(out.bytes);
        out.copy_to_host(guards.data(), guards.size());
        for (int i = 0; i < 16; ++i)
            require(guards[i] == std::byte{0xa5} && guards[guards.size() - i - 1] == std::byte{0xa5},
                    "TP reduction output guard");
    }
    constexpr int rows = 8, width = 513;
    std::vector<float> input(3 * rows * width, -5.f);
    for (int d = 0; d < 3; ++d) for (int t = 0; t < rows; ++t)
        input[(d * rows + t) * width + 7] = t == 0 ? 2.f : float(d + t);
    input[(2 * rows + 4) * width + 511] = INFINITY;
    input[(1 * rows + 5) * width] = NAN;
    auto dx = upload(input);
    DeviceBuffer pairs(3 * rows * sizeof(ops::ArgmaxPair)), ids(rows * sizeof(int));
    for (int d = 0; d < 3; ++d)
        ops::resident_argmax_pairs(static_cast<float*>(dx.p) + d * rows * width,
            static_cast<ops::ArgmaxPair*>(pairs.p) + d * rows, width, rows, d * width, ctx.stream);
    ops::resident_argmax_merge(static_cast<ops::ArgmaxPair*>(pairs.p), static_cast<int*>(ids.p), rows, ctx.stream);
    ctx.synchronize();
    std::vector<int> got(rows);
    ids.copy_to_host(got.data(), ids.bytes);
    for (int t = 0; t < rows; ++t)
        require(got[t] == ((t == 4 || t == 5) ? -1 : (t == 0 ? 7 : 2 * width + 7)),
                "distributed vocabulary ties/nonfinite propagation");
}
} // namespace

int main() {
    try {
        DeviceContext ctx;
        typed_swiglu(ctx);
        owned_swiglu(ctx);
        norm(ctx, 128, 20, false, true, true, true, true, 1);
        norm(ctx, 128, 3, true, false, true, true, true, 1);
        for (int storage : {1, 2, 3}) for (bool center : {false, true}) for (bool gate : {false, true})
            norm(ctx, 4096, 3, center, gate, storage != 3, true, false, storage);
        for (bool alias : {false, true}) pooled_conv(ctx, 129, alias, true, true);
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
        norm(ctx, 128, 65536, false, true, true, false, false);
        row_gather(ctx);
        cases += 2;
        for (int c : {7, 8192})
            for (bool alias : {false, true})
                for (bool packed : {false, true}) {
                    pooled_conv(ctx, c, alias, packed);
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
        tensor_sums(ctx);
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
