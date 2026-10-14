#include "ninfer/ops/dflash_block.h"
#include "core/arena.h"
#include "core/device.h"
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

std::uint16_t bw(double x) {
    auto b = std::bit_cast<unsigned>(static_cast<float>(x));
    return (b + 0x7fffu + ((b >> 16) & 1)) >> 16;
}

double ub(std::uint16_t x) { return std::bit_cast<float>(static_cast<unsigned>(x) << 16); }

double bf(double x) { return ub(bw(x)); }

double val(std::size_t i) {
    unsigned b = static_cast<unsigned>(i) * 747796405u + 2891336453u;
    b          = ((b >> ((b >> 28) + 4)) ^ b) * 277803737u;
    b ^= b >> 22;
    return bf(std::bit_cast<float>((b & 0x807fffffu) | ((122 + b % 5) << 23)));
}

template <class T>
DeviceBuffer upload(const std::vector<T>& x) {
    DeviceBuffer d(x.size() * sizeof(T));
    d.copy_from_host(x.data(), d.bytes);
    return d;
}

template <class T>
std::vector<T> download(const DeviceBuffer& d) {
    std::vector<T> x(d.bytes / sizeof(T));
    d.copy_to_host(x.data(), d.bytes);
    return x;
}

void compare(const std::vector<float>& observed, const std::vector<double>& oracle) {
    double e2 = 0, r2 = 0, peak = 0, error = 0;
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        require(std::isfinite(observed[i]), "nonfinite DFlash block output");
        double e = std::abs(observed[i] - oracle[i]);
        e2 += e * e;
        r2 += oracle[i] * oracle[i];
        peak  = std::max(peak, std::abs(oracle[i]));
        error = std::max(error, e);
    }
    double l2 = std::sqrt(e2 / std::max(r2, 1e-30));
    if (l2 > .0022 || error > 1e-5 + .004 * peak) {
        std::fprintf(stderr, "DFlash l2=%g max=%g peak=%g\n", l2, error, peak);
        throw std::runtime_error("DFlash complete FP64 oracle criterion");
    }
}

void dynamic(DeviceContext& ctx, int t, int stage) {
    std::vector<float> x(t * 4096), d(t * 1024);
    std::vector<std::uint16_t> base(4 * 4096);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = val(i);
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = val(i + 81);
    for (std::size_t i = 0; i < base.size(); ++i) base[i] = bw(val(i + 199));
    auto dx = upload(x), dd = upload(d), db = upload(base);
    DeviceBuffer out(x.size() * 4);
    ops::dflash_dynamic_conv(static_cast<float*>(dx.p), static_cast<float*>(dd.p),
                             static_cast<std::uint16_t*>(db.p), static_cast<float*>(out.p), t,
                             stage, ctx.stream);
    ctx.synchronize();
    std::vector<double> ref(x.size());
    for (int token = 0; token < t; ++token)
        for (int c = 0; c < 4096; ++c) {
            double y = 0;
            for (int tap = 0; tap < 2; ++tap) {
                double v     = token >= tap ? x[(token - tap) * 4096 + c] : 0,
                       k     = ub(base[(stage * 2 + tap) * 4096 + c]),
                       delta = d[token * 1024 + stage * 512 + tap * 256 + c / 16];
                y            = bf(y + bf(k * v));
                y            = bf(delta * v + y);
            }
            ref[token * 4096 + c] = y;
        }
    compare(download<float>(out), ref);
    require(download<float>(dx) == x, "DFlash conv source changed");
}

void rope(DeviceContext& ctx, int heads, int t, int first) {
    std::vector<float> x(t * heads * 128);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = val(i);
    auto dx = upload(x);
    ops::dflash_rope(static_cast<float*>(dx.p), heads, t, first, ctx.stream);
    ctx.synchronize();
    std::vector<double> ref(x.size());
    for (int token = 0; token < t; ++token)
        for (int h = 0; h < heads; ++h)
            for (int c = 0; c < 64; ++c) {
                int off      = (token * heads + h) * 128;
                double angle = (first + token) * std::pow(10000., -double(c) / 64),
                       co = std::cos(angle), si = std::sin(angle);
                ref[off + c]      = x[off + c] * co - x[off + c + 64] * si;
                ref[off + c + 64] = x[off + c + 64] * co + x[off + c] * si;
            }
    compare(download<float>(dx), ref);
}

void codecs(DeviceContext& ctx) {
    constexpr int slots = 2056, first = 32760, t = 8;
    std::vector<std::uint16_t> feature(static_cast<std::size_t>(slots) * 20480);
    for (std::size_t i = 0; i < feature.size(); ++i) feature[i] = bw(val(i));
    auto df = upload(feature);
    DeviceBuffer out(t * 20480 * 4);
    ops::dflash_features(static_cast<std::uint16_t*>(df.p), static_cast<float*>(out.p), slots,
                         first, t, ctx.stream);
    ctx.synchronize();
    auto got = download<float>(out);
    for (int token = 0; token < t; ++token)
        for (int c = 0; c < 20480; ++c)
            require(got[token * 20480 + c] ==
                        ub(feature[static_cast<std::size_t>((first + token) % slots) * 20480 + c]),
                    "DFlash feature-ring exact words");
    std::vector<float> k(t * 1024), v(k.size());
    for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = val(i);
        v[i] = val(i + 77);
    }
    auto dk = upload(k), dv = upload(v);
    DeviceBuffer kc(static_cast<std::size_t>(slots) * 1024 * 2), vc(kc.bytes);
    kc.fill(0xa5);
    vc.fill(0xa5);
    ops::dflash_cache_append(static_cast<float*>(dk.p), static_cast<float*>(dv.p),
                             static_cast<std::uint16_t*>(kc.p), static_cast<std::uint16_t*>(vc.p),
                             slots, first, t, ctx.stream);
    ctx.synchronize();
    auto keys = download<std::uint16_t>(kc), values = download<std::uint16_t>(vc);
    for (int pos = 0; pos < slots; ++pos) {
        int token = -1;
        for (int j = 0; j < t; ++j)
            if ((first + j) % slots == pos) token = j;
        for (int c = 0; c < 1024; ++c) {
            require(keys[pos * 1024 + c] == (token < 0 ? 0xa5a5 : bw(k[token * 1024 + c])),
                    "DFlash key cache words/inactive slots");
            require(values[pos * 1024 + c] == (token < 0 ? 0xa5a5 : bw(v[token * 1024 + c])),
                    "DFlash value cache words/inactive slots");
        }
    }
}

void selector(DeviceContext& ctx) {
    constexpr int vocab = 154880, t = 7, rank = 256;
    std::vector<float> logits(static_cast<std::size_t>(t) * vocab, -100), hidden(t * rank);
    std::vector<std::uint16_t> pre(static_cast<std::size_t>(vocab) * rank), suc(pre.size());
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = val(i) * 2;
    // Only the small set of reachable candidates/anchor needs nonzero codebooks.
    constexpr int anchor = 17;
    for (int c = 0; c < rank; ++c) pre[anchor * rank + c] = bw(val(c + 31));
    for (int token = 0; token < t; ++token)
        for (int j = 0; j < 16; ++j) {
            int id                     = 100 + token * 16 + j;
            logits[token * vocab + id] = j / 16.f;
            for (int c = 0; c < rank; ++c) {
                pre[static_cast<std::size_t>(id) * rank + c] = bw(val(id * rank + c + 9));
                suc[static_cast<std::size_t>(id) * rank + c] = bw(val(id * rank + c + 91));
            }
        }
    auto dl = upload(logits), dh = upload(hidden), dp = upload(pre), ds = upload(suc);
    DeviceBuffer ids(t * 16 * 4), unary(t * 16 * 4), path(t * 4);
    ops::dflash_top16(static_cast<float*>(dl.p), static_cast<std::int32_t*>(ids.p),
                      static_cast<float*>(unary.p), vocab, t, ctx.stream);
    ctx.synchronize();
    auto candidates = download<std::int32_t>(ids);
    auto scores     = download<float>(unary);
    for (int token = 0; token < t; ++token)
        for (int j = 0; j < 16; ++j) {
            require(candidates[token * 16 + j] == 100 + token * 16 + 15 - j &&
                        scores[token * 16 + j] == (15 - j) / 16.f,
                    "DFlash exact top16 full vocabulary");
        }
    ops::dflash_select(static_cast<float*>(dh.p), static_cast<std::uint16_t*>(dp.p),
                       static_cast<std::uint16_t*>(ds.p), static_cast<std::int32_t*>(ids.p),
                       static_cast<float*>(unary.p), static_cast<std::int32_t*>(path.p), anchor, t,
                       ctx.stream);
    ctx.synchronize();
    auto actual  = download<std::int32_t>(path);
    int previous = anchor, changed = 0;
    for (int token = 0; token < t; ++token) {
        double best  = -INFINITY;
        int selected = -1;
        for (int j = 0; j < 16; ++j) {
            int id     = candidates[token * 16 + j];
            double sum = 0;
            for (int c = 0; c < rank; ++c)
                sum += bf(ub(pre[static_cast<std::size_t>(previous) * rank + c]) *
                          hidden[token * rank + c]) *
                       ub(suc[static_cast<std::size_t>(id) * rank + c]);
            double score = scores[token * 16 + j] + bf(sum);
            if (score > best) {
                best     = score;
                selected = id;
            }
        }
        require(actual[token] == selected, "DFlash predecessor-conditioned FP64 codebook path");
        changed += selected != candidates[token * 16];
        previous = selected;
    }
    require(changed > 0, "selector fixture must exercise non-unary candidate choice");
}
} // namespace

int main() {
    try {
        DeviceContext ctx;
        unsigned cases = 0;
        for (int t : {1, 8})
            for (int stage : {0, 1}) {
                dynamic(ctx, t, stage);
                ++cases;
            }
        for (int heads : {8, 32})
            for (int t : {1, 8})
                for (int first : {0, 1, 2048, 32760}) {
                    rope(ctx, heads, t, first);
                    ++cases;
                }
        codecs(ctx);
        ++cases;
        selector(ctx);
        ++cases;
        std::printf("PASS: %u DFlash2 block fixtures; FP64 dynamic-conv/RoPE/path oracle; "
                    "full-vocabulary top16; target-feature and KV rings; no hidden allocation\n",
                    cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
