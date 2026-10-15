#include "ninfer/ops/sparse_latent_attention.h"
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

float val(std::size_t i) {
    unsigned b = static_cast<unsigned>(i) * 747796405u + 2891336453u;
    b          = ((b >> ((b >> 28) + 4)) ^ b) * 277803737u;
    b ^= b >> 22;
    return std::bit_cast<float>((b & 0x807fffffu) | ((122 + b % 5) << 23));
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

void compare(const std::vector<float>& observed, const std::vector<double>& oracle, bool rounded) {
    double e2 = 0, r2 = 0, peak = 0, error = 0;
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        require(std::isfinite(observed[i]), "nonfinite attention output");
        double e = std::abs(observed[i] - oracle[i]);
        e2 += e * e;
        r2 += oracle[i] * oracle[i];
        peak  = std::max(peak, std::abs(oracle[i]));
        error = std::max(error, e);
    }
    double l2 = std::sqrt(e2 / std::max(r2, 1e-30));
    if (l2 > (rounded ? .0022 : 2e-5) ||
        error > (rounded ? 1e-5 + .004 * peak : 2e-6 + 5e-5 * peak)) {
        std::fprintf(stderr, "attention rounded=%d l2=%g max=%g peak=%g\n", rounded, l2, error,
                     peak);
        throw std::runtime_error("complete FP64 attention oracle criterion");
    }
}

void latent(DeviceContext& ctx, int heads, int dim, int t, int count, bool saturated) {
    constexpr int context = 4096;
    std::vector<float> q(static_cast<std::size_t>(t) * heads * dim);
    std::vector<std::uint16_t> kv(context * dim);
    std::vector<std::int32_t> ids(t * count);
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = val(i) * (saturated ? 16 : 1);
    for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = bw(val(i + 71));
    for (int token = 0; token < t; ++token)
        for (int j = 0; j < count; ++j)
            ids[token * count + j] = j % 13 == 0 ? -1 : (j * 31 + token * 17) % context;
    auto dq = upload(q), dk = upload(kv), di = upload(ids);
    DeviceBuffer p(static_cast<std::size_t>(t) * heads * count * 4), out(q.size() * 4);
    const float scale = 1.f / 16;
    ops::sparse_latent_attention(static_cast<float*>(dq.p), static_cast<std::uint16_t*>(dk.p),
                                 static_cast<std::int32_t*>(di.p), static_cast<float*>(p.p),
                                 static_cast<float*>(out.p), heads, dim, t, count, scale,
                                 ctx.stream);
    ctx.synchronize();
    std::vector<double> oracle(q.size()), score(count);
    for (int token = 0; token < t; ++token)
        for (int h = 0; h < heads; ++h) {
            double maximum = -INFINITY, sum = 0;
            for (int j = 0; j < count; ++j) {
                int pos  = ids[token * count + j];
                double s = -INFINITY;
                if (pos >= 0) {
                    s = 0;
                    for (int c = 0; c < dim; ++c)
                        s += double(q[(token * heads + h) * dim + c]) * ub(kv[pos * dim + c]);
                    s *= scale;
                }
                score[j] = s;
                maximum  = std::max(maximum, s);
            }
            for (auto& x : score) {
                x = std::exp(x - maximum);
                sum += x;
            }
            for (int c = 0; c < dim; ++c) {
                double value = 0;
                for (int j = 0; j < count; ++j)
                    if (ids[token * count + j] >= 0)
                        value += score[j] / sum * ub(kv[ids[token * count + j] * dim + c]);
                oracle[(token * heads + h) * dim + c] = value;
            }
        }
    compare(download<float>(out), oracle, false);
}

void draft_attention(DeviceContext& ctx, int first, int t) {
    constexpr int slots = 2056;
    std::vector<float> q(t * 4096), nk(t * 1024), nv(t * 1024);
    std::vector<std::uint16_t> kc(slots * 1024), vc(kc.size());
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = bf(val(i));
    for (std::size_t i = 0; i < nk.size(); ++i) {
        nk[i] = bf(val(i + 37));
        nv[i] = bf(val(i + 83));
    }
    for (std::size_t i = 0; i < kc.size(); ++i) {
        kc[i] = bw(val(i + 191));
        vc[i] = bw(val(i + 213));
    }
    auto dq = upload(q), dk = upload(kc), dv = upload(vc), dnk = upload(nk), dnv = upload(nv);
    DeviceBuffer p(static_cast<std::size_t>(t) * 32 * (2048 + t) * 4), out(q.size() * 4);
    ops::dflash_attention(static_cast<float*>(dq.p), static_cast<std::uint16_t*>(dk.p),
                          static_cast<std::uint16_t*>(dv.p), static_cast<float*>(dnk.p),
                          static_cast<float*>(dnv.p), static_cast<float*>(p.p),
                          static_cast<float*>(out.p), slots, first, t, ctx.stream);
    ctx.synchronize();
    std::vector<double> ref(q.size());
    int begin = std::max(0, first - 2047), context = first - begin, count = context + t;
    std::vector<double> scores(count);
    for (int token = 0; token < t; ++token)
        for (int h = 0; h < 32; ++h) {
            double maximum = -INFINITY, sum = 0;
            for (int j = 0; j < count; ++j) {
                int pos    = j < context ? begin + j : first + j - context;
                double dot = -INFINITY;
                if (first + token - pos < 2048 && pos - (first + token) < 2048) {
                    dot = 0;
                    for (int c = 0; c < 128; ++c) {
                        const auto off = (h / 4) * 128 + c;
                        double key     = j < context ? ub(kc[(pos % slots) * 1024 + off])
                                                     : nk[(j - context) * 1024 + off];
                        dot += double(q[(token * 32 + h) * 128 + c]) * key;
                    }
                    dot *= double(.08838834764831844f);
                }
                scores[j] = dot;
                maximum   = std::max(maximum, dot);
            }
            for (auto& x : scores) {
                x = std::exp(x - maximum);
                sum += x;
            }
            for (int c = 0; c < 128; ++c) {
                double value = 0;
                for (int j = 0; j < count; ++j) {
                    int pos = j < context ? begin + j : first + j - context,
                        off = (h / 4) * 128 + c;
                    value += scores[j] / sum *
                             (j < context ? ub(vc[(pos % slots) * 1024 + off])
                                          : nv[(j - context) * 1024 + off]);
                }
                ref[(token * 32 + h) * 128 + c] = value;
            }
        }
    compare(download<float>(out), ref, true);
}

void pooling(DeviceContext& ctx, int visible, bool masked) {
    const int complete = visible / 4;
    std::vector<std::uint16_t> keys(static_cast<std::size_t>(visible) * 128), gates(keys.size()),
        pools(static_cast<std::size_t>(std::max(complete, 1)) * 128);
    std::vector<float> q(4096), hw(32), bias(512);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i]  = bw(val(i + 13));
        gates[i] = bw(val(i + 81) * 3);
    }
    for (int i = 0; i < 4096; ++i) q[i] = bf(val(i + 19));
    for (int i = 0; i < 32; ++i) hw[i] = val(i + 89);
    for (int i = 0; i < 512; ++i) bias[i] = val(i + 73);
    // CPU builds all preceding complete pools; the public append Op publishes
    // only the last completed pool and its exact stored key/gate words.
    for (int pool = 0; pool < complete; ++pool)
        for (int c = 0; c < 128; ++c) {
            double a[4], maximum = -INFINITY, sum = 0, value = 0;
            for (int j = 0; j < 4; ++j) {
                a[j]    = ub(gates[(pool * 4 + j) * 128 + c]) + bias[j * 128 + c];
                maximum = std::max(maximum, a[j]);
            }
            for (auto& x : a) {
                x = std::exp(x - maximum);
                sum += x;
            }
            for (int j = 0; j < 4; ++j)
                value += bf(bf(a[j] / sum) * ub(keys[(pool * 4 + j) * 128 + c]));
            pools[pool * 128 + c] = bw(value);
        }
    auto dk = upload(keys), dg = upload(gates), dp = upload(pools), dq = upload(q), dh = upload(hw),
         db = upload(bias);
    std::vector<std::uint8_t> allowed((visible + 63) / 64, 1);
    if (masked)
        for (std::size_t i = 0; i < allowed.size(); ++i) allowed[i] = i % 3 == 0;
    auto da = upload(allowed);
    if (complete) {
        const int position = complete * 4 - 1;
        std::vector<float> k(128), g(128);
        for (int c = 0; c < 128; ++c) {
            k[c] = ub(keys[position * 128 + c]);
            g[c] = ub(gates[position * 128 + c]);
        }
        auto ik = upload(k), ig = upload(g);
        ops::index_pool_append(static_cast<float*>(ik.p), static_cast<float*>(ig.p),
                               static_cast<float*>(db.p), static_cast<std::uint16_t*>(dk.p),
                               static_cast<std::uint16_t*>(dg.p), static_cast<std::uint16_t*>(dp.p),
                               position, ctx.stream);
        ctx.synchronize();
        auto actual = download<std::uint16_t>(dp);
        std::vector<float> last(128);
        std::vector<double> oracle(128);
        for (int c = 0; c < 128; ++c) {
            last[c]   = ub(actual[(complete - 1) * 128 + c]);
            oracle[c] = ub(pools[(complete - 1) * 128 + c]);
        }
        compare(last, oracle, true);
        require(download<std::uint16_t>(dk) == keys && download<std::uint16_t>(dg) == gates,
                "pool append changed unrelated key/gate words");
        pools = actual;
    }
    DeviceBuffer ds(std::max(complete, 1) * 4), di(2051 * 4);
    ops::index_pool_scores(static_cast<float*>(dq.p), static_cast<float*>(dh.p),
                           static_cast<std::uint16_t*>(dp.p),
                           masked ? static_cast<std::uint8_t*>(da.p) : nullptr,
                           static_cast<float*>(ds.p), complete, ctx.stream);
    ctx.synchronize();
    auto score = download<float>(ds);
    for (int pool = 0; pool < complete; ++pool) {
        if (masked && !allowed[pool / 16]) {
            require(score[pool] == -INFINITY, "masked pool score");
            continue;
        }
        double ideal = 0, absolute = 0;
        for (int h = 0; h < 32; ++h) {
            double dot = 0;
            for (int c = 0; c < 128; ++c) dot += double(q[h * 128 + c]) * ub(pools[pool * 128 + c]);
            double contribution = std::max(dot * double(.08838834764831844f), 0.) * hw[h] *
                                  double(.17677669529663688f);
            ideal += contribution;
            absolute += std::abs(contribution);
        }
        require(std::abs(score[pool] - ideal) <= 8e-6 * (1 + absolute),
                "index scores complete FP64 dot/head oracle");
    }
    ops::index_pool_select(static_cast<float*>(ds.p), static_cast<std::int32_t*>(di.p), visible,
                           ctx.stream);
    ctx.synchronize();
    auto ids = download<std::int32_t>(di);
    std::vector<int> order(complete);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
    std::vector<int> reference(2051, -1);
    for (int j = 0; j < std::min(complete, 512); ++j)
        if (std::isfinite(score[order[j]]))
            for (int c = 0; c < 4; ++c) reference[j * 4 + c] = order[j] * 4 + c;
    for (int c = 0; c < visible % 4; ++c) reference[2048 + c] = complete * 4 + c;
    require(ids == reference, "exact represented-score top512 pools / partial tail / padding");
}
void selection_edges(DeviceContext& ctx) {
    for (int pools : {1, 31, 256, 511, 512, 513, 8192}) for (int mode : {0, 1, 2, 3}) {
        std::vector<float> score(pools);
        for (int i = 0; i < pools; ++i) {
            score[i] = mode == 0 ? 0.f : static_cast<float>((i * 17) % 31 - 15);
            if (mode == 1 && i % 7) score[i] = -INFINITY;
            if (mode == 2) score[i] = i % 2 ? NAN : -INFINITY;
        }
        if (mode == 3) score.back() = INFINITY;
        auto input = upload(score);
        DeviceBuffer out(2051 * 4 + 32); out.fill(0xa5);
        ops::index_pool_select(static_cast<float*>(input.p), reinterpret_cast<int*>(static_cast<std::byte*>(out.p) + 16),
                               pools * 4, ctx.stream);
        ctx.synchronize();
        std::vector<int> got(2059); out.copy_to_host(got.data(), out.bytes);
        for (int i = 0; i < 4; ++i) require(got[i] == static_cast<int>(0xa5a5a5a5u) && got[got.size()-1-i] == static_cast<int>(0xa5a5a5a5u), "pool selection guards");
        std::vector<int> order;
        if (mode != 3) for (int i = 0; i < pools; ++i) if (std::isfinite(score[i])) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b] || (score[a] == score[b] && a < b); });
        if (order.size() > 512) order.resize(512);
        std::vector<int> expected(2051, -1);
        for (unsigned i = 0; i < order.size(); ++i) for (int c = 0; c < 4; ++c) expected[i*4+c] = order[i]*4+c;
        require(std::equal(expected.begin(), expected.end(), got.begin()+4), "exact finite top-k order/ties/exhaustion");
        auto after = download<float>(input);
        for (int i : order) score[i] = -INFINITY;
        for (int i = 0; i < pools; ++i) require(after[i] == score[i] || (std::isnan(after[i]) && std::isnan(score[i])), "pool score scratch mutation");
    }
}
} // namespace

int main() {
    try {
        DeviceContext ctx;
        selection_edges(ctx);
        unsigned cases = 0;
        for (int v : {1, 3, 4, 7, 2047, 2048, 2051, 32768})
            for (bool mask : {false, true}) {
                pooling(ctx, v, mask);
                ++cases;
            }
        latent(ctx, 64, 512, 1, 2051, false);
        ++cases;
        latent(ctx, 64, 512, 8, 33, false);
        ++cases;
        latent(ctx, 2, 128, 8, 65, true);
        ++cases;
        for (int first : {0, 1, 2048, 2058, 32760})
            for (int t : {1, 8}) {
                draft_attention(ctx, first, t);
                ++cases;
            }
        std::printf("PASS: %u pool/MLA/DFlash attention fixtures; complete independent FP64 "
                    "scores/softmax/value oracle; top512 and tails; K7 width8; resident "
                    "ring/wrap/sliding/GQA\n",
                    cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
