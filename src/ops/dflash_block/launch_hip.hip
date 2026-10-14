#include "ninfer/ops/dflash_block.h"
#include "core/device.h"
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
#include "ops/dflash_block/rotary_frequencies.cuh"

__device__ float ub(std::uint16_t x) { return __uint_as_float(static_cast<unsigned>(x) << 16); }

__device__ std::uint16_t bw(float v) {
    auto b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffu + ((b >> 16) & 1)) >> 16);
}

__device__ float bf(float v) { return ub(bw(v)); }

__global__ void dynamic_conv(const float* x, const float* dynamic, const std::uint16_t* base,
                             float* out, int stage) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c >= 4096) return;
    float y = 0;
    for (int tap = 0; tap < 2; ++tap) {
        const float v      = t >= tap ? x[(t - tap) * 4096 + c] : 0;
        const float kernel = ub(base[(stage * 2 + tap) * 4096 + c]);
        const float d      = dynamic[t * 1024 + stage * 512 + tap * 256 + c / 16];
        y                  = bf(y + bf(kernel * v));
        y                  = bf(fmaf(d, v, y));
    }
    out[t * 4096 + c] = y;
    return;
}

__global__ void rope(float* x, int heads, int first) {
    const int c = threadIdx.x, h = blockIdx.x, t = blockIdx.y;
    auto* row     = x + (static_cast<long long>(t) * heads + h) * 128;
    const float a = row[c], b = row[c + 64], pos = first + t;
    volatile float high = pos * rotary_frequency_hi[c];
    const float low     = fmaf(pos, rotary_frequency_hi[c], -high) + pos * rotary_frequency_lo[c];
    const float ch = cosf(high), sh = sinf(high), cl = cosf(low), sl = sinf(low);
    const float co = ch * cl - sh * sl, si = sh * cl + ch * sl;
    row[c]      = bf(a * co - b * si);
    row[c + 64] = bf(b * co + a * si);
    return;
}

__global__ void features(const std::uint16_t* x, float* y, int slots, int first) {
    int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c < 20480)
        y[static_cast<long long>(t) * 20480 + c] =
            ub(x[static_cast<long long>((first + t) % slots) * 20480 + c]);
    return;
}

__global__ void append_cache(const float* k, const float* v, std::uint16_t* kc, std::uint16_t* vc,
                             int slots, int first) {
    int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c < 1024) {
        const auto dst = static_cast<long long>((first + t) % slots) * 1024 + c;
        kc[dst]        = bw(k[t * 1024 + c]);
        vc[dst]        = bw(v[t * 1024 + c]);
    }
    return;
}

__global__ void attention_scores(const float* q, const std::uint16_t* kc, const float* kn, float* p,
                                 int slots, int first, int tokens) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x, begin = max(0, first - 2047),
              context = first - begin, count = context + tokens, stride = 2048 + tokens;
    const auto* query = q + (t * 32 + h) * 128;
    auto* prob        = p + (t * 32 + h) * stride;
    __shared__ float red[256];
    float maximum = -INFINITY;
    for (int i = lane; i < count; i += 256) {
        int pos   = i < context ? begin + i : first + i - context;
        float dot = -INFINITY;
        if (first + t - pos < 2048 && pos - (first + t) < 2048) {
            dot = 0;
            for (int c = 0; c < 128; ++c) {
                const auto off = (h / 4) * 128 + c;
                float key = i < context ? ub(kc[static_cast<long long>(pos % slots) * 1024 + off])
                                        : kn[(i - context) * 1024 + off];
                dot += query[c] * key;
            }
            dot *= .08838834764831844f;
        }
        prob[i] = dot;
        maximum = fmaxf(maximum, dot);
    }
    red[lane] = maximum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) red[lane] = fmaxf(red[lane], red[lane + s]);
        __syncthreads();
    }
    maximum = red[0];
    __syncthreads();
    float sum = 0;
    for (int i = lane; i < count; i += 256) {
        prob[i] = expf(prob[i] - maximum);
        sum += prob[i];
    }
    red[lane] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    sum = red[0];
    __syncthreads();
    for (int i = lane; i < count; i += 256) prob[i] /= sum;
    return;
}

__global__ void attention_values(const std::uint16_t* vc, const float* vn, const float* p,
                                 float* out, int slots, int first, int tokens) {
    const int h = blockIdx.x, t = blockIdx.y, c = threadIdx.x, begin = max(0, first - 2047),
              context = first - begin, count = context + tokens;
    const auto* prob = p + (t * 32 + h) * (2048 + tokens);
    float sum        = 0;
    for (int i = 0; i < count; ++i) {
        const int off = (h / 4) * 128 + c;
        float value = i < context ? ub(vc[static_cast<long long>((begin + i) % slots) * 1024 + off])
                                  : vn[(i - context) * 1024 + off];
        sum += prob[i] * value;
    }
    out[(t * 32 + h) * 128 + c] = bf(sum);
    return;
}

__global__ void top16(const float* logits, std::int32_t* ids, float* unary, int vocab) {
    const int t = blockIdx.x, lane = threadIdx.x;
    __shared__ float values[256];
    __shared__ int indices[256];
    for (int j = 0; j < 16; ++j) {
        float best = -INFINITY;
        int id     = 0;
        for (int v = lane; v < vocab; v += 256) {
            bool used = false;
            for (int k = 0; k < j; ++k) used |= ids[t * 16 + k] == v;
            if (used) continue;
            float s = logits[static_cast<long long>(t) * vocab + v];
            if (s > best || (s == best && v < id)) {
                best = s;
                id   = v;
            }
        }
        values[lane]  = best;
        indices[lane] = id;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (lane < s &&
                (values[lane + s] > values[lane] ||
                 (values[lane + s] == values[lane] && indices[lane + s] < indices[lane]))) {
                values[lane]  = values[lane + s];
                indices[lane] = indices[lane + s];
            }
            __syncthreads();
        }
        if (!lane) {
            ids[t * 16 + j]   = indices[0];
            unary[t * 16 + j] = values[0];
        }
        __syncthreads();
    }
    return;
}

__global__ void select(const float* hidden, const std::uint16_t* pre, const std::uint16_t* suc,
                       const std::int32_t* ids, const float* unary, std::int32_t* out, int anchor,
                       int tokens) {
    const int lane = threadIdx.x;
    __shared__ int previous;
    __shared__ float scores[16];
    if (!lane) previous = anchor;
    __syncthreads();
    for (int t = 0; t < tokens; ++t) {
        if (lane < 16) {
            const int candidate = ids[t * 16 + lane];
            float sum           = 0;
            for (int c = 0; c < 256; ++c)
                sum +=
                    bf(ub(pre[static_cast<long long>(previous) * 256 + c]) * hidden[t * 256 + c]) *
                    ub(suc[static_cast<long long>(candidate) * 256 + c]);
            scores[lane] = unary[t * 16 + lane] + bf(sum);
        }
        __syncthreads();
        if (!lane) {
            int best = 0;
            for (int j = 1; j < 16; ++j)
                if (scores[j] > scores[best]) best = j;
            previous = ids[t * 16 + best];
            out[t]   = previous;
        }
        __syncthreads();
    }
    return;
}

void shape(int t) {
    if (t < 1 || t > 8) throw std::invalid_argument("DFlash block width must be 1..8");
}
} // namespace

void dflash_dynamic_conv(const float* x, const float* d, const std::uint16_t* b, float* o, int t,
                         int stage, cudaStream_t s) {
    shape(t);
    if (!x || !d || !b || !o || x == o || stage < 0 || stage > 1)
        throw std::invalid_argument("DFlash dynamic-conv operands/stage");
    dynamic_conv<<<dim3(16, t), 256, 0, s>>>(x, d, b, o, stage);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void dflash_rope(float* x, int heads, int t, int first, cudaStream_t s) {
    if (!x || heads < 1 || heads > 32 || t < 1 || t > 32768 || first < 0)
        throw std::invalid_argument("DFlash RoPE operands");
    rope<<<dim3(heads, t), 64, 0, s>>>(x, heads, first);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void dflash_features(const std::uint16_t* x, float* o, int slots, int first, int t,
                     cudaStream_t s) {
    if (!x || !o || slots < 2048 || first < 0 || t < 1 || t > 2048)
        throw std::invalid_argument("DFlash feature extent");
    features<<<dim3(80, t), 256, 0, s>>>(x, o, slots, first);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void dflash_cache_append(const float* k, const float* v, std::uint16_t* kc, std::uint16_t* vc,
                         int slots, int first, int t, cudaStream_t s) {
    if (!k || !v || !kc || !vc || slots < 2048 || first < 0 || t < 1 || t > 2048)
        throw std::invalid_argument("DFlash cache append extent");
    append_cache<<<dim3(4, t), 256, 0, s>>>(k, v, kc, vc, slots, first);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void dflash_attention(const float* q, const std::uint16_t* kc, const std::uint16_t* vc,
                      const float* kn, const float* vn, float* p, float* out, int slots, int first,
                      int t, cudaStream_t s) {
    shape(t);
    if (!q || !kc || !vc || !kn || !vn || !p || !out || slots < 2048 || first < 0)
        throw std::invalid_argument("DFlash attention operands");
    attention_scores<<<dim3(32, t), 256, 0, s>>>(q, kc, kn, p, slots, first, t);
    attention_values<<<dim3(32, t), 128, 0, s>>>(vc, vn, p, out, slots, first, t);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void dflash_top16(const float* x, std::int32_t* ids, float* unary, int v, int t, cudaStream_t s) {
    shape(t);
    if (!x || !ids || !unary || v < 16) throw std::invalid_argument("DFlash top16 operands");
    top16<<<t, 256, 0, s>>>(x, ids, unary, v);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void dflash_select(const float* h, const std::uint16_t* pre, const std::uint16_t* suc,
                   const std::int32_t* ids, const float* unary, std::int32_t* out, int anchor,
                   int t, cudaStream_t s) {
    shape(t);
    if (!h || !pre || !suc || !ids || !unary || !out || anchor < 0)
        throw std::invalid_argument("DFlash selector operands");
    select<<<1, 32, 0, s>>>(h, pre, suc, ids, unary, out, anchor, t);
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops
