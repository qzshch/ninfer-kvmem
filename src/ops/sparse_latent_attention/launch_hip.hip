#include "ninfer/ops/sparse_latent_attention.h"
#include "core/device.h"
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
__device__ float ub(std::uint16_t x) { return __uint_as_float(static_cast<unsigned>(x) << 16); }

__device__ std::uint16_t bw(float v) {
    auto b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffu + ((b >> 16) & 1)) >> 16);
}

__device__ float bf(float v) { return ub(bw(v)); }

__global__ void pool_append(const float* key, const float* gate, const float* bias,
                            std::uint16_t* keys, std::uint16_t* gates, std::uint16_t* pools,
                            int position) {
    const int c               = threadIdx.x;
    keys[position * 128 + c]  = bw(key[c]);
    gates[position * 128 + c] = bw(gate[c]);
    if (position % 4 != 3) return;
    const int begin = position - 3;
    float a[4], m = -INFINITY, sum = 0;
    for (int j = 0; j < 4; ++j) {
        a[j] = ub(gates[(begin + j) * 128 + c]) + bias[j * 128 + c];
        m    = fmaxf(m, a[j]);
    }
    for (int j = 0; j < 4; ++j) {
        a[j] = expf(a[j] - m);
        sum += a[j];
    }
    float value = 0;
    for (int j = 0; j < 4; ++j) value += bf(bf(a[j] / sum) * ub(keys[(begin + j) * 128 + c]));
    pools[(position / 4) * 128 + c] = bw(value);
    return;
}

__global__ void pool_scores(const float* query, const float* weights, const std::uint16_t* pools,
                            const std::uint8_t* allowed, float* scores) {
    const int pool = blockIdx.x, h = threadIdx.x;
    __shared__ float red[64];
    if (allowed && !allowed[pool / 16]) {
        if (!h) scores[pool] = -INFINITY;
        return;
    }
    float sum = 0;
    if (h < 32) {
        for (int c = 0; c < 128; ++c) sum += query[h * 128 + c] * ub(pools[pool * 128 + c]);
        sum = fmaxf(sum * .08838834764831844f, 0.f) * weights[h] * .17677669529663688f;
    }
    red[h] = sum;
    __syncthreads();
    for (int s = 32; s; s /= 2) {
        if (h < s) red[h] += red[h + s];
        __syncthreads();
    }
    if (!h) scores[pool] = red[0];
    return;
}

__global__ void pool_select(float* scores, std::int32_t* ids, int visible) {
    const int lane = threadIdx.x, complete = visible / 4;
    __shared__ float value[256];
    __shared__ int index[256];
    for (int i = lane; i < 2051; i += 256) ids[i] = -1;
    __syncthreads();
    for (int j = 0; j < min(complete, 512); ++j) {
        float best = -INFINITY;
        int id     = 0;
        for (int pool = lane; pool < complete; pool += 256) {
            const auto v = scores[pool];
            if (v > best || (v == best && pool < id)) {
                best = v;
                id   = pool;
            }
        }
        value[lane] = best;
        index[lane] = id;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (lane < s && (value[lane + s] > value[lane] ||
                             (value[lane + s] == value[lane] && index[lane + s] < index[lane]))) {
                value[lane] = value[lane + s];
                index[lane] = index[lane + s];
            }
            __syncthreads();
        }
        if (!lane && isfinite(value[0])) {
            const int p = index[0];
            for (int k = 0; k < 4; ++k) ids[j * 4 + k] = p * 4 + k;
            scores[p] = -INFINITY;
        }
        __syncthreads();
    }
    if (lane < visible % 4) ids[2048 + lane] = (visible / 4) * 4 + lane;
    return;
}

__global__ void attention_scores(const float* q, const std::uint16_t* kv, const std::int32_t* ids,
                                 float* p, int heads, int dim, int count, float scale) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const auto* query = q + (static_cast<long long>(t) * heads + h) * dim;
    auto* prob        = p + (static_cast<long long>(t) * heads + h) * count;
    __shared__ float reduction[256];
    float maximum = -INFINITY;
    for (int i = lane; i < count; i += 256) {
        const int pos = ids[t * count + i];
        float dot     = -INFINITY;
        if (pos >= 0) {
            dot = 0;
            for (int c = 0; c < dim; ++c)
                dot += query[c] * ub(kv[static_cast<long long>(pos) * dim + c]);
            dot *= scale;
        }
        prob[i] = dot;
        maximum = fmaxf(maximum, dot);
    }
    reduction[lane] = maximum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) reduction[lane] = fmaxf(reduction[lane], reduction[lane + s]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    float sum = 0;
    for (int i = lane; i < count; i += 256) {
        float v = isfinite(maximum) ? expf(prob[i] - maximum) : 0;
        prob[i] = v;
        sum += v;
    }
    reduction[lane] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) reduction[lane] += reduction[lane + s];
        __syncthreads();
    }
    sum = reduction[0];
    __syncthreads();
    for (int i = lane; i < count; i += 256) prob[i] = sum > 0 ? prob[i] / sum : 0;
    return;
}

__global__ void attention_values(const std::uint16_t* kv, const std::int32_t* ids, const float* p,
                                 float* out, int heads, int dim, int count) {
    const int h = blockIdx.y, t = blockIdx.z, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= dim) return;
    const auto* prob = p + (static_cast<long long>(t) * heads + h) * count;
    float sum        = 0;
    for (int i = 0; i < count; ++i) {
        int pos = ids[t * count + i];
        if (pos >= 0) sum += prob[i] * ub(kv[static_cast<long long>(pos) * dim + c]);
    }
    out[(static_cast<long long>(t) * heads + h) * dim + c] = sum;
    return;
}
} // namespace

void index_pool_append(const float* k, const float* g, const float* b, std::uint16_t* keys,
                       std::uint16_t* gates, std::uint16_t* pools, int pos, cudaStream_t s) {
    if (!k || !g || !b || !keys || !gates || !pools || pos < 0 || pos >= 32768)
        throw std::invalid_argument("index_pool_append: operands/position");
    pool_append<<<1, 128, 0, s>>>(k, g, b, keys, gates, pools, pos);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void index_pool_scores(const float* q, const float* w, const std::uint16_t* p,
                       const std::uint8_t* allowed, float* scores, int n, cudaStream_t s) {
    if (!q || !w || !p || !scores || n < 0 || n > 8192)
        throw std::invalid_argument("index_pool_scores: operands/count");
    if (n) pool_scores<<<n, 64, 0, s>>>(q, w, p, allowed, scores);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void index_pool_select(float* scores, std::int32_t* ids, int visible, cudaStream_t s) {
    if (!scores || !ids || visible < 1 || visible > 32768)
        throw std::invalid_argument("index_pool_select: operands/count");
    pool_select<<<1, 256, 0, s>>>(scores, ids, visible);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void sparse_latent_attention(const float* q, const std::uint16_t* kv, const std::int32_t* ids,
                             float* p, float* out, int h, int dim, int t, int count, float scale,
                             cudaStream_t s) {
    if (!q || !kv || !ids || !p || !out || h < 1 || h > 128 || dim < 1 || dim > 512 || t < 1 ||
        t > 8 || count < 1 || count > 32768 || !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("sparse_latent_attention: operands/geometry");
    attention_scores<<<dim3(h, t), 256, 0, s>>>(q, kv, ids, p, h, dim, count, scale);
    CUDA_CHECK(cudaGetLastError());
    attention_values<<<dim3((dim + 255) / 256, h, t), 256, 0, s>>>(kv, ids, p, out, h, dim, count);
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops
