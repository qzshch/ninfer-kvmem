#include "ninfer/ops/resident_transform.h"
#include "core/device.h"
#include <cstdint>
#include <stdexcept>
namespace ninfer::ops {
namespace {
__device__ float bf(float x) {
    const auto bits = __float_as_uint(x);
    const auto word = (bits & 0x7fffffffU) > 0x7f800000U
        ? (bits >> 16) | 0x40U : (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
    return __uint_as_float(word << 16);
}
#include "ops/resident_transform/argmax.cuh"
void shape(int width, int rows) {
    if (width < 1 || rows < 1 || rows > 65535)
        throw std::invalid_argument("parallel transform: dimensions");
}
__global__ void moe_partial_kernel(const float* experts, const float* weights, float* out,
                                    int hidden, int active) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c >= hidden) return;
    float sum = 0.f;
    for (int j = 0; j < active; ++j)
        sum += experts[(static_cast<long long>(t) * active + j) * hidden + c] * weights[t * active + j];
    out[static_cast<long long>(t) * hidden + c] = sum;
}
__global__ void tp_sum_kernel(const float* a, const float* b, const float* c, float* out,
                              std::size_t n, bool routed) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    float value = (a[i] + b[i]) + c[i];
    if (routed) value = bf(value) + ((a[n + i] + b[n + i]) + c[n + i]);
    out[i] = bf(value);
}

__global__ void argmax_merge_kernel(const ArgmaxPair* pairs, std::int32_t* ids, int rows) {
    const int t = threadIdx.x;
    if (t >= rows) return;
    auto best = pairs[t];
    bool invalid = best.id < 0;
    for (int d = 1; d < 3; ++d) {
        const auto p = pairs[d * rows + t];
        invalid |= p.id < 0;
        if (p.value > best.value || (p.value == best.value && p.id < best.id)) best = p;
    }
    ids[t] = invalid ? -1 : best.id;
}
} // namespace
void resident_moe_partial(const float* x, const float* weights, float* out, int h, int active,
                          int tokens, cudaStream_t s) {
    if (!x || !weights || !out || h < 1 || active < 1 || active > 32 || tokens < 1 || tokens > 65535)
        throw std::invalid_argument("resident_moe_partial: operands");
    moe_partial_kernel<<<dim3((h + 255) / 256, tokens), 256, 0, s>>>(x, weights, out, h, active);
    CUDA_CHECK(cudaGetLastError());
}

void resident_tp_sum(const float* a, const float* b, const float* c, float* out,
                     std::size_t count, bool routed, cudaStream_t s) {
    if (!a || !b || !c || !out || !count || count > 4096ULL * 1024 || out == a || out == b || out == c)
        throw std::invalid_argument("resident_tp_sum: operands");
    tp_sum_kernel<<<(count + 255) / 256, 256, 0, s>>>(a, b, c, out, count, routed);
    CUDA_CHECK(cudaGetLastError());
}

void resident_argmax_pairs(const float* x, ArgmaxPair* pairs, int width, int rows, int begin,
                           cudaStream_t s) {
    shape(width, rows);
    if (!x || !pairs || begin < 0 || width > INT32_MAX - begin)
        throw std::invalid_argument("resident_argmax_pairs: operands/origin");
    argmax_kernel<<<rows, 256, 0, s>>>(x, nullptr, width, pairs, begin);
    CUDA_CHECK(cudaGetLastError());
}

void resident_argmax_merge(const ArgmaxPair* pairs, std::int32_t* ids, int rows, cudaStream_t s) {
    if (!pairs || !ids || rows < 1 || rows > 8)
        throw std::invalid_argument("resident_argmax_merge: operands/rows");
    argmax_merge_kernel<<<1, 32, 0, s>>>(pairs, ids, rows);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
