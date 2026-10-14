#include "ninfer/ops/indexed_linear.h"
#include "ops/linear/portable/launch.h"
#include "core/device.h"
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {
#include "ops/linear/portable/weight_decode.cuh"

template <QType F>
__global__ void indexed_kernel(const float* x, const void* w, const std::int32_t* ids, float* out,
                               int n, int k, int choices, bool shared) {
    const int row = blockIdx.x, t = blockIdx.y, c = blockIdx.z, lane = threadIdx.x;
    const int bank = ids ? ids[t * choices + c] : c;
    const auto* input =
        x + (static_cast<long long>(t) * (shared ? 1 : choices) + (shared ? 0 : c)) * k;
    float sum = 0;
    for (int col = lane; col < k; col += 256)
        sum += load_weight<F>(w, static_cast<long long>(bank) * n + row, col, k) * input[col];
    __shared__ float partial[256];
    partial[lane] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) partial[lane] += partial[lane + s];
        __syncthreads();
    }
    if (!lane) out[(static_cast<long long>(t) * choices + c) * n + row] = partial[0];
    return;
}

template <QType F>
__global__ void embedding_kernel(const void* w, const std::int32_t* ids, float* out, int k) {
    const int col = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (col < k)
        out[static_cast<long long>(t) * k + col] =
            decode_bf16(encode_bf16(load_weight<F>(w, ids[t], col, k)));
    return;
}

template <QType F>
void launch(const float* x, const Weight& w, const std::int32_t* ids, float* out, int rows,
            int choices, int t, bool shared, cudaStream_t s) {
    if (x)
        indexed_kernel<F><<<dim3(rows, t, choices), 256, 0, s>>>(x, w.qdata, ids, out, rows, w.k,
                                                                 choices, shared);
    else
        embedding_kernel<F><<<dim3((w.k + 255) / 256, t), 256, 0, s>>>(w.qdata, ids, out, w.k);
    return;
}

void dispatch(const float* x, const Weight& w, const std::int32_t* ids, float* o, int n, int c,
              int t, bool shared, cudaStream_t s) {
    detail::validate_portable_linear_weight(w.qtype, w.n, w.k);
    const bool block = interleaved_block_geometry(w.qtype).values != 0;
    if (!w.qdata || !o ||
        w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous) || t < 1 ||
        t > 65535 || c < 1 || c > 65535 || n < 1 || w.n % n)
        throw std::invalid_argument("indexed_linear: invalid bank/extent/layout");
#define DISPATCH(F)                                                                                \
    case QType::F:                                                                                 \
        launch<QType::F>(x, w, ids, o, n, c, t, shared, s);                                        \
        break
    switch (w.qtype) {
        DISPATCH(FP32);
        DISPATCH(FP16);
        DISPATCH(BF16);
        DISPATCH(MXFP4_G32_E8M0);
        DISPATCH(Q8_G32_FP16_BLOCK);
        DISPATCH(IQ3_S_G256_BLOCK);
        DISPATCH(IQ4_XS_G256_BLOCK);
        DISPATCH(Q3_K_G256_BLOCK);
        DISPATCH(Q4_K_G256_BLOCK);
        DISPATCH(Q6_K_G256_BLOCK);
    default:
        throw std::invalid_argument("indexed_linear: unsupported format");
    }
#undef DISPATCH
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace

void indexed_linear_fp32(const float* x, const Weight& w, const std::int32_t* ids, float* o, int n,
                         int c, int t, bool shared, cudaStream_t s) {
    if (!x || x == o || n < 1 || (ids == nullptr && w.n / n < c))
        throw std::invalid_argument("indexed_linear: input or bank coverage");
    dispatch(x, w, ids, o, n, c, t, shared, s);
    return;
}

void embedding_fp32(const Weight& w, const std::int32_t* ids, float* o, int t, cudaStream_t s) {
    if (!ids) throw std::invalid_argument("embedding: missing device ids");
    dispatch(nullptr, w, ids, o, w.n, 1, t, true, s);
    return;
}
} // namespace ninfer::ops
