#include "ninfer/ops/indexed_linear.h"
#include "ops/linear/portable/launch.h"
#include "core/device.h"
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {
#include "ops/linear/portable/weight_decode.cuh"

__device__ float projection_output(float x, bool rounded) {
    return rounded ? decode_bf16(encode_bf16(x)) : x;
}
#ifdef NINFER_HYGON_DTK
#    include "ops/linear/portable/wave64_dot.cuh"

template <QType F>
__global__ void indexed_wave64(const float* x, const void* w, const std::int32_t* ids, float* out,
                               int n, int k, int choices, bool shared, bool rounded, int first, int banks) {
    const int row = blockIdx.x * 4 + threadIdx.x / 64;
    if (row >= n) return;
    const int t = blockIdx.y, c = blockIdx.z;
    const auto bank = static_cast<long long>(ids ? ids[t * choices + c] : c) - first;
    if (bank < 0 || bank >= banks) {
        if ((threadIdx.x & 63) == 0) out[(static_cast<long long>(t) * choices + c) * n + row] = 0.f;
        return;
    }
    const auto* input =
        x + (static_cast<long long>(t) * (shared ? 1 : choices) + (shared ? 0 : c)) * k;
    const float sum = dot_wave64<F, false>(input, w, static_cast<long long>(bank) * n + row, k);
    if ((threadIdx.x & 63) == 0)
        out[(static_cast<long long>(t) * choices + c) * n + row] = projection_output(sum, rounded);
    return;
}
#endif

template <QType F>
__global__ void indexed_kernel(const float* x, const void* w, const std::int32_t* ids, float* out,
                               int n, int k, int choices, bool shared, bool rounded, int first, int banks) {
    const int row = blockIdx.x, t = blockIdx.y, c = blockIdx.z, lane = threadIdx.x;
    const auto bank = static_cast<long long>(ids ? ids[t * choices + c] : c) - first;
    if (bank < 0 || bank >= banks) {
        if (!lane) out[(static_cast<long long>(t) * choices + c) * n + row] = 0.f;
        return;
    }
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
    if (!lane)
        out[(static_cast<long long>(t) * choices + c) * n + row] =
            projection_output(partial[0], rounded);
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
            int choices, int t, bool shared, cudaStream_t s, bool rounded, int first) {
    if (x) {
#ifdef NINFER_HYGON_DTK
        if (rows >= 512 && w.k % 256 == 0) {
            indexed_wave64<F>
                <<<dim3((static_cast<unsigned>(rows) + 3) / 4, t, choices), 256, 0, s>>>(
                    x, w.qdata, ids, out, rows, w.k, choices, shared, rounded, first, w.n / rows);
            return;
        }
#endif
        indexed_kernel<F><<<dim3(rows, t, choices), 256, 0, s>>>(x, w.qdata, ids, out, rows, w.k,
                                                                 choices, shared, rounded, first, w.n / rows);
    } else
        embedding_kernel<F><<<dim3((w.k + 255) / 256, t), 256, 0, s>>>(w.qdata, ids, out, w.k);
    return;
}

void dispatch(const float* x, const Weight& w, const std::int32_t* ids, float* o, int n, int c,
              int t, bool shared, cudaStream_t s, bool rounded, int first = 0) {
    detail::validate_portable_linear_weight(w.qtype, w.n, w.k);
    const bool block = interleaved_block_geometry(w.qtype).values != 0;
    if (!w.qdata || !o ||
        w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous) || t < 1 ||
        t > 65535 || c < 1 || c > 65535 || n < 1 || w.n % n)
        throw std::invalid_argument("indexed_linear: invalid bank/extent/layout");
#define DISPATCH(F)                                                                                \
    case QType::F:                                                                                 \
        launch<QType::F>(x, w, ids, o, n, c, t, shared, s, rounded, first);                        \
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
                         int c, int t, bool shared, cudaStream_t s, bool rounded) {
    if (!x || x == o || n < 1 || (ids == nullptr && w.n / n < c))
        throw std::invalid_argument("indexed_linear: input or bank coverage");
    dispatch(x, w, ids, o, n, c, t, shared, s, rounded);
    return;
}

void indexed_linear_owned_fp32(const float* x, const Weight& w, const std::int32_t* ids, float* out,
                                int rows, int choices, int tokens, bool shared, int first,
                                cudaStream_t stream, bool rounded) {
    if (!x || !ids || x == out || rows < 1 || first < 0)
        throw std::invalid_argument("owned indexed linear: input or expert origin");
    dispatch(x, w, ids, out, rows, choices, tokens, shared, stream, rounded, first);
}

void embedding_fp32(const Weight& w, const std::int32_t* ids, float* o, int t, cudaStream_t s) {
    if (!ids) throw std::invalid_argument("embedding: missing device ids");
    dispatch(nullptr, w, ids, o, w.n, 1, t, true, s, false);
    return;
}
} // namespace ninfer::ops
