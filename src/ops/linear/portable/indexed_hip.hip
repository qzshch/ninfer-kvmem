#include "ninfer/ops/indexed_linear.h"
#include "ops/linear/portable/launch.h"
#include "core/device.h"
#include "ops/linear/portable/mmac_mapping.h"
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {
#include "ops/linear/portable/weight_decode.cuh"
#include "ops/linear/portable/mmac_tiles.cuh"

__device__ float projection_output(float x, bool rounded) {
    return rounded ? decode_bf16(encode_bf16(x)) : x;
}
#ifdef NINFER_HYGON_DTK
#    include "ops/linear/portable/wave64_dot.cuh"

template <QType F, bool InputBf16, bool OutputBf16>
__global__ void indexed_wave64(const void* x, const void* w, const std::int32_t* ids, void* out,
                               int n, int k, int choices, bool shared, bool rounded, int first, int banks) {
    const int row = blockIdx.x * 4 + threadIdx.x / 64;
    if (row >= n) return;
    const int t = blockIdx.y, c = blockIdx.z;
    const auto bank = static_cast<long long>(ids ? ids[t * choices + c] : c) - first;
    if (bank < 0 || bank >= banks) {
        if ((threadIdx.x & 63) == 0) {
            const auto at = (static_cast<long long>(t) * choices + c) * n + row;
            if constexpr (OutputBf16) static_cast<std::uint16_t*>(out)[at] = 0;
            else static_cast<float*>(out)[at] = 0.f;
        }
        return;
    }
    const auto* input =
        static_cast<const unsigned char*>(x) + (static_cast<long long>(t) * (shared ? 1 : choices) + (shared ? 0 : c)) * k * (InputBf16 ? 2 : 4);
    const float sum = dot_wave64<F, InputBf16>(input, w, static_cast<long long>(bank) * n + row, k);
    if ((threadIdx.x & 63) == 0)
        if constexpr (OutputBf16) static_cast<std::uint16_t*>(out)[(static_cast<long long>(t) * choices + c) * n + row] = encode_bf16(sum);
        else static_cast<float*>(out)[(static_cast<long long>(t) * choices + c) * n + row] = projection_output(sum, rounded);
    return;
}
#endif

template <QType F, bool InputBf16, bool OutputBf16>
__global__ void indexed_kernel(const void* x, const void* w, const std::int32_t* ids, void* out,
                               int n, int k, int choices, bool shared, bool rounded, int first, int banks) {
    const int row = blockIdx.x, t = blockIdx.y, c = blockIdx.z, lane = threadIdx.x;
    const auto bank = static_cast<long long>(ids ? ids[t * choices + c] : c) - first;
    if (bank < 0 || bank >= banks) {
        if (!lane) {
            const auto at = (static_cast<long long>(t) * choices + c) * n + row;
            if constexpr (OutputBf16) static_cast<std::uint16_t*>(out)[at] = 0;
            else static_cast<float*>(out)[at] = 0.f;
        }
        return;
    }
    const auto* input =
        static_cast<const unsigned char*>(x) + (static_cast<long long>(t) * (shared ? 1 : choices) + (shared ? 0 : c)) * k * (InputBf16 ? 2 : 4);
    float sum = 0;
    for (int col = lane; col < k; col += 256)
        sum += load_weight<F>(w, static_cast<long long>(bank) * n + row, col, k) * (InputBf16 ? decode_bf16(reinterpret_cast<const std::uint16_t*>(input)[col]) : reinterpret_cast<const float*>(input)[col]);
    __shared__ float partial[256];
    partial[lane] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) partial[lane] += partial[lane + s];
        __syncthreads();
    }
    if (!lane)
        if constexpr (OutputBf16) static_cast<std::uint16_t*>(out)[(static_cast<long long>(t) * choices + c) * n + row] = encode_bf16(partial[0]);
        else static_cast<float*>(out)[(static_cast<long long>(t) * choices + c) * n + row] = projection_output(partial[0], rounded);
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

template <QType F, bool InputBf16, bool OutputBf16>
void launch(const void* x, const Weight& w, const std::int32_t* ids, void* out, int rows,
            int choices, int t, bool shared, cudaStream_t s, bool rounded, int first) {
    if (x) {
// Each indexed assignment has its own bank and one input row. A larger
        // request batch does not make this a multi-row matrix multiplication.
#ifdef NINFER_HYGON_DTK
        if (rows >= 512 && w.k % 256 == 0) {
            indexed_wave64<F, InputBf16, OutputBf16>
                <<<dim3((static_cast<unsigned>(rows) + 3) / 4, t, choices), 256, 0, s>>>(
                    x, w.qdata, ids, out, rows, w.k, choices, shared, rounded, first, w.n / rows);
            return;
        }
#endif
        indexed_kernel<F, InputBf16, OutputBf16><<<dim3(rows, t, choices), 256, 0, s>>>(x, w.qdata, ids, out, rows, w.k,
                                                                 choices, shared, rounded, first, w.n / rows);
    } else
        embedding_kernel<F><<<dim3((w.k + 255) / 256, t), 256, 0, s>>>(w.qdata, ids, static_cast<float*>(out), w.k);
    return;
}

template <bool InputBf16 = false, bool OutputBf16 = false>
void dispatch(const void* x, const Weight& w, const std::int32_t* ids, void* o, int n, int c,
              int t, bool shared, cudaStream_t s, bool rounded, int first = 0) {
    detail::validate_portable_linear_weight(w.qtype, w.n, w.k);
    const bool block = interleaved_block_geometry(w.qtype).values != 0;
    if (!w.qdata || !o ||
        w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous) || t < 1 ||
        t > 65535 || c < 1 || c > 65535 || n < 1 || w.n % n)
        throw std::invalid_argument("indexed_linear: invalid bank/extent/layout");
#define DISPATCH(F)                                                                                \
    case QType::F:                                                                                 \
        launch<QType::F, InputBf16, OutputBf16>(x, w, ids, o, n, c, t, shared, s, rounded, first);                        \
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

void indexed_linear_mixed(const Tensor& x, const Weight& w, const int* ids, Tensor& out,
                           int rows, int choices, int tokens, bool shared, int first, cudaStream_t stream) {
    if (!x.data || x.data == out.data || rows < 1 || first < 0 ||
        (x.dtype != DType::FP32 && x.dtype != DType::BF16) ||
        (out.dtype != DType::FP32 && out.dtype != DType::BF16) ||
        !x.is_contiguous() || !out.is_contiguous() ||
        x.ne[0] != w.k || x.numel() != static_cast<long long>(w.k) * tokens * (shared ? 1 : choices) ||
        out.numel() != static_cast<long long>(rows) * tokens * choices ||
        (!ids && (first || w.n / rows < choices)))
        throw std::invalid_argument("indexed mixed: invalid physical operands/coverage");
    if (x.dtype == DType::BF16) {
        if (out.dtype == DType::BF16) dispatch<true, true>(x.data, w, ids, out.data, rows, choices, tokens, shared, stream, false, first);
        else dispatch<true, false>(x.data, w, ids, out.data, rows, choices, tokens, shared, stream, false, first);
    } else {
        if (out.dtype == DType::BF16) dispatch<false, true>(x.data, w, ids, out.data, rows, choices, tokens, shared, stream, false, first);
        else dispatch<false, false>(x.data, w, ids, out.data, rows, choices, tokens, shared, stream, false, first);
    }
}

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
