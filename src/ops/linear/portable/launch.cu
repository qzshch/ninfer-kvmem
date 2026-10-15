// Native NInfer matrix math for the DeepSeek artifact formats. Apache-2.0.
// DTK groups large rows by wave64; other profiles retain the block reduction.
// No hidden allocation or external engine.
#include "ops/linear/portable/launch.h"
#include "core/device.h"
#include "ops/linear/portable/mmac_mapping.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
#include "ops/linear/portable/weight_decode.cuh"

__device__ float projection_output(float x, bool rounded) {
    return rounded ? decode_bf16(encode_bf16(x)) : x;
}
#include "ops/linear/portable/q8_tiles.cuh"
#include "ops/linear/portable/mmac_tiles.cuh"
#ifdef NINFER_HYGON_DTK
#    include "ops/linear/portable/wave64_dot.cuh"

template <QType Format, bool Bf16, bool OutputBf16 = Bf16>
__global__ void projection_wave64(const void* x, const void* w, void* out, int n, int k,
                                  bool rounded) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * 4 + threadIdx.x / 64;
    if (row >= n) return;
    const auto token = static_cast<std::int64_t>(blockIdx.y);
    const void* input;
    if constexpr (Bf16)
        input = static_cast<const std::uint16_t*>(x) + token * k;
    else
        input = static_cast<const float*>(x) + token * k;
    const float sum = dot_wave64<Format, Bf16>(input, w, row, k);
    if ((threadIdx.x & 63) == 0) {
        if constexpr (OutputBf16)
            static_cast<std::uint16_t*>(out)[token * n + row] = encode_bf16(sum);
        else
            static_cast<float*>(out)[token * n + row] = projection_output(sum, rounded);
    }
    return;
}
#endif

template <QType Format, bool Bf16, bool OutputBf16 = Bf16>
__global__ void projection(const void* x, const void* w, void* out, int n, int k, bool rounded) {
    __shared__ float partial[256];
    const auto row     = static_cast<std::int64_t>(blockIdx.x);
    const auto token   = static_cast<std::int64_t>(blockIdx.y);
    const unsigned tid = threadIdx.x;
    float sum          = 0.0f;
    for (std::int64_t column = tid; column < k; column += 256) {
        float value;
        if constexpr (Bf16) {
            value = decode_bf16(static_cast<const std::uint16_t*>(x)[token * k + column]);
        } else {
            value = static_cast<const float*>(x)[token * k + column];
        }
        sum += load_weight<Format>(w, row, static_cast<int>(column), k) * value;
    }
    partial[tid] = sum;
    __syncthreads();
#ifdef NINFER_HYGON_DTK
    if constexpr (Format == QType::Q8_G32_FP16_BLOCK) {
        // Preserve the 256-column accumulation and reduction tree. Only its
        // first two joins cross wave64 boundaries; finish inside the first
        // wave instead of synchronizing the whole block after every join.
        if (tid < 128) partial[tid] += partial[tid + 128];
        __syncthreads();
        if (tid < 64) {
            float reduced = partial[tid] + partial[tid + 64];
            for (int distance = 32; distance; distance /= 2)
                reduced += __shfl_down(reduced, distance, 64);
            if (tid == 0) partial[0] = reduced;
        }
    } else
#endif
        for (unsigned distance = 128; distance != 0; distance /= 2) {
            if (tid < distance) { partial[tid] += partial[tid + distance]; }
            __syncthreads();
        }
    if (tid == 0) {
        if constexpr (OutputBf16) {
            static_cast<std::uint16_t*>(out)[token * n + row] = encode_bf16(partial[0]);
        } else {
            static_cast<float*>(out)[token * n + row] = projection_output(partial[0], rounded);
        }
    }
    return;
}


#ifdef NINFER_HYGON_DTK
// Four independent activation columns share every represented weight load and
// decode. Each column retains the existing 256-lane accumulation/reduction tree.
// Tails are masked; there is no padding request or duplicated mutable state.
template <QType Format, bool Bf16, bool OutputBf16 = Bf16>
__global__ void projection_columns4(const void* x, const void* w, void* out, int n, int k,
                                    int tokens, bool rounded, int parent_k, int begin) {
    const int row = blockIdx.x, base = blockIdx.y * 4, lane = threadIdx.x;
    float sum[4]{};
    for (int column = lane; column < k; column += 256) {
        const float weight = load_weight<Format>(w, row, begin + column, parent_k);
        for (int t = 0; t < 4; ++t)
            if (base + t < tokens) {
                float value;
                if constexpr (Bf16)
                    value =
                        decode_bf16(static_cast<const std::uint16_t*>(x)[(base + t) * k + column]);
                else
                    value = static_cast<const float*>(x)[(base + t) * k + column];
                sum[t] += weight * value;
            }
    }
    __shared__ float partial[4][256];
    for (int t = 0; t < 4; ++t) partial[t][lane] = sum[t];
    __syncthreads();
    if (lane < 128)
        for (int t = 0; t < 4; ++t) partial[t][lane] += partial[t][lane + 128];
    __syncthreads();
    if (lane < 64)
        for (int t = 0; t < 4; ++t) {
            float value = partial[t][lane] + partial[t][lane + 64];
            for (int distance = 32; distance; distance /= 2)
                value += __shfl_down(value, distance, 64);
            if (!lane && base + t < tokens) {
                if constexpr (OutputBf16)
                    static_cast<std::uint16_t*>(out)[(base + t) * n + row] = encode_bf16(value);
                else
                    static_cast<float*>(out)[(base + t) * n + row] =
                        projection_output(value, rounded);
            }
        }
    return;
}
#endif

template <QType Format, bool InputBf16, bool OutputBf16>
void launch_typed(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream, bool rounded) {
#ifdef NINFER_K100_MMAC
    if (InputBf16 && x.ne[1] >= 16 && w.n >= 128 && w.k >= 128) {
        mmac_projection<Format, InputBf16, OutputBf16><<<dim3((w.n + 63) / 64, (x.ne[1] + 15) / 16), 256, 0, stream>>>(
            x.data, w.qdata, out.data, w.n, w.k, 0, w.k, x.ne[1], rounded);
        return;
    }
#endif
#ifdef NINFER_HYGON_DTK
    if constexpr (Format == QType::Q8_G32_FP16_BLOCK) {
        if (w.n >= 128 && ((w.k >= 512 && x.ne[1] == 1) || x.ne[1] >= 16)) {
            if (x.ne[1] == 1)
                q8_decode_launch<InputBf16, OutputBf16>(x.data, w.qdata, out.data,
                    w.n, w.k, 0, w.k, rounded, stream);
            else
                q8_matrix_tile<InputBf16, OutputBf16><<<dim3((w.n + 7) / 8, (x.ne[1] + 7) / 8), 256, 0, stream>>>(
                    x.data, w.qdata, out.data, w.n, w.k, 0, w.k, x.ne[1], rounded);
            return;
        }
    }
    if (x.ne[1] > 1 && (Format == QType::Q8_G32_FP16_BLOCK || Format == QType::Q6_K_G256_BLOCK) && w.n >= 128) {
        projection_columns4<Format, InputBf16, OutputBf16><<<dim3(w.n, (x.ne[1] + 3) / 4), 256, 0, stream>>>(
            x.data, w.qdata, out.data, w.n, w.k, x.ne[1], rounded, w.k, 0);
        return;
    }
    if (Format != QType::Q8_G32_FP16_BLOCK && w.n >= 512 && w.k % 256 == 0) {
        projection_wave64<Format, InputBf16, OutputBf16><<<dim3((w.n + 3) / 4, x.ne[1]), 256, 0, stream>>>(
            x.data, w.qdata, out.data, w.n, w.k, rounded);
        return;
    }
#endif
    projection<Format, InputBf16, OutputBf16><<<dim3(w.n, x.ne[1]), 256, 0, stream>>>(
        x.data, w.qdata, out.data, w.n, w.k, rounded);
}

template <QType Format>
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream, bool rounded) {
    if (x.dtype == DType::BF16) {
        if (out.dtype == DType::BF16) launch_typed<Format, true, true>(x, w, out, stream, rounded);
        else launch_typed<Format, true, false>(x, w, out, stream, rounded);
    } else {
        if (out.dtype == DType::BF16) launch_typed<Format, false, true>(x, w, out, stream, rounded);
        else launch_typed<Format, false, false>(x, w, out, stream, rounded);
    }
}

template <QType F, bool InputBf16>
__global__ void partial_projection(const void* x, const void* w, float* out, int n,
                                    int parent_k, int begin, int k) {
    const int row = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    float sum = 0.f;
    for (int c = lane; c < k; c += 256)
        sum += load_weight<F>(w, row, begin + c, parent_k) * tile_input<InputBf16>(x, static_cast<long long>(t) * k + c);
    __shared__ float partial[256];
    partial[lane] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) partial[lane] += partial[lane + s];
        __syncthreads();
    }
    if (!lane) out[static_cast<long long>(t) * n + row] = partial[0];
}

#ifdef NINFER_HYGON_DTK
template <QType F, bool InputBf16>
__global__ void partial_wave64(const void* x, const void* weights, float* out, int n,
                               int parent_k, int begin, int k) {
    const int row = blockIdx.x * 4 + threadIdx.x / 64, t = blockIdx.y;
    if (row >= n) return;
    const auto* input = static_cast<const unsigned char*>(x) + static_cast<long long>(t) * k * (InputBf16 ? 2 : 4);
    const float value = dot_wave64<F, InputBf16>(input, weights,
                                            row, k, parent_k, begin);
    if (!(threadIdx.x & 63)) out[static_cast<long long>(t) * n + row] = value;
}
#endif

template <QType F, bool InputBf16>
void partial_launch(const void* x, const Weight& w, float* out, int begin, int k, int t,
                     cudaStream_t s) {
#ifdef NINFER_K100_MMAC
    if (InputBf16 && t >= 16 && w.n >= 128 && k >= 128) {
        mmac_projection<F, InputBf16, false><<<dim3((w.n + 63) / 64, (t + 15) / 16), 256, 0, s>>>(
            x, w.qdata, out, w.n, w.k, begin, k, t, false);
        return;
    }
#endif
    if constexpr (F == QType::Q8_G32_FP16_BLOCK) {
        if (w.n >= 128 && ((k >= 512 && t == 1) || t >= 16)) {
            if (t == 1)
                q8_decode_launch<InputBf16, false>(x, w.qdata, out, w.n, w.k, begin, k, false, s);
            else
                q8_matrix_tile<InputBf16, false><<<dim3((w.n + 7) / 8, (t + 7) / 8), 256, 0, s>>>(
                    x, w.qdata, out, w.n, w.k, begin, k, t, false);
            return;
        }
    }
#ifdef NINFER_HYGON_DTK
    if (t > 1 && (F == QType::Q8_G32_FP16_BLOCK || F == QType::Q6_K_G256_BLOCK) && w.n >= 128) {
        projection_columns4<F, InputBf16, false><<<dim3(w.n, (t + 3) / 4), 256, 0, s>>>(
            x, w.qdata, out, w.n, k, t, false, w.k, begin);
        return;
    }
    if (F != QType::Q8_G32_FP16_BLOCK && w.n >= 512 && k % 256 == 0 && begin % 256 == 0) {
        partial_wave64<F, InputBf16><<<dim3((w.n + 3) / 4, t), 256, 0, s>>>(x, w.qdata, out, w.n, w.k, begin, k);
        return;
    }
#endif
    partial_projection<F, InputBf16><<<dim3(w.n, t), 256, 0, s>>>(x, w.qdata, out, w.n, w.k, begin, k);
}
} // namespace

void portable_partial_linear_launch(const void* x, DType input_type, const Weight& w, float* out, int begin, int k,
                                     int t, cudaStream_t stream) {
    validate_portable_linear_weight(w.qtype, w.n, w.k);
    const auto block = interleaved_block_geometry(w.qtype).values;
    if ((input_type != DType::FP32 && input_type != DType::BF16) || !x || !out || x == out || !w.qdata || begin < 0 || k < 1 || k > w.k || begin > w.k - k ||
        t < 1 || t > 65535 || (block && (begin % block || k % block)) ||
        w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous))
        throw std::invalid_argument("partial linear: invalid represented K interval");
#define PARTIAL(F) case QType::F: \
    if (input_type == DType::BF16) partial_launch<QType::F, true>(x, w, out, begin, k, t, stream); \
    else partial_launch<QType::F, false>(x, w, out, begin, k, t, stream); break
    switch (w.qtype) {
        PARTIAL(FP32); PARTIAL(FP16); PARTIAL(BF16); PARTIAL(MXFP4_G32_E8M0);
        PARTIAL(Q8_G32_FP16_BLOCK); PARTIAL(IQ3_S_G256_BLOCK); PARTIAL(IQ4_XS_G256_BLOCK);
        PARTIAL(Q3_K_G256_BLOCK); PARTIAL(Q4_K_G256_BLOCK); PARTIAL(Q6_K_G256_BLOCK);
        default: throw std::invalid_argument("partial linear: unsupported format");
    }
#undef PARTIAL
    CUDA_CHECK(cudaGetLastError());
}

void validate_portable_linear_weight(QType type, std::int32_t n, std::int32_t k) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("linear: positive weight dimensions required");
    }
    switch (type) {
    case QType::FP32:
    case QType::FP16:
    case QType::BF16:
        return;
    case QType::MXFP4_G32_E8M0:
    case QType::Q8_G32_FP16_BLOCK:
        if (k % 32 == 0) { return; }
        throw std::invalid_argument("linear: block format requires K%32=0");
    case QType::IQ3_S_G256_BLOCK:
    case QType::IQ4_XS_G256_BLOCK:
    case QType::Q3_K_G256_BLOCK:
    case QType::Q4_K_G256_BLOCK:
    case QType::Q6_K_G256_BLOCK:
        if (k % 256 == 0) return;
        throw std::invalid_argument("linear: block format requires K%256=0");
    default:
        throw std::invalid_argument("linear: format has no portable projection implementation");
    }
}

void portable_linear_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                            bool bf16_output) {
    validate_portable_linear_weight(w.qtype, w.n, w.k);
    if (!w.qdata || (x.dtype != DType::BF16 && x.dtype != DType::FP32) || (out.dtype != DType::BF16 && out.dtype != DType::FP32) ||
        x.ne[1] > 65535 || x.ne[1] <= 0) {
        throw std::invalid_argument("linear: invalid data, dtype, or token grid extent");
    }
    const bool block = interleaved_block_geometry(w.qtype).values != 0;
    if (w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous)) {
        throw std::invalid_argument("linear: persistent format/layout mismatch");
    }
    switch (w.qtype) {
    case QType::FP32:
        launch<QType::FP32>(x, w, out, stream, bf16_output);
        break;
    case QType::FP16:
        launch<QType::FP16>(x, w, out, stream, bf16_output);
        break;
    case QType::BF16:
        launch<QType::BF16>(x, w, out, stream, bf16_output);
        break;
    case QType::MXFP4_G32_E8M0:
        launch<QType::MXFP4_G32_E8M0>(x, w, out, stream, bf16_output);
        break;
    case QType::Q8_G32_FP16_BLOCK:
        launch<QType::Q8_G32_FP16_BLOCK>(x, w, out, stream, bf16_output);
        break;
    case QType::IQ3_S_G256_BLOCK:
        launch<QType::IQ3_S_G256_BLOCK>(x, w, out, stream, bf16_output);
        break;
    case QType::IQ4_XS_G256_BLOCK:
        launch<QType::IQ4_XS_G256_BLOCK>(x, w, out, stream, bf16_output);
        break;
    case QType::Q3_K_G256_BLOCK:
        launch<QType::Q3_K_G256_BLOCK>(x, w, out, stream, bf16_output);
        break;
    case QType::Q4_K_G256_BLOCK:
        launch<QType::Q4_K_G256_BLOCK>(x, w, out, stream, bf16_output);
        break;
    case QType::Q6_K_G256_BLOCK:
        launch<QType::Q6_K_G256_BLOCK>(x, w, out, stream, bf16_output);
        break;
    default:
        throw std::logic_error("validated projection type was not dispatched");
    }
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops::detail
