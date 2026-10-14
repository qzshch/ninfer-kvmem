// Native NInfer matrix math for the DeepSeek artifact formats. Apache-2.0.
// One row/token workgroup; no hidden allocation, external engine, or warp-width assumption.
#include "ops/linear/portable/launch.h"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
#include "ops/linear/portable/weight_decode.cuh"

template <QType Format, bool Bf16>
__global__ void projection(const void* x, const void* w, void* out, int n, int k) {
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
    for (unsigned distance = 128; distance != 0; distance /= 2) {
        if (tid < distance) { partial[tid] += partial[tid + distance]; }
        __syncthreads();
    }
    if (tid == 0) {
        if constexpr (Bf16) {
            static_cast<std::uint16_t*>(out)[token * n + row] = encode_bf16(partial[0]);
        } else {
            static_cast<float*>(out)[token * n + row] = partial[0];
        }
    }
    return;
}

template <QType Format>
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(w.n), static_cast<unsigned>(x.ne[1]));
    if (x.dtype == DType::BF16) {
        projection<Format, true><<<grid, 256, 0, stream>>>(x.data, w.qdata, out.data, w.n, w.k);
    } else {
        projection<Format, false><<<grid, 256, 0, stream>>>(x.data, w.qdata, out.data, w.n, w.k);
    }
    return;
}
} // namespace

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

void portable_linear_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_portable_linear_weight(w.qtype, w.n, w.k);
    if (!w.qdata || (x.dtype != DType::BF16 && x.dtype != DType::FP32) || out.dtype != x.dtype ||
        x.ne[1] > 65535 || x.ne[1] <= 0) {
        throw std::invalid_argument("linear: invalid data, dtype, or token grid extent");
    }
    const bool block = interleaved_block_geometry(w.qtype).values != 0;
    if (w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous)) {
        throw std::invalid_argument("linear: persistent format/layout mismatch");
    }
    switch (w.qtype) {
    case QType::FP32:
        launch<QType::FP32>(x, w, out, stream);
        break;
    case QType::FP16:
        launch<QType::FP16>(x, w, out, stream);
        break;
    case QType::BF16:
        launch<QType::BF16>(x, w, out, stream);
        break;
    case QType::MXFP4_G32_E8M0:
        launch<QType::MXFP4_G32_E8M0>(x, w, out, stream);
        break;
    case QType::Q8_G32_FP16_BLOCK:
        launch<QType::Q8_G32_FP16_BLOCK>(x, w, out, stream);
        break;
    case QType::IQ3_S_G256_BLOCK:
        launch<QType::IQ3_S_G256_BLOCK>(x, w, out, stream);
        break;
    case QType::IQ4_XS_G256_BLOCK:
        launch<QType::IQ4_XS_G256_BLOCK>(x, w, out, stream);
        break;
    case QType::Q3_K_G256_BLOCK:
        launch<QType::Q3_K_G256_BLOCK>(x, w, out, stream);
        break;
    case QType::Q4_K_G256_BLOCK:
        launch<QType::Q4_K_G256_BLOCK>(x, w, out, stream);
        break;
    case QType::Q6_K_G256_BLOCK:
        launch<QType::Q6_K_G256_BLOCK>(x, w, out, stream);
        break;
    default:
        throw std::logic_error("validated projection type was not dispatched");
    }
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops::detail
