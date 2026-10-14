// Native NInfer matrix math for the DeepSeek artifact formats. Apache-2.0.
// One row/token workgroup; no hidden allocation, external engine, or warp-width assumption.
#include "ops/linear/portable/launch.h"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
__device__ float decode_half(std::uint16_t word) {
    const auto exponent = (word >> 10) & 31;
    const auto mantissa = word & 1023;
    const float sign = word & 0x8000 ? -1.0f : 1.0f;
    if (exponent == 0) { return sign * ldexpf(static_cast<float>(mantissa), -24); }
    if (exponent == 31) { return sign * __uint_as_float(mantissa ? 0x7fc00000u : 0x7f800000u); }
    return sign * ldexpf(1.0f + mantissa / 1024.0f, exponent - 15);
}
__device__ float decode_bf16(std::uint16_t word) {
    return __uint_as_float(static_cast<std::uint32_t>(word) << 16);
}
__device__ std::uint16_t encode_bf16(float value) {
    const auto bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    }
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
template<QType Format>
__device__ float load_weight(const void* data, std::int64_t row, std::int32_t column, int k) {
    const auto index = row * k + column;
    if constexpr (Format == QType::FP32) { return static_cast<const float*>(data)[index]; }
    if constexpr (Format == QType::BF16) { return decode_bf16(static_cast<const std::uint16_t*>(data)[index]); }
    if constexpr (Format == QType::FP16) { return decode_half(static_cast<const std::uint16_t*>(data)[index]); }
    if constexpr (Format == QType::MXFP4_G32_E8M0) {
        const auto* block = static_cast<const unsigned char*>(data) + (row * (k / 32) + column / 32) * 17;
        const unsigned lane = column % 32;
        const unsigned code = (block[1 + lane % 16] >> (lane < 16 ? 0 : 4)) & 15;
        const unsigned magnitude = code & 7;
        const float value = magnitude < 2 ? magnitude * 0.5f :
            ldexpf(1.0f + (magnitude & 1) * 0.5f, static_cast<int>(magnitude / 2) - 1);
        const float scale = block[0] == 0 ? __uint_as_float(0x00400000u) :
            __uint_as_float(static_cast<unsigned>(block[0]) << 23);
        return (code & 8 ? -value : value) * scale;
    }
    if constexpr (Format == QType::Q8_G32_FP16_BLOCK) {
        const auto* block = static_cast<const unsigned char*>(data) + (row * (k / 32) + column / 32) * 34;
        const auto scale = static_cast<std::uint16_t>(block[0] | (static_cast<unsigned>(block[1]) << 8));
        const auto code = static_cast<const signed char*>(static_cast<const void*>(block + 2))[column % 32];
        return decode_half(scale) * static_cast<float>(code);
    }
    return 0.0f; // unreachable for the finite instantiation set below
}
template<QType Format, bool Bf16>
__global__ void projection(const void* x, const void* w, void* out, int n, int k) {
    __shared__ float partial[256];
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto token = static_cast<std::int64_t>(blockIdx.y);
    const unsigned tid = threadIdx.x;
    float sum = 0.0f;
    for (std::int64_t column = tid; column < k; column += 256) {
        float value;
        if constexpr (Bf16) { value = decode_bf16(static_cast<const std::uint16_t*>(x)[token * k + column]); }
        else { value = static_cast<const float*>(x)[token * k + column]; }
        sum += load_weight<Format>(w, row, static_cast<int>(column), k) * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned distance = 128; distance != 0; distance /= 2) {
        if (tid < distance) { partial[tid] += partial[tid + distance]; }
        __syncthreads();
    }
    if (tid == 0) {
        if constexpr (Bf16) { static_cast<std::uint16_t*>(out)[token * n + row] = encode_bf16(partial[0]); }
        else { static_cast<float*>(out)[token * n + row] = partial[0]; }
    }
    return;
}
template<QType Format>
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
    if (n <= 0 || k <= 0) { throw std::invalid_argument("linear: positive weight dimensions required"); }
    switch (type) {
    case QType::FP32: case QType::FP16: case QType::BF16: return;
    case QType::MXFP4_G32_E8M0: case QType::Q8_G32_FP16_BLOCK:
        if (k % 32 == 0) { return; }
        throw std::invalid_argument("linear: block format requires K%32=0");
    default: throw std::invalid_argument("linear: format has no portable projection implementation");
    }
}
void portable_linear_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_portable_linear_weight(w.qtype, w.n, w.k);
    if (!w.qdata || (x.dtype != DType::BF16 && x.dtype != DType::FP32) || out.dtype != x.dtype ||
        x.ne[1] > 65535 || x.ne[1] <= 0) {
        throw std::invalid_argument("linear: invalid data, dtype, or token grid extent");
    }
    const bool block = w.qtype == QType::MXFP4_G32_E8M0 || w.qtype == QType::Q8_G32_FP16_BLOCK;
    if (w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous)) {
        throw std::invalid_argument("linear: persistent format/layout mismatch");
    }
    switch (w.qtype) {
    case QType::FP32: launch<QType::FP32>(x,w,out,stream); break;
    case QType::FP16: launch<QType::FP16>(x,w,out,stream); break;
    case QType::BF16: launch<QType::BF16>(x,w,out,stream); break;
    case QType::MXFP4_G32_E8M0: launch<QType::MXFP4_G32_E8M0>(x,w,out,stream); break;
    case QType::Q8_G32_FP16_BLOCK: launch<QType::Q8_G32_FP16_BLOCK>(x,w,out,stream); break;
    default: throw std::logic_error("validated projection type was not dispatched");
    }
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops::detail
