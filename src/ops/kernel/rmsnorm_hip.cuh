// Hygon DTK implementation of the existing RMSNorm mathematical contract.
// Project modifications: Apache-2.0. No CUDA warp-size or instruction assumptions.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

__device__ inline float rms_hip_load_bf16(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ inline std::uint16_t rms_hip_store_bf16(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    }
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// One workgroup per row, FP32 sum of squares, one final BF16 RNE boundary.
// All workgroup threads participate, including odd-width and short-row tails.
// This correctness baseline is not a tuned matrix/warp-specific implementation.
template <bool Offset, bool Gated>
__global__ void rmsnorm_hip_kernel(const std::uint16_t* x, const std::uint16_t* weight,
                                   const std::uint16_t* z, std::uint16_t* out,
                                   std::int32_t d, float eps) {
    constexpr unsigned kBlock = 256;
    __shared__ float sums[kBlock];
    const unsigned tid = threadIdx.x;
    const auto base = static_cast<std::int64_t>(blockIdx.x) * d;
    float sum = 0.0f;
    for (std::int64_t column = tid; column < d; column += kBlock) {
        const float value = rms_hip_load_bf16(x[base + column]);
        sum += value * value;
    }
    sums[tid] = sum;
    __syncthreads();
    for (unsigned distance = kBlock / 2; distance != 0; distance /= 2) {
        if (tid < distance) { sums[tid] += sums[tid + distance]; }
        __syncthreads();
    }
    const float inverse = rsqrtf(sums[0] / static_cast<float>(d) + eps);
    for (std::int64_t column = tid; column < d; column += kBlock) {
        const float value = rms_hip_load_bf16(x[base + column]);
        float gain = rms_hip_load_bf16(weight[column]);
        if constexpr (Offset) { gain += 1.0f; }
        float result = value * inverse * gain;
        if constexpr (Gated) {
            const float gate = rms_hip_load_bf16(z[base + column]);
            result *= gate / (1.0f + expf(-gate));
        }
        out[base + column] = rms_hip_store_bf16(result);
    }
    return;
}
} // namespace ninfer::ops::detail
