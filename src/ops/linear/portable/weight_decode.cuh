// Native resident operand decoding. Include inside an implementation namespace.
#pragma once

__device__ float decode_half(std::uint16_t word) {
    const auto exponent = (word >> 10) & 31;
    const auto mantissa = word & 1023;
    const float sign    = word & 0x8000 ? -1.0f : 1.0f;
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

#include "ops/linear/portable/block256.cuh"

template <QType Format>
__device__ float load_weight(const void* data, std::int64_t row, std::int32_t column, int k) {
    const auto index = row * k + column;
    if constexpr (Format == QType::FP32) { return static_cast<const float*>(data)[index]; }
    if constexpr (Format == QType::BF16) {
        return decode_bf16(static_cast<const std::uint16_t*>(data)[index]);
    }
    if constexpr (Format == QType::FP16) {
        return decode_half(static_cast<const std::uint16_t*>(data)[index]);
    }
    if constexpr (Format == QType::MXFP4_G32_E8M0) {
        const auto* block =
            static_cast<const unsigned char*>(data) + (row * (k / 32) + column / 32) * 17;
        const unsigned lane      = column % 32;
        const unsigned code      = (block[1 + lane % 16] >> (lane < 16 ? 0 : 4)) & 15;
        const unsigned magnitude = code & 7;
        const float value        = magnitude < 2 ? magnitude * 0.5f
                                                 : ldexpf(1.0f + (magnitude & 1) * 0.5f,
                                                          static_cast<int>(magnitude / 2) - 1);
        const float scale = block[0] == 0 ? __uint_as_float(0x00400000u)
                                          : __uint_as_float(static_cast<unsigned>(block[0]) << 23);
        return (code & 8 ? -value : value) * scale;
    }
    if constexpr (Format == QType::Q8_G32_FP16_BLOCK) {
        const auto* block =
            static_cast<const unsigned char*>(data) + (row * (k / 32) + column / 32) * 34;
        const auto scale =
            static_cast<std::uint16_t>(block[0] | (static_cast<unsigned>(block[1]) << 8));
        const auto code =
            static_cast<const signed char*>(static_cast<const void*>(block + 2))[column % 32];
        return decode_half(scale) * static_cast<float>(code);
    }
    if constexpr (Format == QType::IQ3_S_G256_BLOCK || Format == QType::IQ4_XS_G256_BLOCK ||
                  Format == QType::Q3_K_G256_BLOCK || Format == QType::Q4_K_G256_BLOCK ||
                  Format == QType::Q6_K_G256_BLOCK) {
        constexpr int bytes = Format == QType::IQ4_XS_G256_BLOCK ? 136
                              : Format == QType::Q4_K_G256_BLOCK ? 144
                              : Format == QType::Q6_K_G256_BLOCK ? 210
                                                                 : 110;
        const auto* block =
            static_cast<const unsigned char*>(data) + (row * (k / 256) + column / 256) * bytes;
        return decode_block256<Format>(block, column % 256);
    }
    return 0.0f; // unreachable for the finite instantiation set below
}
