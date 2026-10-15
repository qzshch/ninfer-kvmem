// Private DTK projection primitive. Include after weight_decode.cuh in a namespace.
#pragma once

// A group of four adjacent represented operands shares the packed scale, grid
// index and sign byte. Decode the metadata once, rather than four scalar calls.
template <QType F>
__device__ float4 represented_four(const void* weights, std::int64_t row, int column, int k) {
    if constexpr (F == QType::IQ3_S_G256_BLOCK) {
        const auto* b =
            static_cast<const unsigned char*>(weights) + (row * (k / 256) + column / 256) * 110;
        const int c = column % 256, group = c / 32, sub = c / 4;
        const unsigned grid       = b[2 + sub] | (((b[66 + group] >> (sub % 8)) & 1u) << 8);
        const unsigned magnitudes = iq3_s_grid[grid], signs = b[74 + c / 8] >> (c % 8);
        const float scale =
            block_half(b) *
            static_cast<float>(1 + 2 * ((b[106 + group / 2] >> (4 * (group % 2))) & 15u));
        float4 out;
        out.x =
            scale * static_cast<float>((signs & 1 ? -1 : 1) * static_cast<int>(magnitudes & 255));
        out.y = scale * static_cast<float>((signs & 2 ? -1 : 1) *
                                           static_cast<int>((magnitudes >> 8) & 255));
        out.z = scale * static_cast<float>((signs & 4 ? -1 : 1) *
                                           static_cast<int>((magnitudes >> 16) & 255));
        out.w =
            scale * static_cast<float>((signs & 8 ? -1 : 1) * static_cast<int>(magnitudes >> 24));
        return out;
    } else if constexpr (F == QType::IQ4_XS_G256_BLOCK) {
        const auto* b =
            static_cast<const unsigned char*>(weights) + (row * (k / 256) + column / 256) * 136;
        const int c = column % 256, group = c / 32;
        const unsigned high   = b[2] | (static_cast<unsigned>(b[3]) << 8);
        const int scale_code  = static_cast<int>(((b[4 + group / 2] >> (4 * (group % 2))) & 15u) |
                                                 (((high >> (2 * group)) & 3u) << 4)) -
                                32;
        const float scale     = block_half(b) * static_cast<float>(scale_code);
        const unsigned packed = *reinterpret_cast<const unsigned*>(b + 8 + group * 16 + c % 16);
        const unsigned shift  = c % 32 < 16 ? 0 : 4;
        return {scale * iq4_values[(packed >> shift) & 15],
                scale * iq4_values[(packed >> (shift + 8)) & 15],
                scale * iq4_values[(packed >> (shift + 16)) & 15],
                scale * iq4_values[(packed >> (shift + 24)) & 15]};
    } else if constexpr (F == QType::Q8_G32_FP16_BLOCK) {
        const auto* b =
            static_cast<const unsigned char*>(weights) + (row * (k / 32) + column / 32) * 34;
        const float scale = block_half(b);
        const auto* codes = reinterpret_cast<const signed char*>(b + 2 + column % 32);
        return {scale * codes[0], scale * codes[1], scale * codes[2], scale * codes[3]};
    } else
        return {load_weight<F>(weights, row, column, k),
                load_weight<F>(weights, row, column + 1, k),
                load_weight<F>(weights, row, column + 2, k),
                load_weight<F>(weights, row, column + 3, k)};
}

// The four accumulators correspond to the original 256-thread row's lanes
// 4*lane+[0,1,2,3]. Adjacent operands share packed block metadata. Reduce the
// four columns independently before the final two additions, retaining the
// original row's reduction tree rather than changing its arithmetic grouping.
template <QType Format, bool Bf16>
__device__ float dot_wave64(const void* input, const void* weights, std::int64_t row, int k) {
    const int lane = threadIdx.x & 63;
    float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
    for (int base = 0; base < k; base += 256) {
        const int column = base + 4 * lane;
        float x0, x1, x2, x3;
        if constexpr (Bf16) {
            const auto* x = static_cast<const std::uint16_t*>(input) + column;
            x0            = decode_bf16(x[0]);
            x1            = decode_bf16(x[1]);
            x2            = decode_bf16(x[2]);
            x3            = decode_bf16(x[3]);
        } else {
            const auto* x = static_cast<const float*>(input) + column;
            x0            = x[0];
            x1            = x[1];
            x2            = x[2];
            x3            = x[3];
        }
        const float4 weights4 = represented_four<Format>(weights, row, column, k);
        a += weights4.x * x0;
        b += weights4.y * x1;
        c += weights4.z * x2;
        d += weights4.w * x3;
    }
    for (int distance = 32; distance; distance /= 2) {
        a += __shfl_down(a, distance, 64);
        b += __shfl_down(b, distance, 64);
        c += __shfl_down(c, distance, 64);
        d += __shfl_down(d, distance, 64);
    }
    return (a + c) + (b + d);
}
