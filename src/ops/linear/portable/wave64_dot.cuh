// Private DTK projection primitive. Include after weight_decode.cuh in a namespace.
#pragma once

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
        a += load_weight<Format>(weights, row, column, k) * x0;
        b += load_weight<Format>(weights, row, column + 1, k) * x1;
        c += load_weight<Format>(weights, row, column + 2, k) * x2;
        d += load_weight<Format>(weights, row, column + 3, k) * x3;
    }
    for (int distance = 32; distance; distance /= 2) {
        a += __shfl_down(a, distance, 64);
        b += __shfl_down(b, distance, 64);
        c += __shfl_down(c, distance, 64);
        d += __shfl_down(d, distance, 64);
    }
    return (a + c) + (b + d);
}
