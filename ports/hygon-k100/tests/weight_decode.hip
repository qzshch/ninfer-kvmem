#include "core/device.h"
#include "core/arena.h"
#include "core/weight_view.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <array>
#include <stdexcept>
#include <vector>

namespace {
using ninfer::QType;
#include "ops/linear/portable/weight_decode.cuh"
#include "ops/linear/portable/wave64_dot.cuh"

__global__ void decode_all(float* output) {
    const unsigned word = blockIdx.x * blockDim.x + threadIdx.x;
    output[word]        = decode_half(static_cast<std::uint16_t>(word));
    return;
}
__global__ void decode_quads(const void* encoded, float* output) {
    const auto v = represented_four<QType::Q6_K_G256_BLOCK>(encoded, blockIdx.x, threadIdx.x * 4, 256);
    const int at = (blockIdx.x * 64 + threadIdx.x) * 4;
    output[at] = v.x; output[at+1] = v.y; output[at+2] = v.z; output[at+3] = v.w;
}
void quad_oracle(ninfer::DeviceContext& device) {
    constexpr int rows = 17;
    std::vector<unsigned char> encoded(rows * 210);
    std::mt19937 rng(20373);
    for (auto& b : encoded) b = static_cast<unsigned char>(rng());
    const std::array<unsigned, 5> multipliers{0x0001,0x2000,0x3007,0xb055,0x3801};
    for (int row = 0; row < rows; ++row) {
        const auto word = multipliers[row % multipliers.size()];
        encoded[row * 210 + 208] = word & 255; encoded[row * 210 + 209] = word >> 8;
    }
    ninfer::DeviceBuffer input(encoded.size()), output(rows * 256 * sizeof(float));
    input.copy_from_host(encoded.data(), input.bytes);
    decode_quads<<<rows,64,0,device.stream>>>(input.p, static_cast<float*>(output.p));
    CUDA_CHECK(cudaGetLastError()); device.synchronize();
    std::vector<float> actual(rows * 256); output.copy_to_host(actual.data(), output.bytes);
    for (int row = 0; row < rows; ++row) {
        const auto* b = encoded.data() + row * 210;
        const unsigned word = b[208] + 256u * b[209], e = (word >> 10) & 31u;
        const double base = (word & 32768u ? -1. : 1.) *
                            (e ? std::ldexp(1024. + (word & 1023u), static_cast<int>(e)-25)
                               : std::ldexp(static_cast<double>(word & 1023u), -24));
        for (int c = 0; c < 256; ++c) {
            const int part = c / 128;
            const int lower = (b[part * 64 + c % 64] >> (c % 128 < 64 ? 0 : 4)) & 15;
            const int upper = (b[128 + part * 32 + c % 32] >> (2 * ((c % 128) / 32))) & 3;
            const int subscale = static_cast<signed char>(b[192 + c / 16]);
            const float expected = static_cast<float>(base * subscale * (lower + 16 * upper - 32));
            if (std::bit_cast<unsigned>(actual[row*256+c]) != std::bit_cast<unsigned>(expected))
                throw std::runtime_error("Q6 quartet differs from independent represented FP64 value");
        }
    }
}
} // namespace

int main() try {
    ninfer::DeviceContext device;
    quad_oracle(device);
    ninfer::DeviceBuffer output(65536 * sizeof(float));
    decode_all<<<256, 256, 0, device.stream>>>(static_cast<float*>(output.p));
    CUDA_CHECK(cudaGetLastError());
    device.synchronize();
    std::vector<float> values(65536);
    output.copy_to_host(values.data(), output.bytes);
    for (unsigned word = 0; word < 65536; ++word) {
        const int exponent = (word >> 10) & 31, fraction = word & 1023;
        if (exponent == 31 && fraction) {
            if (!std::isnan(values[word])) throw std::runtime_error("FP16 NaN decode");
            continue;
        }
        double expected = exponent == 31 ? std::numeric_limits<double>::infinity()
                          : exponent     ? std::ldexp(1. + fraction / 1024., exponent - 15)
                                         : std::ldexp(static_cast<double>(fraction), -24);
        if (word & 0x8000) expected = -expected;
        if (std::bit_cast<std::uint32_t>(values[word]) !=
            std::bit_cast<std::uint32_t>(static_cast<float>(expected)))
            throw std::runtime_error("FP16 decode differs from independent FP64 representation");
    }
    std::puts("PASS: all 65536 FP16 encodings; independent FP64 representation; exact "
              "finite/zero/infinity bits and NaN category");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
