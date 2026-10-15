#include "core/device.h"
#include "core/arena.h"
#include "core/weight_view.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using ninfer::QType;
#include "ops/linear/portable/weight_decode.cuh"

__global__ void decode_all(float* output) {
    const unsigned word = blockIdx.x * blockDim.x + threadIdx.x;
    output[word]        = decode_half(static_cast<std::uint16_t>(word));
    return;
}
} // namespace

int main() try {
    ninfer::DeviceContext device;
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
