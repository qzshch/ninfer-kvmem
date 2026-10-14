// Hygon DTK launcher; uses the unchanged NInfer wrapper and Op contracts.
#include "ops/launcher/rmsnorm.h"
#include "ops/kernel/rmsnorm_hip.cuh"
#include "core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, std::int32_t /*multiprocessor_count*/,
                    cudaStream_t stream) {
    const std::int32_t d = x.ne[0];
    if (d <= 0) { throw std::invalid_argument("rmsnorm: ne[0] must be positive"); }
    const std::int64_t rows = out.numel() / d;
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds HIP grid limit");
    }
    const auto* input = static_cast<const std::uint16_t*>(x.data);
    const auto* gains = static_cast<const std::uint16_t*>(weight.data);
    const auto* gate = z ? static_cast<const std::uint16_t*>(z->data) : nullptr;
    auto* output = static_cast<std::uint16_t*>(out.data);
    const auto grid = static_cast<unsigned>(rows);
    if (z != nullptr) {
        rmsnorm_hip_kernel<false, true><<<grid, 256, 0, stream>>>(input, gains, gate, output, d, eps);
    } else if (unit_offset) {
        rmsnorm_hip_kernel<true, false><<<grid, 256, 0, stream>>>(input, gains, nullptr, output, d, eps);
    } else {
        rmsnorm_hip_kernel<false, false><<<grid, 256, 0, stream>>>(input, gains, nullptr, output, d, eps);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops::detail
