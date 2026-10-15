#include "core/indexed_transfer.h"
#include "core/device.h"
#include <stdexcept>

namespace ninfer {
namespace {
__global__ void transfer(unsigned char* compact, unsigned char* persistent, const std::int32_t* ids,
                         std::size_t bytes, std::size_t stride, bool scatter) {
    const auto offset = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (offset >= bytes) return;
    auto* a = compact + static_cast<std::size_t>(blockIdx.y) * bytes + offset;
    auto* b = persistent + static_cast<std::size_t>(ids[blockIdx.y]) * stride + offset;
    if (scatter)
        *b = *a;
    else
        *a = *b;
    return;
}
} // namespace

void indexed_transfer(void* compact, void* persistent, const std::int32_t* indices,
                      std::size_t bytes, std::size_t stride, int rows, bool scatter,
                      cudaStream_t stream) {
    if (!compact || !persistent || !indices || !bytes || stride < bytes || rows < 1 || rows > 8)
        throw std::invalid_argument("indexed_transfer: invalid rows/operands");
    transfer<<<dim3((bytes + 255) / 256, rows), 256, 0, stream>>>(
        static_cast<unsigned char*>(compact), static_cast<unsigned char*>(persistent), indices,
        bytes, stride, scatter);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer
