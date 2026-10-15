#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer {
// Raw compact-row transfer; rows have identical byte widths. Device indices
// select persistent rows. Source/destination must not overlap.
void indexed_transfer(void* compact, void* persistent, const std::int32_t* indices,
                      std::size_t row_bytes, std::size_t persistent_stride, int rows, bool scatter,
                      cudaStream_t stream);
} // namespace ninfer
