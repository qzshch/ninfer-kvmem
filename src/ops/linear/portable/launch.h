#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
void validate_portable_linear_weight(QType type, std::int32_t n, std::int32_t k);
void portable_linear_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                            bool bf16_output = false);
} // namespace ninfer::ops::detail
