#pragma once
#include "core/weight.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {
// Device-indexed banks: weight [banks*rows,K], ids [T,choices], x [T,K]
// (shared_input) or [T,choices,K], out [T,choices,rows]. All activations and
// output are FP32. A null ids pointer means bank=choice, useful for headwise
// maps. Caller provides valid ids in [0,banks); no weight/index host read,
// repack, allocation or activation rounding occurs. Operands are disjoint.
void indexed_linear_fp32(const float* x, const Weight& bank, const std::int32_t* ids, float* out,
                         int rows, int choices, int tokens, bool shared_input, cudaStream_t stream);
// Decode selected embedding rows directly to BF16-represented FP32 values.
// ids [T] stay on device, out [T,K]; no complete embedding expansion.
void embedding_fp32(const Weight& embedding, const std::int32_t* ids, float* out, int tokens,
                    cudaStream_t stream);
} // namespace ninfer::ops
