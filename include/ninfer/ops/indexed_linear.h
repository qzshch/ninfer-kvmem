#pragma once
#include "core/weight.h"
#include "core/arena.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace ninfer::ops {
// Device-indexed banks: weight [banks*rows,K], ids [T,choices], x [T,K]
// (shared_input) or [T,choices,K], out [T,choices,rows]. All activations and
// output are FP32. A null ids pointer means bank=choice, useful for headwise
// maps. Caller provides valid ids in [0,banks); no weight/index host read,
// repack, allocation or input activation rounding occurs. Operands are disjoint.
// bf16_output=true rounds each final dot to BF16 (RNE), promoted into FP32
// storage; false retains the unrounded FP32 output.
void indexed_linear_fp32(const float* x, const Weight& bank, const std::int32_t* ids, float* out,
                         int rows, int choices, int tokens, bool shared_input, cudaStream_t stream,
                         bool bf16_output = false);
// EP variant: stored banks cover [bank_begin,bank_begin+bank.n/rows).
// Global expert ids outside this interval write exact zero, without reading
// input or weight data. Owned choices retain the complete-dot ideal above.
void indexed_linear_owned_fp32(const float* x, const Weight& bank, const std::int32_t* ids,
                                float* out, int rows, int choices, int tokens, bool shared_input,
                                int bank_begin, cudaStream_t stream, bool bf16_output = false);
// Matrix form of the owned-bank projection. Group global ids on device, reuse
// encoded weight tiles across up to eight assignments, scatter each result to
// its original [token,choice] row. tokens<=1024, choices<=32, banks<=288.
// The complete-dot ideal and output boundary match the owned GEMV form.
// Caller-owned call-scoped workspace; no host routing read or persistent repack.
std::size_t indexed_linear_grouped_workspace_capacity_bytes(int banks, int choices, int tokens);
void indexed_linear_grouped_fp32(const float* x, const Weight& bank, const std::int32_t* ids,
                                  float* out, int rows, int choices, int tokens, bool shared_input,
                                  int bank_begin, WorkspaceArena& workspace, cudaStream_t stream,
                                  bool bf16_output = false);
// Decode selected embedding rows directly to BF16-represented FP32 values.
// ids [T] stay on device, out [T,K]; no complete embedding expansion.
void embedding_fp32(const Weight& embedding, const std::int32_t* ids, float* out, int tokens,
                    cudaStream_t stream);
} // namespace ninfer::ops
