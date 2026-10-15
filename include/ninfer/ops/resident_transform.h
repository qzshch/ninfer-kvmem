#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {
// DTK resident mathematical primitives. FP32 storage carries BF16-represented
// model activations; norm reductions/gates remain FP32, final model boundaries
// explicitly round to BF16. No routine allocates or transfers to the host.
void round_bf16(float* values, std::size_t count, cudaStream_t stream);
void pack_bf16(const float* input, std::uint16_t* output, std::size_t count, cudaStream_t stream);
void unpack_bf16(const std::uint16_t* input, float* output, std::size_t count, cudaStream_t stream);
// rows of width values. Optional gain/bias are direct FP32/FP16/BF16 vectors;
// gain absent => unweighted. center=true => LayerNorm, otherwise RMSNorm.
// Optional gate applies sigmoid after norm/gain. Exact input/output alias valid.
void resident_norm(const float* input, const Tensor* gain, const Tensor* bias, const float* gate,
                   float* output, int width, int rows, float epsilon, bool center, bool bf16_output,
                   cudaStream_t stream);
// gate/up BF16-represented inputs; limit>0 clips gate above limit, up to +/-limit.
// The SiLU result and subsequent product each have BF16 semantic boundaries.
void resident_swiglu(const float* gate, const float* up, float* output, std::size_t count,
                     float limit, cudaStream_t stream);
void resident_add(const float* a, const float* b, float* out, std::size_t count,
                  cudaStream_t stream);
void replicate_streams(const float* input, float* out, int hidden, int tokens, cudaStream_t stream);
void mean_streams(const float* input, float* out, int hidden, int tokens, cudaStream_t stream);
// Depthwise causal 4-tap convolution followed by SiLU. State [channels,3]
// holds oldest..newest BF16-represented values. Input/output may alias exactly.
// Kernel is a direct vector [channels,1,4]. T processed in order on device.
void resident_causal_conv(const float* input, const Tensor& kernel, float* state, float* output,
                          int channels, int tokens, cudaStream_t stream);
// The same one-token transition for independent rows [rows,channels]. State is
// [slot_count,channels,3]; device slots[rows] are valid, distinct physical slots.
// Only selected slots change. Input/output may alias exactly; all other operands
// are disjoint. The caller owns state, selectors and their lifetime through stream completion.
void resident_causal_conv_batch(const float* input, const Tensor& kernel, float* states,
                                const std::int32_t* slots, float* output, int channels,
                                int rows, cudaStream_t stream);
// Sigmoid top-k with bias for selection only; normalized original scores times
// routed_scale. Deterministic lower-id ties. Single router group is explicit.
void resident_router(const float* logits, const float* bias, std::int32_t* ids, float* weights,
                     int experts, int active, int tokens, float routed_scale, cudaStream_t stream);
// FP32 sum of routed expert outputs weighted by their FP32 routing probability,
// plus the shared output, then BF16. Expert outputs are [T,active,hidden].
void resident_moe_sum(const float* experts, const float* weights, const float* shared, float* out,
                      int hidden, int active, int tokens, cudaStream_t stream);
// Device argmax, one int per row, lower-id tie; values [rows,width]. A row
// containing any nonfinite value publishes -1, never a fabricated token id.
void resident_argmax(const float* values, std::int32_t* ids, int width, int rows,
                     cudaStream_t stream);
// Retrieval metadata only: arithmetic mean across rows of a BF16 matrix, or
// sum across rows of an FP32 matrix. Outputs [width] stay on the device.
void resident_mean_bf16(const std::uint16_t* values, float* out, int width, int rows,
                        cudaStream_t stream);
void resident_sum_rows(const float* values, float* out, int width, int rows, cudaStream_t stream);
} // namespace ninfer::ops
