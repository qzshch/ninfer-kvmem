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
// Typed storage forms of the same norm/SwiGLU/packed-conv ideals. Contiguous
// FP32 or BF16 tensors; norm uses width=x.ne[0] and rows=numel/width. FP32 norm
// outputs retain FP32; BF16 outputs use the existing final semantic cast. A gate
// has the input dtype/extent. SwiGLU and conv require one matching dtype and
// preserve both existing BF16 arithmetic boundaries even with FP32 storage.
// Exact same-dtype input/output alias is valid. Conv state remains FP32.
void resident_norm(const Tensor& input, const Tensor* gain, const Tensor* bias, const Tensor* gate,
                   Tensor& output, float epsilon, bool center, cudaStream_t stream);
void resident_swiglu(const Tensor& gate, const Tensor& up, Tensor& output, float limit, cudaStream_t stream);
void resident_causal_conv_packed(const Tensor& input, const Tensor& kernel, float* states,
                                 const std::int32_t* slots, const std::int32_t* offsets,
                                 Tensor& output, int sequences, cudaStream_t stream);
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
// Packed temporal form of the same transition. FP32 state is
// [slot_count,channels,3]. Input/output may alias exactly; all other operands
// are disjoint. Offsets[sequences+1] partition [0,M) into nonempty
// ordered spans; slots[sequences] select valid distinct states. Input/output
// are [M,channels]. The same mathematical state transition runs in increasing
// time within each span, without copying state to compact execution rows.
void resident_causal_conv_packed(const float* input, const Tensor& kernel, float* states,
                                 const std::int32_t* slots, const std::int32_t* offsets,
                                 float* output, int channels, int sequences, cudaStream_t stream);
// Exact FP32 row gather, output [rows,width] = input[indices[row],:]. Device
// indices are valid for the input extent. Input, indices and output are disjoint.
void resident_gather_rows(const float* input, const std::int32_t* indices, float* output,
                          int width, int rows, cudaStream_t stream);
// Sigmoid top-k with bias for selection only; normalized original scores times
// routed_scale. Deterministic lower-id ties. Single router group is explicit.
void resident_router(const float* logits, const float* bias, std::int32_t* ids, float* weights,
                     int experts, int active, int tokens, float routed_scale, cudaStream_t stream);
// FP32 sum of routed expert outputs weighted by their FP32 routing probability,
// plus the shared output, then BF16. Expert outputs are [T,active,hidden].
void resident_moe_sum(const float* experts, const float* weights, const float* shared, float* out,
                      int hidden, int active, int tokens, cudaStream_t stream);
// BF16 expert storage; same shared-plus-weighted sum and final BF16 boundary.
void resident_moe_sum(const Tensor& experts, const float* weights, const float* shared, float* out,
                      int hidden, int active, int tokens, cudaStream_t stream);
// Local EP weighted sum, unrounded FP32. No shared expert contribution.
void resident_moe_partial(const float* experts, const float* weights, float* out,
                          int hidden, int active, int tokens, cudaStream_t stream);
// BF16 expert storage, same unrounded FP32 weighted sum. x [hidden,active,T].
void resident_moe_partial(const Tensor& experts, const float* weights, float* output,
                          int hidden, int active, int tokens, cudaStream_t stream);
// Three device-local K/EP contributions [count] -> BF16-represented FP32.
// routed=true: each input is [2,count], shared K-partial then EP weighted sum;
// shared contributions are BF16-rounded before adding the routed contributions.
// Operands are disjoint. No allocation or transport inside this Op.
void resident_tp_sum(const float* a, const float* b, const float* c, float* out,
                     std::size_t count, bool routed, cudaStream_t stream);
// Device argmax, one int per row, lower-id tie; values [rows,width]. A row
// containing any nonfinite value publishes -1, never a fabricated token id.
void resident_argmax(const float* values, std::int32_t* ids, int width, int rows,
                     cudaStream_t stream);
struct ArgmaxPair { float value; std::int32_t id; };
// Local vocabulary interval [begin,begin+width). Any nonfinite value marks the
// pair invalid (-1); merge propagates invalidity from every rank, and uses the
// global lower-id tie rule. pairs for merge are [3,rows], output ids [rows].
void resident_argmax_pairs(const float* values, ArgmaxPair* pairs, int width, int rows,
                           int begin, cudaStream_t stream);
void resident_argmax_merge(const ArgmaxPair* pairs, std::int32_t* ids, int rows,
                           cudaStream_t stream);
// Retrieval metadata only: arithmetic mean across rows of a BF16 matrix, or
// sum across rows of an FP32 matrix. Outputs [width] stay on the device.
void resident_mean_bf16(const std::uint16_t* values, float* out, int width, int rows,
                        cudaStream_t stream);
void resident_sum_rows(const float* values, float* out, int width, int rows, cudaStream_t stream);
} // namespace ninfer::ops
