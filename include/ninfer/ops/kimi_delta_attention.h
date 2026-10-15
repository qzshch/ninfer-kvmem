#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

enum class KdaDecayParameter : std::uint8_t { LogRate, NegativeRate };

/**
 * Caller-owned transient capacity for every T in [min_tokens,max_tokens]. D=128,
 * Hqk>=1, Hv>=Hqk and Hv%Hqk==0. Returns zero when all calls use scratch-free recurrence.
 */
[[nodiscard]] std::size_t kimi_delta_attention_workspace_capacity_bytes(std::int32_t qk_heads,
                                                                        std::int32_t value_heads,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

/**
 * Kimi Delta Attention, independently for every value head h. Its shared query/key head is
 * qh=floor(h/(Hv/Hqk)). Normalize each represented q/k row with epsilon 1e-6, then define
 *
 *   a_h          = exp(A_log[h])
 *   log_alpha[c] = lower_bound * sigmoid(a_h * (g[c,h,t] + dt_bias[c,h]))
 *   alpha[c]     = exp(log_alpha[c])
 *   b            = sigmoid(beta[h,t]).
 *
 * Starting from logical FP32 state S_h[value,key], for t in increasing order:
 *
 *   decayed      = S_h * diag(alpha)
 *   delta        = b * (v[:,h,t] - decayed * k[:,qh,t])
 *   S_h          = decayed + outer(delta, k[:,qh,t])
 *   ideal[:,h,t] = scale * S_h * q[:,qh,t].
 *
 * Contiguous represented inputs are q/k BF16 [128,Hqk,T], v/g/out BF16 [128,Hv,T],
 * beta BF16 [Hv,T], A_log FP32 [Hv], dt_bias FP32 [128,Hv], state FP32 [128,128,Hv].
 * Physical state index is ((h*128+value)*128+key). Hqk>=1, Hv>=Hqk, Hv%Hqk==0, T>=1.
 * lower_bound is finite in [-5,0]; scale is 1/sqrt(128). Inputs/out are non-overlapping.
 * With NegativeRate (DTK), the same FP32 parameter stores -a_h directly. This
 * preserves GLM's source representation without log/exp weight rewriting.
 * State input/output may be disjoint or exactly alias; no other operand overlaps state.
 *
 * The independent oracle evaluates the complete recurrence naively in FP64 from the represented
 * BF16 inputs and FP32 parameters/state, with no private staging or output rounding. Recurrent
 * FP32 SIMT and chunked BF16/TF32 MMA are private arithmetic profiles qualified directly against
 * that oracle. Running and published state stay FP32; the output epilogue rounds once to BF16.
 * Q/K normalization, activated gates, triangular solves, residuals and accumulators stay FP32;
 * chunked decay-weighted Q/K operands have private BF16 storage. Chunked state prediction and
 * small matrix products use TF32 MMA; Q readout and state updates use BF16 MMA with FP32
 * accumulators. State snapshots and Delta stay FP32, converting only their BF16 MMA operands.
 * Transcendental approximations and reduction association are implementation details.
 * The Hygon DTK ordinary overload and pooled one-token updates use scratch-free FP32 recurrence.
 * The packed temporal overload below supplies a separate caller-owned-workspace route.
 * These shared-memory reductions have no CUDA warp-width dependency; CUDA chunked dispatch
 * described above remains unchanged.
 *
 * workspace supplies the capacity query above and is scoped to the call; the Op allocates no
 * device storage. execution supplies the stream and physical SM count for launch decomposition.
 * Chunked execution supports partial final chunks. This overload publishes state after all T
 * tokens into the same state storage.
 */
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, WorkspaceArena& workspace, Tensor& state,
                          Tensor& out, DeviceExecutionView execution,
                          KdaDecayParameter decay_parameter = KdaDecayParameter::LogRate);

/** Distinct-state form of the same mathematical contract; exact state alias is also allowed. */
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, WorkspaceArena& workspace,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          DeviceExecutionView execution,
                          KdaDecayParameter decay_parameter = KdaDecayParameter::LogRate);

/**
 * Scratch-free one-token update for B independent state-pool slots, using the same head mapping.
 * q/k BF16 [128,Hqk,1,B], v/g/out BF16 [128,Hv,1,B], beta BF16 [Hv,1,B],
 * A_log FP32 [Hv], dt_bias FP32 [128,Hv], states FP32 [128,128,Hv,Slots], device I32 slots [B].
 * B is in [1,8]. Caller-provided active slots are valid and distinct. Each row reads and writes
 * its selected state. Inputs, out, selectors and state do not overlap.
 * decay_parameter has the same represented-parameter meaning as the ordinary overload.
 * NegativeRate is supported by DTK; other implementations reject it before launching.
 */
void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& A_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       Tensor& states, const Tensor& slots, Tensor& out,
                                       cudaStream_t stream,
                                       KdaDecayParameter decay_parameter = KdaDecayParameter::LogRate);

/**
 * DTK packed temporal form of the same recurrence for R independent state slots.
 * q/k BF16 [128,Hqk,M], v/g/out BF16 [128,Hv,M], beta BF16 [Hv,M]. States are
 * FP32 [128,128,Hv,Slots]. Device I32 slots[R] are valid and distinct, R=1..8.
 * Device I32 offsets[R+1] start at zero, end at M and increase strictly. Sequence
 * r consumes input rows [offsets[r],offsets[r+1]) in time order and only updates
 * states[slots[r]]. Unselected states are unchanged. Both decay parameterizations
 * are explicit. Other operands follow the ordinary overload's dtype, alias,
 * numerical and represented-input rules. No allocation or host metadata readback.
 * M<=32768. DTK selects FP32 time tiles from the rows/sequence execution
 * geometry, with cumulative diagonal
 * decays, a block-local triangular solve and FP32 state prediction/update.
 * Q/K, solves and delta remain FP32; only the public output rounds to BF16.
 * workspace is caller-owned, disjoint call-scoped storage sized by the query
 * below. Short-span geometry prepares head-only FP32 vectors once before the value-row
 * recurrence. Non-DTK rejects execution.
 */
std::size_t kimi_delta_attention_packed_workspace_capacity_bytes(
    std::int32_t qk_heads, std::int32_t value_heads, std::int32_t sequences, std::int32_t rows);
void kimi_delta_attention_packed_update(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
    const Tensor& A_log, const Tensor& dt_bias, float lower_bound, float scale,
    WorkspaceArena& workspace, Tensor& states, const Tensor& slots, const Tensor& offsets, Tensor& out,
    cudaStream_t stream, KdaDecayParameter decay_parameter);

} // namespace ninfer::ops
