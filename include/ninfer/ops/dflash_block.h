#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {
// DFlash2 two-tap, 16-channel-group dynamic convolution. input/output [T,4096]
// are disjoint BF16-represented FP32. dynamic [T,2,2,256] and base [2,2,4096]
// are BF16-represented FP32 / stored BF16 respectively. Stage 0=prepare,1=finish.
// At each tap, round base*value, sum, then fused dynamic*value+sum to BF16.
void dflash_dynamic_conv(const float* input, const float* dynamic, const std::uint16_t* base,
                         float* output, int tokens, int stage, cudaStream_t stream);
// In-place half-split RoPE, head dimension128, theta10000, final BF16 rounding.
// Input [T,heads,128]; positions are consecutive starting at first_position.
void dflash_rope(float* values, int heads, int tokens, int first_position, cudaStream_t stream);
// Gather target mHC-mean taps [ring_slots,20480] BF16 into [T,20480] FP32.
void dflash_features(const std::uint16_t* ring, float* output, int ring_slots, int first_position,
                     int tokens, cudaStream_t stream);
void dflash_cache_append(const float* key, const float* value, std::uint16_t* key_cache,
                         std::uint16_t* value_cache, int slots, int first_position, int tokens,
                         cudaStream_t stream);
// Noncausal sliding-window2048 GQA: q [T,32,128], noise k/v [T,8,128]
// FP32/BF16-represented; context k/v [slots,8,128] BF16 ring at positions
// before first_position. Scratch [T,32,2048+T], output [T,4096], BF16 rounded.
void dflash_attention(const float* query, const std::uint16_t* context_key,
                      const std::uint16_t* context_value, const float* noise_key,
                      const float* noise_value, float* probabilities, float* output, int slots,
                      int first_position, int tokens, cudaStream_t stream);
// Exact FP32 top16 per [T,vocab] row, lower-id ties. Output ids/unary [T,16].
void dflash_top16(const float* logits, std::int32_t* ids, float* unary, int vocab, int tokens,
                  cudaStream_t stream);
// Greedy DFlash2 candidate path. hidden [T,256] BF16-represented FP32;
// codebooks [vocab,256] BF16, candidates/unary [T,16], output [T] I32.
// Each position conditions on the previous selected token (initial anchor).
void dflash_select(const float* hidden, const std::uint16_t* predecessor,
                   const std::uint16_t* successor, const std::int32_t* candidates,
                   const float* unary, std::int32_t* output, int anchor, int tokens,
                   cudaStream_t stream);
} // namespace ninfer::ops
