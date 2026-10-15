#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {
// Packed query rows. slots/positions are device arrays [tokens];
// persistent caches are [resident_lanes, capacity, width]. Positions are the
// append frontier before this transition. C=capacity in 1..32768, B=tokens in
// 1..32768. Caller supplies valid slots and positions in [0,C). A slot may
// repeat for a temporal span, but each (slot,position) written by append is
// unique and all preceding history is initialized. Input writes complete before
// a completed four-token pool reads any of them. Each query sees only pools
// completed at its own position; physically written future rows stay invisible.
// Buffers are device-resident, naturally aligned and disjoint. No allocation or
// host metadata round trip; values outside each declared write region survive.
// Append latent [B,512], key/gate [B,128] FP32 to BF16 caches; bias [4,128].
// The pooling formula and BF16 boundaries are identical to index_pool_append.
void resident_index_append(const float* latent, const float* key, const float* gate,
                           const float* bias, std::uint16_t* latents, std::uint16_t* keys,
                           std::uint16_t* gates, std::uint16_t* pools, const std::int32_t* slots,
                           const std::int32_t* positions, int capacity, int tokens,
                           cudaStream_t stream);
// Retrieval state means [lanes,ceil(C/128),128] stores FP32 arithmetic means
// of each completed 128-token BF16 key block. At a newly completed block, update
// only that mean with an ordered FP32 sum. query [B,32,128] sums heads in FP32;
// score each full block by FP64 cosine, cast score FP32 (zero norm -> zero).
// Select one sink, one most recent block (including partial), then remaining
// budget by descending score/lower block-id ties. NaN scores are unranked.
// allowed [B,ceil(C/64)] marks those blocks' pages. Prior full means must be
// initialized. Means completed by later rows in the same packed call are not
// eligible for an earlier query. No request lifetime/frontier changes in this Op.
void resident_retrieval(const float* query, const std::uint16_t* keys, float* means,
                        std::uint8_t* allowed, const std::int32_t* slots,
                        const std::int32_t* positions, int capacity, int tokens, int budget_blocks,
                        cudaStream_t stream);
// Batch form of index_pool_scores + index_pool_select: query [B,32,128],
// weights [B,32], cache pools [lanes,ceil(C/4),128]. tier in 4..32768 must cover
// every visible frontier; it sizes masked work only. Scratch scores [B,8192],
// ids [B,2051]. Optional allowed [B,ceil(C/64)] uses the same mask semantics.
// Optional selected_counts [B,2] receives the number of valid prefix token IDs
// (0..2048) and tail IDs (0..3). Other IDs remain -1; no host readback.
void resident_pool_select(const float* query, const float* weights, const std::uint16_t* pools,
                          const std::uint8_t* allowed, float* scores, std::int32_t* ids,
                          const std::int32_t* slots, const std::int32_t* positions, int capacity,
                          int tier, int tokens, cudaStream_t stream, std::int32_t* selected_counts = nullptr);
// Same append/retrieval/selection ideals with physical BF16 latent/key/gate/query
// inputs. Head weights, bias, means, scores and reductions retain FP32.
void resident_index_append(const std::uint16_t* latent, const std::uint16_t* key, const std::uint16_t* gate,
                           const float* bias, std::uint16_t* latents, std::uint16_t* keys,
                           std::uint16_t* gates, std::uint16_t* pools, const std::int32_t* slots,
                           const std::int32_t* positions, int capacity, int tokens,
                           cudaStream_t stream);
void resident_retrieval(const std::uint16_t* query, const std::uint16_t* keys, float* means,
                        std::uint8_t* allowed, const std::int32_t* slots,
                        const std::int32_t* positions, int capacity, int tokens, int budget_blocks,
                        cudaStream_t stream);
void resident_pool_select(const std::uint16_t* query, const float* weights, const std::uint16_t* pools,
                          const std::uint8_t* allowed, float* scores, std::int32_t* ids,
                          const std::int32_t* slots, const std::int32_t* positions, int capacity,
                          int tier, int tokens, cudaStream_t stream, std::int32_t* selected_counts = nullptr);
// Batch MLA: query/output [B,heads,512], cache [lanes,C,512] BF16,
// ids [B,2051] distinct visible offsets or -1; probabilities [B,heads,2051].
// Complete softmax(Q*KV^T/16)*KV, FP32 score/probability/value accumulation.
// When supplied, selected_counts is the matching selector output: valid prefix
// at IDs[0..prefix), tail at IDs[2048..2048+tail); all other IDs are -1.
// Without counts, arbitrary valid/-1 ID ordering remains supported.
void resident_sparse_attention(const float* query, const std::uint16_t* latents,
                               const std::int32_t* ids, float* probabilities, float* out,
                               const std::int32_t* slots, int capacity, int tokens,
                               cudaStream_t stream, int heads = 64, const std::int32_t* selected_counts = nullptr);
// GLM DSA index mathematics. All cache arrays are caller-owned device buffers.
// Append one BF16-represented key/gate [128] into full-position BF16 arrays and
// publish a complete four-token pool when position%4==3. Position bias [4,128]
// is FP32. Probabilities/products/pool output have BF16 semantic boundaries.
void index_pool_append(const float* key, const float* gate, const float* position_bias,
                       std::uint16_t* keys, std::uint16_t* gates, std::uint16_t* pools,
                       int position, cudaStream_t stream);
// query [32,128], head_weights [32] FP32; pools [complete_pools,128] BF16.
// score=sum_h weights[h]/sqrt(32)*relu(dot(query[h],pool)/sqrt(128)).
// Optional allowed_pages is a device byte mask over 64-token pages. Masked
// complete pools receive -inf. Scores are FP32 and caller-owned.
void index_pool_scores(const float* query, const float* head_weights, const std::uint16_t* pools,
                       const std::uint8_t* allowed_pages, float* scores, int complete_pools,
                       cudaStream_t stream);
// Greedy top min(512,complete_pools) complete pools, expanding each into four
// token ids, followed by the current incomplete tail. ids has exactly 2051
// entries; unused slots are -1. scores are destructive scratch; lower-id ties.
void index_pool_select(float* scores, std::int32_t* ids, int visible_tokens, cudaStream_t stream);
// Absorbed MLA attention on resident compressed BF16 latent KV [context,D].
// query [T,H,D] is FP32; indices [T,count] include -1 padding. Scratch
// probabilities [T,H,count] and output [T,H,D] are FP32. Indices reference
// valid, distinct visible positions. No expanded full-context K/V is stored.
// Ideal: softmax(scale * Q @ KV.T) @ KV. All buffers disjoint.
void sparse_latent_attention(const float* query, const std::uint16_t* kv,
                             const std::int32_t* indices, float* probabilities, float* output,
                             int heads, int dim, int queries, int count, float scale,
                             cudaStream_t stream);
} // namespace ninfer::ops
