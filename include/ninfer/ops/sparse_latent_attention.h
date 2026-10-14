#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {
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
