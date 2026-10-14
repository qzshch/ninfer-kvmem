#pragma once
#include <cuda_runtime.h>

namespace ninfer::ops {

// Four-stream mHC with caller-owned buffers, no allocation. Activations are
// BF16-represented values carried in FP32 words; coefficients remain FP32.
// tokens>=1, hidden>=1. Coefficients: [T,24] = pre[4],post[4],comb[4,4].
// projection is the FP32 [T,24] dot of unweighted RMS-normalized flattened
// input with the stored mHC matrix. base[24], scale[3] are FP32. Coefficient
// generation uses row softmax+eps, column normalization, then iters-1 row/column
// normalizations. collapse = BF16(sum(pre*stream)); expand = BF16(
// BF16(BF16(post)*layer_output) + BF16(BF16(comb).T @ residual)).
// All operands must be disjoint, except expand may replace residual exactly.
void hyper_connection_coefficients(const float* projection,const float* base,const float* scale,
                                   float* coefficients,int tokens,int iterations,float epsilon,
                                   cudaStream_t stream);
void hyper_connection_collapse(const float* residual,const float* coefficients,float* output,
                               int hidden,int tokens,cudaStream_t stream);
void hyper_connection_expand(const float* residual,const float* layer_output,
                             const float* coefficients,float* output,int hidden,int tokens,
                             cudaStream_t stream);

} // namespace ninfer::ops
