#include "ninfer/ops/linear.h"
#include "ops/linear/portable/launch.h"
#include <stdexcept>

namespace ninfer::ops {
void linear_mixed(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (!x.data || !out.data || x.data == out.data || x.ne[0] != w.k || out.ne[0] != w.n ||
        x.ne[1] != out.ne[1] || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 ||
        out.ne[3] != 1 || !x.is_contiguous() || !out.is_contiguous())
        throw std::invalid_argument("linear_mixed: contiguous disjoint [K,T] -> [N,T] required");
    detail::portable_linear_launch(x, w, out, stream);
}
void linear_partial(const Tensor& x, const Weight& w, Tensor& out, int begin, cudaStream_t stream) {
    if (out.dtype != DType::FP32 || x.ne[1] != out.ne[1] || out.ne[0] != w.n ||
        x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1 ||
        !x.is_contiguous() || !out.is_contiguous())
        throw std::invalid_argument("linear_partial: compact input -> FP32 partial required");
    detail::portable_partial_linear_launch(x.data, x.dtype, w, static_cast<float*>(out.data),
                                           begin, x.ne[0], x.ne[1], stream);
}

void linear_partial_fp32(const float* x, const Weight& w, float* out, int begin,
                          int columns, int tokens, cudaStream_t stream) {
    detail::portable_partial_linear_launch(x, DType::FP32, w, out, begin, columns, tokens, stream);
}
void linear_fp32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                 bool bf16_output) {
    if (x.dtype != DType::FP32 || out.dtype != DType::FP32 || !x.data || !out.data ||
        x.ne[0] != w.k || out.ne[0] != w.n || x.ne[1] != out.ne[1] || x.ne[2] != 1 ||
        x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1 || !x.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("linear_fp32: expected contiguous FP32 [K,T] -> [N,T]");
    }
    detail::portable_linear_launch(x, w, out, stream, bf16_output);
}
} // namespace ninfer::ops
