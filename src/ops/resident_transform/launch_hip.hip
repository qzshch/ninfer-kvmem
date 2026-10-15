#include "ninfer/ops/resident_transform.h"
#include "core/device.h"
#include <cmath>
#include <climits>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops {
namespace {
__device__ float unbf(std::uint16_t v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }

__device__ std::uint16_t bfword(float x) {
    unsigned b = __float_as_uint(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((b >> 16) | 0x40);
    return static_cast<std::uint16_t>((b + 0x7fffu + ((b >> 16) & 1)) >> 16);
}

__device__ float bf(float v) { return unbf(bfword(v)); }

__device__ float sig(float v) { return 1.f / (1.f + expf(-v)); }

__device__ float direct(const void* data, DType type, int c) {
    if (type == DType::FP32) return static_cast<const float*>(data)[c];
    const auto v = static_cast<const std::uint16_t*>(data)[c];
    if (type == DType::BF16) return unbf(v);
    const int e = (v >> 10) & 31, m = v & 1023;
    return (v & 0x8000 ? -1.f : 1.f) *
           (e ? ldexpf(1.f + m / 1024.f, e - 15) : ldexpf(static_cast<float>(m), -24));
}

template <class T> __device__ float represented(T x) {
    if constexpr (std::is_same_v<T, float>) return x;
    else return unbf(x);
}
template <class T> __device__ T stored(float x, bool rounded = true) {
    if constexpr (std::is_same_v<T, float>) return rounded ? bf(x) : x;
    else return bfword(x);
}

template <int Mode>
__global__ void convert_kernel(const void* x, void* y, std::size_t n) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    if constexpr (Mode == 0) static_cast<float*>(y)[i] = bf(static_cast<const float*>(x)[i]);
    if constexpr (Mode == 1)
        static_cast<std::uint16_t*>(y)[i] = bfword(static_cast<const float*>(x)[i]);
    if constexpr (Mode == 2)
        static_cast<float*>(y)[i] = unbf(static_cast<const std::uint16_t*>(x)[i]);
}

template <class Input, class Output>
__global__ void norm_kernel(const Input* x, const void* gain, DType gt, const void* bias, DType bt,
                            const Input* gate, Output* y, int n, float eps, bool center,
                            bool round) {
    const int r = blockIdx.x, lane = threadIdx.x;
    const auto off = static_cast<long long>(r) * n;
    __shared__ float red[256], mean, inv;
    float sum = 0;
    for (int c = lane; c < n; c += 256) sum += center ? represented(x[off + c]) : represented(x[off + c]) * represented(x[off + c]);
    red[lane] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    if (!lane) {
        mean = center ? red[0] / n : 0;
        inv  = center ? 0 : rsqrtf(red[0] / n + eps);
    }
    __syncthreads();
    if (center) {
        sum = 0;
        for (int c = lane; c < n; c += 256) {
            float d = represented(x[off + c]) - mean;
            sum += d * d;
        }
        red[lane] = sum;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (lane < s) red[lane] += red[lane + s];
            __syncthreads();
        }
        if (!lane) inv = rsqrtf(red[0] / n + eps);
        __syncthreads();
    }
    for (int c = lane; c < n; c += 256) {
        float value = (represented(x[off + c]) - mean) * inv;
        if (gain) value *= direct(gain, gt, c);
        if (bias) value += direct(bias, bt, c);
        if (gate) value *= sig(represented(gate[off + c]));
        y[off + c] = stored<Output>(value, round);
    }
    return;
}

template <class T> __device__ T swiglu_value(T a, T b, float limit) {
    const float g = limit > 0 ? fminf(represented(a), limit) : represented(a);
    const float u = limit > 0 ? fminf(fmaxf(represented(b), -limit), limit) : represented(b);
    const float activated = bf(static_cast<float>(static_cast<double>(g) / (1. + exp(-static_cast<double>(g)))));
    return stored<T>(activated * u);
}

template <int Mode, class T>
__global__ void element_kernel(const T* a, const T* b, T* o, std::size_t n, float limit) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    if constexpr (Mode == 0) o[i] = swiglu_value(a[i], b[i], limit);
    if constexpr (Mode == 1) o[i] = stored<T>(represented(a[i]) + represented(b[i]));
}

template <class T>
__global__ void owned_swiglu_kernel(const T* a, const T* b, T* out, const int* ids,
                                    int width, long long rows, int first, int banks, float limit) {
    const auto c = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (c >= width) return;
    for (long long row = blockIdx.y; row < rows; row += gridDim.y) {
        const auto i = row * width + c;
        const auto bank = static_cast<long long>(ids[row]) - first;
        if (bank < 0 || bank >= banks) out[i] = T{};
        else out[i] = swiglu_value(a[i], b[i], limit);
    }
}

template <bool Mean>
__global__ void streams_kernel(const float* x, float* y, int h) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c >= h) return;
    if constexpr (Mean) {
        float v = 0;
        for (int j = 0; j < 4; ++j) v += x[(static_cast<long long>(t) * 4 + j) * h + c];
        y[static_cast<long long>(t) * h + c] = bf(v * .25f);
    } else
        for (int j = 0; j < 4; ++j)
            y[(static_cast<long long>(t) * 4 + j) * h + c] = x[static_cast<long long>(t) * h + c];
}

template <class T>
__global__ void conv_kernel(const T* x, const void* weight, DType type, float* state, T* o,
                            int channels, int t, const std::int32_t* slots = nullptr,
                            const std::int32_t* offsets = nullptr) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= channels) return;
    if (slots) {
        state += static_cast<long long>(slots[blockIdx.y]) * channels * 3;
        const int begin = offsets[blockIdx.y];
        t = offsets[blockIdx.y + 1] - begin;
        x += static_cast<long long>(begin) * channels;
        o += static_cast<long long>(begin) * channels;
    }
    float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    const float w0 = direct(weight, type, c * 4), w1 = direct(weight, type, c * 4 + 1),
                w2 = direct(weight, type, c * 4 + 2), w3 = direct(weight, type, c * 4 + 3);
    // Preserve product and addition residuals before the observable BF16 cast.
    // Volatile staging prevents contraction/reassociation of the error-free
    // transforms; no device FP64 arithmetic assumption is needed.
    for (int j = 0; j < t; ++j) {
        const auto off        = static_cast<long long>(j) * channels + c;
        const float v         = represented(x[off]);
        const float values[4] = {s0, s1, s2, v}, weights[4] = {w0, w1, w2, w3};
        float sum = 0, correction = 0;
        for (int i = 0; i < 4; ++i) {
            volatile float product = values[i] * weights[i];
            const float low        = fmaf(values[i], weights[i], -product);
            volatile float next    = sum + product;
            volatile float back    = next - sum;
            volatile float first   = sum - (next - back);
            volatile float second  = product - back;
            correction += low + (first + second);
            sum = next;
        }
        const float z = bf(sum + correction);
        o[off] = stored<T>(static_cast<float>(static_cast<double>(z) / (1. + exp(-static_cast<double>(z)))));
        s0 = s1;
        s1 = s2;
        s2 = v;
    }
    state[c * 3]     = s0;
    state[c * 3 + 1] = s1;
    state[c * 3 + 2] = s2;
}

__global__ void gather_rows(const float* input, const int* ids, float* output, int width) {
    const int c = blockIdx.x * 256 + threadIdx.x, row = blockIdx.y;
    if (c < width)
        output[static_cast<long long>(row) * width + c] =
            input[static_cast<long long>(ids[row]) * width + c];
}

__global__ void router_serial_kernel(const float* logits, const float* bias, std::int32_t* ids,
                                     float* weights, int experts, int active, float scale) {
    if (threadIdx.x) return;
    const int t = blockIdx.x;
    float selected[32], sum = 0;
    for (int j = 0; j < active; ++j) {
        int best    = -1;
        float value = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int k = 0; k < j; ++k) used |= ids[t * active + k] == e;
            if (used) continue;
            float s = sig(logits[t * experts + e]) + bias[e];
            if (s > value) {
                value = s;
                best  = e;
            }
        }
        ids[t * active + j] = best;
        selected[j]         = sig(logits[t * experts + best]);
        sum += selected[j];
    }
    for (int j = 0; j < active; ++j) weights[t * active + j] = selected[j] * scale / (sum + 1e-20f);
}

__device__ void router_max(float& value, int& id, int distance) {
    const float other  = __shfl_down(value, distance, 64);
    const int other_id = __shfl_down(id, distance, 64);
    if (other > value || (other == value && other_id < id)) {
        value = other;
        id    = other_id;
    }
    return;
}

__global__ void router_kernel(const float* logits, const float* bias, std::int32_t* ids,
                              float* weights, int experts, int active, float scale) {
    __shared__ float probability[512], rank[512], wave_value[4];
    __shared__ int wave_id[4];
    const int t = blockIdx.x, thread = threadIdx.x, lane = thread & 63, wave = thread / 64;
    for (int e = thread; e < experts; e += 256) {
        const float p  = sig(logits[t * experts + e]);
        probability[e] = p;
        rank[e]        = p + bias[e];
    }
    __syncthreads();
    float sum = 0.f;
    for (int j = 0; j < active; ++j) {
        float value = -INFINITY;
        int id      = INT_MAX;
        for (int e = thread; e < experts; e += 256) {
            const float score = rank[e];
            if (score > value || (score == value && e < id)) {
                value = score;
                id    = e;
            }
        }
        for (int distance = 32; distance; distance /= 2) router_max(value, id, distance);
        if (!lane) {
            wave_value[wave] = value;
            wave_id[wave]    = id;
        }
        __syncthreads();
        if (!wave) {
            value = lane < 4 ? wave_value[lane] : -INFINITY;
            id    = lane < 4 ? wave_id[lane] : INT_MAX;
            router_max(value, id, 2);
            router_max(value, id, 1);
            if (!lane) {
                ids[t * active + j] = id;
                sum += probability[id];
                rank[id] = -INFINITY;
            }
        }
        __syncthreads();
    }
    // Keep the selected-score sum and final normalization in the original order.
    if (!thread)
        for (int j = 0; j < active; ++j)
            weights[t * active + j] = probability[ids[t * active + j]] * scale / (sum + 1e-20f);
    return;
}

template <class T>
__global__ void moe_sum_kernel(const T* x, const float* w, const float* shared, float* o, int h,
                               int active) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c >= h) return;
    float v = shared[static_cast<long long>(t) * h + c];
    for (int j = 0; j < active; ++j)
        v += represented(x[(static_cast<long long>(t) * active + j) * h + c]) * w[t * active + j];
    o[static_cast<long long>(t) * h + c] = bf(v);
}

#include "ops/resident_transform/argmax.cuh"

void shape(int width, int rows) {
    if (width < 1 || rows < 1 || rows > 65535)
        throw std::invalid_argument("resident_transform: dimensions");
}

template <bool Bf>
__global__ void reduce_rows_kernel(const void* x, float* o, int width, int rows) {
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= width) return;
    float sum = 0;
    for (int r = 0; r < rows; ++r) {
        const auto i = static_cast<long long>(r) * width + c;
        if constexpr (Bf)
            sum += unbf(static_cast<const std::uint16_t*>(x)[i]);
        else
            sum += static_cast<const float*>(x)[i];
    }
    o[c] = Bf ? sum / rows : sum;
    return;
}

void direct_weight(const Tensor* w, int values) {
    if (!w) return;
    if (!w->data || w->numel() != values || !w->is_contiguous() ||
        (w->dtype != DType::FP32 && w->dtype != DType::BF16 && w->dtype != DType::FP16))
        throw std::invalid_argument("resident_transform: direct weight/shape required");
}

void count(std::size_t n) {
    if (!n || n > static_cast<std::size_t>(INT32_MAX) * 256)
        throw std::invalid_argument("resident_transform: count");
}
} // namespace

void round_bf16(float* x, std::size_t n, cudaStream_t s) {
    count(n);
    convert_kernel<0><<<(n + 255) / 256, 256, 0, s>>>(x, x, n);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void pack_bf16(const float* x, std::uint16_t* o, std::size_t n, cudaStream_t s) {
    count(n);
    convert_kernel<1><<<(n + 255) / 256, 256, 0, s>>>(x, o, n);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void unpack_bf16(const std::uint16_t* x, float* o, std::size_t n, cudaStream_t s) {
    count(n);
    convert_kernel<2><<<(n + 255) / 256, 256, 0, s>>>(x, o, n);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_norm(const float* x, const Tensor* g, const Tensor* b, const float* gate, float* o,
                   int n, int rows, float eps, bool center, bool round, cudaStream_t s) {
    // Norm uses grid.x, unlike the two-dimensional row transforms. In
    // particular 1024 time rows times 64 KDA heads is a legal 65536-row norm.
    if (n < 1 || rows < 1) throw std::invalid_argument("resident_norm: dimensions");
    direct_weight(g, n);
    direct_weight(b, n);
    if (!x || !o || !(eps > 0) || !std::isfinite(eps))
        throw std::invalid_argument("resident_norm: operands");
    norm_kernel<<<rows, 256, 0, s>>>(x, g ? g->data : nullptr, g ? g->dtype : DType::FP32,
                                     b ? b->data : nullptr, b ? b->dtype : DType::FP32, gate, o, n,
                                     eps, center, round);
    CUDA_CHECK(cudaGetLastError());
    return;
}

namespace {
void activation(const Tensor& x) {
    if (!x.data || !x.is_contiguous() || x.numel() < 1 ||
        (x.dtype != DType::BF16 && x.dtype != DType::FP32))
        throw std::invalid_argument("resident transform: contiguous FP32/BF16 activation required");
}
}
void resident_norm(const Tensor& x, const Tensor* g, const Tensor* b, const Tensor* gate,
                   Tensor& out, float eps, bool center, cudaStream_t stream) {
    activation(x); activation(out);
    const auto width = x.ne[0];
    const auto rows = x.numel() / width;
    if (out.ne[0] != width || out.numel() != x.numel() || !(eps > 0) || !std::isfinite(eps) ||
        (x.data == out.data && x.dtype != out.dtype))
        throw std::invalid_argument("resident typed norm: extent/alias/epsilon");
    direct_weight(g, width); direct_weight(b, width);
    if (gate) {
        activation(*gate);
        if (gate->numel() != x.numel() || gate->dtype != x.dtype)
            throw std::invalid_argument("resident typed norm: gate storage/extent");
    }
    const auto launch = [&]<class Input, class Output>() {
        norm_kernel<<<rows, 256, 0, stream>>>(static_cast<const Input*>(x.data),
            g ? g->data : nullptr, g ? g->dtype : DType::FP32,
            b ? b->data : nullptr, b ? b->dtype : DType::FP32,
            gate ? static_cast<const Input*>(gate->data) : nullptr,
            static_cast<Output*>(out.data), width, eps, center, out.dtype == DType::BF16);
    };
    if (x.dtype == DType::BF16) {
        if (out.dtype == DType::BF16) launch.template operator()<std::uint16_t, std::uint16_t>();
        else launch.template operator()<std::uint16_t, float>();
    } else {
        if (out.dtype == DType::BF16) launch.template operator()<float, std::uint16_t>();
        else launch.template operator()<float, float>();
    }
    CUDA_CHECK(cudaGetLastError());
}
void resident_swiglu(const Tensor& a, const Tensor& b, Tensor& out, float limit, cudaStream_t stream) {
    activation(a); activation(b); activation(out);
    if (a.dtype != b.dtype || a.dtype != out.dtype || a.numel() != b.numel() || a.numel() != out.numel())
        throw std::invalid_argument("resident typed swiglu: matching physical tensors required");
    const auto n = a.numel(); count(n);
    if (a.dtype == DType::BF16)
        element_kernel<0><<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const std::uint16_t*>(a.data),
            static_cast<const std::uint16_t*>(b.data), static_cast<std::uint16_t*>(out.data), n, limit);
    else element_kernel<0><<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const float*>(a.data),
            static_cast<const float*>(b.data), static_cast<float*>(out.data), n, limit);
    CUDA_CHECK(cudaGetLastError());
}
void resident_swiglu_owned(const Tensor& a, const Tensor& b, const int* ids,
                           int first, int banks, Tensor& out, float limit, cudaStream_t stream) {
    activation(a); activation(b); activation(out);
    if (!ids || first < 0 || banks < 1 || first > INT_MAX - banks ||
        a.ne[0] != b.ne[0] || a.ne[0] != out.ne[0] || a.dtype != b.dtype ||
        a.dtype != out.dtype || a.numel() != b.numel() || a.numel() != out.numel())
        throw std::invalid_argument("resident owned swiglu: operands/range");
    const auto n = a.numel(); count(n);
    const auto rows = n / a.ne[0];
    const dim3 grid(static_cast<unsigned>((a.ne[0] + 255ULL) / 256),
                    static_cast<unsigned>(rows < 65535 ? rows : 65535));
    if (a.dtype == DType::BF16)
        owned_swiglu_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const std::uint16_t*>(a.data), static_cast<const std::uint16_t*>(b.data),
            static_cast<std::uint16_t*>(out.data), ids, a.ne[0], rows, first, banks, limit);
    else owned_swiglu_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const float*>(a.data), static_cast<const float*>(b.data),
            static_cast<float*>(out.data), ids, a.ne[0], rows, first, banks, limit);
    CUDA_CHECK(cudaGetLastError());
}
void resident_causal_conv_packed(const Tensor& x, const Tensor& kernel, float* state,
                                 const int* slots, const int* offsets, Tensor& out,
                                 int sequences, cudaStream_t stream) {
    activation(x); activation(out);
    const int channels = x.ne[0]; shape(channels, sequences); direct_weight(&kernel, channels * 4);
    if (x.dtype != out.dtype || x.numel() != out.numel() || !state || !slots || !offsets)
        throw std::invalid_argument("resident typed conv: storage/state/spans");
    const auto launch = [&]<class T>() {
        conv_kernel<<<dim3((channels + 255) / 256, sequences), 256, 0, stream>>>(
            static_cast<const T*>(x.data), kernel.data, kernel.dtype, state,
            static_cast<T*>(out.data), channels, 1, slots, offsets);
    };
    if (x.dtype == DType::BF16) launch.template operator()<std::uint16_t>();
    else launch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}

void resident_swiglu(const float* a, const float* b, float* o, std::size_t n, float limit,
                     cudaStream_t s) {
    count(n);
    element_kernel<0><<<(n + 255) / 256, 256, 0, s>>>(a, b, o, n, limit);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_add(const float* a, const float* b, float* o, std::size_t n, cudaStream_t s) {
    count(n);
    element_kernel<1><<<(n + 255) / 256, 256, 0, s>>>(a, b, o, n, 0);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void replicate_streams(const float* x, float* o, int h, int t, cudaStream_t s) {
    shape(h, t);
    streams_kernel<false><<<dim3((h + 255) / 256, t), 256, 0, s>>>(x, o, h);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void mean_streams(const float* x, float* o, int h, int t, cudaStream_t s) {
    shape(h, t);
    streams_kernel<true><<<dim3((h + 255) / 256, t), 256, 0, s>>>(x, o, h);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_causal_conv(const float* x, const Tensor& w, float* state, float* o, int c, int t,
                          cudaStream_t s) {
    shape(c, t);
    direct_weight(&w, c * 4);
    if (!x || !o || !state) throw std::invalid_argument("resident_conv: operands");
    conv_kernel<<<(c + 255) / 256, 256, 0, s>>>(x, w.data, w.dtype, state, o, c, t);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_causal_conv_packed(const float* x, const Tensor& w, float* state,
                                 const std::int32_t* slots, const std::int32_t* offsets,
                                 float* o, int c, int sequences, cudaStream_t s) {
    shape(c, sequences);
    direct_weight(&w, c * 4);
    if (!x || !o || !state || !slots || !offsets)
        throw std::invalid_argument("resident_conv_packed: operands");
    conv_kernel<<<dim3((c + 255) / 256, sequences), 256, 0, s>>>(
        x, w.data, w.dtype, state, o, c, 1, slots, offsets);
    CUDA_CHECK(cudaGetLastError());
}

void resident_gather_rows(const float* x, const std::int32_t* ids, float* out,
                          int width, int rows, cudaStream_t s) {
    shape(width, rows);
    if (!x || !ids || !out || x == out)
        throw std::invalid_argument("resident_gather_rows: operands");
    gather_rows<<<dim3((width + 255) / 256, rows), 256, 0, s>>>(x, ids, out, width);
    CUDA_CHECK(cudaGetLastError());
}

void resident_router(const float* x, const float* b, std::int32_t* ids, float* w, int e, int k,
                     int t, float scale, cudaStream_t s) {
    shape(e, t);
    if (!x || !b || !ids || !w || k < 1 || k > 32 || k > e || !std::isfinite(scale))
        throw std::invalid_argument("resident_router: operands");
    if (e <= 512)
        router_kernel<<<t, 256, 0, s>>>(x, b, ids, w, e, k, scale);
    else
        router_serial_kernel<<<t, 1, 0, s>>>(x, b, ids, w, e, k, scale);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_moe_sum(const float* x, const float* w, const float* shared, float* o, int h, int k,
                      int t, cudaStream_t s) {
    shape(h, t);
    if (k < 1 || !x || !w || !shared || !o)
        throw std::invalid_argument("resident_moe_sum: operands");
    moe_sum_kernel<<<dim3((h + 255) / 256, t), 256, 0, s>>>(x, w, shared, o, h, k);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_moe_sum(const Tensor& x, const float* weights, const float* shared, float* out,
                      int hidden, int active, int tokens, cudaStream_t stream) {
    activation(x); shape(hidden, tokens);
    if (x.dtype != DType::BF16 || x.numel() != static_cast<long long>(hidden) * active * tokens ||
        active < 1 || active > 32 || !weights || !shared || !out)
        throw std::invalid_argument("resident typed expert sum: operands");
    moe_sum_kernel<<<dim3((hidden + 255) / 256, tokens), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), weights, shared, out, hidden, active);
    CUDA_CHECK(cudaGetLastError());
}

void resident_argmax(const float* x, std::int32_t* ids, int w, int t, cudaStream_t s) {
    shape(w, t);
    if (!x || !ids) throw std::invalid_argument("resident_argmax: operands");
    argmax_kernel<<<t, 256, 0, s>>>(x, ids, w);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_mean_bf16(const std::uint16_t* x, float* o, int w, int t, cudaStream_t s) {
    shape(w, t);
    if (!x || !o) throw std::invalid_argument("resident_mean_bf16: operands");
    reduce_rows_kernel<true><<<(w + 255) / 256, 256, 0, s>>>(x, o, w, t);
    CUDA_CHECK(cudaGetLastError());
    return;
}

void resident_sum_rows(const float* x, float* o, int w, int t, cudaStream_t s) {
    shape(w, t);
    if (!x || !o) throw std::invalid_argument("resident_sum_rows: operands");
    reduce_rows_kernel<false><<<(w + 255) / 256, 256, 0, s>>>(x, o, w, t);
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops
