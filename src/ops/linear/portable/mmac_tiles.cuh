// Hygon FP32 MMAC, with original represented weights and FP32 accumulation.
// Register contract: https://github.com/HYGON-AI/composable_kernel-das
// include/ck/utility/hcu_mmac.hpp and
// warp/mmac_gemm.hpp (16x16x4/8 f32). No dependency on that implementation.
// Include after weight_decode.cuh inside a launcher's private namespace.
#if defined(NINFER_K100_MMAC)
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx928__)
#error "NInfer K100 MMAC requires the vendor gfx928 compiler target"
#endif
#if __has_builtin(__builtin_hcu_mmac_16x16x8_f32)
#define NINFER_MMAC_K8 1
#elif !__has_builtin(__builtin_hcu_mmac_16x16x4_f32)
#error "DTK compiler lacks Hygon FP32 MMAC builtins; build with NINFER_K100_MMAC=OFF"
#endif
#endif
using MmacFloat4 = float __attribute__((ext_vector_type(4)));
using MmacFloat2 = float __attribute__((ext_vector_type(2)));

template <bool Bf16> __device__ float mmac_input(const void* x, long long at) {
    if constexpr (Bf16) return decode_bf16(static_cast<const std::uint16_t*>(x)[at]);
    else return static_cast<const float*>(x)[at];
}
template <bool Bf16> __device__ void mmac_output(void* out, long long at, float value, bool rounded) {
    if constexpr (Bf16) static_cast<std::uint16_t*>(out)[at] = encode_bf16(value);
    else static_cast<float*>(out)[at] = rounded ? decode_bf16(encode_bf16(value)) : value;
}

// Four wave64 groups share sixteen activation rows. The encoded weight tile
// exists only as call-local LDS; neither global repacking nor A16 weight casts.
// assignments, when present, scatter back to the original expert-choice rows.
template <QType F, bool InputBf16, bool OutputBf16>
__device__ void mmac_tile(const void* x, const void* weights, void* out,
                          int n, int parent_k, int begin, int k, int tokens,
                          int token0, long long weight_row0, bool rounded,
                          const int* assignments = nullptr, int input_choices = 1) {
    using Map = ::ninfer::ops::detail::MmacF32Mapping;
    constexpr int TileK = 128, Stride = TileK + 1;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const int row0 = blockIdx.x * 64, row = row0 + wave * 16 + Map::row(lane);
    __shared__ float a[64][Stride], b[16][Stride];
    MmacFloat4 sum{0.f, 0.f, 0.f, 0.f};
    for (int base = 0; base < k; base += TileK) {
        for (int at = tid; at < 64 * TileK; at += 256) {
            const int r = at / TileK, c = at % TileK;
            a[r][c] = row0 + r < n && base + c < k
                ? load_weight<F>(weights, weight_row0 + row0 + r, begin + base + c, parent_k) : 0.f;
        }
        for (int at = tid; at < 16 * TileK; at += 256) {
            const int t = at / TileK, c = at % TileK;
            const bool valid = token0 + t < tokens && base + c < k;
            const auto source = valid && assignments ? assignments[token0 + t] / input_choices : token0 + t;
            b[t][c] = valid ? mmac_input<InputBf16>(x, static_cast<long long>(source) * k + base + c) : 0.f;
        }
        __syncthreads();
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(NINFER_MMAC_K8)
        for (int c = 0; c < TileK; c += 8) {
            const int column = c + Map::contraction(lane, 2);
            MmacFloat2 lhs{a[wave * 16 + Map::row(lane)][column], a[wave * 16 + Map::row(lane)][column + 1]};
            MmacFloat2 rhs{b[Map::row(lane)][column], b[Map::row(lane)][column + 1]};
            sum = __builtin_hcu_mmac_16x16x8_f32(lhs, rhs, sum);
        }
#else
        for (int c = 0; c < TileK; c += 4)
            sum = __builtin_hcu_mmac_16x16x4_f32(a[wave * 16 + Map::row(lane)][c + Map::contraction(lane, 1)],
                                                b[Map::row(lane)][c + Map::contraction(lane, 1)], sum);
#endif
#endif
        __syncthreads();
    }
    for (int reg = 0; reg < 4; ++reg) {
        const int t = token0 + Map::column(lane, reg);
        if (row < n && t < tokens) {
            const auto output_row = assignments ? assignments[t] : t;
            mmac_output<OutputBf16>(out, static_cast<long long>(output_row) * n + row, sum[reg], rounded);
        }
    }
}

template <QType F, bool InputBf16, bool OutputBf16>
__global__ void mmac_projection(const void* x, const void* weights, void* out,
                                int n, int parent_k, int begin, int k, int tokens, bool rounded) {
    mmac_tile<F, InputBf16, OutputBf16>(x, weights, out, n, parent_k, begin, k,
                                       tokens, blockIdx.y * 16, 0, rounded);
}

template <QType F, bool InputBf16, bool OutputBf16>
__global__ void mmac_indexed(const void* x, const void* weights, const int* ids, void* out,
                             int n, int k, int choices, bool shared, bool rounded, int first, int banks) {
    const int t = blockIdx.y, c = blockIdx.z;
    const auto bank = static_cast<long long>(ids ? ids[t * choices + c] : c) - first;
    const auto output_row = static_cast<long long>(t) * choices + c;
    if (bank < 0 || bank >= banks) {
        const int row = blockIdx.x * 64 + threadIdx.x;
        if (threadIdx.x < 64 && row < n) mmac_output<OutputBf16>(out, output_row * n + row, 0.f, false);
        return;
    }
    const auto input_row = static_cast<long long>(t) * (shared ? 1 : choices) + (shared ? 0 : c);
    const auto* input = static_cast<const unsigned char*>(x) + input_row * k * (InputBf16 ? 2 : 4);
    auto* output = static_cast<unsigned char*>(out) + output_row * n * (OutputBf16 ? 2 : 4);
    mmac_tile<F, InputBf16, OutputBf16>(input, weights, output, n, k, 0, k, 1, 0, bank * n, rounded);
}
#undef NINFER_MMAC_K8
#endif
