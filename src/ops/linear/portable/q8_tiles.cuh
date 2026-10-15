// Q8 block-interleaved operands stay encoded in global memory. Include inside
// the portable launcher's private namespace after weight_decode.cuh.
template <bool Bf16>
__device__ float tile_input(const void* x, long long i) {
    if constexpr (Bf16) return decode_bf16(static_cast<const std::uint16_t*>(x)[i]);
    else return static_cast<const float*>(x)[i];
}
template <bool Bf16, bool OutputBf16 = Bf16>
__global__ void q8_decode_tile(const void* x, const void* weight, void* out,
                               int n, int parent_k, int begin, int k, bool rounded) {
    constexpr int Rows = 4, TileK = 512;
    const int lane = threadIdx.x, group = lane / 16, pair = lane % 16;
    const int first_row = blockIdx.x * Rows;
    const auto* bytes = static_cast<const unsigned char*>(weight);
    __shared__ float input[TileK], scales[Rows][16];
#ifdef NINFER_HYGON_DTK
    __shared__ float partial[Rows][128];
#else
    __shared__ float partial[Rows][256];
#endif
    float sum[Rows]{};
    for (int base = 0; base < k; base += TileK) {
        for (int j = lane; j < TileK; j += 256)
            input[j] = base + j < k ? tile_input<Bf16>(x, base + j) : 0.f;
        if (!pair)
            for (int r = 0; r < Rows; ++r) {
                const bool valid = first_row + r < n && base + group * 32 < k;
                const auto block = (static_cast<long long>(first_row + r) * parent_k +
                                    begin + base + group * 32) / 32;
                scales[r][group] = valid
                    ? decode_half(*reinterpret_cast<const unsigned short*>(bytes + block * 34)) : 0.f;
            }
        __syncthreads();
        for (int r = 0; r < Rows; ++r) {
            if (first_row + r >= n || base + group * 32 >= k) continue;
            const auto block = (static_cast<long long>(first_row + r) * parent_k +
                                begin + base + group * 32) / 32;
            // Every 34-byte block and code pair is naturally two-byte aligned.
            const auto code = *reinterpret_cast<const unsigned short*>(
                bytes + block * 34 + 2 + pair * 2);
            const float a = scales[r][group] * static_cast<signed char>(code & 255);
            const float b = scales[r][group] * static_cast<signed char>(code >> 8);
            sum[r] += a * input[group * 32 + pair * 2];
            sum[r] += b * input[group * 32 + pair * 2 + 1];
        }
        __syncthreads();
    }
#ifdef NINFER_HYGON_DTK
    // Keep the original 256-lane tree: joins128 and64 cross wave64;
    // the remaining joins stay within the first wave. The two shared regions
    // are disjoint while the lower half reads and the next quarter publishes.
    if (lane >= 128)
        for (int r = 0; r < Rows; ++r) partial[r][lane - 128] = sum[r];
    __syncthreads();
    if (lane < 128)
        for (int r = 0; r < Rows; ++r) {
            sum[r] += partial[r][lane];
            if (lane >= 64) partial[r][lane] = sum[r];
        }
    __syncthreads();
    if (lane < 64)
        for (int r = 0; r < Rows; ++r) {
            float v = sum[r] + partial[r][lane + 64];
            for (int distance = 32; distance; distance /= 2)
                v += __shfl_down(v, distance, 64);
            if (!lane && first_row + r < n) {
                if constexpr (OutputBf16)
                    static_cast<std::uint16_t*>(out)[first_row + r] = encode_bf16(v);
                else static_cast<float*>(out)[first_row + r] = projection_output(v, rounded);
            }
        }
#else
    for (int r = 0; r < Rows; ++r) partial[r][lane] = sum[r];
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (lane < stride)
            for (int r = 0; r < Rows; ++r) partial[r][lane] += partial[r][lane + stride];
        __syncthreads();
    }
    if (lane < Rows && first_row + lane < n) {
        const float v = partial[lane][0];
        if constexpr (OutputBf16)
            static_cast<std::uint16_t*>(out)[first_row + lane] = encode_bf16(v);
        else static_cast<float*>(out)[first_row + lane] = projection_output(v, rounded);
    }
#endif
}

template <bool Bf16, bool OutputBf16 = Bf16>
__global__ void q8_matrix_tile(const void* x, const void* weight, void* out,
                               int n, int parent_k, int begin, int k, int tokens, bool rounded) {
    constexpr int Rows = 8, Columns = 8, TileK = 128;
    const int lane = threadIdx.x, part = lane % 4, cell = lane / 4;
    const int r = cell % Rows, t = cell / Rows;
    const int row0 = blockIdx.x * Rows, token0 = blockIdx.y * Columns;
    __shared__ float a[Columns][TileK], b[Rows][TileK], partial[256];
    const auto* bytes = static_cast<const unsigned char*>(weight);
    float sum = 0.f;
    for (int base = 0; base < k; base += TileK) {
        for (int j = lane; j < Columns * TileK; j += 256) {
            const int m = j / TileK, c = j % TileK;
            a[m][c] = token0 + m < tokens && base + c < k
                ? tile_input<Bf16>(x, static_cast<long long>(token0 + m) * k + base + c) : 0.f;
        }
        for (int j = lane; j < Rows * TileK; j += 256) {
            const int nr = j / TileK, c = j % TileK;
            b[nr][c] = row0 + nr < n && base + c < k
                ? load_weight<QType::Q8_G32_FP16_BLOCK>(bytes, row0 + nr, begin + base + c, parent_k)
                : 0.f;
        }
        __syncthreads();
        for (int c = part; c < TileK; c += 4) sum += b[r][c] * a[t][c];
        __syncthreads();
    }
    partial[lane] = sum;
    __syncthreads();
    if (!part && row0 + r < n && token0 + t < tokens) {
        const float v = ((partial[lane] + partial[lane + 1]) + partial[lane + 2]) + partial[lane + 3];
        const auto index = static_cast<long long>(token0 + t) * n + row0 + r;
        if constexpr (OutputBf16)
            static_cast<std::uint16_t*>(out)[index] = encode_bf16(v);
        else
            static_cast<float*>(out)[index] = projection_output(v, rounded);
    }
}
