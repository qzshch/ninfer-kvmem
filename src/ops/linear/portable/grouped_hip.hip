#include "ninfer/ops/indexed_linear.h"
#include "ninfer/ops/resident_transform.h"
#include "ops/linear/portable/launch.h"
#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/portable/mmac_mapping.h"
#include <climits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
#include "ops/linear/portable/weight_decode.cuh"
#include "ops/linear/portable/mmac_tiles.cuh"
struct Task { int bank, begin, count; };
static_assert(sizeof(Task) == 3 * sizeof(int));

__global__ void collect_routes(const int* ids, int* counts, int* assignments,
                                int first, int banks, int count) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= count) return;
    const auto bank = static_cast<long long>(ids[i]) - first;
    if (bank < 0 || bank >= banks) return;
    const int at = atomicAdd(counts + bank, 1);
    assignments[static_cast<long long>(bank) * count + at] = i;
}

__global__ void group_tasks(const int* counts, Task* tasks, int* total, int banks, int columns) {
    int used = 0;
    for (int bank = 0; bank < banks; ++bank)
        for (int at = 0; at < counts[bank]; at += columns)
            tasks[used++] = {bank, at, min(columns, counts[bank] - at)};
    *total = used;
}

#ifdef NINFER_K100_MMAC
template <QType F, bool InputBf16, bool OutputBf16>
__global__ void grouped_mmac(const void* x, const void* weights, void* out,
                              const int* assignments, const Task* tasks, const int* total,
                              int n, int k, int choices, int capacity, bool shared, bool rounded) {
    if (blockIdx.y >= *total) return;
    const auto task = tasks[blockIdx.y];
    mmac_tile<F, InputBf16, OutputBf16>(x, weights, out, n, k, 0, k, task.count, 0,
        static_cast<long long>(task.bank) * n, rounded,
        assignments + static_cast<long long>(task.bank) * capacity + task.begin, shared ? choices : 1);
}
#endif

template <QType F, bool InputBf16, bool OutputBf16>
__global__ void grouped_projection(const void* x, const void* weights, void* out,
                                    const int* assignments, const Task* tasks, const int* total,
                                    int n, int k, int choices, int capacity, bool shared, bool rounded) {
    if (blockIdx.y >= *total) return;
    constexpr int TileK = 128;
    const auto task = tasks[blockIdx.y];
    const int lane = threadIdx.x, part = lane % 4, cell = lane / 4;
    const int row = cell % 8, token = cell / 8, row0 = blockIdx.x * 8;
    __shared__ float a[8][TileK], b[8][TileK], sums[256];
    __shared__ int indices[8];
    if (lane < 8) indices[lane] = lane < task.count
        ? assignments[static_cast<long long>(task.bank) * capacity + task.begin + lane] : 0;
    __syncthreads();
    float sum = 0.f;
    for (int base = 0; base < k; base += TileK) {
        for (int j = lane; j < 8 * TileK; j += 256) {
            const int t = j / TileK, c = j % TileK;
            const auto input_row = shared ? indices[t] / choices : indices[t];
            a[t][c] = t < task.count && base + c < k
                ? (InputBf16 ? decode_bf16(static_cast<const std::uint16_t*>(x)[static_cast<long long>(input_row) * k + base + c])
                            : static_cast<const float*>(x)[static_cast<long long>(input_row) * k + base + c]) : 0.f;
            b[t][c] = row0 + t < n && base + c < k
                ? load_weight<F>(weights, static_cast<long long>(task.bank) * n + row0 + t, base + c, k) : 0.f;
        }
        __syncthreads();
        for (int c = part; c < TileK; c += 4) sum += b[row][c] * a[token][c];
        __syncthreads();
    }
    sums[lane] = sum;
    __syncthreads();
    if (!part && row0 + row < n && token < task.count) {
        const float v = ((sums[lane] + sums[lane + 1]) + sums[lane + 2]) + sums[lane + 3];
        const auto at = static_cast<long long>(indices[token]) * n + row0 + row;
        if constexpr (OutputBf16) static_cast<std::uint16_t*>(out)[at] = encode_bf16(v);
        else static_cast<float*>(out)[at] = rounded ? decode_bf16(encode_bf16(v)) : v;
    }
}
} // namespace

std::size_t indexed_linear_grouped_workspace_capacity_bytes(int banks, int choices, int tokens) {
    if (banks < 1 || banks > 288 || choices < 1 || choices > 32 || tokens < 1 || tokens > 1024)
        throw std::invalid_argument("grouped linear: routing geometry");
    const auto count = static_cast<std::size_t>(choices) * tokens;
    return (banks + banks * count + 3 * ((count + 7) / 8 + banks) + 1) * sizeof(int);
}

namespace {
struct Routes {
    int* assignments;
    Task* tasks;
    int* total;
    int banks, choices, tokens, columns, capacity;
};

Routes prepare_routes(WorkspaceArena& work, int banks, int choices, int tokens,
                       int columns, cudaStream_t stream) {
    const auto bytes = indexed_linear_grouped_workspace_capacity_bytes(banks, choices, tokens);
    const int count = choices * tokens;
    auto* counts = static_cast<int*>(work.alloc_bytes(bytes).data);
    auto* assignments = counts + banks;
    auto* tasks = reinterpret_cast<Task*>(assignments + banks * count);
    auto* total = reinterpret_cast<int*>(tasks + (count + 7) / 8 + banks);
    CUDA_CHECK(cudaMemsetAsync(counts, 0, banks * sizeof(int), stream));
    return {assignments, tasks, total, banks, choices, tokens, columns,
            (count + columns - 1) / columns + banks};
}

void fill_routes(const Routes& routes, const int* ids, int first, cudaStream_t stream) {
    const int count = routes.choices * routes.tokens;
    auto* counts = routes.assignments - routes.banks;
    collect_routes<<<(count + 255) / 256, 256, 0, stream>>>(ids, counts, routes.assignments,
                                                         first, routes.banks, count);
    group_tasks<<<1, 1, 0, stream>>>(counts, routes.tasks, routes.total, routes.banks, routes.columns);
    CUDA_CHECK(cudaGetLastError());
}

bool matrix_columns(bool bf16, int rows, int k) {
#ifdef NINFER_K100_MMAC
    return bf16 && rows >= 128 && k >= 128;
#else
    (void)bf16; (void)rows; (void)k;
    return false;
#endif
}
} // namespace

template <bool InputBf16, bool OutputBf16>
void grouped_impl(const void* x, const Weight& w, const int* ids, void* out,
                                  int rows, int choices, int tokens, bool shared, int first,
                                  WorkspaceArena& work, cudaStream_t stream, bool rounded,
                                  const Routes* prepared = nullptr) {
    detail::validate_portable_linear_weight(w.qtype, w.n, w.k);
    const auto block = interleaved_block_geometry(w.qtype).values;
    if (!x || !ids || !out || x == out || !w.qdata || rows < 1 || w.n % rows || first < 0 ||
        w.layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous))
        throw std::invalid_argument("grouped linear: operands/layout");
    const int banks = w.n / rows;
    (void)indexed_linear_grouped_workspace_capacity_bytes(banks, choices, tokens);
    const int count = choices * tokens;
    if (first > INT_MAX - banks) throw std::invalid_argument("grouped linear: bank origin overflow");
    auto scope = work.scope();
    const auto routes = prepared ? *prepared : prepare_routes(work, banks, choices, tokens,
        matrix_columns(InputBf16, rows, w.k) ? 16 : 8, stream);
    if (routes.banks != banks || routes.choices != choices || routes.tokens != tokens ||
        (routes.columns == 16 && !matrix_columns(InputBf16, rows, w.k)))
        throw std::invalid_argument("grouped linear: prepared routing geometry");
    if (!prepared) fill_routes(routes, ids, first, stream);
    CUDA_CHECK(cudaMemsetAsync(out, 0, static_cast<std::size_t>(count) * rows * (OutputBf16 ? 2 : 4), stream));
#ifdef NINFER_K100_MMAC
#define GROUPED(F) case QType::F: \
    if (routes.columns == 16) \
        grouped_mmac<QType::F, InputBf16, OutputBf16><<<dim3((rows + 63) / 64, routes.capacity), 256, 0, stream>>>( \
            x, w.qdata, out, routes.assignments, routes.tasks, routes.total, rows, w.k, choices, count, shared, rounded); \
    else grouped_projection<QType::F, InputBf16, OutputBf16><<<dim3((rows + 7) / 8, routes.capacity), 256, 0, stream>>>( \
            x, w.qdata, out, routes.assignments, routes.tasks, routes.total, rows, w.k, choices, count, shared, rounded); break
#else
#define GROUPED(F) case QType::F: grouped_projection<QType::F, InputBf16, OutputBf16> \
    <<<dim3((rows + 7) / 8, routes.capacity), 256, 0, stream>>>(x, w.qdata, out, routes.assignments, routes.tasks, routes.total, \
        rows, w.k, choices, count, shared, rounded); break
#endif
    switch (w.qtype) {
        GROUPED(FP32); GROUPED(FP16); GROUPED(BF16); GROUPED(MXFP4_G32_E8M0);
        GROUPED(Q8_G32_FP16_BLOCK); GROUPED(IQ3_S_G256_BLOCK); GROUPED(IQ4_XS_G256_BLOCK);
        GROUPED(Q3_K_G256_BLOCK); GROUPED(Q4_K_G256_BLOCK); GROUPED(Q6_K_G256_BLOCK);
        default: throw std::invalid_argument("grouped linear: unsupported represented format");
    }
#undef GROUPED
    CUDA_CHECK(cudaGetLastError());
}
void indexed_linear_grouped_fp32(const float* x, const Weight& w, const int* ids, float* out,
                                  int rows, int choices, int tokens, bool shared, int first,
                                  WorkspaceArena& work, cudaStream_t stream, bool rounded) {
    grouped_impl<false, false>(x, w, ids, out, rows, choices, tokens, shared, first, work, stream, rounded);
}
void indexed_linear_grouped(const Tensor& x, const Weight& w, const int* ids, Tensor& out,
                             int rows, int choices, int tokens, bool shared, int first,
                             WorkspaceArena& work, cudaStream_t stream) {
    if (!x.data || !out.data || !x.is_contiguous() || !out.is_contiguous() ||
        x.ne[0] != w.k || x.numel() != static_cast<long long>(w.k) * tokens * (shared ? 1 : choices) ||
        out.numel() != static_cast<long long>(rows) * tokens * choices ||
        (x.dtype != DType::BF16 && x.dtype != DType::FP32) ||
        (out.dtype != DType::BF16 && out.dtype != DType::FP32))
        throw std::invalid_argument("grouped linear: physical tensor storage/extent");
    const auto launch = [&]<bool Input, bool Output>() {
        grouped_impl<Input, Output>(x.data, w, ids, out.data, rows, choices, tokens,
                                    shared, first, work, stream, false);
    };
    if (x.dtype == DType::BF16) {
        if (out.dtype == DType::BF16) launch.template operator()<true, true>();
        else launch.template operator()<true, false>();
    } else {
        if (out.dtype == DType::BF16) launch.template operator()<false, true>();
        else launch.template operator()<false, false>();
    }
}

std::size_t indexed_swiglu_linear_workspace_capacity_bytes(DType storage, int banks,
    int intermediate_rows, int choices, int tokens) {
    if ((storage != DType::BF16 && storage != DType::FP32) || intermediate_rows < 1)
        throw std::invalid_argument("indexed swiglu linear: storage/intermediate extent");
    const auto routes = indexed_linear_grouped_workspace_capacity_bytes(banks, choices, tokens);
    const auto elements = static_cast<std::size_t>(intermediate_rows) * choices * tokens;
    return 2 * elements * (storage == DType::BF16 ? 2 : 4) + routes + 3 * 255;
}

void indexed_swiglu_linear(const Tensor& x, const Weight& gate, const Weight& up,
    const Weight& down, const int* ids, Tensor& out, int intermediate_rows, int output_rows,
    int choices, int tokens, int first, float limit, WorkspaceArena& work, cudaStream_t stream) {
    if (intermediate_rows < 1 || output_rows < 1 || gate.n % intermediate_rows ||
        gate.k != up.k || down.k != intermediate_rows || up.n != gate.n || first < 0 ||
        !x.data || !out.data || !ids || x.data == out.data || !x.is_contiguous() ||
        !out.is_contiguous() || x.dtype != out.dtype || x.ne[0] != gate.k || out.ne[0] != output_rows ||
        x.numel() != static_cast<long long>(gate.k) * tokens ||
        out.numel() != static_cast<long long>(output_rows) * tokens * choices)
        throw std::invalid_argument("indexed swiglu linear: operands/geometry");
    const int banks = gate.n / intermediate_rows;
    (void)indexed_swiglu_linear_workspace_capacity_bytes(x.dtype, banks, intermediate_rows, choices, tokens);
    if (first > INT_MAX - banks || static_cast<long long>(down.n) != static_cast<long long>(banks) * output_rows)
        throw std::invalid_argument("indexed swiglu linear: bank interval");
    for (const auto* weight : {&gate, &up, &down}) {
        detail::validate_portable_linear_weight(weight->qtype, weight->n, weight->k);
        const auto block = interleaved_block_geometry(weight->qtype).values;
        if (!weight->qdata || weight->layout != (block ? QuantLayout::BlockInterleaved : QuantLayout::Contiguous))
            throw std::invalid_argument("indexed swiglu linear: weight storage");
    }
    auto scope = work.scope();
    const int count = choices * tokens;
    auto g = work.alloc(x.dtype, {intermediate_rows, count});
    auto u = work.alloc(x.dtype, {intermediate_rows, count});
    Routes routes{};
    if (tokens >= 16) {
        const bool matrix = matrix_columns(x.dtype == DType::BF16, intermediate_rows, gate.k) &&
                            matrix_columns(x.dtype == DType::BF16, output_rows, intermediate_rows);
        routes = prepare_routes(work, banks, choices, tokens, matrix ? 16 : 8, stream);
        fill_routes(routes, ids, first, stream);
    }
    const auto project = [&](const Tensor& input, const Weight& bank, Tensor& result, int rows, bool shared) {
        if (tokens >= 16) {
            if (x.dtype == DType::BF16)
                grouped_impl<true, true>(input.data, bank, ids, result.data, rows, choices,
                    tokens, shared, first, work, stream, true, &routes);
            else grouped_impl<false, false>(input.data, bank, ids, result.data, rows, choices,
                    tokens, shared, first, work, stream, true, &routes);
        } else if (x.dtype == DType::BF16)
            indexed_linear_mixed(input, bank, ids, result, rows, choices, tokens, shared, first, stream);
        else indexed_linear_owned_fp32(static_cast<const float*>(input.data), bank, ids,
            static_cast<float*>(result.data), rows, choices, tokens, shared, first, stream, true);
    };
    project(x, gate, g, intermediate_rows, true);
    project(x, up, u, intermediate_rows, true);
    resident_swiglu_owned(g, u, ids, first, banks, g, limit, stream);
    project(g, down, out, output_rows, false);
}
} // namespace ninfer::ops
