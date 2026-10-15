#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include "ops/linear_attention/kimi_delta_attention/chunked_dtk.h"

namespace ninfer::ops::detail::kimi_delta_attention {
namespace {
__device__ float bf_value(const __nv_bfloat16* p, long long i) {
    return __uint_as_float(static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(p)[i]) << 16);
}
__device__ void store_bf(__nv_bfloat16* p, long long i, float value) {
    const auto u = __float_as_uint(value);
    reinterpret_cast<std::uint16_t*>(p)[i] = (u & 0x7fffffffU) > 0x7f800000U
        ? (u >> 16) | 0x40U : (u + 0x7fffU + ((u >> 16) & 1U)) >> 16;
}
__device__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// The r-th span starts its packets at floor(offset[r]/16)+r. Adjacent
// intervals cannot overlap; ceil(M/16)+R-1 packets per head cover all spans.
__device__ long long packet_index(Arguments a, int sequence, int chunk, int head) {
    return (static_cast<long long>(dtk_packet_base(a.sequence_offsets[sequence], sequence) + chunk) * a.value_heads) + head;
}

__global__ void prepare_vectors(Arguments a, DtkChunk* packets) {
    const int h = blockIdx.x, chunk = blockIdx.y, seq = blockIdx.z, tid = threadIdx.x;
    const int start = a.sequence_offsets[seq] + chunk * 16;
    const int end = a.sequence_offsets[seq + 1];
    if (start >= end) return;
    const int length = min(16, end - start), row = tid / 16, lane = tid % 16;
    const int qh = h / (a.value_heads / a.qk_heads);
    auto& p = packets[packet_index(a, seq, chunk, h)];
    __shared__ float prefix[16][128], qr[16][16], kr[16][16];
    if (tid < 128) {
        const float rate = a.negative_rate ? -a.a_log[h] : expf(a.a_log[h]);
        float sum = 0.f;
        for (int t = 0; t < 16; ++t) {
            if (t < length) {
                const auto index = (static_cast<long long>(start + t) * a.value_heads + h) * 128 + tid;
                sum += a.lower_bound * sigmoid(rate * (bf_value(a.g, index) + a.dt_bias[h * 128 + tid]));
            }
            prefix[t][tid] = sum;
        }
        p.gamma[tid] = expf(sum);
    }
    if (tid < 16) p.beta[tid] = tid < length
        ? sigmoid(bf_value(a.beta, static_cast<long long>(start + tid) * a.value_heads + h)) : 0.f;
    float q[8]{}, k[8]{}, qs = 0.f, ks = 0.f;
    for (int j = 0; j < 8; ++j) {
        if (row < length) {
            const auto index = (static_cast<long long>(start + row) * a.qk_heads + qh) * 128 + lane * 8 + j;
            q[j] = bf_value(a.q, index); k[j] = bf_value(a.k, index);
        }
        qs += q[j] * q[j]; ks += k[j] * k[j];
    }
    qr[row][lane] = qs; kr[row][lane] = ks;
    __syncthreads();
    for (int distance = 8; distance; distance /= 2) {
        if (lane < distance) {
            qr[row][lane] += qr[row][lane + distance]; kr[row][lane] += kr[row][lane + distance];
        }
        __syncthreads();
    }
    const float iq = rsqrtf(qr[row][0] + kQkL2NormEps), ik = rsqrtf(kr[row][0] + kQkL2NormEps);
    for (int j = 0; j < 8; ++j) {
        const int c = lane * 8 + j;
        const float g = prefix[row][c], decay = expf(g), qv = q[j] * iq, kv = k[j] * ik;
        p.kd[row][c] = kv * decay; p.qd[row][c] = qv * decay;
        p.prefix[row][c] = g;
        p.kr[row][c] = kv * expf(prefix[length - 1][c] - g);
    }
}

__global__ void prepare_solve(Arguments a, DtkChunk* packets) {
    const int h = blockIdx.x, chunk = blockIdx.y, seq = blockIdx.z, tid = threadIdx.x;
    if (a.sequence_offsets[seq] + chunk * 16 >= a.sequence_offsets[seq + 1]) return;
    auto& p = packets[packet_index(a, seq, chunk, h)];
    const int start = a.sequence_offsets[seq] + chunk * 16;
    const int length = min(16, a.sequence_offsets[seq + 1] - start);
    const int time = tid / 16, lane = tid % 16;
    const int qh = h / (a.value_heads / a.qk_heads);
    __shared__ float q[16][128], k[16][128], qr[16][16], kr[16][16];
    float qv[8]{}, kv[8]{}, qs = 0.f, ks = 0.f;
    for (int j = 0; j < 8; ++j) {
        if (time < length) {
            const auto at = (static_cast<long long>(start + time) * a.qk_heads + qh) * 128 + lane * 8 + j;
            qv[j] = bf_value(a.q, at); kv[j] = bf_value(a.k, at);
        }
        qs += qv[j] * qv[j]; ks += kv[j] * kv[j];
    }
    qr[time][lane] = qs; kr[time][lane] = ks;
    __syncthreads();
    for (int distance = 8; distance; distance /= 2) {
        if (lane < distance) { qr[time][lane] += qr[time][lane + distance]; kr[time][lane] += kr[time][lane + distance]; }
        __syncthreads();
    }
    const float iq = rsqrtf(qr[time][0] + kQkL2NormEps), ik = rsqrtf(kr[time][0] + kQkL2NormEps);
    for (int j = 0; j < 8; ++j) {
        q[time][lane * 8 + j] = qv[j] * iq; k[time][lane * 8 + j] = kv[j] * ik;
    }
    __syncthreads();
    const int row = tid / 16, col = tid % 16;
    __shared__ float lower[16][16], inverse[16][16];
    float kk = 0.f, qk = 0.f;
    if (row >= col) for (int c = 0; c < 128; ++c) {
        // Evaluate the causal diagonal ratio directly. Reciprocal cumulative
        // decays would amplify private underflow even for the unit diagonal.
        const float decay = expf(p.prefix[row][c] - p.prefix[col][c]);
        kk += (k[row][c] * k[col][c]) * decay;
        qk += (q[row][c] * k[col][c]) * decay;
    }
    lower[row][col] = row > col ? p.beta[row] * kk : 0.f;
    p.mqk[row][col] = row >= col ? qk : 0.f;
    __syncthreads();
    for (int r = 0; r < 16; ++r) {
        if (tid < 16) {
            float value = r == tid ? 1.f : 0.f;
            for (int j = 0; j < r; ++j) value -= lower[r][j] * inverse[j][tid];
            inverse[r][tid] = value;
        }
        __syncthreads();
    }
    p.solve[row][col] = inverse[row][col] * p.beta[col];
}

__global__ void consume_chunks(Arguments a, const DtkChunk* packets, const std::int32_t* slots) {
    const int h = blockIdx.x, value0 = blockIdx.y * 8, seq = blockIdx.z, tid = threadIdx.x;
    const int v = tid / 32, c = tid % 32, begin = a.sequence_offsets[seq], end = a.sequence_offsets[seq + 1];
    const auto state0 = ((static_cast<long long>(slots[seq]) * a.value_heads + h) * 128 + value0 + v) * 128;
    float state[4];
    for (int j = 0; j < 4; ++j) state[j] = a.state_in[state0 + c + j * 32];
    __shared__ DtkChunk packet;
    __shared__ float reduce_k[256], reduce_q[256], residual[16][8], initial[16][8], delta[16][8];
    for (int start = begin, chunk = 0; start < end; start += 16, ++chunk) {
        const int length = min(16, end - start);
        const auto* source = reinterpret_cast<const uint4*>(packets + packet_index(a, seq, chunk, h));
        auto* target = reinterpret_cast<uint4*>(&packet);
        for (int i = tid; i < kDtkChunkDataBytes / sizeof(uint4); i += 256) target[i] = source[i];
        __syncthreads();
        for (int t = 0; t < length; ++t) {
            float pk = 0.f, pq = 0.f;
            for (int j = 0; j < 4; ++j) {
                pk += state[j] * packet.kd[t][c + j * 32]; pq += state[j] * packet.qd[t][c + j * 32];
            }
            reduce_k[tid] = pk; reduce_q[tid] = pq;
            __syncthreads();
            for (int distance = 16; distance; distance /= 2) {
                if (c < distance) { reduce_k[tid] += reduce_k[tid + distance]; reduce_q[tid] += reduce_q[tid + distance]; }
                __syncthreads();
            }
            if (!c) {
                const auto at = (static_cast<long long>(start + t) * a.value_heads + h) * 128 + value0 + v;
                residual[t][v] = bf_value(a.v, at) - reduce_k[tid]; initial[t][v] = reduce_q[tid];
            }
        }
        __syncthreads();
        const int time = tid / 8, value = tid % 8;
        if (time < length) {
            float sum = 0.f;
            for (int j = 0; j <= time; ++j) sum += packet.solve[time][j] * residual[j][value];
            delta[time][value] = sum;
        }
        __syncthreads();
        if (time < length) {
            float sum = initial[time][value];
            for (int j = 0; j <= time; ++j) sum += packet.mqk[time][j] * delta[j][value];
            const auto at = (static_cast<long long>(start + time) * a.value_heads + h) * 128 + value0 + value;
            store_bf(a.out, at, sum * a.scale);
        }
        for (int j = 0; j < 4; ++j) {
            float sum = state[j] * packet.gamma[c + j * 32];
            for (int t = 0; t < length; ++t) sum += delta[t][v] * packet.kr[t][c + j * 32];
            state[j] = sum;
        }
        __syncthreads();
    }
    for (int j = 0; j < 4; ++j) a.state_out[state0 + c + j * 32] = state[j];
}
} // namespace

void launch_packed_chunks(const Arguments& args, const std::int32_t* slots, int sequences,
                           void* storage, cudaStream_t stream) {
    auto* packets = static_cast<DtkChunk*>(storage);
    const dim3 prepare(args.value_heads, (args.tokens + 15) / 16, sequences);
    prepare_vectors<<<prepare, 256, 0, stream>>>(args, packets);
    prepare_solve<<<prepare, 256, 0, stream>>>(args, packets);
    consume_chunks<<<dim3(args.value_heads, 16, sequences), 256, 0, stream>>>(args, packets, slots);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::kimi_delta_attention
