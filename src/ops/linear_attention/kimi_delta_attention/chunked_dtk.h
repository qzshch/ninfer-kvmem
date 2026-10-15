#pragma once
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail::kimi_delta_attention {
inline constexpr int kDtkTimeTile = 16;
struct alignas(16) DtkPrepared {
    float q[128], k[128], alpha[128], beta;
};
struct alignas(256) DtkChunk {
    float kd[16][128], qd[16][128], kr[16][128];
    float gamma[128], solve[16][16], mqk[16][16];
};
inline constexpr std::size_t kDtkChunkDataBytes = offsetof(DtkChunk, mqk) + sizeof(DtkChunk::mqk);
static_assert(kDtkChunkDataBytes % 16 == 0);
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
constexpr int dtk_packet_base(int begin, int sequence) { return begin / 16 + sequence; }
inline constexpr bool dtk_use_chunks(int rows, int sequences) { return rows >= kDtkTimeTile * sequences; }
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline int dtk_packet_sequence(const int* offsets, int sequences, int packet) {
    for (int r = 0; r < sequences; ++r) {
        const int base = dtk_packet_base(offsets[r], r);
        const int chunks = (offsets[r + 1] - offsets[r] + 15) / 16;
        if (packet >= base && packet - base < chunks) return r;
    }
    return -1;
}
inline std::size_t dtk_packed_workspace_bytes(int heads, int tokens, int sequences) {
    if (heads < 1 || tokens < 1 || tokens > 32768 || sequences < 1 || sequences > 8 || sequences > tokens)
        throw std::invalid_argument("DTK KDA packed workspace geometry");
    if (!dtk_use_chunks(tokens, sequences)) return static_cast<std::size_t>(tokens) * heads * sizeof(DtkPrepared);
    return static_cast<std::size_t>((tokens + 15) / 16 + sequences - 1) * heads * sizeof(DtkChunk);
}
} // namespace ninfer::ops::detail::kimi_delta_attention
