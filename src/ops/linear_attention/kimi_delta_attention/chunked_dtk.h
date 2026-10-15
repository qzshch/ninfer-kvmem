#pragma once
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail::kimi_delta_attention {
inline constexpr int kDtkTimeTile = 16;
struct alignas(256) DtkChunk {
    float kd[16][128], qd[16][128], prefix[16][128], kr[16][128];
    float gamma[128], beta[16], solve[16][16], mqk[16][16];
};
inline constexpr std::size_t kDtkChunkDataBytes = offsetof(DtkChunk, mqk) + sizeof(DtkChunk::mqk);
static_assert(kDtkChunkDataBytes % 16 == 0);
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
constexpr int dtk_packet_base(int begin, int sequence) { return begin / 16 + sequence; }
inline std::size_t dtk_packed_workspace_bytes(int heads, int tokens, int sequences) {
    if (heads < 1 || tokens < 1 || tokens > 32768 || sequences < 1 || sequences > 8 || sequences > tokens)
        throw std::invalid_argument("DTK KDA packed workspace geometry");
    if (tokens < kDtkTimeTile) return 0;
    return static_cast<std::size_t>((tokens + 15) / 16 + sequences - 1) * heads * sizeof(DtkChunk);
}
} // namespace ninfer::ops::detail::kimi_delta_attention
