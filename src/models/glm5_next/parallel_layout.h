#pragma once
#include <array>
#include <stdexcept>

namespace ninfer::models::glm5_next {
struct RowPartition { int begin, count; };
inline constexpr std::array<RowPartition, 3> head_rows{{{0,24},{24,20},{44,20}}};
inline constexpr std::array<RowPartition, 3> vocabulary_rows{{{0,51712},{51712,51712},{103424,51456}}};
inline constexpr std::array<RowPartition, 3> shared_rows{{{0,768},{768,768},{1536,512}}};
inline RowPartition head_partition(int rank) { return head_rows.at(rank); }
inline RowPartition vocabulary_partition(int rank) { return vocabulary_rows.at(rank); }
inline RowPartition mlp_partition(int intermediate, int rank) {
    if (rank < 0 || rank >= 3) throw std::out_of_range("GLM rank");
    if (intermediate == 12288) return {rank * 4096, 4096};
    if (intermediate == 2048) return shared_rows[rank];
    throw std::invalid_argument("GLM MLP extent");
}
} // namespace ninfer::models::glm5_next
