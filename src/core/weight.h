#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    FP16                = 9,
    MXFP4_G32_E8M0     = 10,
    Q8_G32_FP16_BLOCK  = 11,
    IQ3_S_G256_BLOCK   = 12,
    IQ4_XS_G256_BLOCK  = 13,
    Q3_K_G256_BLOCK    = 14,
    Q4_K_G256_BLOCK    = 15,
    Q6_K_G256_BLOCK    = 16,
};

struct InterleavedBlockGeometry {
    std::uint32_t values=0, bytes=0, base_scale_offset=0;
    DType base_scale_dtype=DType::FP16;
};

inline constexpr InterleavedBlockGeometry interleaved_block_geometry(QType type) {
    switch(type) {
    case QType::MXFP4_G32_E8M0:return {32,17,0,DType::U8};
    case QType::Q8_G32_FP16_BLOCK:return {32,34,0,DType::FP16};
    case QType::IQ3_S_G256_BLOCK:return {256,110,0,DType::FP16};
    case QType::IQ4_XS_G256_BLOCK:return {256,136,0,DType::FP16};
    case QType::Q3_K_G256_BLOCK:return {256,110,108,DType::FP16};
    case QType::Q4_K_G256_BLOCK:return {256,144,0,DType::FP16};
    case QType::Q6_K_G256_BLOCK:return {256,210,208,DType::FP16};
    default:return {};
    }
}

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    BlockInterleaved    = 4,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

} // namespace ninfer
