#include "artifact/formats.h"

#include "artifact/schema.h"

#include <array>

namespace ninfer::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{QType::BF16, std::string_view{"bf16"}},
    std::pair{QType::FP32, std::string_view{"fp32"}},
    std::pair{QType::INT32, std::string_view{"int32"}},
    std::pair{QType::Q4_G64_FP16, std::string_view{"q4_g64_fp16"}},
    std::pair{QType::Q5_G64_FP16, std::string_view{"q5_g64_fp16"}},
    std::pair{QType::Q6_G64_FP16, std::string_view{"q6_g64_fp16"}},
    std::pair{QType::Q8_G32_FP16, std::string_view{"q8_g32_fp16"}},
    std::pair{QType::NVFP4, std::string_view{"nvfp4"}},
    std::pair{QType::FP8_E4M3FN_ROW_BF16, std::string_view{"fp8_e4m3fn_row_bf16"}},
    std::pair{QType::FP16, std::string_view{"fp16"}},
    std::pair{QType::MXFP4_G32_E8M0, std::string_view{"mxfp4_g32_e8m0"}},
    std::pair{QType::Q8_G32_FP16_BLOCK, std::string_view{"q8_g32_fp16_block"}},
    std::pair{QType::IQ3_S_G256_BLOCK, std::string_view{"iq3_s_g256_block"}},
    std::pair{QType::IQ4_XS_G256_BLOCK, std::string_view{"iq4_xs_g256_block"}},
    std::pair{QType::Q3_K_G256_BLOCK, std::string_view{"q3_k_g256_block"}},
    std::pair{QType::Q4_K_G256_BLOCK, std::string_view{"q4_k_g256_block"}},
    std::pair{QType::Q6_K_G256_BLOCK, std::string_view{"q6_k_g256_block"}},
};
constexpr std::array kLayouts = {
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous_le_v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row_split_k128_v1"}},
    std::pair{QuantLayout::RowScale, std::string_view{"row_scale_v1"}},
    std::pair{QuantLayout::BlockInterleaved, std::string_view{"block_interleaved_v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4, std::string_view{"block_scale_k16_m128x4_v1"}},
};

} // namespace

QType parse_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept {
    for (const auto& [value, spelling] : kFormats) {
        if (value == format) { return spelling; }
    }
    return {};
}

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninfer::artifact
