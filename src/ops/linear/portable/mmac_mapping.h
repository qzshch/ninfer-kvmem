#pragma once
#if defined(__CUDACC__) || defined(__HIPCC__)
#define NINFER_MMAC_HD __host__ __device__
#else
#define NINFER_MMAC_HD
#endif
namespace ninfer::ops::detail {
// Hygon FP32 MMAC default C layout. Four lane groups reduce K and
// distribute C columns; this differs from the FP16 MMAC group size.
struct MmacF32Mapping {
    NINFER_MMAC_HD static constexpr int row(int lane) { return lane % 16; }
    NINFER_MMAC_HD static constexpr int contraction(int lane, int width, int element = 0) {
        return (lane / 16) * width + element;
    }
    NINFER_MMAC_HD static constexpr int column(int lane, int reg) { return reg * 4 + lane / 16; }
};
} // namespace ninfer::ops::detail
#undef NINFER_MMAC_HD
