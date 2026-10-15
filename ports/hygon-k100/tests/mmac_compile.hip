#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx928__)
#error "This compiler probe requires gfx928"
#endif
using F4 = float __attribute__((ext_vector_type(4)));
using F2 = float __attribute__((ext_vector_type(2)));
extern "C" __global__ void ninfer_k100_fp32_mmac_probe(const float* a, const float* b, float* out) {
    const int lane = threadIdx.x;
    F4 c{0, 0, 0, 0};
#if __has_builtin(__builtin_hcu_mmac_16x16x8_f32)
    F2 x{a[2 * lane], a[2 * lane + 1]}, y{b[2 * lane], b[2 * lane + 1]};
    c = __builtin_hcu_mmac_16x16x8_f32(x, y, c);
#elif __has_builtin(__builtin_hcu_mmac_16x16x4_f32)
    c = __builtin_hcu_mmac_16x16x4_f32(a[lane], b[lane], c);
#else
#error "Vendor compiler has no supported Hygon FP32 MMAC builtin"
#endif
    for (int j = 0; j < 4; ++j) out[lane * 4 + j] = c[j];
}
#endif
