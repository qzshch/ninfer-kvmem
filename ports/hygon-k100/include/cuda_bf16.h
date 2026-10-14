// Storage-only bridge for native wrappers. No NVIDIA arithmetic API is emulated.
#pragma once
#include "dtk_runtime.h"
#include <cstdint>
using __nv_bfloat16 = std::uint16_t;
static_assert(sizeof(__nv_bfloat16) == 2);
