#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath "$HERE/../..")
DTK_ROOT=${DTK_ROOT:-/opt/dtk}
source "$DTK_ROOT/env.sh"
set -u
export PATH="/opt/hyhal/bin:$DTK_ROOT/bin:$DTK_ROOT/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/hyhal/lib:/opt/hyhal/lib64:$DTK_ROOT/lib:$DTK_ROOT/lib64:$DTK_ROOT/hip/lib:$DTK_ROOT/llvm/lib:${LD_LIBRARY_PATH:-}"
# Per-card tests use HIP_VISIBLE_DEVICES; peer qualification needs all three.
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
BUILD=${K100_BUILD_DIR:-"$ROOT/build-k100-core"}
cmake -S "$HERE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_HIP_COMPILER="$DTK_ROOT/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$DTK_ROOT" -DCMAKE_HIP_ARCHITECTURES=gfx928 \
  -DNINFER_K100_REAL_WEIGHT_SAMPLE="${NINFER_K100_REAL_WEIGHT_SAMPLE:-}" \
  -DBUILD_TESTING=ON
cmake --build "$BUILD" -j "${BUILD_JOBS:-24}"
ctest --test-dir "$BUILD" --output-on-failure --no-tests=error \
  --output-junit "$BUILD/core-results.xml"
