#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath "$HERE/../..")
DTK_ROOT=${DTK_ROOT:-/opt/dtk}
source "$DTK_ROOT/env.sh"
set -u
export PATH="/opt/hyhal/bin:$DTK_ROOT/bin:$DTK_ROOT/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/hyhal/lib:/opt/hyhal/lib64:$DTK_ROOT/lib:$DTK_ROOT/lib64:$DTK_ROOT/hip/lib:$DTK_ROOT/llvm/lib:${LD_LIBRARY_PATH:-}"
export NINFER_WEIGHT_READ_THREADS=${NINFER_WEIGHT_READ_THREADS:-4}
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
BUILD=${K100_BUILD_DIR:-"$ROOT/build-k100-core"}
exec "$BUILD/ninfer_k100_generate" "$@"
