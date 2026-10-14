#!/usr/bin/env bash
set -eo pipefail
SOURCE=$(realpath "${1:?provide the pinned mx-llama.cpp checkout}")
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DTK_ROOT=${DTK_ROOT:-/opt/dtk}
source "$DTK_ROOT/env.sh"
set -u
export PATH="/opt/hyhal/bin:$DTK_ROOT/bin:$DTK_ROOT/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/hyhal/lib:/opt/hyhal/lib64:$DTK_ROOT/lib:$DTK_ROOT/lib64:$DTK_ROOT/hip/lib:$DTK_ROOT/llvm/lib:${LD_LIBRARY_PATH:-}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 71a0624fb36aaaa2df24ccf428236493c813a441
if git -C "$SOURCE" apply --reverse --check "$HERE/dtk-gfx928.patch" 2>/dev/null; then
    echo 'Diagnostic patch already applied'
else
    git -C "$SOURCE" apply --check "$HERE/dtk-gfx928.patch"
    git -C "$SOURCE" apply "$HERE/dtk-gfx928.patch"
fi
BUILD="$SOURCE/build-k100-diagnostic"
cmake -S "$SOURCE" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_HIP_COMPILER="$DTK_ROOT/llvm/bin/clang++" \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx928 -DCMAKE_HIP_ARCHITECTURES=gfx928 \
    -DCMAKE_PREFIX_PATH="$DTK_ROOT" -DCMAKE_HIP_FLAGS=-Wno-return-type \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF \
    -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=OFF -DGGML_HIP_MMQ_MFMA=OFF \
    -DGGML_HIP_RCCL=OFF -DGGML_CUDA_FA=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$BUILD" -j "${BUILD_JOBS:-24}" --target llama-server llama-cli llama-bench test-backend-ops
for backend in ROCm0 ROCm1 ROCm2; do
    "$BUILD/bin/test-backend-ops" test -b "$backend" -o MUL_MAT,MUL_MAT_ID \
        -p 'type_a=mxfp4,type_b=f32,.*m=(16|32|64|128|256|2880),n=(1|2|8|16|32),k=(32|64|128|256|512|1024|2880),' \
        > "$BUILD/$backend.txt" 2>&1
    grep -F '40/40 tests passed' "$BUILD/$backend.txt"
done
