#!/usr/bin/env bash
# Compile/disassemble only. This script never starts a GPU kernel or a server.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
compiler=${HIPCXX:-${DTK_ROOT:-/opt/dtk}/bin/hipcc}
if [[ ! -x "$compiler" ]]; then
    echo "DTK compiler unavailable: set HIPCXX or DTK_ROOT" >&2
    exit 2
fi
output=$(mktemp -d)
trap 'rm -f -- "$output/probe.s"; rmdir -- "$output"' EXIT
"$compiler" --offload-arch=gfx928 --cuda-device-only -std=c++20 -O2 -S \
    "$here/tests/mmac_compile.hip" -o "$output/probe.s"
if ! grep -Eq 'v_mmac(_f32)?_16x16x(4|8)(_f32|f32)' "$output/probe.s"; then
    echo "No Hygon FP32 MMAC instruction in compiler output" >&2
    exit 1
fi
echo "PASS: vendor gfx928 compiler emitted FP32 MMAC; GPU correctness remains a separate check"
