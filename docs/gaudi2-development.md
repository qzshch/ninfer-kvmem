# Gaudi2 development

Gaudi2 work lives on `feat/gaudi2` in
[qzshch/ninfer-kvmem](https://github.com/qzshch/ninfer-kvmem). The first executable
milestone is bounded C++ preparation of official Qwen block-FP8 weight tiles,
with host numerical and checkpoint-offset tests. HPU model inference is not yet
available. Current CUDA binaries do not acquire Gaudi2 support from this change.

```sh
python scripts/prepare-backends.py --backend ninfer
python scripts/prepare-backends.py --backend ninfer --check
cmake -S backends/ninfer -B build-gaudi2-host -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_GAUDI2_PREPARATION_ONLY=ON
cmake --build build-gaudi2-host --config Release -j
ctest --test-dir build-gaudi2-host -C Release --output-on-failure
```

This target requires CMake 3.28+, a C++20 compiler and Python 3.11+, with no CUDA
or SynapseAI dependency. The prepared backend's `docs/gaudi2-preparation.md`
defines the CLI, numeric profile, bounded allocations, tests and remaining work.
Backend changes are stored in the wrapper's
[versioned integration patch](../backends/patches/ninfer-kvmem.patch), not a
different upstream submodule commit; run preparation before opening those files.

The source `F8_E4M3` bytes retain E4M3FN's finite range up to 448 and explicit
FP32 multipliers per 128x128 block. They are decoded tile by tile to BF16 using
round-to-nearest-even. This is preparation for W8A16 dequantization/BF16 MME,
with no native Gaudi FP8/W8A8 computation or performance claim. A 128x128 aligned
tile uses 49,156 bytes of payload buffers; process/allocator/file-cache overhead
is separate. The default 64 MiB tile payload ceiling is enforced before reading.

The read-only [checkpoint inventory](../scripts/gaudi2/README.md) validates local
safetensors metadata and emits native tile commands with explicit absolute
plane offsets. It does not download weights or create a full BF16 model copy.

Windows MSVC host checks cover exact FP8 decoding, BF16 cast boundaries,
nonaligned tile scales, bounded allocation and malformed inputs. CI is configured
to build and run the same tests on Linux and Windows. Real-checkpoint accuracy, SynapseAI runtime/TPC/MME,
GDN, full-KV ordinary AR and later KVMem device integration still require
implementation and validation on Gaudi2. No throughput result is available yet.
