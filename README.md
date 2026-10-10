# ninfer-kvmem

English | [简体中文](README.zh-CN.md)

**A NInfer + KVMem integration for single-GPU long-context inference, concurrent requests and agent workflows, maintained by [qzshch](https://github.com/qzshch).**

This project builds on [NInfer](https://github.com/Neroued/ninfer)'s C++/CUDA engine and integrates [KVMem](https://github.com/kvmem/kvmem-llama.cpp)'s bounded GPU working set, Host history and query-based retrieval with native execution and prefix reuse. The focus is large logical contexts within limited VRAM, multiple lanes, DFlash2 and vision running together, and regression checks based on real agent requests.

`main` tracks the current daily build. Measurements below were recorded on **2026-10-10**, using qualified code `7e9f4e13`. Later documentation and attribution changes do not represent additional performance improvements.

## Core changes

| Change | Purpose | Current status |
|---|---|---|
| Multi-lane KVMem | Give each execution lane an independent sparse GPU window while retaining logical history in quota-managed shared Host storage | Daily profile: 2 lanes, each with a 36K window and a 256K logical limit |
| Native prefix reuse | Reuse KV and complete continuation state to reduce repeated prefill; share native prefix leases instead of maintaining another cache ledger | Cold/warm, append, fork, cancellation, recovery and resource retirement checked |
| KVMem + DFlash2 + vision | Integrate retrieval, speculative verification and image/video state through the same native execution path | DFlash2 K7 and image/video inference checked; MTP3 also has cross-window regression coverage |
| Long-history instructions and tools | Retain System/Developer instructions, tool definitions, required media and tail pages; reject an undersized window explicitly | Original DSH failure requests replayed in bounded smoke tests; premature migration of uncommitted MTP lookahead pages fixed |
| Observable performance and capacity | Emit per-lane prefill/decode, prefix reuse, selection, transfers and replay metrics to JSONL | Daily dashboard receives both lanes; logical occupancy and fixed Host backing remain separate |
| Modular KVMem core | Move retrieval, window and capacity policy into an independently buildable `kvmem_core`, connected through a NInfer adapter | Reduces coupling to CUDA, models and frontends; a source-level interface, not a dynamic plugin ABI |
| Selected upstream integration | Adopt NInfer structured output and strict tool constraints; adapt video sampling validation and Responses lifecycle handling | JSON/supported schemas, GBNF, choice/regex supported; unsupported schemas fail explicitly |

NInfer supplies model mathematics, CUDA kernels, graphs, scheduling and protocol foundations. KVMem supplies the working-set and retrieval design. This project maintains their integration and fixes for the combinations above. Upstream capabilities retain their original attribution; see the [source version manifest](backends/versions.json) and [backend integration contract](docs/maintainer/backend-integration.md).

## Current daily profile

| Item | Configuration |
|---|---|
| Qualified platform | RTX 5090 32 GiB, Linux/WSL2, CUDA 13.2.86, GCC 13.3 |
| Measured model | Huihui Qwen3.8-27B NVFP4 `.ninfer` with DFlash2 companion weights |
| Concurrency / logical limit | 2 lanes; 262,144 tokens per lane |
| GPU working window | 36,864 tokens per lane: 576 pages × 64 tokens |
| Shared Device KV pool | 77,824 tokens, including execution growth slack; distinct from a lane's window |
| KV / speculation / media | FP8 / DFlash2, 7 draft tokens / vision enabled |
| Host / prefill | Shared 18 GiB pageable Host quota / 1,024-token chunks |
| Prefix reuse | NInfer Native cache; HiCache disabled |

A 36K resident window is about **14.1%** of the 256K logical limit. VRAM also holds weights, state, graphs and workspace. This is a capacity design, not evidence of an 85.9% VRAM reduction or lossless quality. The Host quota is shared; configuring two 256K limits does not qualify simultaneous full-history peaks. Resource pressure can trigger eviction, pause or replay.

## Measurements and limits

The latest comparison used **A → B → B → A**: A is the frozen previous daily build (`ae5134c7` + r2 telemetry), B is the qualified current build. Weights, configuration, public prompts and seed were held fixed. Each request published 256 tokens, with two repetitions per build. Both services were configured for 2 lanes; **single-request rows have one active request, not a service configured for one lane**.

The metric is **actual output tokens for the batch ÷ batch wall time, in tok/s**, including prefill, queueing, cache transfers and decode. Long prompts contain approximately 54.6K tokens, exceeding the 36K GPU window. Ranges cover both repetitions:

| Workload | Baseline A, cold | Current B, cold | Baseline A, warm | Current B, warm |
|---|---:|---:|---:|---:|
| Short, one request | 159.1–167.2 | 160.6–167.2 | 72.5–133.2 | 133.2–141.3 |
| Short, two requests | 201.0–214.1 | 203.5–212.8 | 138.3–268.6 | 260.0–275.3 |
| Long, one request | 23.4–24.1 | 24.0–24.7 | 48.0–72.8 | 66.1–70.6 |
| Long, two requests | 17.2–19.9 | 24.0–24.4 | 83.4–103.1 | 102.4–105.0 |

- **Concurrency helps, but does not guarantee twice the throughput.** Current cold short requests reached 203.5–212.8 with two active requests and 160.6–167.2 with one. Workload, verification batches and prefill affect the result.
- **Prefix hits reduce repeated prefill.** Warm long requests reused about 54,624 tokens. Current long two-request throughput was 24.0–24.4 cold and 102.4–105.0 warm. This is a batch-level difference, including less input computation and restoration costs, not a pure decode speedup ratio.
- **Modularization has not established a general speedup.** Cold short and cold long single-request results overlap substantially. Long two-request measurements improved, but there are only two repetitions, warm results vary and no component ablation establishes causality. Cold short pairs also shared 29 cached tokens, so they were not strictly zero-hit runs.
- **Quality qualification remains bounded.** Token IDs matched across builds and repetitions in all four single-request cases. Parallel baseline repetitions themselves differed. Dynamic-concurrency token determinism, formal multi-round semantic quality equivalence and full simultaneous two-256K peaks remain unqualified. Sparse retrieval is not full-history dense attention.

See the [2026-10-10 qualification report](docs/reports/2026-10-10-modular-kvmem-constraints.md) (Chinese) for methods, fixes, adverse findings and limits. Upstream model evaluation scores are not this build's scores; kernel microbenchmarks are not end-to-end gains.

## Stability checks

Completed checks: independent CPU-only core 1/1; CPU/protocol and independent oracles 18/18; GPU mathematics/graph oracles 4/4; real-model configurations 15/15; and 25 generated tool-argument sets validated independently with Draft 2020-12. Real-model coverage includes none/MTP3/DFlash2 K7 cross-window cold/warm constraints, tools, snapshot/replay/cancel, vision and Native transactions.

The full daily profile also passed JSON Schema, strict tools, four-frame video and Responses continuation after parent deletion. Two original DSH failure requests were replayed concurrently with their original output caps in bounded streaming smoke tests, then cancelled. These tests did not execute external tools or generate complete long answers. Private prompts and SSE payloads are not published.

## Build and run

This build targets 64-bit Linux/WSL2 and `sm_120a`; the qualified GPU is the RTX 5090. Requirements: CUDA, CMake ≥ 3.28, a C++20 compiler, Ninja, pkg-config, FFmpeg development libraries and libcurl ≥ 7.85. This qualification used CUDA 13.2.86; other GPUs and operating systems were not qualified by these tests.

```bash
git clone https://github.com/qzshch/ninfer-kvmem.git
cd ninfer-kvmem
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Prepare a v3 `.ninfer` model separately; see [weight conversion](docs/weight-conversion.md). DFlash2 requires compatible companion weights in the artifact. A text-only or MTP-only artifact cannot enable DFlash2 directly.

The following **direct HTTP server example** uses the daily capacity settings and the public alias `ninfer-kvmem`. Substitute your actual model path. The 18 GiB pageable Host setting was tested on this machine; choose memory type and quota for your hardware.

```bash
mkdir -p logs
NINFER_WEIGHT_READ_THREADS=2 ./build/apps/ninfer-serve \
  models/qwen3_8_27b_huihui_abliterated_nvfp4_dflash2.ninfer \
  --host 0.0.0.0 --port 8080 --model-id ninfer-kvmem \
  --max-context 262144 --kv-dtype fp8 --kv-capacity 77824 \
  --kvmem-window-pages 576 --max-concurrency 2 \
  --host-context-mib 18432 --host-context-memory pageable \
  --prefill-chunk 1024 --device-state-slots 0 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --vision \
  --log-stats-interval-ms 1000 --pending-timeout-ms 1800000 \
  --request-log-jsonl logs/ninfer.jsonl
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ninfer-kvmem","messages":[{"role":"user","content":"Explain prefix caching in one sentence."}],"max_tokens":64}'
```

See [serving](docs/serving.md) for APIs, structured outputs and tools. Executable `--help` is authoritative for option spelling and defaults.

## Branches and experiments

| Branch | Purpose |
|---|---|
| [`main`](https://github.com/qzshch/ninfer-kvmem/tree/main) | Current daily build and ongoing maintenance |
| [`HiCache`](https://github.com/qzshch/ninfer-kvmem/tree/HiCache) | Optional RAM/disk cache experiment; separate from the daily build |
| [`Native-KVMem-Baseline`](https://github.com/qzshch/ninfer-kvmem/tree/Native-KVMem-Baseline) | Historical 2026-10-08 Native + KVMem comparison baseline |
| [`Modular-Integration`](https://github.com/qzshch/ninfer-kvmem/tree/Modular-Integration) | Historical 2026-10-10 modular/constraint integration snapshot, already included in `main` |

HiCache is not part of the current daily build. Non-CP GDN recurrence fusion, cross-lane projection packing and NVFP4 rounding experiments are also not enabled as default optimizations. Output RMSNorm + SiLU gating is already fused in the production GDN path; it is distinct from the recurrence experiment. Older `master`, `feature/kvmem` and PR branches retain their historical or review roles.

Documentation:

- [Guide index](docs/README.md): CLI, serving, conversion, tests and maintenance.
- [Current qualification report](docs/reports/2026-10-10-modular-kvmem-constraints.md) (Chinese).
- [KVMem / Native cache contract](docs/maintainer/sparse-native-context-cache.md): selection, transfers, instruction retention, Host quotas and recovery.
- [Backend integration and update method](docs/maintainer/backend-integration.md), [upstream source versions](backends/versions.json).

Inherited model cards and general guides describe upstream artifacts and interfaces. This README and project-specific reports describe this build's measurements and defaults.

## Attribution and licenses

NInfer-derived code and project-owned modifications use **[Apache License 2.0](LICENSE)**. Original copyright notices, third-party licenses and notices are retained. Changes and provenance are recorded in Git history, [NOTICE](NOTICE) and the [source and license notes](docs/upstream-and-licenses.md) (Chinese).

| Source | Attribution and terms |
|---|---|
| [NInfer](https://github.com/Neroued/ninfer) | Neroued and NInfer contributors; [Apache-2.0](https://github.com/Neroued/ninfer/blob/81c8ce093b2c1646a87566a8e59d807fcf0ec95c/LICENSE). Model execution, CUDA and protocol foundations come from upstream |
| [KVMem](https://github.com/kvmem/kvmem-llama.cpp) / [KVMem-qw3](https://github.com/kvmem/kvmem-qw3) | KVMem authors and contributors. The pinned port's [README license statement](https://github.com/kvmem/kvmem-llama.cpp/blob/d9ae944b39f55f77f5434edb96b9fb037217a0a4/README.md#license) declares Apache-2.0 treatment; that revision has no standalone root LICENSE. llama.cpp's MIT terms are not a blanket license for all KVMem code |
| [XGrammar](third_party/xgrammar/README.ninfer.md) | Apache-2.0; its [LICENSE](third_party/xgrammar/LICENSE) and [NOTICE](third_party/xgrammar/NOTICE) are retained |
| Other vendored dependencies and templates | Retain their own terms, including MIT for llama-jinja, cpp-httplib, spdlog and nlohmann JSON; see the [component list](docs/upstream-and-licenses.md#第三方组件) |
| Models and draft weights | Distributed separately; follow each original, abliterated, quantized and companion model's model card and license. The source license does not replace weight licenses |

For the KVMem research design, cite [KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852), Di Chai, Leye Wang, Zeshen Su, Zhiguo Xia and Zhihang Yu, 2026. This is an independently maintained integration, not an official upstream release or endorsement.
