# Experimental Prefill + Decode projection sharing

This branch migrates the tested mixed changes onto official KVMem master
`df555e9e`, including merged #160, its preparation-boundary follow-up #164,
MTP Prefill accounting #165, and later upstream changes. This migration has
pinned-patch/CPU publication checks only; it has not been built or GPU-retested.
The historical measurements below do not qualify this migrated head.
The exact tested source is retained on `experiment/pd-mixed-tested-20261021`.
The ordinary execution path remains the default; no daily service is switched.
The historical branch preserves the engine source of the tested `2f2c00d2` candidate. Its
historical KVMem base is `4ca449ed96af1fcdae51159df51bdd8f22896a75`, the Host
preparation change submitted as [#160](https://github.com/kvmem/kvmem/pull/160).
The integration remains in `backends/patches/ninfer-kvmem.patch`; the NInfer
submodule is pinned to `08fd7019925f43b4b42539400afd461b70f7144a`.

The `main` branch is now the mixed-compute development baseline, promoted from
`experiment/pd-mixed`. The `upstream` branch preserves the unmodified official
baseline after #160 and its follow-up fixes. Future comparisons use that branch
as the control; synchronize official updates there before integrating them into
the mixed mainline. This branch promotion does not add GPU acceptance or enable
the experimental switches in a running process.
The historical branch is retained so the measurements below remain tied to
the source that produced them. It does not include later upstream fixes.

## Implementation

- Share Prefill and Decode linear work in MLP, FullAttention input/output
  projections, and GDN input/output projections. Keep attention and recurrent
  state updates separate. One executor and one shared workspace remain.
- Cache fixed-shape joint CUDA Graph segments, with optional Graph/native
  state comparisons for diagnostics.
- Separate the standalone Prefill chunk from the joint token budget. Reserve
  the actual ordinary MTP verification width, while retaining the wider ngram
  reservation, and scale Decode cadence with the configured chunk.
- Reuse a bounded checkpoint-buffer pool. Optionally retain one lazy Host
  reconstruction scratch buffer per Program; keep its lease alive through CUDA
  cleanup fences, and retire reuse if draining fails.
- Add optional phase/allocation tracing and Graph/state verification. Expensive
  Graph/state checks are disabled in the performance configuration; lightweight
  Host phase tracing is retained.

Mixed execution is opt-in and currently restricted to the experimental MTP
path. It is not a DFlash2 or Vision acceptance result. Shared mode 0 changes
Decode activation quantization to the shared A8 route; Graph/native agreement
does not prove equality to the original independent Decode path.

## Experimental configuration

Use the regular KVMem build and preparation commands. The adjacent JSON file
records the relevant CLI arguments and environment settings. Supply an explicit
model path; no model or request data is included. Environment switches must be
set before starting the process. Unset them to use the ordinary execution path.
Do not treat this recipe as a production default or a globally optimal chunk.

## Completed evidence on the historical source

Hardware: RTX 5090 D 32 GB under WSL2. Model: 27B Huihui NVFP4; INT8 KV;
three lanes, 262144 logical tokens per lane; B36864/R16384; adaptive MTP4 and
ngram15; CPU Vision configured, with Vision acceptance deferred.

One unchanged real request pair combined a hot 44K Decode with a cold 103K
Prefill. Ordinary A/B/B/A comparison used physical chunk 256 and joint budget
256, identical request bytes/sampling settings and cache preparation. In the
predeclared early concurrent phase:

| Metric | Official + Host | Mixed candidate | Change |
|---|---:|---:|---:|
| Prefill tokens/s | 1825.451 | 1910.042 | +4.634% |
| Decode tokens/s | 71.943 | 80.071 | +11.298% |
| Decode rounds/s | 28.448 | 30.414 | +6.911% |
| Nonempty SSE event gap P99, ms | 155.316 | 125.187 | -19.398% |

P99 is an event-gap metric, not per-token ITL. These results cover one request
pair and the measured concurrent interval. They are not a universal end-to-end
speedup or a comparison against the best official chunk. Profiled measurements
are kept separate from ordinary wall-clock results. Paired Nsight analysis
found less linear-kernel time, partly offset by attention/GDN and non-compute
intervals; the whole difference cannot be uniquely attributed to one feature.

At reconstruction boundaries, later pinned-buffer allocations fell from
172--205 ms to 3--4 us with retained scratch. The first allocation remains and
the retained Host cost is about 146.8 MiB for this model. Four independent GPU
state fold/restore byte comparisons passed. These allocation events were outside
the early throughput window, so they do not explain its complete speedup.

Quality checks on the original GDN route: both arms passed 22/22 retrieval
semantic cases. The original strict result remains 14/22 because eight old
zero-cache expectations conflict with legal common-prefix reuse. The original
verdicts were preserved. Six fact/tool cases passed in the native-entry protocol,
but only five target Decode cases were proven to participate in actual mixing.
The complementary target-Prefill quality run was interrupted at user request;
its partial result is not acceptance evidence. Full model quality, broader real
request diversity, and Vision remain unaccepted.

Other measured candidates were not adopted:

- C512/J512: one ordinary pair gave Prefill +0.42%, Decode rounds +3.43%,
  event-gap P99 +2.99%; no clear overall improvement.
- C256/J272 padding: GDN cost decreased, but larger projections/layout work
  offset it; P99 increased about 21%.
- Global GDN two-stage: independent FP64 checks passed and GDN was faster,
  but a fixed-seed greedy output-selection case regressed on both arms.
- Global eager CUDA loading increased resources substantially. Standalone
  attention preloading remains diagnostic and is not part of this patch.

C768 alternatives, quota tuning, best-official versus best-mixed selection,
and full performance/quality acceptance are incomplete. MTP joint-cost filtering
is an optional diagnostic experiment in the source and remains disabled here.

## Publication checks

The snapshot was reconstructed from the official pinned integration and
hash-verified against the tested source. Private request/response bodies, profiler
reports, credentials, model weights and local binaries are excluded.

CPU checks for this publication:

```sh
python scripts/prepare-backends.py --backend ninfer --check
python scripts/public_evidence.py --check
python tests/test_public_evidence.py
python tests/test_validation_portability.py
python tests/test_ci_changes.py
git diff --check
```

These publication checks do not replace the uncompleted GPU/model acceptance.
