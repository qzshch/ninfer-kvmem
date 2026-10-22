# Selective INT8 prefill attention

This ports the grouped INT8 attention and natural merge from NInfer commit
`64d10fbf9f8e6bfc79ab0570b94674c5b9997929` into an isolated namespace. It retains
the pinned backend and existing KV append, output gate and decode implementations.
The grouped kernel shares KV reads across query tiles and uses FP32 PV partials.

The route is restricted to compute capability 12.0 with 170 SMs, H24/KV4/D256,
INT8 Group64, batch one, query widths 224/240/246/251, and a declared visible-key
capacity of 12,288–53,248. Other shapes and devices, width 256, decode, and the
existing explicit fast-prompt path retain their previous routes. Set
`NINFER_SELECTIVE_INT8_PREFILL=0` before startup to disable it; the default is one.
The switch is immutable during the process.

The separate FP32 partials require up to 16,644,096 bytes (15.87 MiB) of transient
workspace in the measured matrix. The selected graph grows from two to three
nodes, including the unchanged append. Workspace planning reserves these bytes;
the execution scope releases them for subsequent operations.

## Validation scope

The same final operator binary was run with the switch off/on in ABBA order:
five warmups and 30 CUDA-event repetitions for each width at contexts 12,288,
36,864, 49,152 and 52,992. The conservative reduction compares the slowest
candidate repetition with the fastest baseline repetition:

| Query width | Conservative attention latency reduction |
| --- | --- |
| 224 | 42.1–49.5% |
| 240 | 19.9–22.1% |
| 246 | 19.8–21.9% |
| 251 | 19.7–21.9% |

These are operator measurements, not end-to-end model throughput improvements.
Some baseline repetitions drifted about 10%; all repetitions are retained.
The unchanged width-256 negative control also appeared faster in some cases and
does not support a gain claim.

Both routes passed 50 numerical/guard cases: 40 append/cached Graph cases across
four widths and five context depths, four eager append cases, and six fully
masked/partially valid/full-valid cases with the output gate. The 44 large cases
evaluate the unchanged FP64 reference at first/middle/last query positions;
the six masked cases evaluate all outputs. Cache writes, inputs, canaries and
workspace bounds are checked over their full arrays. This is sampled numerical
qualification, not full-output FP64 coverage for every long shape or a full-model
quality evaluation.

Paired Nsight Systems 2026.5.1 captures each contain 700 Graph launches. The
candidate capture contains 576 grouped-kernel and 576 merge executions, while
the four width-256 controls retain 144 legacy attention executions. Ordinary
timing and profiling runs are separate. The non-Blackwell fallback also compiled
for SM80. No new kernel is selected there.

Artifact identities:

- Benchmark binary SHA256: `66d83ea826ee233d058042830be34768aeaba329d2063c9b5da0450c011332e6`.
- Oracle binary SHA256: `598f324e82146f748d7a9604e5a5b9a960d146f428f27b497e942c2a1feb934f`.
- Campaign: `selective-int8-actual-port-v1`.

The private archive preserves source/object manifests, commands, the reference
adaptation, all CSV repetitions, clock samples, raw Nsight reports, SQLite and
failed preparation attempts. Acceptance follows the
[performance evidence contract](../performance-evidence.md).
