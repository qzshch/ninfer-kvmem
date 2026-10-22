# Performance evidence contract v1

Tests must establish a declared claim, not merely finish. The executable contract
is `scripts/performance_evidence.py`; its version is part of every plan and result.
Raw requests, responses and traces remain in the operator's private artifact store.
Public reports contain the evidence index, assertions, scope and limitations.

Before a GPU run, freeze a plan containing both source manifests, binary hashes,
complete recipe hashes and the calibrated device profile; original request body
hashes; cache preparation and arrival/dependency protocols; the single treatment;
metrics and acceptance limits; A/A evidence; ABBA order; and the logical sampling
trigger and target event. Hash the actual HTTP bytes, including system/history,
tool schemas/results, sampling settings and output caps. Do not add a seed.

Start from [the example plan](examples/performance-plan.json). Its placeholder
hashes intentionally fail preflight until replaced. The receipt must contain
`plan_sha256`, `started_unix`, `run_completed`, `runs`, `quality`, `measurements`
and `artifacts`. Run identities use the same fields as the plan's arms; actual
request hashes, cache/arrival hashes and observed collector fields are required.
Measurements identify arm, metric, request, actual phase, context band, cache
class/frontier, profiling status and numeric value. The complete minimal receipt
and failure examples are exercised in `tests/test_performance_evidence.py`.

```
python scripts/performance_evidence.py preflight --plan campaign.json
python scripts/performance_evidence.py seal --plan campaign.json --seal frozen.json
python scripts/performance_evidence.py evaluate --plan campaign.json --seal frozen.json \
  --receipt run.json --artifact-root artifacts --output verdict.json
```

`record-run` performs the same preflight before an owned finite test driver. The
driver must record and clean up all children in its own `finally`, including
cancellation and timeout. Collectors are observers: failures mark evidence
incomplete and never stop an inference service. Keep raw failed attempts.

The plan's `metrics` each declare a `key`, `direction` (`higher` or `lower`),
`availability` and whether they support a `claim`. An unavailable timer needs a
reason and cannot support a claim. This prevents, for example, treating prefix
planning time as an exclusive prefix-restore time. Enabled collectors declare
their supported fields; each run must record fields actually observed.

| Scope | Minimum evidence |
| --- | --- |
| Operator latency | FP64 reference, cache/input/output/workspace guards, eager and Graph qualification, ordinary CUDA-event ABBA, matching detailed Nsight |
| Host allocation | Actual allocation/capture event inside both ROIs, API/NVTX/kernel timeline, lease integrity and Host-memory cost; no whole-engine speed claim |
| Engine performance | 30 distinct original requests from eight sessions; P/D/PP/DD/PD observed; three independent real pairs for each concurrent class; exact-head quality/tools and cache validation |

Ordinary measurements and profiled runs are separate. Record profiler version,
command, target PID/boot/start identity, actual API/kernel/memcpy/Graph/NVTX table
coverage, tail integrity, loss warnings and clock witnesses. A file existing or
a profiler exiting successfully is insufficient. A trace taken before the
target allocation cannot establish its removal. Repeating one pair does not add
independent workload coverage. Hardware-internal DRAM/SM/stall claims require
their hardware counters; a Systems timeline alone cannot establish those claims.

Every engine run needs a unique HTTP/native request join and a per-request ledger:
prompt tokens, cache tokens, uncached tokens and elapsed Prefill time, Decode
tokens/rounds/time, TTFT, queue and total latency, finish/error, and SSE gap
P95/P99/maximum. SSE gaps are not per-token ITL. Keep replay work separately.
Classify P/D/PP/DD/PD from actual simultaneous progress, including zero-progress
intervals, and match context bands and cache frontiers when comparing requests.

Report cache restore/bind separately when its exclusive timer exists. Report
uncached Prefill tokens/s, Decode tokens/s **and** rounds/s and useful output per
round; cache hit fraction is a separate metric. Do not count reused tokens as
computed Prefill. Equal aggregate hit fractions do not prove equal hit positions.
Unmatched cache/context/phase strata remain explicitly unqualified. Do not use
whole-request duration alone to identify a kernel or scheduling cause.

Attribute Host/control/checkpoint/pure-P/joint-PD time with a mutually exclusive
ledger (`exclusive_ledger` provides priority-based interval partitioning).
Inclusive submit time, wait time and nested capture time must not be added twice
or labelled GPU idle. Include relevant stalls and memory peaks, not only kernels.

Archive sources, binaries, commands, raw samples, traces/SQLite, outputs and
failures. The artifact manifest records each relative path and SHA256 plus the
archive URI and SHA256 read back from that copy. Archive receipts are evidence,
not a substitute for checking files.

The verdict separates `run_completed`, `evidence_complete`, `quality_pass` and
`performance_gain`. Missing data yields `evidence_incomplete`; complete results
can still be `inconclusive` or `regression`. No inherited pass from another head,
workload, recipe or profiler window. Change a frozen plan by creating a new
campaign, preserving the original and explaining the change.

Validator revision 2 records the evaluator SHA256 and revision in seals and
verdicts. Preserve the original evaluator and verdict when fixing a validator;
an offline re-evaluation must identify both versions and its reason. Never alter
frozen request bytes, measurements or acceptance limits to obtain a pass.

Gain requires the slowest candidate repetition to beat the fastest control by
the frozen noise floor in every claimed stratum. Report control spread and the
median difference as well. Retain fallback negative controls even when they
appear faster through drift; they do not support an operator-change claim.

Cross-domain clock witnesses are mandatory for Host-allocation and engine
claims. A CUDA-event-only operator claim has no Host-clock join: paired Nsight
still verifies the actual kernel route, replay count and trace tail. Its node
spans are diagnostic and do not replace ordinary CUDA-event timings.
