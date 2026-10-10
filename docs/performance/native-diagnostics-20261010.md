# Native DFlash2 rejection diagnostics

Archived experiment instructions, 2026-10-10. These commands apply to the local
diagnostic experiment worktree described in the accompanying performance report;
the report publication does not add these probe targets or scripts to `feature/kvmem`.

This is an opt-in, eager-only diagnostic of six fixed public code, math and prose
prefixes. It executes one real K7/top16 transaction per prefix, with two identical
lanes. The duplicate lanes test batch consistency; they are not independent
acceptance samples. It is not a serving performance or answer-quality benchmark.

Build `ninfer_qwen3_5_native_transactions_test` and `ninfer_candidate_selector_test`
with the same CUDA/toolchain settings as the engine. Run the independent selector
operator test first. With an artifact containing a DFlash2 component:

```bash
mkdir -p /path/to/new-evidence
NINFER_TEST_ARTIFACT=/path/to/model.ninfer \
NINFER_NATIVE_TRANSACTIONS_PACK=1 \
NINFER_NATIVE_TRANSACTIONS_TARGET_WIDTH=1 \
NINFER_TARGET_TRACE_TEACHER_STEPS=32 \
NINFER_DFLASH_ROUND_TRACE=1 \
NINFER_DFLASH_SELECTOR_TRACE=1 \
NINFER_PACK_DIAGNOSTIC_DIR=/path/to/new-evidence \
./build/tests/ninfer_qwen3_5_native_transactions_test dflash2 \
  > /path/to/native.log 2>&1
python tests/models/qwen3_5/analyze_dflash_round.py \
  /path/to/new-evidence /path/to/native.log /path/to/new-summary.json
```

Use a new evidence directory/output file for every run. The probe refuses to
overwrite raw evidence. Teacher steps are bounded to 128. Check available host,
commit and device capacity before loading a real artifact; supervise the process
and retain a safe resource margin.

The probe preserves actual candidate IDs, unary scores, proposal q, selected
tokens, target logits and the represented selector hidden/codebook values. The
Python analysis independently computes `unary + dot(predecessor * hidden,
successor)` in FP64 and checks the real choice against a conservative FP32
forward-error envelope. It counts only the accepted prefix and first rejection;
positions after that rejection do not constitute observed rejection events.
It also verifies conditional ordinary and projection-profile counterfactuals.

Only this native diagnostic creates selector capture buffers. Serving leaves the
optional capture empty and performs no new copy, allocation, capture query or
synchronization. Do not use probe timings for throughput. Changing the target
artifact generates a different teacher trajectory, so comparing acceptance
counts across artifacts alone cannot attribute a difference to fine-tuning.
