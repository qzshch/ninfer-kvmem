# Resident checkpoint buffers

Checkpoint capture uses pinned host images. A free-list of two cached images can
allocate and release pinned memory during concurrent Prefill/Decode as live
checkpoint references outlast those two cached entries. Allocation and page
population can synchronously leave the GPU idle.

`NINFER_KVMEM_CHECKPOINT_RESIDENT=1` reserves and touches the bounded working
set before serving requests. It retains the original direct D2H/H2D copies and
last-shared-reference lifetime. No model arithmetic, KV selection, or prompt
representation changes. The default remains `0`.

Resident mode also covers the MTP reconstruction backup when
`NINFER_EXPERIMENT_REWRITE_SCRATCH_REUSE` is unset or `0`. Explicit `1` and `2`
keep pooled-per-use and retained-single-backup behavior respectively. With
resident mode off, the existing experimental settings retain their behavior.

The KVMem memory summary includes catalog images, idle checkpoint-pool buffers,
and the independent retained reconstruction backup in `checkpoint_bytes`.
Borrowed buffers leave the idle count until their last shared reference returns,
so the same allocation is not counted as both an idle buffer and a live image.

Capacity is five checkpoint kinds times active plus retained windows, two
transient leases (capture replacement and idle I/O), and reconstruction scratch.
MTP reconstruction uses one image: lanes are backed up serially on the executor.
Unexpected extra live leases retain the existing safe allocation fallback.
The tested three-lane, one-retained-window, retained-MTP configuration reserves
23 images, 3,540,948,992 bytes of host memory. This is total pool reservation,
not incremental memory relative to the previous dynamically allocated images.
It adds no device KV/workspace allocation; platform pinned-memory limits apply.
Startup time and host memory are the tradeoff; it is disabled by default.

The standalone `ninfer_checkpoint_buffer_resident_test` checks 23 independent
leases through 100 churn cycles without hot allocation/free, shared aliases,
close lifetime, prewarm bounds, and failure cleanup. It keeps checks enabled in
Release builds. ASan/UBSan and original-request GPU comparisons are recorded
separately. Earlier private paired traces motivated this change; their timing
results do not qualify this revision or the separate public implementation.
Exact-head public quality/cache/cancellation and allocation traces are tracked
in upstream PR #168. No whole-workload throughput increase is claimed.

The sizing and lease helpers are shared with upstream PR #168. Cold-storage
materialization keeps its original allocation path.
