# Resident checkpoint buffers

Checkpoint capture uses pinned host images. A free-list of two cached images can
allocate and release pinned memory during concurrent Prefill/Decode as live
checkpoint references outlast those two cached entries. Allocation and page
population can synchronously leave the GPU idle.

`NINFER_KVMEM_CHECKPOINT_RESIDENT=1` reserves and touches the bounded working
set before serving requests. It retains the original direct D2H/H2D copies and
last-shared-reference lifetime. No model arithmetic, KV selection, or prompt
representation changes. The default remains `0`.

Capacity is five checkpoint kinds times active plus retained windows, two
transient leases (capture replacement and idle I/O), and reconstruction scratch.
MTP reconstruction uses one image: lanes are backed up serially on the executor.
Unexpected extra live leases retain the existing safe allocation fallback.
The tested three-lane, one-retained-window, retained-MTP configuration reserves
23 images, 3,540,948,992 bytes of host memory. This is total pool reservation,
not incremental memory relative to the previous dynamically allocated images.
It adds no device KV/workspace allocation; platform pinned-memory limits apply.
Startup time and host memory are the
tradeoff; this experiment is not automatically enabled.

The standalone `ninfer_checkpoint_buffer_resident_test` checks 23 independent
leases through 100 churn cycles without hot allocation/free, shared aliases,
close lifetime, prewarm bounds, and failure cleanup. It keeps checks enabled in
Release builds. ASan/UBSan and original-request GPU comparisons are recorded
separately. A paired late-Prefill Nsight ROI removed two 216–219 ms allocations
and associated 229–233 ms GPU gaps. This does not establish a whole-workload
throughput increase or complete model-quality acceptance.

The sizing and lease helpers are shared with upstream PR #168. Cold-storage materialization keeps its original allocation path. This implementation adds no device KV/workspace buffers; platform-specific pinned-memory limits still apply.
