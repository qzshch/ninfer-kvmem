# Sparse working sets and tiered Native history

This extension is based on upstream NInfer 68c54356, with the source-wait admission
contract from 911d34db adapted to sparse physical working sets. Native checkpoint identity,
StateImages, immutable source leases, persistent KV directories, private replay
points and incremental execution permits remain authoritative. Storage tiers do
not introduce another prefix catalog or another owner of model state.

## Three representations, one logical history

* Device: KVMem's bounded per-lane Main KV working set. Missing logical pages are
  `kPagedKVPageHole` and are masked before attention loads; original positions and
  causal boundaries are retained. A page selected by another active row cannot
  be recycled. The working set and its growth allowance are separate from the
  pool's shared physical capacity.
* Host: optional bounded RAM hot KV blocks with protected/probation queues,
  checksums and write-back. A configured resident quota separately bounds lazy
  pinned StateImages and uniquely owned retrieval features. Completed MeanK blocks
  are immutable across aliases and detach under copy-on-write.
* Disk: one private file per Host KV arena, extended only by actual payload writes.
  The configured logical quota does not preallocate an empty file of that size.
  Cold extents use the same
  address ledger as StateImages, pause snapshots and in-flight destinations.
  Resident State reservations grow from the high end of that ledger while cold
  KV takes low free extents. State still consumes the unified quota, but does not
  create unwritten file gaps ahead of cold payloads.
  Fixed pinned staging slots bridge CUDA and the file worker. File payloads are
  valid only together with Native's complete checkpoint metadata and StateImage.

The unified Host logical quota is not a promise that that many bytes remain in
RAM. In file mode, RAM hot capacity and resident State/metadata capacity are
separate physical limits. Staging, checksum/epoch tables, prompt preparation,
allocator overhead and process memory also require headroom. Runtime exposes each
component; do not sum a typed arena's capacity with the shared arena's capacity.

## Configuration

`--kvmem-window-pages N` enables sparse execution; each page covers 64 tokens.
Zero preserves dense behavior and its startup capacity invariant. Sparse startup
requires at least the per-lane working set, a prefill chunk and growth slack.
`--kv-capacity` counts the shared physical pool's token capacity, whereas
`--max-context` bounds each sequence's logical position range. Main pool growth
and recovery reservations are bounded without removing logical history.

`--host-context-mib` is upstream's single logical Host quota. No obsolete
`--host-kv-mib` or `--host-state-slots` alias is added to this base.

`--kv-file-dir` enables instance-local file KV. `--hicache-ram-mib` adds a RAM
hot tier (whole MiB), and `--hicache-state-mib` bounds resident State/metadata.
`--hicache-prefetch` enables bounded queued read hints. `--hicache-write-through`
enables idle asynchronous dirty writeback. Both policies need nonzero RAM hot
capacity. Defaults leave these experimental I/O policies off. File backing also
works with dense Native execution; KVMem is optional.

This is an in-process runtime cache. Files are unlinked when their owner retires.
Abrupt process termination does not run that retirement and can leave private
scratch files behind. Such files do not constitute recoverable checkpoints.
Cleanup must first establish that no live instance owns the affected files.
Independent processes and engine restarts do not share a checkpoint namespace.
Pointing two engines at the same directory creates distinct private files and
never implies prefix sharing.

## Transfers and publication

1. Reserve every destination while retaining all source capabilities.
2. Schedule copy/IO work. The worker accesses file, RAM and staging only; it does
   not call CUDA or mutate Native logical pages, directories or StateImages.
3. Drain the required completion boundary and check file errors.
4. Publish valid replicas, then retire obsolete representations and staging.

An execution row's pinned Host page-table shadow stays immutable until its last
asynchronous upload completes. Reusing the same CUDA stream alone does not protect
the Host source. Each row fences that source before a later publication writes it,
and destruction drains outstanding uploads before freeing the shadow.

Incoming sparse placement adds temporary destination claims to the already
reserved growth capacity, then restores the exact prior reservation. Submission
or checksum failure drains submitted work and aborts unpublished Device replicas.
Outgoing placement stages every necessary Host extent before changing a row.
Epoch invalidation precedes extent reuse. Callbacks, cancellation and cleanup
retain their capabilities through completion.

File writeback batches span multiple demand jobs. Transfer completion means
readable bytes and their integrity records, not crash durability. Before a write
would exceed 64 MiB plus one system page of conservatively charged filesystem
residency, the worker waits for completed range page writeback with
Linux `sync_file_range`, then requests page-cache eviction. Adjacent ranges are
coalesced. This does not flush metadata or the device write cache, and does not
provide crash durability; that is outside this instance-local scratch contract. An idle
50-ms boundary, explicit `drain()` or destruction also flushes a partial batch.
This fixed budget is additional headroom beyond RAM hot capacity, StateImages and
pinned staging; it does not grow with cold capacity. The currently read staging
range temporarily adds clean file pages which are immediately advised away.
The charge describes the Linux client page cache. On DrvFs, it does not bound the
Windows filesystem cache or establish physical-device completion; monitor total
Windows memory headroom separately. A zero charge is an IO retirement boundary,
not evidence of a physically cold disk read.
`filesystem_pending_bytes` reports the charge and
`filesystem_write_budget_bytes` reports its limit; both are gauges. Writeback errors are checked before the charge retires. File errors
poison the instance and prevent publication of an unchecked restore. Explicit
drain must still settle all IO and request eviction before backing retirement.

An asynchronous bind pins the selected resident donor pages until the final
synchronous fork takes over. A logical checkpoint lease alone cannot protect a
physical replica from another lane's placement. Under temporary capacity pressure,
retrieval may retire idle replicas with current Host backups; a changed-query bind
can similarly retire only unselected donor replicas. Active aliases and pending
source pins prevent both forms of reclamation, and logical history remains intact.

An in-flight restore also blocks execution aliases that could retrieve its pending
Device pages. Admission can introduce this dependency after unit reservation, so
Control, Decode and Prefill check it again when assembling the actual batch.
Unrelated histories keep their overlap. Execution failure drains an outstanding
context transaction before releasing histories that may share its page leases.

Native unit admission forecasts imminent sparse Host copies and feature growth
before GPU execution. This follows upstream's incremental admission rather than
reserving each request's maximum future context. Resource pressure may reclaim,
pause, replay or reject a request; it must not silently exceed a tier's quota.
The first query replay reserves growth from its future rewind frontier before
rewinding the unpublished probe. Its present mapped extent is not that baseline;
using it would admit zero growth and later allocate after a competing binding.
This reservation also works at the full logical context boundary and does not
rewind or publish any row during a failed admission.
The final query probe can replace most of the working set after scoring. Its
permit therefore forecasts unbacked or stale resident replicas, the possibly
rewritten tail and new growth pages. Ordinary rolling-window demand alone is not
sufficient for this placement burst. Scores are unavailable before execution, so
this boundary uses a conservative backup bound; actual copies still follow the
selected set. A denied permit leaves the frontier and physical claims unchanged.

## Long prefill and prefix reuse

Prefill maps only its cursor/chunk growth and rolls the committed history at
boundaries. Query features use the original pre-RoPE, post-QK-normalized domain.
A long query probe is unpublished: it does not add a generated token or penalty
history. Its GDN/local-draft checkpoint permits a bounded query-suffix replay
with the retrieved history. Capture groups in the unpublished suffix are deferred
until canonical replay reaches them. Replay has its own cursor and respects
Native execution/capture boundaries.

The query-boundary checkpoint also saves the partial Key-sum ring. Canonical
replay rebuilds Key means while leaving the completed Query sum unchanged. This
keeps an image or rewrite checkpoint's Key frontier aligned with its actual stored
KV, including a block split across probe, replay and later prompt extension. Paused
query checkpoints carry this ring in the same explicitly budgeted Host snapshot.
They include the continuation hidden row needed by MTP. At the beginning of query
replay, a requested pause snapshot first rewinds the unpublished probe to its
canonical boundary; this preserves original prefix KV instead of recomputing it
with a different sparse selection. The vacated lane retires its CPU feature
references once ResumeState takes ownership.

Native checkpoints retain a COW retrieval index and partial key accumulators.
An identical full typed input retains its canonical selected pages even when Native
recovers at an earlier image/rewrite boundary. A different or extended input uses
its own new query. Equality includes media identity, not just tokens or session keys.
Text-only suffix replay does not require a live Vision session after the image has
already been recovered. Pause/resume rebinding restores features and any unpublished
query checkpoint into the selected lane before its snapshot is released.
Final probe placement covers the complete first canonical replay window at the saved
query boundary, not an overlapping recent range at the unpublished probe end. This
prevents a later unit from expanding an undersized set after competing bindings
consume the pool margin. The active row protects the admitted history through rewind;
probe suffix pages may retire because replay rewrites them. Main and MTP use their
respective budgets.
An owned pause snapshot restores its current retrieved history even before the
public prompt is complete; it does not use unrelated cached-donor input equality.
This covers paused private query replay and active generation, and the binding
collects the complete window before granting the execution lane.
Canonical scored features omit the rolling recent share. Binding derives and
admits the complete first executing window (including sink, recent pages, media
groups and the writable tail) before publishing the lane. A pressure restore must
not admit only scored pages and discover the rest after another row fills the pool.

Caching reuses previously valid state; sparse attention itself is approximate
relative to dense full history. Byte-preserving tier copies do not establish
answer-quality equivalence. Test tier equality with identical selections and
separately evaluate selection quality against a dense/reference configuration.

## Observability and qualification

Per-request JSONL `kvmem` separates prefill, retrieval, replay and decode placement
(Main/backend bytes, pages, existing waits and host wall time), key/query capture,
selection and replay work. Periodic `hicache` contains lifetime IO/hit/prefetch
counters plus instantaneous occupancy and pending work. File-worker counters are
read live even when the Engine has no runnable request; model resource/lane state
remains the last execution publication. This read touches atomics only and does
not wake the Engine or synchronize CUDA. Take counter differences
when comparing phases; pending work is a gauge. Collection reads bounded counters
and does not add a CUDA timing synchronization solely for telemetry.
`computed_prefill_tokens` counts the admitted prompt suffix once. Additional query
execution is counted separately by `kvmem.replay.tokens`; add both for total prompt
execution work, and report their costs separately rather than advancing the public
prompt frontier twice.

Weight materialization can use `NINFER_WEIGHT_READ_THREADS=2` or `4` (default `1`).
It retains the original bounded pinned staging ring and serial upload order.
Workers only read immutable file ranges after all direct descriptors are prepared;
the uploading caller waits for CUDA completion before a slot can be refilled.
This controls startup IO, independently of prefix-cache prefetch or Prefill kernels.

New qualification covers shared-row residency, partial-tail COW, exact reservation
retention, unified/lazy Host quotas, metadata aliases, insufficient Host rollback,
corrupt restore, file callback faults and numerical hole masking against the
existing FP64 attention oracle. Real-model tests must supply the tested artifact
explicitly and run serially on the single GPU. Preserve failed trials, exact source
and binary hashes, resource guards and cold/warm computed/reused token counts.

## Design references

* [NInfer Native ownership and scheduling](resource-scheduling-and-context-cache.md)
  defines this implementation's checkpoint and resource contracts.
* [KVMem](https://github.com/kvmem/kvmem-llama.cpp) provides the bounded working-set
  and query-based historical retrieval design. This adapter executes those
  mechanisms inside NInfer; it does not link the llama.cpp runtime.
