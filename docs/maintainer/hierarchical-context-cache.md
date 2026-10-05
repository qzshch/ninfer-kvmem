# Hierarchical context cache with KVMem

The local hierarchy adds an independently bounded RAM hot set to the existing
file-backed Host KV arena. It can run in the same Engine as KVMem sparse paging,
multi-lane generation, vision, and the MTP, DFlash2 or DSpark backends. It stores
and transfers exact KV payload bytes; it does not change sampling, quantization,
attention selection or model operators.

## Tiers and ownership

* **GPU:** Program owns logical pages, active aliases, concrete reservations and
  the KVMem working window. HiCache does not change the active entitlement.
* **RAM:** `--hicache-ram-mib` reserves a separate, managed KV hot set. It is
  independent of the two fixed pinned transfer slots and the filesystem page
  cache. A protected read set uses three quarters of the hot-block allowance;
  newly written and prefetched blocks enter the probationary set. Reads promote
  blocks, while streaming writes do not evict the protected read set first.
* **Disk:** `--kv-file-dir` supplies a private, mode-0600, bounded positional
  byte store. `--host-kv-mib` specifies its logical address-space capacity.
  RAM replicas do **not** add to this logical capacity. The file is removed
  when the Engine is destroyed. The filesystem page cache is explicitly evicted
  after clean I/O, avoiding an uncontrolled fourth memory tier under WSL.

Complete reusable checkpoints remain Program objects: KV, StateImage/GDN,
backend ring state, MeanK selection index and capture accumulator, frontier,
token/media digests and backend/storage identity. These objects and their
references stay in the existing bounded typed pools/catalogs. StateImages and
MeanK are not independently reconstructed from an incomplete disk KV record.
Requests still pass the normal checkpoint identity, frontier, media and fork
validation before reuse. A hit is not permission to publish a partially restored
execution table or skip the required query/tail replay.

The hierarchy is local to a running Engine. Restart-persistent prefix catalogs,
cross-instance matching and distributed storage backends need a separate complete
checkpoint serialization format; a leftover byte file alone is not such a cache.

## Migration and publication

Host KV writes first copy through fixed pinned staging into the managed RAM owner.
The byte worker computes a checksum for each 256-byte sector before completing
the transfer. Under the default **write-back** policy, dirty data reaches disk
when RAM eviction needs that space. `--hicache-write-through` instead lets the
worker flush one block at a time when there is no queued demand work, retaining
a clean RAM copy. Both policies have fixed RAM capacity and no extra unbounded
dirty-buffer queue. Writes that fail do not publish a successful cold replica.

A demand restore reads RAM when present and otherwise validates disk data before
promoting it into RAM and copying it through pinned staging to GPU. CUDA stream
completion and Program error checks precede logical-replica publication. Existing
source/destination leases, StateImage forks and global transition boundaries
remain authoritative. File-backed restores currently use complete restore fences;
the pinned-RAM-only layer-consumption mechanism retains its existing eligibility
checks.

`--hicache-prefetch` starts best-effort disk-to-RAM hints for pinned prefix source
extents before the GPU restore batch, and bounded look-ahead during KVMem restores.
The queue has at most 128 block hints, a call looks ahead by at most two transfer
chunks, and demand transfers have priority. Allocation invalidation increments
block epochs, so queued hints cannot resurrect old generations. Hints change only
physical byte residency; they do not mutate a logical checkpoint or publish a GPU
page. Prefetching can waste I/O and must be measured against the same configuration
with prefetch disabled.

## Configuration

For example, add the following to an existing KVMem-enabled generation command:

```sh
--host-kv-mib 16384 --kv-file-dir /var/tmp/ninfer-hicache \
--hicache-ram-mib 8192 --hicache-prefetch
```

This means a 16-GiB logical disk-backed arena and an 8-GiB managed RAM hot set,
plus the existing StateImage/index/catalog memory and 32-MiB pinned staging.
`--hicache-ram-mib 0` retains file-only storage. Omitting `--kv-file-dir` retains
the original pinned Host KV arena. Prefetch and write-through require a nonzero
RAM hot set, and RAM cannot exceed the logical Host KV capacity. KVMem GPU window,
lane count, logical context and backend are configured independently.

Choose capacities using the Engine's memory summary and the machine's actual RAM,
Windows commit, free disk space and active-context requirements. Sparse logical
file length does not reserve physical filesystem space. An I/O failure poisons
the backing instance; corrupt, missing or failed data must never become a valid
checkpoint. Cancelling a request does not cancel another lane's leases.

## Prefill measurements

The primary benefit is avoided recomputation after a reused prefix survives RAM
pressure, followed by reduced exposed restoration wait. Cold inputs with no
matching prefix still require Prefill. Reused input tokens divided by latency are
effective input throughput, not faster GPU Prefill kernels.

The `file_kv` JSONL record separates application transfer bytes, managed RAM
hit/miss bytes, positional disk read/write bytes and time, hot-set capacity,
resident/dirty bytes, evictions, prefetched/consumed/wasted bytes and live
transfer/prefetch/writeback/callback counts. Disk timings and CUDA restoration
timings may overlap; do not sum them into a fabricated critical-path saving.

Compare identical requests under A/A and ABBA runs, recording TTFT with queue time
separate, computed Prefill/replay and reused tokens, cold/warm/append/pressure
revisit, multi-lane batch wall time and decode pauses. A larger storage pool and
different inactive GPU-replica reclamation can change residency; such comparisons
are whole-mode comparisons, not isolated I/O experiments.
