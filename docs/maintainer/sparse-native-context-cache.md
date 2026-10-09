# Sparse KVMem and Native context cache

This extension uses upstream NInfer 68c54356 and adapts the source-wait admission
contract from 911d34db to sparse physical working sets. Native owns checkpoint
identity, StateImages, immutable source leases, persistent KV directories and
private replay points. KVMem changes physical placement rather than adding a
second prefix catalog. RAM hot pools and disk KV backing are absent from this build.

## Capacity and ownership

`--kvmem-window-pages N` enables a per-lane sparse working set, with 64 tokens
per page. Zero retains dense execution. `--max-context` bounds logical positions
per sequence; `--kv-capacity` bounds the shared physical Device pool. The pool
must cover all windows, prefill chunks and growth slack. A larger logical context
does not imply that its complete KV remains resident on the GPU.

`--host-context-mib` bounds the shared Native Host quota for StateImages, KV
replicas, pause snapshots, pending destinations and unique retrieval metadata.
Backing is allocated at startup; pinned is the default and pageable is an explicit
experimental memory kind. External CPU retrieval buffers are
charged once through shared ownership; aliases never charge a second copy.
`host_context_resident_bytes` reports fixed backing plus those metadata buffers.
For pageable backing, logical allocation size does not measure physical RSS.

With two lanes, 576 pages per window, prefill chunk 1024 and shared KV capacity
77824 tokens, each working set is 36864 tokens. A Host quota of 8192 MiB does
not guarantee simultaneously retained histories of 262144 tokens per lane.
Host pressure can evict optional history or invoke private replay. Full combined
capacity must be qualified against actual State/KV layouts and host commit margin.

## Transfers and recovery

Reserve destinations while retaining sources, complete CUDA transfers, publish
valid replicas, then retire obsolete ownership. Page-table upload shadows remain
immutable until each row's previous DMA completes. Sparse offload reserves Host
extents before changing the execution row; refusal leaves the original placement
and growth claims intact. Restore admission counts selected missing physical pages
and required growth/tail slack, rather than the complete logical prefix.

Frontend annotates proven rendered System/Developer blocks, including generated
tool definitions, as token ranges without changing tokenization. Quoted control
markers in user content cannot create instruction ranges. These ranges include
late instructions and are remapped through media placeholder expansion.

KVMem keeps all materialized instruction pages, the sink, latest complete media
group and two tail pages before selecting other history. An instruction boundary
sharing a media page retains that complete group. Admission rejects a mandatory
set that cannot fit the configured working window; it never silently truncates
tool definitions. Raw token prompts without instruction ranges retain their old
selection policy. Custom templates without proven structural instruction blocks
do not receive inferred instruction ranges.

Final query probing forecasts Host backup and future growth. Canonical binding
includes scored history, instructions, sink, recent, media and tail pages. The first replay
cannot discover an unreserved history promotion. Snapshot recovery restores its
saved selection; a donor selection is inherited only under proven query identity.
Prefill, query replay, decode, checkpoint restore and execution-unit Host quotes
use the same instruction retention policy, including the MTP KV address space.
Sparse Prefill returns to scheduling after each physical chunk, including partial
chunks at typed rewrite boundaries. The next unit rolls and quotes its working
set before executing; a warm suffix cannot carry the preceding chunk's recency
pages past a boundary where cold query replay would have retired them.
Detached resume state owns its instruction ranges; retiring a lane clears them.
Incomplete query rewind refuses without moving the frontier when required Device
growth cannot be claimed. Cancellation returns unpublished reservations and
private metadata while preserving reusable history.

## Observability and qualification

JSONL schema 26 retains Native admission, per-lane prefill/decode, KVMem copy,
selection and replay counters. Host logical occupancy and resident backing remain
separate. There are no RAM hot-cache or disk IO fields in this build.

The real KVMem fixture covers cold/warm requests, concurrent shared-prefix forks,
media changes, appended turns, query-probe cancellation, exact greedy output
recovery and lane counter conservation. Native transaction tests additionally
cover pause/replay, partial query recovery and resource-pressure refusal. Sparse
attention selects a bounded history and does not establish quality equivalence
to dense full-history attention. Numerical Op qualification is a separate contract.
