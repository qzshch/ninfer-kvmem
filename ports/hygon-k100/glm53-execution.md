# GLM DTK execution contract

The experimental Hygon profile implements GLM-5.3-Flash text generation through
the native public NInfer Engine. The implementation is a finite platform route,
separate from the default CUDA Engine. It does not establish CUDA feature parity.

## Ownership and placement

Model and Parameters own immutable selected v3 weights and frontend resources.
Program owns device workspaces, recurrent state, compressed MLA KV, retrieval
metadata, companion resources and device Graphs. Runtime owns request admission,
cancellation, deadlines and result publication. Ops implement mathematics and
state transitions; Core supplies storage, streams, events and transfers.

The default PP plan uses three resident stages: [0,16), [16,31), [31,45).
Ordinary generation also accepts `--tp-ep`, which distributes attention heads,
dense projections and resident expert banks across all three devices. It cannot
be combined with DFlash2 or MTP. `--graph` captures the local Decode mathematics;
peer transfers and their event dependencies execute between those graphs.
PP places embedding and output head on device zero and executes stages serially.
TP/EP broadcasts embeddings and merges the three vocabulary partitions' argmax.
Encoded weights remain encoded; the loader does not expand the complete model
to BF16. All selected model weights, KV and recurrent state remain on devices.
Host input/output staging does not provide an offload fallback.

## Ordinary generation

The implemented profile is Aggregate greedy text generation, with one to eight
resident request lanes, independent state and context frontiers, and a maximum
context capacity of 32768 tokens. Ordinary batches use actual compact request
columns. Ordinary prompt consumption uses bounded temporal chunks controlled by
prefill_chunk, with at most 1024 total input rows per unit. The native launcher
exposes --prefill-chunk and --pack-prefill for ordinary generation. Intermediate
prompt rows advance context without requesting a next-token prediction.

Graph families cover actual request count and bounded context tiers. Program
captures device stages and the output head lazily. Request slots, token IDs and
frontiers are metadata inputs. Graph mode and eager mode implement the same
mathematics. Graph resources consume device memory and must fit the resource
budget. Temporal prefill currently executes eagerly; Graph replay applies to
ordinary Decode. `--decode-frames 2 --lanes N` (N>=2, ordinary only) permits
independent units in two frames, each with its own streams,
activations, scratch, pinned metadata/results and Graphs. The second frame is
Decode-only. Requests ready together remain one compact Decode batch; later
independent arrivals can fill the other frame. A sequence cannot have two
pending writers. Runtime publishes a
completed Prefill or Decode frame independently, and can refill a Decode frame
while another request is still prefilling; cancellation drains its device writes before
releasing the lane. Extra frame scratch is included in the memory summary.

KVMem block means, selection and restrictions execute on the devices in the
ordinary route. Request lanes own their respective recurrent and KV histories.
This profile does not expose HTTP serving, streaming, vision, general prefix
cache scheduling or CausalScoring from the default CUDA product.

Ordinary activations at existing BF16 arithmetic boundaries use physical BF16
storage, including KDA, MLP/expert and MLA projection intermediates. Raw mHC,
recurrent state, router probabilities, attention accumulations and TP/EP partial
sums remain FP32. `--fp32-activations` retains the FP32-storage control for
native whole-model comparisons.

The optional build flag `-DNINFER_K100_MMAC=ON` selects Hygon FP32 MMAC for
BF16-input projections with reusable matrix columns, including grouped experts
and K-partials. Single-column Decode retains its dedicated vector path.
It requires actual vendor gfx928 builtins and fails compilation when they are
absent. Encoded weights are decoded to FP32 only in transient shared memory.
Use `check-mmac-isa.sh` for a compiler-only ISA check, then run the independent
GPU oracles and native model comparisons. The default SIMT build remains the
control; enabling matrix instructions alone does not establish a speedup.

## Companion routes

DFlash2 uses the GLM-specific companion and K7 verification in a single-lane eager
route. It cannot be combined with the ordinary Graph batch route. Native MTP K1
uses the target artifact's selected MTP weights, with serial target verification
and rejection fallback. Availability of either route is not a speedup guarantee.
The appropriate companion must be present in the converted artifact.

## Invocation and verification

```bash
bash ports/hygon-k100/run-native.sh --artifact /models/glm.ninfer \
  --prompt-file /inputs/prompt.txt --ordinary --graph --lanes 1
```

Use the native generator's --help for exact arguments. Numerical operator tests
use independent represented-input oracles. Real generation and lifecycle checks
must exercise the public Engine; operator tests or an external engine do not
establish end-to-end model behavior. Generated reports are local outputs.
