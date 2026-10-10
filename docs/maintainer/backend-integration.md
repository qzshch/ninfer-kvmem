# KVMem backend integration

NInfer remains the serving and numerical backend. The portable `kvmem/` library
owns retrieval features, scoring, retention policies and capacity arithmetic.
It compiles as C++20 without CUDA, NInfer, model weights or a frontend.

| Boundary | Owner |
|---|---|
| Mean-K, block scoring and ordered selections | `kvmem_core` |
| Instruction/media retention, rolling windows and Host peak arithmetic | `kvmem_core` |
| Frontend token ranges, Vision groups and request error translation | `program/retrieval/adapter.h` |
| Physical page tables, transfers, StateImages and CUDA Graphs | NInfer Core/Program |
| Prefix identity, checkpoint catalog, waiting leases and publication | Native Engine |
| Structured sampling, grammar transactions and tool constraints | NInfer Text/Frontend/Program |
| Optional RAM/disk extension | Separate HiCache branch; disabled in this build |

The backend owns a metadata quota and supplies a `MetadataBudget` claim callback.
Each independently allocated feature block retains one shared RAII charge.
Checkpoint aliases share that charge; copy-on-write obtains a new one before
modifying the source. Refusal preserves the old values and ownership. The quota
owner outlives all lanes and checkpoints. There is no parallel memory ledger.

The current page contract is 64 tokens. The NInfer bridge checks its physical
page size at compile time. A backend with a different page geometry must adapt
the policy explicitly, rather than reinterpret the resulting page indexes.

This is a source-level integration boundary, not a loadable plugin ABI. Adding
another backend requires a backend-specific adapter for capture, placement and
state transactions; it does not require copying the retrieval algorithms.
The NInfer numerical model and Graph ownership are intentionally retained.

## Upstream imports

`backends/versions.json` fixes the upstream references and records which mechanisms
are imported, already present or intentionally not applicable. Keep it and this
contract synchronized with future imports. Do not follow an upstream branch
silently or replace a working source tree with the official KVMem NInfer package:
that package currently rejects DFlash and video and uses a different NInfer fork.

Update NInfer in independent commits: dependency vendor, grammar/sampling,
JSON/schema, tool constraints, then composition. Resolve both compile conflicts
and semantic conflicts in frontend instruction ranges, sparse replay boundaries,
Host admission, sampling masks and checkpoint leases. Run the portable suite,
affected protocol/schema oracles and actual model lifecycle tests before promotion.

KVMem's GDN output RMS normalization plus SiLU gating is already fused in
NInfer's `gated_rmsnorm` production Op. It is distinct from the disabled non-CP
recurrence fusion experiment. KVMem's system-turn merge is a bridge for llama.cpp
templates; NInfer preserves independently typed System and Developer turns,
including late instructions, instead of moving them into a single leading turn.
Both `instructions` and input messages must survive Responses conversion.

NInfer prepares Graph families for all configured batch sizes and owns shared
prefix checkpoints across lanes. Verify idle/reactivated lanes, delete/evict while
an in-flight response holds its parent, cancellation and quota return. KVMem's
llama.cpp session directories and orphan-directory garbage collection do not
exist in this native build; adding a second disk session catalog would increase
coupling and is not part of this integration.

## Independent core checks

```sh
cmake -S kvmem -B build-kvmem-core -DBUILD_TESTING=ON
cmake --build build-kvmem-core -j
ctest --test-dir build-kvmem-core --output-on-failure
```

The same target is included in the normal NInfer CTest build. Numerical and
end-to-end tests remain backend-specific; passing this CPU suite alone does not
establish quality equivalence to full dense attention or a speed increase.
