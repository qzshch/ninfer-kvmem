# Hygon K100 DTK port

This experimental branch provides a native NInfer implementation for Hygon K100
using the vendor DTK HIP toolchain and the gfx928 compiler target. Hygon is the
hardware vendor; HIP interface names do not imply AMD hardware compatibility.

The GLM text route uses the public Engine, v3 .ninfer artifacts, resident encoded
weights, device recurrent state, compressed MLA KV and KVMem. See the
[execution contract](glm53-execution.md) for the finite supported profile.

## Build

```bash
bash ports/hygon-k100/build-and-test.sh
```

DTK_ROOT defaults to /opt/dtk. K100_BUILD_DIR and BUILD_JOBS override the build
directory and parallelism. The recipe builds Core, native operators, numerical
tests and ninfer_k100_generate. Peer tests require three visible devices.
No model downloads, driver changes or external inference engines are involved.

## Run

Use an explicit converted v3 artifact and an input file:

```bash
bash ports/hygon-k100/run-native.sh --artifact /models/glm.ninfer \
  --prompt-file /inputs/prompt.txt --ordinary --graph --lanes 1
```

The launcher forwards arguments to the native generator. Consult its --help
for context, output and KVMem controls. Model and companion weights are external
prerequisites and are not distributed with this repository. Check their licenses.
Capacity must be established per device with all runtime resources included.
There is no CPU or disk weight, state or KV offload route in this profile.

## Source references

The Hygon adaptation uses DTK interface conventions. Useful upstream references
include [HYGON-AI/apex-das](https://github.com/HYGON-AI/apex-das) and
[ninfer-gfx906](https://github.com/JCraigWasTaken/ninfer-gfx906).
Copied-code provenance and licensing remain in NOTICE and the component notices.
The optional [external diagnostic recipe](external/README.md) is separate from
the native NInfer engine.

Generated measurements, execution transcripts and internal design records belong
outside this public checkout. They are not distributed as product documentation.
