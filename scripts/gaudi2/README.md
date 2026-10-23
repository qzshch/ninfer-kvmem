# Gaudi2 source inspection

`inspect_fp8_checkpoint.py` reads local safetensors headers and the checkpoint
index without loading tensor payloads. It validates the official block128
FP8 representation, pairs each E4M3FN weight with its F32 `weight_scale_inv`,
and records absolute byte offsets, including pairs in different shards.
Headers are limited to 16 MiB per shard; unknown tensor dtypes and layouts fail
explicitly. This tool does not convert weights to `.ninfer` or run inference.

```sh
python scripts/gaudi2/inspect_fp8_checkpoint.py /models/Qwen3.8-27B-FP8 \
  --output checkpoint-inventory.json
```

To produce a command for the native, bounded-memory tile reader:

```sh
python scripts/gaudi2/inspect_fp8_checkpoint.py /models/Qwen3.8-27B-FP8 \
  --tensor model.language_model.layers.0.self_attn.q_proj.weight \
  --tile 0 0 128 128 --tile-output q-proj-tile.bf16 \
  --tile-executable /path/to/ninfer-gaudi2-fp8-tile \
  --output tile-plan.json
```

Choose a tensor actually listed in the inventory. `tile_command` in the output
is an argument array, ready to pass to `subprocess.run` without a shell:

```python
import json
import subprocess

with open('tile-plan.json', encoding='utf-8') as stream:
    plan = json.load(stream)
subprocess.run(plan['tile_command'], check=True)
```

The native tool multiplies E4M3FN code values by the stored block scale and
rounds to BF16 using round-to-nearest, ties-to-even. This prepares a W8A16
operand; it does not perform Gaudi2 FP8 MME arithmetic. Inventory validation
checks the representation and file extents; the native reader checks numerical
values for the selected tile. Neither check establishes whole-model quality.

Build and numerical-test instructions are in
[the backend preparation guide](../../backends/ninfer/docs/gaudi2-preparation.md).

```sh
python tests/test_gaudi2_checkpoint.py
```

Format references:
[safetensors](https://github.com/huggingface/safetensors#format),
[official checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).
