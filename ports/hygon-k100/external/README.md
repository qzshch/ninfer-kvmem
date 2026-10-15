# Optional external diagnostic recipe

This recipe builds a separate GGML diagnostic and does not implement NInfer.
It does not download models or change drivers. It expects an explicit clean
checkout of mxxm-t/mx-llama.cpp at
71a0624fb36aaaa2df24ccf428236493c813a441.

```bash
source /opt/dtk/env.sh
bash ports/hygon-k100/external/build-and-test.sh /path/to/mx-llama.cpp
```

DTK_ROOT defaults to /opt/dtk and BUILD_JOBS controls parallelism. The patch is
revision-bound. Its source and changes retain the included MIT attribution.
The recipe's conservative feature switches are build settings, not performance
recommendations. Native NInfer acceptance must use the public Engine directly.
