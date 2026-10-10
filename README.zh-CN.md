# ninfer-kvmem

[English](README.md) | 简体中文

**面向单卡长上下文、多请求与工具调用的 NInfer + KVMem 集成构筑，由 [qzshch](https://github.com/qzshch) 维护。**

本项目以 [NInfer](https://github.com/Neroued/ninfer) 的 C++/CUDA 推理引擎为基础，将 [KVMem](https://github.com/kvmem/kvmem-llama.cpp) 的有界 GPU 工作集、Host 历史存储与查询检索机制接入原生执行和前缀缓存。重点是：在有限显存中使用较大的逻辑上下文，支持多 lane、DFlash2 与视觉共同运行，并持续验证真实 agent 请求中的缓存、工具和会话生命周期。

`main` 是当前日常构筑的维护入口；旧 `master`、`feature/kvmem` 和实验分支保留历史用途。本页数据更新于 **2026-10-10**，对应已运行验证的代码 `7e9f4e13`。后续首页与文档、归属标注修改不代表新增性能优化。

## 本项目的核心改动

| 改动 | 实际作用 | 当前状态 |
|---|---|---|
| 多 lane KVMem | 每个执行 lane 使用独立的稀疏 GPU 窗口，逻辑历史存入共享、受配额管理的 Host 存储 | 日常使用 2 lane；每路 36K 窗口、256K 逻辑上限 |
| 与 Native 前缀缓存结合 | 复用已有 KV 和完整继续执行状态，减少重复 Prefill；共享 prefix lease，避免再建立第二套缓存账本 | 冷/热、追加、分叉、取消、恢复与资源归还已验证 |
| KVMem + DFlash2 + Vision | 将检索、投机验证、图像/视频状态接入同一原生执行路径 | DFlash2 草稿 7 + 图像/视频已实际推理验证；MTP3 也有跨窗口回归 |
| 长历史指令与工具保留 | 保留 System/Developer、工具定义、必要媒体和尾部页；窗口不足时明确拒绝，不静默丢弃工具定义 | 原始 DSH 故障请求已受保护重放；修复 MTP 未提交 lookahead 页提前迁出问题 |
| 可追踪的性能与容量 | JSONL 输出 per-lane Prefill/Decode、缓存、选页、搬运与 replay 数据 | 日常面板已验证收到双 lane 数据；可区分逻辑占用和 Host 固定 backing |
| 模块化 KVMem | 检索、窗口及容量策略拆成可独立构建的 `kvmem_core`，由 NInfer adapter 对接 | 降低策略代码与 CUDA/模型/前端的耦合；是源码接口，不是动态插件 ABI |
| 上游能力适配 | 吸收 NInfer 的结构化输出、严格工具约束；适配视频采样校验及 Responses 生命周期 | 支持 JSON/受支持 Schema、GBNF、choice/regex；不支持的 Schema 明确报错 |

NInfer 提供模型数学、CUDA kernels、Graph、原生调度及协议基础；KVMem 提供工作集与检索设计。本项目维护它们在上述组合中的集成和修复，不把上游能力标为本项目原创。具体来源和适配边界见 [版本清单](backends/versions.json) 与 [后端集成说明](docs/maintainer/backend-integration.md)。

## 当前日常配置

| 项目 | 配置 |
|---|---|
| 验证平台 | RTX 5090 32 GiB、Linux/WSL2、CUDA 13.2.86、GCC 13.3 |
| 实测模型 | Huihui Qwen3.8-27B NVFP4 `.ninfer`，附 DFlash2 companion 权重 |
| 并发 / 逻辑上限 | 2 lane；每路 262,144 token |
| GPU 工作窗口 | 每路 36,864 token，即 576 页 × 64 token |
| 共享 Device KV 池 | 77,824 token，其中包含执行所需增长余量；不等于单路窗口 |
| KV / 投机 / 视觉 | FP8 / DFlash2 草稿 7 / Vision 启用 |
| Host / Prefill | 18 GiB pageable 共享 Host 配额 / chunk 1,024 |
| 缓存 | NInfer Native 前缀复用；HiCache 关闭 |

36K 驻留窗口是 256K 逻辑上限的约 **14.1%**；显存还要容纳权重、State、Graph 与 workspace。这是上下文容量设计，不是“显存降低 85.9%”或“质量无损”的证明。Host 也是共享配额；配置两路 256K 上限，不等于已经证明两路完整 256K 实体历史同时达到峰值。资源压力可能触发缓存淘汰、暂停或 replay。

## 实测收益与边界

最近一次比较采用 **A → B → B → A**：A 为冻结的上一日常构筑（`ae5134c7` + r2 遥测），B 为本页当前构筑。使用相同权重、配置、公开输入、seed，每请求实际发布 256 token，每版本两次重复。两者都配置为 2 lane；表中的单请求是只有一个活跃请求，**不是配置为 1 lane 的实验**。

指标为 **整批实际输出 token ÷ 整批耗时，单位 tok/s**，包含 Prefill、排队、缓存搬运和 Decode。长输入约 54.6K，超过 36K GPU 窗口。下表给出重复范围：

| 负载 | 基线冷态 A | 当前冷态 B | 基线热态 A | 当前热态 B |
|---|---:|---:|---:|---:|
| 短单请求 | 159.1–167.2 | 160.6–167.2 | 72.5–133.2 | 133.2–141.3 |
| 短双请求 | 201.0–214.1 | 203.5–212.8 | 138.3–268.6 | 260.0–275.3 |
| 长单请求 | 23.4–24.1 | 24.0–24.7 | 48.0–72.8 | 66.1–70.6 |
| 长双请求 | 17.2–19.9 | 24.0–24.4 | 83.4–103.1 | 102.4–105.0 |

- **并发有收益，但不保证翻倍。** 当前短双请求冷态总吞吐为 203.5–212.8，短单为 160.6–167.2；负载、验证批次和 Prefill 都会影响结果。
- **前缀命中能减少重复 Prefill。** 本轮长热态复用约 54,624 token；当前长双请求冷态 24.0–24.4，热态 102.4–105.0。这是该案例的整批吞吐差异，包含少算输入和恢复成本，不能当作纯 Decode 提速倍数。
- **此次模块化没有证明普遍加速。** 冷短、冷长单请求基本重合；长双请求记录更好，但只有两次重复，热态又有明显波动，没有消融归因。短双冷态还复用了 29 token 公共前缀，并非严格零命中。
- **质量验收仍有边界。** 四类单请求的重复和跨版本 token IDs 一致；双请求在基线自身重复中也有差异。动态并发逐 token 一致性、正式多轮语义质量及完整双256K峰值尚未证明。稀疏检索也不等于全历史 dense attention。

完整方法、修复、负面结果和资格范围见 [2026-10-10 验证报告](docs/reports/2026-10-10-modular-kvmem-constraints.md)。没有将上游模型测评成绩作为本构筑的质量成绩，也没有把组件 microbenchmark 写成 E2E 提速。

## 稳定性验证

本轮通过：独立无 CUDA core 1/1、CPU/协议及独立 oracle 18/18、GPU 数学/Graph 4/4、真实模型配置 15/15，以及独立 Draft 2020-12 校验的 25 组实际工具参数。真实模型覆盖 none/MTP3/DFlash2 K7 的跨窗口冷/热约束、工具、snapshot/replay/cancel，以及视觉和 Native 事务。

日常全参数构筑还验证了 JSON Schema、strict tool、四帧视频、Responses 父响应删除后子响应续接，以及两条原始 DSH 故障请求并行流式重放。DSH 重放保留原输出上限，只做有界冒烟并取消，没有执行外部工具或生成完整长答案。私有提示词和 SSE 不上传。

## 构建与启动

当前构筑支持 64 位 Linux/WSL2，针对 `sm_120a`，实测 GPU 为 RTX 5090。需要 CUDA、CMake ≥ 3.28、C++20 编译器、Ninja、pkg-config、FFmpeg 开发库及 libcurl ≥ 7.85。本项目本轮使用 CUDA 13.2.86；没有据此验证其他 GPU/系统。

```bash
git clone https://github.com/qzshch/ninfer-kvmem.git
cd ninfer-kvmem
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

模型需另行准备为 v3 `.ninfer`，见 [权重转换](docs/weight-conversion.md)。启用 DFlash2 时，artifact 必须包含兼容的 companion 权重；普通 text-only 或只含 MTP 的文件不能直接启用 DFlash2。

下面是日常参数的**直接 HTTP 服务示例**，使用公开别名 `ninfer-kvmem`；按实际位置替换模型路径。18 GiB pageable Host 是本机已测配置，内存类型和配额应按硬件调整。

```bash
mkdir -p logs
NINFER_WEIGHT_READ_THREADS=2 ./build/apps/ninfer-serve \
  models/qwen3_8_27b_huihui_abliterated_nvfp4_dflash2.ninfer \
  --host 0.0.0.0 --port 8080 --model-id ninfer-kvmem \
  --max-context 262144 --kv-dtype fp8 --kv-capacity 77824 \
  --kvmem-window-pages 576 --max-concurrency 2 \
  --host-context-mib 18432 --host-context-memory pageable \
  --prefill-chunk 1024 --device-state-slots 0 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --vision \
  --log-stats-interval-ms 1000 --pending-timeout-ms 1800000 \
  --request-log-jsonl logs/ninfer.jsonl
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ninfer-kvmem","messages":[{"role":"user","content":"用一句话解释前缀缓存。"}],"max_tokens":64}'
```

API、结构化输出及工具参数见 [Serving](docs/serving.md)；完整选项以可执行文件 `--help` 为准。

## 分支、实验与文档

| 分支 | 用途 |
|---|---|
| [`main`](https://github.com/qzshch/ninfer-kvmem/tree/main) | 当前日常构筑与持续维护 |
| [`HiCache`](https://github.com/qzshch/ninfer-kvmem/tree/HiCache) | 可选 RAM/磁盘缓存实验，与日常构筑分开 |
| [`Native-KVMem-Baseline`](https://github.com/qzshch/ninfer-kvmem/tree/Native-KVMem-Baseline) | 2026-10-08 Native + KVMem 历史对照基线 |
| [`Modular-Integration`](https://github.com/qzshch/ninfer-kvmem/tree/Modular-Integration) | 2026-10-10 模块化与约束接入快照，已包含在 `main` 中 |

HiCache RAM/磁盘扩展不属于当前 `main` 日常构筑。非 CP GDN recurrence 融合、跨 lane 投影打包和 NVFP4 数值舍入实验也没有作为默认优化启用。GDN 输出 RMSNorm + SiLU gating 已在原生产路径融合，与 recurrence 实验不同。旧 `master`、`feature/kvmem` 与 PR 分支保留历史或评审用途。

- [文档入口](docs/README.md)：CLI、Serving、权重转换、测试和维护说明。
- [本构筑实测报告](docs/reports/2026-10-10-modular-kvmem-constraints.md)：资格、ABBA 与未解决边界。
- [KVMem / Native 缓存契约](docs/maintainer/sparse-native-context-cache.md)：选页、搬运、指令保留、Host 配额和恢复。
- [后端模块化与更新方法](docs/maintainer/backend-integration.md)、[上游版本清单](backends/versions.json)。

继承的模型卡与通用指南记录上游能力；本构筑的性能与默认配置以本页及本项目报告为准。

## 上游归属与开源许可

本项目的 NInfer 衍生代码和项目自有改动沿用 **[Apache License 2.0](LICENSE)**。保留原有版权、第三方许可证与 NOTICE；改动来源由 Git 历史、[NOTICE](NOTICE) 和 [上游及许可说明](docs/upstream-and-licenses.md) 记录。

| 来源 | 归属及许可 |
|---|---|
| [NInfer](https://github.com/Neroued/ninfer) | Neroued 与 NInfer contributors；[Apache-2.0](https://github.com/Neroued/ninfer/blob/81c8ce093b2c1646a87566a8e59d807fcf0ec95c/LICENSE)。模型执行、CUDA 与协议基础来自上游 |
| [KVMem](https://github.com/kvmem/kvmem-llama.cpp) / [KVMem-qw3](https://github.com/kvmem/kvmem-qw3) | KVMem 作者与 contributors；固定版本移植仓的 [README 许可声明](https://github.com/kvmem/kvmem-llama.cpp/blob/d9ae944b39f55f77f5434edb96b9fb037217a0a4/README.md#license) 说明按 Apache-2.0 处理。该版本未单独提供根 LICENSE；不将 llama.cpp 的 MIT 泛化为整个 KVMem 的许可 |
| [XGrammar](third_party/xgrammar/README.ninfer.md) | Apache-2.0，保留其 [LICENSE](third_party/xgrammar/LICENSE) 和 [NOTICE](third_party/xgrammar/NOTICE) |
| 其他 vendored 依赖及模板 | 保留各自的许可证；例如 llama-jinja、cpp-httplib、spdlog、nlohmann JSON 的 MIT，详见 [许可清单](docs/upstream-and-licenses.md#第三方组件) |
| 模型与草稿权重 | 独立分发，须分别遵守原模型、消融模型、量化及 companion 权重的 model card/许可；源码许可不替代权重许可 |

KVMem 的研究设计请引用 [KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)，Di Chai、Leye Wang、Zeshen Su、Zhiguo Xia、Zhihang Yu，2026。本项目是独立集成与维护构筑，不代表上游官方发行或背书。
