# 上游来源、修改与许可

本文件是 ninfer-kvmem 的归属与分发说明，不能替代各组件的许可证正文。主代码与项目自有改动沿用 [Apache License 2.0](../LICENSE)，第三方组件保留各自条款和版权。

## NInfer

引擎来自 [Neroued/ninfer](https://github.com/Neroued/ninfer)。当前集成以 `68c54356fd490ab329bd1475d48957f886bb7dd1` 为上游结构基线，选择性适配后续缓存等待与约束能力；本轮复核至 `81c8ce093b2c1646a87566a8e59d807fcf0ec95c`，不是把该 head 全量合并。

上游 [LICENSE](https://github.com/Neroued/ninfer/blob/81c8ce093b2c1646a87566a8e59d807fcf0ec95c/LICENSE) 为 Apache-2.0。本仓保留许可证正文，Git 历史保留作者及提交来源。修改过的项目源文件带有分发修改标注；这些标注不表示所有内容均由 qzshch 创作。上游 cherry-pick 与本地改动的具体归属应查看提交历史及 [版本清单](../backends/versions.json)。

本仓主要扩展 GPU/Host 工作集、检索和指令保留、Native 缓存结合、多 lane 生命周期、投机与视觉适配、遥测、读取准备及可独立构建的 KVMem 策略模块。继承的模型数学、kernels、Graph、协议与约束能力归属于其原作者；README 将本地集成和上游能力分别说明。

## KVMem

本项目参考并适配 [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 的设计和适用机制，复核 master `d9ae944b39f55f77f5434edb96b9fb037217a0a4`、多后端分支 `ae48ec008e77f15aa8518cb85ca35a138f2db8de`。保留 bounded GPU working set、Host 历史和查询检索的来源归属；NInfer 物理状态、checkpoint 和协议接口由本项目适配，没有整体替换为上游另一个 NInfer fork。

固定版本移植仓 [README 的 License 部分](https://github.com/kvmem/kvmem-llama.cpp/blob/d9ae944b39f55f77f5434edb96b9fb037217a0a4/README.md#license) 声明：KVMem-qw3 是 Apache-2.0，该移植按相同条款处理，除非该树后续加入 LICENSE。此次核对的树未提供独立根 LICENSE，因此准确引用 README 声明；不虚构一个已存在的许可文件。[KVMem-qw3 的 LICENSE](https://github.com/kvmem/kvmem-qw3/blob/master/LICENSE) 提供其 Apache-2.0 正文。llama.cpp 保持其自身 MIT 条款；本仓没有打包 llama.cpp 推理运行时。重新吸收上游时，应复核固定版本的许可与 NOTICE。

研究来源：[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)。作者：Di Chai、Leye Wang、Zeshen Su、Zhiguo Xia、Zhihang Yu。

```bibtex
@misc{chai2026kvmem,
  title = {{KVMem}: Virtualizing Million-Token Agent Workspaces on a Consumer {GPU}},
  author = {Di Chai and Leye Wang and Zeshen Su and Zhiguo Xia and Zhihang Yu},
  year = {2026},
  eprint = {2609.04852},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2609.04852}
}
```

## 第三方组件

| 组件 | 本仓保留的许可 / 来源说明 |
|---|---|
| XGrammar | [Apache-2.0 LICENSE](../third_party/xgrammar/LICENSE)、[NOTICE](../third_party/xgrammar/NOTICE)、[固定来源与适配](../third_party/xgrammar/README.ninfer.md) |
| XGrammar / DLPack | [LICENSE](../third_party/xgrammar/3rdparty/dlpack/LICENSE)，来源由 XGrammar 的 README.ninfer.md 固定 |
| XGrammar / picojson | [原头文件内许可](../third_party/xgrammar/3rdparty/picojson/picojson.h) |
| llama-jinja | [MIT LICENSE](../third_party/llama-jinja/LICENSE)、[来源说明](../third_party/llama-jinja/README.ninfer.md) |
| cpp-httplib | [MIT LICENSE](../third_party/cpp-httplib/LICENSE) |
| spdlog | [MIT LICENSE](../third_party/spdlog/LICENSE)、[来源说明](../third_party/spdlog/README.ninfer.md) |
| nlohmann JSON | [MIT LICENSE](../third_party/nlohmann/LICENSE.MIT) |
| utf8proc | [LICENSE.md](../third_party/utf8proc/LICENSE.md)，包含软件及 Unicode 数据的许可说明 |
| Chat templates | [Apache-2.0 LICENSE](../tools/chat_templates/LICENSE) |

根 [NOTICE](../NOTICE) 汇总来源；原组件的许可证、NOTICE 和头文件版权没有被替换。分发源码时应携带这些文件；另行打包二进制时也应附带适用的许可证及 NOTICE。CUDA、系统库和模型权重不因本仓主许可而改变其自身条款。

## 权重与项目身份

模型、Huihui 消融权重、量化来源和 DFlash2 companion 权重独立分发，使用或再分发需分别遵守各来源的 model card/许可。仓内的上游模型卡用于说明来源，不表示本项目重新授予权重许可。

ninfer-kvmem 是 qzshch 维护的集成构筑，不是 NInfer、KVMem 或模型作者的官方发行；保留项目名称用于说明来源，不暗示上游背书。
