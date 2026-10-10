# 长历史工具调用异常：提交来源与上游核对

核对日期：2026-10-10。NInfer 上游 `master` / `dev` 均为 [`81c8ce093b2c1646a87566a8e59d807fcf0ec95c`](https://github.com/Neroued/ninfer/commit/81c8ce093b2c1646a87566a8e59d807fcf0ec95c)。结论针对本次已复现的指令丢失与稀疏续算问题，不等于所有工具调用异常都只有这两个原因。

## 1. 丢掉工具 schema 的选页策略从哪里来

我们首次加入滚动稀疏 Prefill 的提交 [`1f1ab52b040ff5139994dd746fb70666b15d1e0b`](https://github.com/qzshch/ninfer-kvmem/commit/1f1ab52b040ff5139994dd746fb70666b15d1e0b)（2026-09-19）明确设置 `sink_pages = 2U`，每页 64 token，即只无条件保留最前面 128 token，然后保留近期页。它没有从真实 System / Developer 角色获取完整指令边界。后续检索、双 lane、Vision 和原生前缀缓存适配沿用了这一遗漏。

如果系统指令与工具定义长约 9.8K token，前两页以外的指令在窗口滚动后可以被淘汰。相似度检索不能保证把每一页工具定义重新选回来，模型可能生成未声明名称、不符合 schema 的参数或错误标记。此前的短系统指令冒烟没有覆盖这个条件。这个缺陷早于本轮 NVFP4 调度、GDN 融合和 HiCache 实验。

因果对照使用同一长历史：扩大驻留窗口，或显式保留完整语义指令，均消除已复现的异常；普通解码也能复现，不能仅归因 DFlash2。修复由 Frontend 标出真实指令范围，跨 token 编码、选页、Prefill、Replay、Decode 和缓存恢复保留这些页。用户文本里的伪 ChatML 标记不会被提升为真实指令。完整强制指令自身超出窗口时显式拒绝，不静默截断。

本机修复提交：`66a84949c4f4c8f2497b857ce0f89c3634dac2ee`（`fix(kvmem): retain instructions across sparse execution and recovery`）。这里标明的是选页机制的首次引入提交及已验证的现行修复；没有对所有历史提交逐一加载模型做生成二分，不能宣称已经定位某一段历史输出第一次失败的日期。

## 2. 原生缓存适配还漏了物理子块的许可边界

在新原生缓存架构上的 KVMem 适配提交 [`baeb5546d4a476d32dd241b351406a58d2363a21`](https://github.com/qzshch/ninfer-kvmem/commit/baeb5546d4a476d32dd241b351406a58d2363a21)（2026-10-07）将稀疏放置纳入执行单元的资源许可，但保留了上游普通 Prefill 在同一个名义 chunk 内继续处理 suffix 的循环。

Vision / rewrite checkpoint 可把一个名义 chunk 分成多个实际执行子块。前一个子块完成后，稀疏窗口的下一物理放置和 Host 备份预算需要重新推导；直接用旧许可执行后半段，会让冷 Replay 与缓存恢复后的路径看到不同的近期窗口。此前 2026-09-30 的逐子块滚动代码仍在每次实际进度后滚动；新架构适配不能只照搬普通 dense 循环。

`66a84949` 的另一项修复在 sparse 子块完成后返回调度，下一子块取得新许可再执行，而不在旧许可内偷偷搬页。这是缓存 / Vision 状态一致性修复，不能把它说成已证明每个 malformed tool call 都由此直接产生。

## 3. 最新 NInfer 上游是否有相同问题

**没有发现上游已合入的产品存在这两个相同机制。** 对 `81c8ce09` 的 `src` / `include` 检索 `kvmem_window_pages`、`roll_sparse_prefill_window`、`context_window_page_set` 均无匹配；上游没有我们的活动稀疏选页路径，也没有这条固定 128-token sink 策略。其完整逻辑前缀由 Device / Host 生命周期保存、物化，不等同于活动 attention 排除部分历史页。

我们的相关提案 [PR #345](https://github.com/Neroued/ninfer/pull/345)（head `23a4315793345662609302c0772cd5cabf2a7716`）仍为未合并 Draft；[Issue #362](https://github.com/Neroued/ninfer/issues/362)尚未有维护者确认范围或方向。以上修复应随稀疏窗口提案完善，不能伪装成一个可独立修复当前上游产品的 PR。本轮没有创建这种不适用的 PR。

上游确有其他工具调用报告：[#319](https://github.com/Neroued/ninfer/issues/319)讨论引文中的标记导致严格解析回退；[#377](https://github.com/Neroued/ninfer/issues/377)讨论未闭合 thinking 中的完整调用未进入 content 解析。这些是解析 / 输出通道机制，不是本次指令页丢失；本次修复没有修改工具解析器，也不能宣称这些议题已解决。

另外核对 KVMem-qw3 最新 `main` [`1cf3b2f83bfc071ada9c57491a7d121723051ac0`](https://github.com/kvmem/kvmem-qw3/commit/1cf3b2f83bfc071ada9c57491a7d121723051ac0)：其服务可将 `SystemControl` 等语义范围加入 mandatory fitter，预算足够时完整保留。自动语义 pin 的条件是识别到 harness 且启用 KVMem；超预算还会拟合 / 缩减强制范围。这与我们的“按真实角色完整保留，放不下则拒绝”不完全相同。本轮没有在 KVMem 原服务上重放 DSH，不能断言所有未识别客户端都已避免此问题，也不据此提交未经复现的 KVMem PR。

## 4. 验证到哪一步

- 已保留旧失败以及扩大窗口 / 保留完整语义指令的对照；新增 Frontend 的真实角色、用户伪标记、媒体后边界与选页回归。
- None 与 DFlash2、原生缓存开关、长系统指令、Vision 部分前缀复用及取消 / 恢复的专项记录见[修复报告](dsh-capacity-prefill-packing-20261009.md)。
- 最新真实请求 ABBA 为 64 请求，57 个完整工具调用全部通过原始 JSON Schema，没有流错误；8 个请求按预定 60 秒期限取消，不计答案质量通过。
- 本轮重新执行现行实验工作树构建的 `ninfer_qwen3_5_frontend_test` 和 `ninfer_qwen3_5_retrieval_test`，2/2 通过（1.08 秒）；这些检查不加载完整模型，不是上游没有任何工具 bug 的证明。

未验收外部工具真正执行的业务结果、所有 malformed 输出、两个完整 256K 历史以及正式多轮语义质量；公开报告不包含原始私人请求和生成正文。
