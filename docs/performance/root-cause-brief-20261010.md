# 本轮根因结论与社区核对

**修复与实验分开交付。** 日常服务恢复了语义指令保留、稀疏物理分块滚动和双路配置；NVFP4打包代码已编入r2服务，但参数默认OFF、当前也OFF。非CP GDN融合、HiCache继续OFF，舍入实验没有装入服务。下面区分已验证机制和仍需调查的原因，详细数字在[完整诊断](nvfp4-dynamic-and-rounding-20261010.md)。

## 目前能确认什么

| 问题 | 本机因果证据 | 社区如何理解 | 处理结果 |
|---|---|---|---|
| DSH长请求被Host容量拒绝、另一请求等待 | 36K是GPU工作窗口，256K完整历史及未来生成仍要Host预算；旧Host池/重复参数不能保证两个大输出预算同时准入 | KVMem同样区分工作集与历史容量，不能仅增大logical context | 日常Host18GiB pageable，输出预算未缩短；两个原始请求经两级代理重放，3个双路产出区间、无返回错误。两个满256K物理历史仍未验收 |
| 长历史工具调用异常 | 旧128-token sink丢掉实际System/Developer/工具schema；稀疏Vision/replay改写物理页后，后续子块需重新滚动映射 | KVMem强调mandatory system/current-query；旧NInfer搜索时限导致reprefill是另一机制，当前代码不能照搬旧补丁 | 保留真实语义角色，不把User里的伪ChatML升级为指令；滚动许可按实际物理子块处理。真实重放的2个完整工具调用，名字合法、参数通过原始JSON Schema；不执行工具 |
| 为什么NVFP4打包没提高长Prefill | 48K×2逐kernel中gate/up少59ms，down多98ms、down量化多12ms；attention/GDN仍逐路，成本近似相同 | [NInfer #263](https://github.com/Neroued/ninfer/issues/263)的融合已用于本机1024/2048；[#277](https://github.com/Neroued/ninfer/issues/277)的短输入增块结果不能推广到所有长请求 | None/DFlash2各完成52批受限ABBA，长C2耗时+0.87%/−0.36%；没有可确认的整机提速，默认关闭 |
| GDN融合为什么更慢 | 原工作区主要命中L2、DRAM未饱和；续算两路线都1 CTA/SM，active warps反而增大，不能说occupancy下降；准备/同步/求逆算术同时改变 | 融合减少kernel/物化并不保证本硬件快，也不自动保持浮点关联顺序 | 单路48K TTFT+2.15%、整请求+2.92%；没有已验证正收益条件，非严格旧输出等价，默认关闭 |
| DFlash拒绝是不是selector算错 | Huihui/原版各84位置，实际条件选择都恰好等于独立CPU FP64 argmax；Huihui四首拒绝中一处条件项翻转unary，其余三处错误候选本来就获更高unary分数 | [官方机制](https://inco.ai/blog/dflash2/)是unary加前驱条件项；正确token在top16不保证被选中。suffix decay指K块内后段，不等于总上下文效应 | 本组定位为预测/条件打分不匹配，未发现selector实现错误；原版也发生，不能仅归因Huihui |
| 接受率高是否就与ordinary一致 | 两目标权重都捕捉到已接受token与同前缀ordinary不同；固定NV投影A16恢复ordinary，固定草稿接受反而减少 | [NInfer #265](https://github.com/Neroued/ninfer/issues/265)也报告对齐方案随权重有正负接受收益，不支持“更高精度必然更快” | 接受数、目标输出一致性和实际吞吐分别验收，不用单一接受率替代质量 |

## 三个仍不能断言的原因

1. **down GEMM的最终硬件原因。** 连续算子及真实存储权重控制没有复现整机退化。连续synthetic组件NCU显示down激活量化的宽批DRAM成本变大，但down矩阵L2命中与总读量改善。新的真实模型单遍采样也显示DRAM读量减少、时钟较低，尚未得到实际L2数据。实际down与本进程memcpy时间交集为零，不能简单归为L2容量不足或KVMem同时复制争带宽。下一步需固定真实activation、workspace布局、前序工作集和时钟，按同一实际kernel跨层比较。
2. **Huihui对长期接受率的净影响。** 两模型本轮自己生成的teacher轨迹不同，六前缀接受数不能量化消融影响；已有证据只说明同类拒绝/数值路线问题也出现在原版。下一步需要固定token前缀，并分离head、量化、penalty与随机接受，保留官方配置区别。
3. **完整容量与正式质量。** 48K测试通过不代表两个256K完整历史峰值通过；byte-exact有限夹具不代表全体输出严格无损。残差舍入实验有可重复数学截断反例，仍隔离，不能晋升默认。

社区另有 [SGLang #39254](https://github.com/sgl-project/sglang/pull/39254)：packed草稿`fc`被普通Linear漏载、保持随机初始化，作者和另一用户报告恢复接受；核对时仍Open。当前NInfer Binder/Materializer对缺失、形状及完整上传会硬报错，没有同一静默随机路径。排除这一机制并不证明本机所有转换权重都正确，也不意味该PR可直接移植。

## 避免误读性能面板

- 打包请求的`prefill_seconds`记录共同经历的整批时间，不能把两路相加当GPU成本；可相加的work timing按token分摊，native断言其和等于整批。
- 优化对照用整批实算token/墙钟、末路TTFT和完整GPU时间线。SSE事件间隔不是物理Prefill chunk耗时，也不是decode内核耗时。
- 旧packed路径缺外层Prefill NVTX，现已补齐；此前不完整范围的0.771对9.301秒不是90%提速。
- 正常JSONL已恢复，两级面板都收到新数据、自动显示2lane，解析错误0。闲置后`jsonlFresh`变false并不表示日志文件没开，活动请求已验证能更新。

## 日常构筑与验证边界

2lane，每路36864-token窗口、262144逻辑上下文，FP8、DFlash2 K7、Vision、Native prefix cache ON、Host18GiB pageable、prefill_chunk1024。源码/二进制选择舍入实验之前的r2，独立实验仍在分支中。

原两个故障请求保留238904/238874输出预算，经NAS→Windows→WSL重放，两路产生输出且无返回错误；取消后槽归还。完整工具调用通过原始schema，红色图片与并行文本通过。上述是有保护的功能重放，不是完成全部长答案、执行工具、性能A/B或满256K容量证明。

10月10日新增历史DSH请求对照：同一r2、Host18GiB、原始请求体/大输出预算，较小请求对OFF/ON/ON/OFF共16请求全部完成，15个完整工具调用全部通过原始JSON Schema。冷Prefill GPU工作耗时约+0.41%、末路TTFT约+1.01%，没有可确认提速；热缓存只计算14token、命中32990token，四臂均未打包。请求原本缺seed，每次由服务随机生成，冷整批墙钟−11.57%与热整批+9.60%都伴随不同输出长度，不能归为打包收益或退化，也不能作为输出无损证明。档案中109个DSH形状请求只有11个body小于128KiB，且用户/测试来源未完全标记，尚不能代表“日常短请求占多数”。

完整八请求及六历史请求的首次重放都因旧12GiB commit线提前停止，失败记录保留。首次触发时commit余量11.92GiB、可用物理23.45GiB、显存1604MiB、swap0，没有OOM证据。用户指出过于保守后，测试/日常守护改成12GiB预警、8GiB硬停止；运行期如最近2～5秒增长速度预测5秒后余量低于4GiB则提前停止。有限启动上传不作持续增长外推，启动准入及全部硬线仍保留。物理8GiB、显存768MiB、WSL可用1.5GiB/swap256MiB及磁盘线不变。10个压力决策及启动边界检查通过；忽略TERM的测试目标/采集器14.61秒内精确回收，另一睡眠进程保留。

原八请求计划补测已完成四臂、32批、64请求，56正常完成、8按预先设定60秒期限取消，57个完整工具调用全部通过原始schema，没有流错误或新硬线触发。最低commit余量9.006GiB、物理21.739GiB、显存2676MiB、swap0；862次资源采样中124次低于旧12GiB线，因此放宽确实恢复了此前被保护提前切断的测试覆盖。两组实际打包的冷请求，GPU Prefill工作耗时分别+0.83%/+0.79%、末路TTFT+1.31%/+0.21%，仍没有确认提速。其他首轮/热轮均pack=0，TTFT观测−23.06%到+5.34%不作为打包效果；其中首轮第三对已经自然命中20665token，“cold”仅是首次重放标签，不能称完全无缓存。原请求缺seed、输出长度变化，以及8次未完成长答案仍限制吞吐与质量结论。

早期实际模型NCU采集首次启动失败，ASCII路径的第二次已进入真实down kernel，但显存降到767MiB、采集停滞；TERM回收超过旧超时，现已修复有界KILL及退出状态竞争。该时段后WSL uptime重置、journal报告非干净关闭，原因尚未确定；没有主动发出WSL shutdown/reset命令，不能把它写成已确认引擎OOM。[NVIDIA说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)指出多遍kernel replay会保存/恢复可访问内存，单遍不需要。后续先用小组件确认单遍，才用于大模型，已经取得真实时钟/Tensor及DRAM读量。组合访存需要五遍、L2命中率单独也需两遍，均在小组件处拦住，未取得真实L2数据。

同一安全r2、48K×2/None/FP8/Vision/Host4/NativeOFF，在一个第0层down采样：1024-token为133.38µs/2.500GHz/Tensor active78.63%，2048-token为272.54µs/2.417GHz/79.59%。相对**2×单个1024样本**耗时+2.17%，时钟−3.31%，duration×frequency约−1.21%；另一组DRAM采样由2×60.68MB降至73.08MB（−39.78%），耗时仍+1.92%。它支持本采样中读量确实减少、乘加工作并未减少、轻微退化与时钟变化吻合；不能称原整trace的+98ms已全部归因于降频，不能仅由Tensor active推出实际FLOPS利用率。每设置/组一次、前序prompt frontier不同、未锁时钟，全层L2/等待/功耗限频仍有缺口。OFF首次收尾还暴露启动器实际`/bin/sh`与旧image记录不符，报告离线恢复、失败保留；记录真实启动器后，ON及DRAM采集/回收通过。日常服务最终恢复，实验仍OFF。
