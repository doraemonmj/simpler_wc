# 中等规模 paged attention 的 L0 端到端尝试（2026-10-26）

用例在 `tests/st/a5/tensormap_and_ringbuffer/dfx/l0_onboard/test_l0_paged_attention.py`，沿用现有 batch paged attention 的四个真实计算 kernel（QK、softmax、PV、online update）和 golden。参数为 batch 4、head 16、head dim 128、block size 128、context 512，K/V 输入各 512 KiB；runtime 实际产生 16 个不同 task ID。这比此前只写 108 个 float 的 MIX 标记用例更接近真实负载，同时适合这台 EVB 做限时验证。

四个专用 kernel 在入口和出口按函数分别发出 region 100/101、102/103、104/105、106/107 的 L0 标记。普通 PA kernel 仅增加受 `SIMPLER_A5_ONBOARD_L0` 宏控制的空操作 hook，未打开宏时不改变计算。L0 专用源文件定义该宏并包含原 kernel 源，保持 `kernel_entry` 在 ELF `.text` 起始处。最初的外层 wrapper 使入口不在 `.text` 起始处，编译后被 loader 校验拒绝；已改正，并以普通 onboard 基线验证。

| 受限运行 | 结果 | 诊断产物 |
| --- | --- | --- |
| `task_20261026_110558_744100`，`baseline` | 普通 PA ST + golden **通过**（10.49 s）；chip swimlane 有 16 个不同 runtime task ID、16 条核任务记录，单核最多 5 条 | [baseline](../../../src/a5/platform/testdata/l0_onboard/pa_medium_20261026/baseline) |
| `task_20261026_110733_746288`，`capture` | PA 计算与 golden **通过**（7.15 s）；L0 的 `prof_drv_start(17)` **失败**，没有 PA 的 L0 原始数据 | [capture_attempt](../../../src/a5/platform/testdata/l0_onboard/pa_medium_20261026/capture_attempt) |

两次运行都经 `task-submit --device 0 --max-time 180` 获取设备锁，无设备或服务器 reset。第二次的 `l0_status.json` 记录 `capture=unavailable, stage=prof_drv_start, channel=17`。channel 17 已用于先前的全核 L0 捕获；这台驱动此前还在其他 channel 观察到成功 `prof_stop` 后无法二次 `prof_drv_start`。本次失败与其一致，但没有证明根因。所有已知 BIU channel 11–28 均已在前面实验中使用；不能把其他类型的 profiling channel 当作 L0 BIU channel 尝试。

`simpler_setup.tools.l0_task_join` 已增加多 task 归属规则：同一物理子核的 PA 入口/出口 region 必须完整成对，并与该逻辑核的 runtime task 记录数相同、顺序无重叠，才按核内执行顺序标为 `ordered_marker_pair`；不完整则不给 task ID。这个规则已用离线双任务数据测试，**尚无真实 PA L0 数据验证**。capture 用例现已加入 L0 产物断言；再遇到 channel 启动失败时，测试会明确失败，不会只因 golden 正确而显示为完整 L0 通过。

要完成真实 PA L0 泳道，需要先解决 BIU channel 可重开问题，或获得新的可用 BIU channel。随后再跑 `capture` 并检查 raw、marker 配对、task ID 和 golden，不能用此前单任务 L0 raw 替代 PA 数据。

当前端到端链路已在代码中接通，但只有单任务 MIX 用例验证过完整采集和关联。PA 的 `capture` 用例在 channel 启动失败时会因缺少 L0 产物而失败；其 golden 通过不能代表 L0 通过。每个物理子核分配 1 MiB，自管 GM 总量约 108 MiB；完成后按写指针保存原始字节，再离线关联任务。写指针小于采集起点、超过缓冲容量或不对齐时会报告异常，但若缓冲区写满后硬件回卷或覆盖旧记录，当前版本没有可靠的溢出证明，不能保证长时间、大量标记的采集完整性。重启后先验证 channel 能否重新启动，再用 PA capture 检查完整链路和数据量；在确认回卷语义前，结果应限于未接近容量的短运行。
