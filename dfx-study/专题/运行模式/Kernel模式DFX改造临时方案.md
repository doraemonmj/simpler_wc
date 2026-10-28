# #2202：Kernel 模式 DFX 现状与剩余工作

核对基线：上游 main `54cff755b`（2026-10-26）；集成分支代码参考 `17e342455`。[Issue 正文](https://github.com/hw-native-sys/simpler/issues/2202)和[负责人交接评论](https://github.com/hw-native-sys/simpler/issues/2202#issuecomment-6330596304)定义的是 **专用 Kernel 模式的 DFX 生命周期与导出合同**。旧源码行号是历史锚点，不能用来判断当前 main 已支持。

## 已有实现

| 路径 | 已确认状态 |
| ---- | ---------- |
| main 的 borrowed native eager | `prepare_run`/`launch_run_on_stream` 逐次执行 DFX admission 与收尾。公开 Worker 路径已有测试；本地 A2A3 验证见[记录](Kernel模式DFX用户路径真机验证.md)。这不能证明专用 Kernel 图路径。 |
| main 的专用 Kernel launch | `simpler_kernel_mode_supported()` 返回 0；`simpler_kernel_mode_launch()` 拒绝调用。 |
| 集成分支 | [#2311](https://github.com/hw-native-sys/simpler/pull/2311)、[#2327](https://github.com/hw-native-sys/simpler/pull/2327)、[#2329](https://github.com/hw-native-sys/simpler/pull/2329)已合入 **`feat/kernel-mode-integration-test`**：Chip Swimlane/dep_gen、`begin_dfx`/`end_dfx` 窗口、launch 边界和多次 launch 的泳道拆分。显式 `end_dfx` 导出两通道；正常 close 的未结束窗口兜底只明确导出 Chip Swimlane，未调用 dep_gen 导出。它们尚未合入 main。 |
| 待合入的采集修复 | [#2493](https://github.com/hw-native-sys/simpler/pull/2493)修复 HBG 的 DFX 初始化和多 callable 采集，目标仍是集成分支。PR 报告 A3 三个 profiling step、每 step 两次 launch 的结果；PR 仍 open，A5 未做真机验证。 |
| 待合入 main 的专用执行 | [#2647](https://github.com/hw-native-sys/simpler/pull/2647)现在是整合 #2626/#2646 的单个 HBG 合入 PR，只开放 A2/A3 专用执行，明确拒绝该路径 diagnostics；[TMR capture PR](https://github.com/hw-native-sys/simpler/pull/2625)也明确 diagnostics disabled。两者仍 open。 |

main 的 collector 对象可跨 run 复用；设备缓冲区按形状在 native run 的 `arm_collectors_for_run()` 中初始化或重建，`start_shared_collectors_for_run()` 按 `run_epoch` 做采集准入，`teardown_shared_collectors_after_run()` 按执行完成证据收尾。图 replay 不会重新经过这套 host 逐次流程，且图仍引用设备地址时不能按形状重建并释放缓冲区。因此应复用 collector 的实现，给专用 Kernel context 建立捕获前准备与会话所有权，不能直接调用整套逐次 run 流程。

集成分支已有的采集窗口和导出逻辑值得复用，但其 collector 与当前 main 的按 run epoch 保留、异步写出实现不同。不能直接照搬旧分支的 `quiesce()`/导出代码，也不应重新实现相同的边界拆分。

适配时的代码落点：main 的 `simpler_kernel_mode_init()` 进入 `DeviceRunnerBase::init_kernel_context()`，此时尚未初始化 DFX 缓冲区。集成分支在 context init 中按静态配置初始化泳道/dep_gen 区域，在 `DeviceRunnerBase::begin_kernel_dfx()` 启动采集窗口，在 `end_kernel_dfx()` 排空并导出；A2A3 的 `finalize_collectors()` 只对未结束窗口的泳道做兜底。新执行路径如果直到 callable 准备阶段才知道缓冲区尺寸，就必须在该阶段、捕获前完成初始化。相关实现分别在 [C API](../../../src/common/platform/onboard/host/c_api_shared.cpp)、[公共 runner](../../../src/common/platform/onboard/host/device_runner_base.cpp)、[A2A3 runner](../../../src/a2a3/platform/onboard/host/device_runner.cpp)。

## 第 2 项：main 上可复用资源的边界（A2/A3）

`DeviceRunnerBase` 持有 collector 对象，A2A3 runner 用 `mem_alloc_` 分配并注册它们的设备区。`arm_collectors_for_run()` 按 `{AICore 数、AICPU 线程数、launch 线程数}` 检查形状；不匹配时可能调用 `finalize_collectors()` 释放旧区，再执行各通道的 `init_*`，把地址写入本次 `KernelArgs`。所以 **collector 对象常驻不等于设备地址永久稳定**。图还引用这些地址时，Kernel context 必须禁止重建或持有能阻止释放的资源令牌。[A2A3 初始化与释放](../../../src/a2a3/platform/onboard/host/device_runner.cpp) · [形状状态](../../../src/common/platform/onboard/host/device_runner_base.h)

| 通道 | 可复用的初始化及设备地址 | 当前产物出口与 Kernel 路径注意点 |
| ---- | ------------------------ | -------------------------------- |
| Chip Swimlane | `init_chip_swimlane()`；记录区、rotation table、按 run 取得的 terminal bank | `teardown_shared_collectors_after_run()` 的普通导出或 retained writer；replay 的边界和 bank 不能假设 host 每次重新分配。 |
| Args Dump | `init_args_dump()`；dump 区、terminal bank | `close_args_dump_run_boundary()`；retained 失败会保留 backing，不能在 close 中无条件释放。 |
| PMU | `init_pmu()`；PMU 区、terminal bank | `close_pmu_run_boundary()`，另有设备 PMU lease；采集窗口需与 lease 一同结束。 |
| dep_gen | 非 HBG Host 图路径用 `init_dep_gen()` 建设备记录区 | 设备图在 `emit_device_dep_gen_graph()` 输出；HBG Host 图在 bind 阶段由 `emit_host_dep_gen_graph()` 输出，须分别核对，不能把它当成通用 teardown 的一部分。 |
| Scope Stats | `init_scope_stats()`；统计区、terminal bank | `close_scope_stats_run_boundary()`；Host HBG 与设备记录的产物内容不同，需按 runtime 验证。 |
| device wall | `ensure_device_wall_buffer()` 建地址并追加初始化镜像，run 完成时 `read_device_wall_ns()` | main 已把初始化镜像安排进 launch metadata；专用图是否捕获这次发布、每次 replay 是否重置，须在目标执行路径验证。 |

现有 `start_shared_collectors_for_run()` 使用 `run_epoch`、`run_begin()`/`begin_run()` 和 terminal bank 做本次准入；`teardown_shared_collectors_after_run()` 依赖设备完成证据收尾。**这两函数不能在 Kernel init/close 用一个假 run 参数直接调用。** main 的 `simpler_kernel_mode_init(config)` 目前只验证并消费配置来建立执行拓扑，没有把 DFX 配置保存为 context 会话状态。专用路径要明确配置、输出目录、collector 区、bank 和后台写出器的所有者，且仅在图不再引用设备区、设备完成已证明后释放。[当前 init](../../../src/common/platform/onboard/host/c_api_shared.cpp) · [逐次准入/收尾](../../../src/common/platform/onboard/host/device_runner_base.cpp)

| 生命周期 | Kernel context 需要持有或阻止提前释放的资源 |
| -------- | ------------------------------------------- |
| 捕获前至最后一张图销毁并释放令牌 | DFX 配置、稳定的设备记录区及映射、图内会读取的 `KernelArgs`/terminal bank 地址；形状变化不能触发 `finalize_collectors()` 释放这些地址。 |
| 采集窗口开始至最后一次 replay 完成 | collector 线程、计数/边界状态，以及启用 PMU 时的设备 lease；窗口结束后核对记录并交给现有导出器。 |
| 导出完成至 context close | 等待后台写出器完成并处理失败，再调用 `finalize_collectors()`；之后 `finalize_common()` 才能释放 persistent args、内部 stream/事件和 allocator。 |

[预热 PR #2633](https://github.com/hw-native-sys/simpler/pull/2633) 已提议把五通道的幂等初始化与逐次 `KernelArgs` 地址发布拆开，并在 `Worker.init` 预热；它仍 open，且相对当前 main 有冲突。此改动与缓冲区初始化整理重叠，后续应先核定是否复用其拆分，再补 **Kernel context 的配置保存、图资源持有、replay 边界和 close 导出**，避免重复抽取同一批 `init_*`。

## 应满足的合同

1. **初始化与捕获前：** 复用现有 collector，在配置与尺寸确定后准备缓冲区、线程和稳定设备地址；这些动作可以在 context init 或后续准备阶段完成，但必须早于捕获。确保图执行读到正确的 DFX 开关与地址；`FREEZE` 时不拆掉图会引用的资源。
2. **多次 replay：** 每次设备执行能产生正确归属的记录。replay 不会重新调用 host 的 `begin_run`；容量、丢弃统计和重复 task ID 必须有明确处理。`device_wall` 的逐次 host 复位是否需要移到图内，仍须按最终实现验证或限定语义。
3. **结束与关闭：** 显式 `end_dfx` 或 context close 在已证明设备工作完成后排空、核对并导出；图仍引用缓冲区时不能释放。集成分支的 close 兜底缺少 dep_gen 导出，且导出失败的传播与重试仍需核定，不能把现状当作完整合同。

## 最小实施与验收顺序

1. 以实际合入 main 的专用执行代码为基线，逐项对照集成分支的 DFX 初始化、窗口、边界、导出和 #2493 修复，列出需要适配的差异。先确定 A2/A3 HBG 与 TMR 各自支持范围；A5 单列。
2. 优先贯通已有的 Chip Swimlane/dep_gen：捕获真实 callable，多次 replay，结束窗口或销毁图并 close，检查文件、记录归属、丢弃数、重复 task ID 和调用方资源。覆盖未结束窗口的 close 兜底及失败重试。
3. 再决定 Args Dump、PMU、Scope Stats、`device_wall` 的接入或明确暂不支持。核对 Program/native eager 回归；没有公开接口端到端证据前，不关闭 #2202。

原本单独的图捕获长文已并入此处；历史内部 stream 探针只能证明捕获可行性，不能代替上述 DFX 验收。[捕获探针](../../../tests/st/a2a3/kernel_capture/test_kernel_capture.py)
