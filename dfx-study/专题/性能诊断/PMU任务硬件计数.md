# PMU：任务硬件计数

PMU 回答“核函数执行时，计算流水线、访存或缓存发生了什么”。泳道图能看到任务耗时，却不能解释核内为何慢；PMU 把硬件事件归属到单个任务。完整事件表和架构差异见[原设计文档](../../../docs/dfx/pmu-profiling.md)。

## 1. 在哪里采集

Host 为本轮选择一组 PMU 事件并分配缓冲。AICPU 初始化时给每个核配置事件选择器。AICore 在核函数前打开计数、结束后关闭计数，使计数窗口尽量对应核函数执行，而不包含空闲轮询。任务完成时形成一条记录：a2a3 由 AICPU 读取该核计数器；a5 由 AICore 读取计数器到暂存槽，再由 AICPU 在完成通知处核对任务并提交。实现见 [a2a3 AICPU](../../../src/a2a3/platform/shared/aicpu/pmu_collector_aicpu.cpp)、[a5 AICore](../../../src/a5/platform/include/aicore/pmu_collector_aicore.h)。

```text
Host 选事件 → AICPU 配置 PMU → AICore 核函数窗口计数
                                 ↓ 完成时读取并关联任务
                         设备缓冲 → Host Collector → pmu.csv
```

## 2. 采了什么，为什么需要

`<output_prefix>/pmu.csv` 一条记录对应一次任务执行。公共列及含义：

| 数据 | 用途 |
| ---- | ---- |
| `task_id`、`func_id` | 找到是哪次任务、哪个核函数，便于与泳道图关联。 |
| `core_id`、`core_type`、`thread_id` | 知道执行核类型和负责它的调度线程；不同核不能混成一个总数。 |
| `pmu_total_cycles` | 给同一计数窗口的事件量提供周期参照。 |
| 按事件组命名的计数列、`event_type` | 解读流水线忙碌、访存请求、缓存命中或资源冲突；列名和含义由架构及事件组决定。 |
| `run_epoch` | 区分同一进程中的不同运行轮次。 |

`--enable-pmu` 默认选流水线利用率事件组（值 `2`）；例如改为 `--enable-pmu 4` 采访存组。一次运行只配置一组事件，想对比其他组需要在相同负载下另跑一轮。a2a3 有 8 个事件计数槽，a5 有 10 个，不能按列的位置跨架构解释；应查各架构的事件定义。

## 3. 怎样读结果

同一任务的流水线忙碌计数偏低，可以继续排查访存或等待；访存/缓存事件偏高，可以定位相应核函数，而非把整轮运行误当成一个热点。PMU 计数描述硬件事件，不能直接给出瓶颈原因或任务间等待时间；任务时序需结合[泳道图](芯片泳道任务时间线.md)。

设备缓冲由 [PmuCollector](../../../src/a2a3/platform/shared/host/pmu_collector.cpp) 持续接收，运行结束时合并 CSV 并核对产生、收到和丢弃的记录数。跨轮后台采集时，读取前调用 `Worker.flush_diagnostics()`；文件缺失或计数丢失时，不应把现有行当成完整样本。
