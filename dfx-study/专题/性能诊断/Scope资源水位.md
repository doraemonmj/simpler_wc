# Scope Stats：Scope 资源水位

Scope Stats 回答“哪个 scope 使任务窗口、堆、依赖池或 TensorMap 接近容量”。它目前由 `tensormap_and_ringbuffer` runtime 的 scope 调用点产生记录；其他 runtime 即使接受开关也不产生 scope 样本。详细定义见[原设计文档](../../../docs/dfx/scope-stats.md)。

## 1. 在哪里采集

开启 `--enable-scope-stats` 后，每个 `SIMPLER_SCOPE` 进入时调用 `scope_stats_begin`，退出时调用 `scope_stats_end`。运行时先读取本 scope 所在 ring 的任务窗口 head/tail、堆 head/tail、依赖池 top/tail，以及全局 TensorMap 当前使用量，再把这些整数交给平台 Collector；容量则在初始化时注册。退出快照取在 scope 结束清理前，避免把清理后的低占用误认为 scope 的结束状态。采样位置见 [Orchestrator](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/orchestrator.cpp)；[平台接口](../../../src/common/platform/include/aicpu/scope_stats_collector_aicpu.h)只接收数值，不读取 runtime 内部结构。

```text
进入 scope：快照 begin ── scope 内分配与释放 ── 快照 end：退出 scope
       ↓                                                ↓
             设备缓冲 → Host Collector → scope_stats.jsonl
```

## 2. 采了什么，为什么需要

`<output_prefix>/scope_stats/scope_stats.jsonl` 首行是本轮元数据，后续每行是一个 `begin` 或 `end` 样本。

| 数据 | 用途 |
| ---- | ---- |
| `site`、`phase`、`depth`、`ring` | 把进入/退出样本归到源码位置和 ring，定位压力来自哪个 scope。 |
| `task_window_start/end`、`heap_start/end`、`dep_pool_start/end` | 保存各资源的累计分配前沿和释放边界，用两个边界算净占用及 scope 内分配量。 |
| `tensormap` | 记录全局 TensorMap 在这个边界的当前使用项数；它不是按 ring 分配的。 |
| 首行的 `*_max`、`total`、`dropped`、`fatal` | 用本轮真实容量计算使用比例，并判断记录是否完整、运行是否出现 fatal。 |
| `run_epoch`、`buf_seq` | 区分运行与缓冲批次，帮助关联和核对。 |

对按 ring 管理的资源，令 `head` 为分配前沿、`tail` 为释放边界：

- `end.head - begin.tail`：scope 高水位指标，包含进入时积压及 scope 内全部分配；它不是逐时刻真实峰值，中途释放也不扣除。
- `end.head - end.tail`：退出时实际仍占用的量。
- `end.head - begin.head`：这个 scope 内的总分配量。

## 3. 怎样读结果

[ScopeStatsCollector](../../../src/common/platform/shared/host/scope_stats_collector.cpp) 排空样本并导出 JSONL；[`scope_stats_plot.py`](../../../simpler_setup/tools/tmr/scope_stats_plot.py) 配对 begin/end，按资源与 ring 画报告。先找接近容量的资源及对应 `site`，再比较“高水位”“退出占用”“分配量”：高水位高而退出占用低，可能是 scope 内吞吐高、释放也快。采样只在边界，不能还原每时刻占用曲线；若 `dropped > 0`，报告可能漏掉真正的高压 scope。
