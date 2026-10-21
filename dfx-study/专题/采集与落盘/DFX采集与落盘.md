# DFX 采集与落盘

这一专题说明 DFX 为什么要分层采集，以及运行中的记录怎样到达 Host、形成可供分析的原始产物。整体流程见[DFX 业务流程](../../总览/DFX业务流程.md)，各模块细节见[DFX 框架总览](../../总览/DFX框架总览.md)第 2 节。

## 1. 目的与设计逻辑

DFX 要保留足以解释运行问题的证据，同时控制采集对主流程的影响。因此，编译期决定诊断代码是否存在，进程期设定全局策略，单次运行配置决定本轮启用哪些能力及输出位置；不同模块在掌握事实的位置记录，再通过适合自己的路径交给 Host。

记录必须归属到具体运行，并在停止采集后排空、核对产生与丢弃数量。只有文件存在，不能说明本轮证据完整。默认模式在 `run()` 返回前完成本轮应有产物的导出；在 Worker 初始化时启用 `collect_across_runs` 后，支持跨轮保留的模块可以在下一轮设备运行期间后台发布文件，读取前需调用 `Worker.flush_diagnostics()`。

## 2. 各类记录如何落盘

| 路径 | 记录与模块 | 形成的原始产物 |
| ---- | ---------- | -------------- |
| 日志与固定时间槽 | STRACE 直接写 Host 日志；Device phases 由 Host 读回固定槽后写入阶段标记 | 含 `[STRACE]` 标记的日志 |
| 公共 Profiling Framework | Chip Swimlane、PMU、Args Dump、Scope Stats，以及设备侧构图时的 dep_gen | `chip_swimlane_records.json`、`pmu.csv`、`args_dump.json` 与可选的 `args.bin`、`scope_stats.jsonl`、`deps.json` |
| Host 构图直录 | dep_gen 在 Host 构图时直接记录依赖 | `deps.json` |

公共框架只负责持续产生的设备记录如何交接，不解释记录内容。各模块的 Host Collector 按记录类型和运行归属整理、核对并导出。Chip Swimlane 的 AICore 与 AICPU 原始记录在落盘时尚未按任务关联；设备侧 dep_gen 需在确认记录完整后于 Host 重放生成依赖图。

## 3. 公共框架如何交接设备记录

```text
空闲缓冲 → Device 写入 → 待消费队列 → Host drain → 模块 Collector
    ↑                                      │
    └────────── Host 回收缓冲 ──────────────┘
```

一块缓冲在任一时刻只属于 Device 或 Host 一方。Host 消费跟不上时，生产者可能受到背压；长时间无进展才保护性丢弃记录，并计入丢弃数。a2a3 主要通过共享映射读取缓冲，a5 使用 Host shadow 和显式复制。停止时要排空尾部数据，让 Collector 处理最后一批记录，再核对产生、采集和丢弃计数。

这些步骤得到的是原始证据。任务关联、时间转换和指标分析属于[产物处理](../产物分析/DFX产物处理.md)。
