# DFX 产物处理

这一专题说明原始产物为什么还需要处理，以及仓库工具怎样把它们整理成诊断结果。采集和落盘过程见[DFX 采集与落盘](../采集与落盘/DFX采集与落盘.md)，各工具的完整处理链见[DFX 框架总览](../../总览/DFX框架总览.md)第 3 节。

## 1. 目的与处理逻辑

落盘文件保存的是分散的运行事实。仓库工具按任务、运行和时间信息关联这些记录，再计算耗时、依赖、资源水位等指标，生成便于检查的表、报告或时间线。复杂的关联与归因放到离线阶段，避免增加运行时采集路径的负担。

分析前需确认文件属于目标运行、记录是否完整、任务标识能否关联，以及时间戳是否在可比较的时钟域。跨轮后台导出时，读取前调用 `Worker.flush_diagnostics()`。记录缺失、时钟未对齐或采集开销明显时，图中的空档不能直接作为性能结论。

## 2. 各类产物如何处理

| 要看什么 | 输入 | 工具与结果 |
| -------- | ---- | ---------- |
| 阶段耗时 | 含 `[STRACE]` 的日志 | `strace_timing.py` 生成阶段树或耗时表；`hbg_bind_phases` 和 `phase_time_split.py` 进一步分析 Host 阶段。 |
| 任务时间线 | `chip_swimlane_records.json`，可选 `deps.json`、`name_map*.json` | `swimlane_converter.py` 关联 AICore 与 AICPU 记录并生成 `merged_swimlane.json`；`sched_overhead_analysis.py` 和 `critical_path.py` 分析调度开销与关键路径。 |
| 任务依赖 | `deps.json` | `deps_viewer.py` 生成可检索文本或依赖图；`wait_reduction_sim.py` 生成 WAIT 边约简的模拟报告。 |
| 参数与核内行为 | `args_dump.json`、可选 `args.bin` | `dump_viewer.py` 检查参数；`core_swimlane.py` 结合 SceneTest 文件与 Dump 参数生成单任务模拟重放和核内 trace，需具备相应模拟器环境。 |
| 资源水位 | `scope_stats.jsonl` | `scope_stats_plot.py` 生成资源高水位 HTML 报告。 |
| 硬件计数 | `pmu.csv` | 当前仓库没有专用离线分析脚本，需结合相应架构的事件定义解释 CSV。 |

工具的参数和输出格式见 [simpler_setup/tools/README.md](../../../simpler_setup/tools/README.md)。

## 3. 读结果时注意边界

- 时间线说明任务何时执行；任务间空档的原因还要结合依赖和资源等记录判断。
- dep_gen 采集可能扰动提交时序。分析调度开销时，通常复用同一计算图另一次运行的 `deps.json`。
- Scope Stats 的高水位用于定位资源压力，不等于逐时刻的实际占用曲线。
- Core Swimlane 是单任务隔离模拟重放，不包含 AICPU 调度和跨任务等待。
