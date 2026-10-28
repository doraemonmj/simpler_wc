# 借用设备的逐次 DFX：A2A3 真机记录

验证日期：2026-10-22；本地工作树提交：`52665dc95`。这份记录只对应公开 `Worker` 的 **native eager** 路径，不证明 #2202 的专用 Kernel capture/replay 或 close 导出合同，也未在当前 main 重新运行。

[验证脚本](验证脚本/probe.py)使用 `Worker.init(borrowed_context_generation=...)` 借用调用方 context，通过 `Worker.submit(..., caller_stream=...)` 提交同一个 callable。每次检查计算结果、DFX 文件与 `run_epoch`，关闭后再检查调用方 context、stream 和显存仍可使用。[Program 对照脚本](验证脚本/owned_pmu_control.py)只用于异常对照。

| 场景 | HBG | TMR |
| ---- | --- | --- |
| DFX 关闭；关闭后调用方资源 | 通过 | 通过 |
| Chip Swimlane + dep_gen | 两次调用各产出泳道与依赖图，epoch 正确 | 同左 |
| Args Dump `full` | 3 条参数记录，128 字节 payload | 同左 |
| PMU | 1 行，epoch 正确 | 最终完整序列通过 |
| Scope Stats | 不适用 | 3 行 JSONL |

最终完整重跑：`task_20261022_205144_25798302151`，设备 0，HBG 连续 6 次、TMR 连续 7 次调用均通过，退出码 0；产物分别在 `/tmp/simpler-kernel-dfx-user-20261022/final/host_build_graph/` 和 `/tmp/simpler-kernel-dfx-user-20261022/final/tensormap_and_ringbuffer/`。每个作业都在 `task-submit` 分配设备后执行架构预检。

一次 TMR `Args Dump → PMU` 序列在设备 1 出现 AICPU 异常（`task_20261022_204428_23299011558`、`task_20261022_204659_242227416334`）；后续同设备对照及最终完整重跑均通过，尚未稳定复现。它不能直接归因于 #2202。

复现时从仓库根目录进入 `task-submit`，先运行 `.claude/skills/onboard-arch-precheck/check.sh a2a3`，再运行 `.venv/bin/python dfx-study/专题/运行模式/验证脚本/probe.py --runtime host_build_graph --device "$TASK_DEVICE" --output <新目录>`；TMR 改为 `--runtime tensormap_and_ringbuffer`。设备查询、预检和脚本必须留在同一分配作业内。
