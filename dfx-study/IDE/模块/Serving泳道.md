# Serving Strace Swimlane

`serving-strace-swimlane.json` 可独立打开，用于查看 Serving 任务时间线，不依赖 Chip Swimlane 目录中的 `deps.json` 或 `name_map*.json`。公开 PyPTO3 用户文档说明性能面板按 `WorkerProcess` 汇总任务数、平均、最大和最小耗时。

本机安装包的 `dist/common/serving-strace.js` 按文件名中的 `strace-swimlane` 识别该输入，`dist/command/registerCommand.js` 把它送入泳道视图，并通过 `isServingStrace` 区分普通 Chip 任务。底层仍读取 trace 事件：`dist/process.js` 也会根据事件 `cat === 'strace.host'` 识别 Serving 类型。字段形状见[数据结构](../数据结构.md#serving-strace)。

Serving trace 与 `chip_swimlane_records.json` 是两种输入；前者的 `WorkerProcess` 统计不能直接解释为 AICore Worker View 或 AICPU Scheduler 阶段。
