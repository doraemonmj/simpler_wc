# Chip Swimlane

## 输入到视图

PyPTO 3.0 的主输入是原始 `chip_swimlane_records.json`。本机扩展 `dist/command/registerCommand.js` 识别文件中的 `aicore_tasks`，`dist/handler/swimGraphMsgHandler.js` 调用 `dist/converter/perfSwimlaneConverter.js`，把原始 device cycle 记录归一成微秒域的任务，再转换为前端绘制的 trace 事件。`deps.json` 用于边和函数身份恢复，`name_map*.json` 将 Func ID 映射到名称。

公开 PyPTO3 用户文档描述 Worker View、Scheduler View、AICPU Scheduler、AICPU Orchestrator 四类视图。它们取决于采集级别及输入实际存在的数据；只有 AICore 任务时不能推导出完整调度阶段。选择任务可看详情、连线，性能面板可统计 Kernel 耗时、Setup、连续单依赖等。关键路径高亮需 CPM 结果。数据形状见[输入数据结构](../数据结构.md#chip-swimlane-原始记录)。

## 简化处理链

```text
aicore_tasks + metadata.clock_freq_hz
  → cycle 换算为微秒、按 Core 排列 Worker 任务
  + deps.json::tasks[].kernel_ids → 恢复 Func ID
  + name_map*.json::callable_id_to_name → 函数名
  + 调度/编排记录 → 相应阶段视图
  + CPM_*.json → 关键路径高亮
```

当前扩展的原始格式判断主要看顶层是否有 `aicore_tasks[]` 且没有 `tasks[]`；基础验证只检查数组存在，不能替代对记录列数、时钟和关联键的完整校验。

## 当前 simpler schema 的兼容边界

simpler 当前导出样例见 `tests/ut/cpp/common/platform/test_chip_swimlane_run_export.cpp`，其中 `aicore_tasks` 行为七列，最后一列是 `run_epoch`；`scheduler_tasks.records` 也是带 epoch 的行，调度阶段放在 `scheduler_records.streams`。simpler 离线工具 `simpler_setup/tools/swimlane_converter.py` 按 `(run_epoch, core_id, reg_task_id)` 关联记录。

本机插件 `4.0.36` 的 `normalizeRawRecords` 读取 `aicore_tasks` 前六列，但 AICPU join 使用旧字段 `aicpu_tasks` 和键 `(core_id, reg_task_id)`，阶段读取旧字段 `aicpu_scheduler_phases`。它没有按上述新 `scheduler_tasks` / `scheduler_records` schema 完整消费。结果是 Worker 任务可能可见，Scheduler 阶段、dispatch/finish 和多轮同 ID 任务的归属不能仅凭文件可打开就认定正确。需要用当前 simpler 真实产物在 VS Code 中验证后才能给出端到端兼容结论。

若分析依赖、时间和多轮归属要求严格，可先用 simpler 自带的 `swimlane_converter.py` 生成 Perfetto trace 做交叉核对；这属于验证和另一查看路径，不是 PyPTO 3.0 IDE 打开原始文件的前置步骤。
