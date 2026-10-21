# DepGen：任务依赖图

DepGen 回答“任务为什么要等前驱，以及这条边由哪个张量或显式依赖产生”。泳道图只采时间，不携带完整依赖边；`deps.json` 是离线查看依赖与绘制箭头的输入。完整语义见[原设计文档](../../../docs/dfx/dep-gen.md)。

## 1. 两条采集路径

| Runtime | 采集点与做法 | 为什么这样做 |
| ------- | ------------ | ------------ |
| `tensormap_and_ringbuffer` | AICPU 的 `submit_task` 在依赖查询前记录本次提交输入：任务 ID、scope 模式、张量及参数角色、显式依赖与种类、核 ID、block 数等。Host Collector 收到 `DepGenRecord` 后，在内存中重放依赖计算。 | 设备侧构图时 Host 看不到提交过程；在提交入口保留输入，才能在之后还原已完成任务的后继边。 |
| `host_build_graph` | Host 的 `submit_task_common` 建立任务，`compute_task_fanin` 的标注钩子在实际连边时记录来源与张量。 | 构图本来就在 Host，可直接记图，无需设备缓冲和重放。 |

设备侧采集点见 [Orchestrator](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/orchestrator.cpp)；Host 直录见 [Host graph](../../../src/common/host_build_graph/host/dep_gen_host_graph.cpp)。

```text
设备构图：submit 输入 → 设备缓冲 → Host 内存记录 → 双路重放校验 → deps.json
Host 构图：submit 与依赖计算 ────────────────→ 直接记录 → deps.json
```

## 2. 采了什么，为什么需要

设备侧原始记录是提交输入，最终文件是推导出的图。Host 重放一条路径调用正式的 `compute_task_fanin`，另一条路径生成带张量说明的边；每条记录都比较两条路径的前驱及依赖标志，结果不一致就不输出图。丢记录也不能得到完整图，故先核对采集计数。重放实现见 [`dep_gen_replay.cpp`](../../../src/common/tensormap_and_ringbuffer/host/dep_gen_replay.cpp)。

`<output_prefix>/deps.json` 主要有三组：

| 字段 | 说明与用途 |
| ---- | ---------- |
| `tasks[]`：`task_id`、`scope`、`kernel_ids`、`block_num`、`args` | 识别任务、提交模式及参数，便于解释不同任务为何相连。 |
| `tensors[]`：张量标识、底层缓冲、版本、类型和大小 | 区分同一缓冲的不同版本，避免把张量复用误判成同一数据。 |
| `edges[]`：`pred`、`succ`、`source`、`arg`，可带张量切片和 `flags` | 指明前驱/后继、关联参数及来源：显式依赖、创建者保留或 TensorMap 命中。T&R 的 `flags` 还区分 `wait`（顺序约束）和 `retain`（生命周期保留）；HBG 不提供该字段。 |

`deps.json` 保存构造时的边；T&R 后续的 WAIT 边约简不会回写这里。因此图上有边，不代表运行时一定为该边单独等待。用 [`deps_viewer.py`](../../../simpler_setup/tools/deps_viewer.py) 查看图，再把同一图的依赖边与[泳道图](芯片泳道任务时间线.md)时间线关联。设备侧 DepGen 记录每次提交，可能扰动提交时序；性能分析常把另一次同图的依赖文件与计时文件配合使用。
