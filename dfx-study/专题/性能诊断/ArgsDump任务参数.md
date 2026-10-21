# ArgsDump：任务参数快照

ArgsDump 回答“核函数实际收到什么输入、产生什么输出”。数值、shape 或 stride 出错时，按任务和参数位置查看快照，比只看错误发生后的张量更容易找到第一处异常。详细模式和文件格式见[原设计文档](../../../docs/dfx/args-dump.md)。

## 1. 在哪里采集

Scheduler 在任务下发前调用 `dump_args_for_task(BEFORE_DISPATCH)`，采 INPUT 和 INOUT；任务完成通知后调用 `dump_args_for_task(AFTER_COMPLETION)`，采 OUTPUT 和 INOUT。标量只在下发前记录。AICore 在启用 Dump 时还执行写入屏障。调用点可查 [T&R dispatch](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp)、[T&R completion](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp)与[采集设计](../../../docs/dfx/args-dump.md#52-where-dump-calls-are-wired-in)。

```text
下发前：INPUT / INOUT / 标量 → 执行核函数 → 完成后：OUTPUT / INOUT
                    ↓                                  ↓
                 元数据缓冲 + 可选 payload arena → Host Collector
```

## 2. 采了什么，为什么需要

`<output_prefix>/args_dump/args_dump.json` 是清单；需要保存张量元素时，`args.bin` 放二进制 payload，清单用 `bin_offset`、`bin_size` 指向它。

| 数据 | 用途 |
| ---- | ---- |
| `task_id`、`func_id`、`arg_index` | 把快照定位到任务和参数槽；混合任务的 `func_id` 是核 ID 数组。 |
| `role`、`stage`、`kind` | 区分输入/输出/双向参数、下发前/完成后、张量/标量，避免比较错误阶段。 |
| `dtype`、`shape`、`strides`、`start_offset`、`is_contiguous` | 按逻辑视图解释张量；同一底层缓冲的切片或非连续视图不能只看原始字节。 |
| `bin_offset`、`bin_size`、标量值 | 在二进制文件中定位元素，或直接读取清单中的标量。 |
| `truncated`、`dropped_records` | 判断 payload 截断或元数据丢失，避免把缺失误解为正常值。 |

## 3. 选择采集范围与读结果

| 模式 | JSON 清单 | 张量 payload |
| ---- | --------- | ------------ |
| `--dump-args` / `partial` | 只记录 `CoreTaskArgs::dump(...)` 标记的参数 | 同样只存标记参数。 |
| `--dump-args hybrid` | 每个任务的参数元数据 | 只存标记参数；没有选中的张量时可没有 `args.bin`。 |
| `--dump-args full` | 每个任务的参数元数据 | 保存所有参数的张量数据。 |

[`dump_viewer.py`](../../../simpler_setup/tools/dump_viewer.py) 可按任务、阶段、角色和位置查清单并读取 payload。`hybrid` 的完整元数据也可供 Core Swimlane 做单任务模拟重放。Dump 会复制张量数据，尤其 `full` 会占用设备缓冲和磁盘空间；`truncated` 或 `dropped_records` 非零时，应先判断快照是否足以支持结论。

检查快照可见性时还要注意当前实现顺序：[HBG 执行器](../../../src/a2a3/runtime/host_build_graph/aicore/aicore_executor.cpp)在 FIN 前执行屏障，[T&R 执行器](../../../src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp)在 FIN 后执行。T&R 的“完成后”指 Scheduler 观察到 FIN 后采集，单凭这一阶段名称不能证明所有核内写入已在采样前完成屏障。
