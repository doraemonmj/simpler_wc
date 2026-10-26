# A5 onboard L0 与 runtime task ID：单任务端到端 demo

2026-10-26 在 36-cluster EVB 的 device 0 上，普通 `SceneTestCase` 的 36-block MIX 用例同时打开 `enable_l0` 和 `enable_chip_swimlane=1`。运行 `task_20261026_103635_730184` 通过，耗时 6.84 s；全程使用 `task-submit --device 0 --max-time 180`，未做 reset 或重启。采集使用 channel 17；这台驱动的 channel 曾表现出一次启动后无法再次启动，因此不要把这个 channel 当作可重复运行资源。

## 运行链路

1. `TestL0Onboard` 的 case 配置同时申请 L0 与 chip swimlane。L0 先配置 108 个物理子核的自管 GM，再启动 profiling channel；普通 runtime 启动一个 36-block MIX task。
2. task 完成后，L0 collector 安全撤销 GM 映射，按本次 `[capture_start_offset, valid_bytes)` 保存每核原始 `.bin` 和已解码 `.json`；chip swimlane 保存 `chip_swimlane_records.json` 中的 `aicore_tasks`。
3. 这次上板运行先采到两路原始数据，之后用同一次运行的产物生成 `l0_task_join.json` 和 `l0_task_trace.json`。现已把关联步骤接入 `SceneTestCase.finalize_diagnostic_outputs`：后续运行只要两个文件都存在就自动执行。L0 未开启时这条分支不执行；仅开启 L0 时仍可单独保存原始数据。

关联工具先把 A5 物理子核编号映射到 runtime 逻辑核编号，并校验核类型、run epoch、L0 block 与物理簇、原始 `.bin` 的 START_STAMP 和每个事件的 pipe/region 字节。若 L0 cycle 落在该核唯一的 runtime task 区间，则用 `matched_interval`；若未对齐但该核在同一 run 只有一条 task 记录且 block 匹配，则用 `unique_core_task`。其余情况保留未匹配状态及 `task_id: null`，不猜测任务。

本次共 108 条 AICore task 记录，均带有真实 runtime `task_token=0`、`reg_task_id=1`、`run_epoch=1`。108 个物理子核都生成了原始字节；360 条 L0 pipe 事件全部关联为 `unique_core_task`。例如物理核 0 的首条事件是 `pipe=0, region=10, block=0, cycle=325969270003172`，关联到逻辑核 0 的 `task_id=0`。事件覆盖 36 个 AIC 的 72 条 scalar 标记，以及 72 个 AIV 的 144 条 scalar 和 144 条 MTE3 标记；全部 360 条事件的 block 与簇一致。

**时间限制**：原始 L0 cycle 比对应核的 runtime task 起点大约晚 386–389 千个 cycle，且无法找到一个统一偏移让全部事件落入 task 时间窗。因此本次 task ID 归属依靠“同 run、同核唯一 task、block 一致”，不能证明两个相邻 task 的边界区分。`l0_task_trace.json` 把 runtime 与 L0 放在独立进程轨道，并各自以首个事件为原点；横向位置不能跨轨道比较。下一个多 task 用例需要时钟标定或额外的任务边界标记，才能严格按时间区分归属。

## 保存与回放

原始产物、两份关联输出及逐文件 SHA-256 清单在 [channel17_task_join_20261026](../../../src/a5/platform/testdata/l0_onboard/channel17_task_join_20261026)。`l0_core*.bin` 是驱动写指针界定的完整前缀，包含初始化前缀；`capture_start_offset` 后才是本次新记录。

同一批原始产物无需访问硬件即可重新生成关联：

```bash
python -m simpler_setup.tools.l0_task_join \
  src/a5/platform/testdata/l0_onboard/channel17_task_join_20261026
```

验证新 ST 配置的普通命令如下；channel 17 已消耗，在这台 EVB 上再次运行前必须先确认并配置新的可用 channel：

```bash
task-submit --device 0 --max-time 180 --run \
  './.venv/bin/python -m pytest -q -s tests/st/a5/tensormap_and_ringbuffer/dfx/l0_onboard/test_l0_onboard.py --platform a5 --manual include'
```
