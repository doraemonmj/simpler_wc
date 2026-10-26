# A5 onboard L0 原始采集样本

此目录保存 **perfmon 自管 GM 的有效字节前缀**，用于离线解码与方案复核。`.bin` 是原始字节；同名 `.hex` 是逐字节等价的文本备份（每行 32 字节），不是解析后的事件。数据来自 device 0 / CANN 9.1.0 / driver 7.0.t9.0.B848 的 36-cluster EVB，均在设备锁下采集，无设备或服务器 reset。

| 文件 | 来源 | 有效长度 | SHA-256（`.bin`） |
| --- | --- | ---: | --- |
| `channel25_core91_one_task_from_log.bin` | 2026-10-23 单任务，cluster 27 / AIV1 core 91 / channel 25；按当时日志中完整的 64 字节 hex 逐字节还原 | 64 B | `ffaff8b129b31bc6f9e0a3a5b68ff3a8ca3be73ca8df288f4cc9fe19e19b0cb6` |
| `channel27_core106_two_tasks.bin` | 2026-10-23 双任务，cluster 35 / AIV0 core 106 / channel 27；直接复制本次 GM D2H 导出的 `.bin` | 128 B | `e2964bad46ee47e672e92b9c4b8f3ad4aa1350cbd0da29ef00116b8afe2eefc4` |
| `channel28_core107_st_20261026.bin` | 2026-10-26 普通 72-AIV ST，cluster 35 / AIV1 core 107 / channel 28 | 288 B | `6f7f7eef63e6850c639bc7e7a6c06df587a429e43d0be346589d54db3175586a` |

### 2026-10-26 全核采集

三个目录分别保存一次 36-block MIX 普通 ST 的**按硬件写指针导出的逐核原始前缀**（每核 `.bin`）、同名 live 解码 JSON、覆盖全部 108 核的 `l0_survey.json` 和 `.bin` 的 `SHA256SUMS`。部分前缀包含上次会话写指针留下的初始化 `A5` 空档，不能把整个 `[0,wptr)` 都算成本次有效记录。

| 目录 | channel | 已配置/产出数据 | 导出前缀 | 本次新写入记录 | region 10/11 事件 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `channel26_all_physical_20261026` | 26 | 35/108 | 10,016 B | 9,536 B | 118 |
| `channel23_all_physical_20261026` | 23 | 35/108 | 19,744 B | 9,728 B | 118 |
| `channel20_all_physical_20261026` | 20 | **108/108** | 48,544 B | **28,800 B** | **360** |

### 2026-10-26 runtime task ID 关联样本

[`channel17_task_join_20261026`](../../../src/a5/platform/testdata/l0_onboard/channel17_task_join_20261026) 保存了一次通过的 36-block MIX ST 的 108 核原始 `.bin`、live 解码 JSON、`l0_survey.json`、同一次运行的 `chip_swimlane_records.json`，以及自动关联的 `l0_task_join.json` 和 `l0_task_trace.json`。这次 360 条 L0 事件都关联到 runtime 的真实 `task_token=0`；关联状态为 `unique_core_task`，因为两路原始 cycle 未直接对齐。详细条件和限制见 [端到端 demo 说明](l0_task_join_demo_2026_10.md)。目录中的 `SHA256SUMS` 覆盖归档时的全部 222 个文件。

[`pa_medium_20261026`](../../../src/a5/platform/testdata/l0_onboard/pa_medium_20261026) 保存中等规模 paged attention 的 baseline 和 L0 capture 尝试产物。两次 PA 计算与 golden 均通过且各有 16 个 runtime task ID；capture 尝试的 channel 17 在 `prof_drv_start` 失败，因此**该目录没有 PA L0 原始数据**。详见 [实验记录](l0_paged_attention_attempt_2026_10.md)。

channel 26/23 在 `prof_drv_start` 后预检发现相同的 73 核 `gen=1,len=0,base=0`，未改写这些核，只采其余 35 核。channel 20 则在 `prof_drv_start` **之前**配置全部 108 核的 GM，启动后逐核核实配置仍有效，正常 ST 完成后逐核确认撤销。36 个 AIC 各有 2 条 scalar pipe 标记；72 个 AIV 各有 2 条 scalar 和 2 条 MTE3 pipe 标记。所有事件的 region 和 block 均匹配用例预期；每份 raw 用独立离线解码器复核，与 live JSON 的 `events` 完全一致。

channel 20 的 [窗口审计](../../../src/a5/platform/testdata/l0_onboard/channel20_all_physical_20261026/l0_window_audit.json)逐核列出初始化前缀与实际写入窗口。35 个带 `A5` 前缀的核，其前缀长度与 channel 23 同核上次写指针 **35/35 完全相等**；channel 23 对 channel 26 也为 35/35。108 个本次窗口都从完整 START_STAMP 起始，内部无 `A5` 空洞，单核长 224–288 B。当前 `.bin` 保留完整导出前缀，不裁掉这些空档，方便复核寄存器状态残留。

第二份数据的硬件写指针在任务 1 后为 64，任务 2 后为 128。两个区间 `[0,64)`、`[64,128)` 分别包含该任务的 START_STAMP 和 pipe 标记。两次任务均为 72-block AIV kernel；任务 ID 1/2 是独立采集程序的顺序编号，不是 runtime 调度 task ID。详细寄存器、标记和运行记录见 [实验报告](../../../src/a5/platform/docs/l0_evb_onboard_probe_2026_10.md)。

2026-10-23 的同名 `.json` 是用 `tools/l0_decode_offline.cpp` 从归档 `.bin` 生成的事件列表，`source=archive`；2026-10-26 普通 ST 输出的同名 `.json` 标记为 `source=live`。`task_window` 仅按已记录的 64 字节边界划分两次独立实验任务，不是 runtime task ID。

```bash
c++ -std=c++17 -O2 -Isrc/a5/platform/include src/a5/platform/tools/l0_decode_offline.cpp -o build/l0_decode_offline
build/l0_decode_offline src/a5/platform/testdata/l0_onboard/channel27_core106_two_tasks.bin \
  src/a5/platform/testdata/l0_onboard/channel27_core106_two_tasks.json 64
```

复核原始文件：

```bash
sha256sum src/a5/platform/testdata/l0_onboard/*.bin
python3 -c 'from pathlib import Path; d=Path("src/a5/platform/testdata/l0_onboard"); [(lambda b,h: print(p.name, "OK" if b == bytes.fromhex(h) else "MISMATCH"))(p.read_bytes(), (d/(p.stem+".hex")).read_text()) for p in d.glob("*.bin")]'
```

这里没有保存完整的 1 MiB GM 分配区：采集时其余字节仍为初始化哨兵 `0xA5`，硬件写指针界定的有效原始记录分别只有 64/128 字节。
