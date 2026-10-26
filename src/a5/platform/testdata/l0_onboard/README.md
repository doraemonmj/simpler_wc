# A5 onboard L0 capture evidence

This directory preserves raw bytes from device 0 on a 36-cluster EVB. The
`l0_core*.bin` files contain the complete GM prefix selected by the hardware
write pointer. For each core, only bytes from `capture_start_offset` onward
belong to that run; earlier bytes may be initialization padding left by a
previous channel session. The paired JSON files contain decoded pipe stamps.

| Dataset | Result |
| --- | --- |
| `channel26_all_physical_20261026` | 35 of 108 cores captured |
| `channel23_all_physical_20261026` | 35 of 108 cores captured |
| `channel20_all_physical_20261026` | 108 of 108 cores captured; 360 pipe stamps |
| `channel17_task_join_20261026` | 108 cores captured; 360 stamps joined to one runtime task ID |
| `pa_medium_20261026/baseline` | Paged attention golden passed; 16 runtime task IDs; no L0 capture requested |
| `pa_medium_20261026/capture_attempt` | Paged attention golden passed; channel 17 failed to start, so no PA L0 bytes |

Each full-core dataset includes a `SHA256SUMS` file. The smaller single-core
samples at the top level preserve the initial channel experiments. Detailed
records and limitations are in the [Chinese experiment notes](../../../../../docs/zh-cn/a5-onboard-l0/raw_data.md)
and the [EVB probe report](../../docs/l0_evb_onboard_probe_2026_10.md).

The current collector allocates 1 MiB per physical subcore (about 108 MiB
total). It rejects invalid write pointers, but has not verified how the
hardware reports a full or wrapped buffer. A long capture can therefore lose
older records without a reliable overflow signal. The paged-attention L0
capture remains unverified because channel 17 failed to restart.
