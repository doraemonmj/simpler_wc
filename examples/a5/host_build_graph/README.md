# a5 — `host_build_graph` examples

Examples in this directory build their task graph on the AICPU with the A5
`host_build_graph` runtime.

| Example | What it demonstrates |
| ------- | -------------------- |
| [`qwen3_14b_decode/`](qwen3_14b_decode/) | Full 40-layer Qwen3-14B decode: record one decoder-layer Graph, then replay it for layers 1 through 39. Onboard only. |

Run onboard cases only through `task-submit`. The Qwen example's README has the
exact command and A5 kernel-porting details.
