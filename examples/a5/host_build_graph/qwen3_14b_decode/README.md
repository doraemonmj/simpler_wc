# Qwen3-14B 40-layer Graph Execution on A5

This is the A5 `host_build_graph` variant of the full Qwen3-14B decode example.
Layer 0 records and executes one decoder-layer task DAG. Layers 1 through 39
hit the cached definition and replay it as one outer Graph task with new tensor
bindings.

The graph orchestration is shared with the
[a2a3 HBG example](../../../a2a3/host_build_graph/qwen3_14b_decode/README.md),
while all 37 in-core entries are compiled for A5. The model inputs and torch
golden are shared with the A5 `tensormap_and_ringbuffer` variant.

Run hardware only through `task-submit` on an A5 host:

```bash
task-submit --device auto --device-num 1 \
  --run '.claude/skills/onboard-arch-precheck/check.sh a5 && pytest examples/a5/host_build_graph/qwen3_14b_decode --platform a5 --device "$TASK_DEVICE"'
```
