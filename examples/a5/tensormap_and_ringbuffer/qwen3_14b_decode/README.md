# Qwen3-14B 40-layer decode on A5

This is the A5 `tensormap_and_ringbuffer` variant of the full-model Qwen3-14B
decode example. It uses the same model inputs, CANN fused-attention extern, and
torch golden as the
[a2a3 example](../../../a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/README.md),
but runs with the A5 runtime and carries A5-specific generated-kernel fixes.

The workload contains all 40 decoder layers in one orchestration dispatch and
checks both the final hidden state and every layer's KV-cache update.

## A5 kernel differences

The A5 target is not source-compatible with the harvested A2/A3 codegen in two
places:

- A5 L0A tiles use `BLayout::ColMajor`; all 18 AIC projection kernels under
  `kernels/aic/` carry that architecture-specific layout.
- A5 vector barriers accept the MTE2/MTE3/ALL range, so the 10 affected AIV
  kernels under `kernels/aiv/` use `pipe_barrier(PIPE_ALL)` instead of the
  A2/A3-only `PIPE_V` form.

The A5 TMR launch API also names its SPMD width `set_core_num`, reflected in the
local orchestration source. Unchanged AIV kernels and the CANN fused-attention
tree continue to come from the shared a2a3 harvest; they compile unchanged for
`dav-c310`.

Run hardware only through `task-submit` on an A5 host:

```bash
task-submit --device auto --device-num 1 \
  --run '.claude/skills/onboard-arch-precheck/check.sh a5 && pytest examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode --platform a5 --device "$TASK_DEVICE"'
```
