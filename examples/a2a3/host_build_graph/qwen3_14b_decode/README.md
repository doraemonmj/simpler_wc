# Qwen3-14B 40-layer decode with Graph Execution

This scene runs the complete Qwen3-14B decoder stack with the
`host_build_graph` runtime. It uses the same 37 AIC/AIV kernels, CANN fused
attention implementation, input fixture, and torch golden as
[`tensormap_and_ringbuffer/qwen3_14b_decode`](../../tensormap_and_ringbuffer/qwen3_14b_decode/README.md).
The host-build-graph-specific orchestration records one decoder layer as a
Graph and reuses it across all 40 layers.

## Graph shape

The invocation contains a short ordinary prefix, 40 Graph submissions, and a
short ordinary suffix:

```text
paged-attention tiling + input preparation
  -> layer 0: record decoder DAG off the ring, submit one GRAPH task
  -> layers 1..39: reuse the cached Definition, submit one GRAPH task each
  -> copy final hidden state to output
```

The recorded layer contains the fixed decoder topology. Every replay updates
the current layer's tensor addresses, including its weight slices, KV-cache
slice, hidden-state buffers, and scratch buffers. The tensor shapes, strides,
dtypes, directions, and alias partition remain fixed, as required by Graph
Execution.

Each invocation therefore contributes one outer task-window entry per decoder
layer instead of roughly 265 ordinary task entries. The first invocation also
uses one outer `GRAPH` task because recording happens in host-only metadata off
the ring.

## Temporary storage

Graph boundaries cannot contain runtime-allocated outputs. The orchestration
allocates the layer's boundary storage before recording, then passes existing
buffers into the Graph:

- two hidden-state and normalized-state slots are used as a ping-pong pair;
- one BF16 and one FP32 scratch arena are shared by every layer;
- the decoder dependency chain and shared inout workspace serialize Graph
  execution, so a later layer cannot overwrite storage still used by an
  earlier layer.

This keeps the temporary live set flat in layer count and uses the default
host-build-graph ring heap. It does not need the 512 MiB heap override required
when all 40 layers are expanded as ordinary host-built tasks.

## Parameter regime

The fixture matches the original Qwen scene:

| Parameter | Value |
| --------- | ----: |
| batch | 16 |
| maximum sequence length | 5500 |
| decode sequence length | 3500 |
| decoder layers | 40 |

The output hidden state and all 40 layers' KV-cache writes are checked at
`RTOL=5e-2` and `ATOL=1e-1`.

## Run

On a shared device host, use `task-submit` as described by the repository's
onboard testing guide:

```bash
task-submit --device auto --device-num 1 --run \
  ".claude/skills/onboard-arch-precheck/check.sh a2a3 && \
  pytest examples/a2a3/host_build_graph/qwen3_14b_decode \
  --platform a2a3 --device \$TASK_DEVICE"
```
