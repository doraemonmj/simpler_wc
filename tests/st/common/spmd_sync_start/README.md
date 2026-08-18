# Shared SPMD Sync-Start Kernels

Slow incore kernels used to keep cores occupied while sync-start placement and
early-dispatch behavior are observed. The sources have the same signature and
behavior on a2a3 and a5; orchestration remains local to each test contract.

Supported platforms: `a2a3sim`, `a2a3`, `a5sim`, and `a5`.

Consumers:

- `tests/st/host_build_graph_wide_dispatch/`
- `tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_mix_spill/`
- `tests/st/a5/tensormap_and_ringbuffer/spmd_sync_start_mix_spill/`
- `tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_early_dispatch/`
- `tests/st/a5/tensormap_and_ringbuffer/spmd_sync_start_early_dispatch/`
- `tests/st/a2a3/tensormap_and_ringbuffer/dfx/chip_swimlane/`
- `tests/st/a5/tensormap_and_ringbuffer/dfx/chip_swimlane/`

The `kernel_spmd_write_slow.cpp` source is core-neutral and is compiled as AIC or
AIV according to the consuming callable.
