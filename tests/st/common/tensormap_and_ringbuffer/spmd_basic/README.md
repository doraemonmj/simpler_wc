# Shared TRB SPMD Context Kernels

Orchestration and MIX kernels that report `block_idx`, `block_num`, and
`sub_block_id` into separate cache-line slots. The contract is shared across
architectures and specific to the `tensormap_and_ringbuffer` runtime.

Supported platforms: `a2a3sim`, `a2a3`, `a5sim`, and `a5`.

Consumers:

- `tests/st/a2a3/tensormap_and_ringbuffer/spmd_basic/`
- `tests/st/a5/tensormap_and_ringbuffer/spmd_basic/`
