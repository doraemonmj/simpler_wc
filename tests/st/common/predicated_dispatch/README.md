# Shared Predicated-Dispatch Kernels

Orchestration and AIC kernels for delayed dispatch-predicate evaluation. The
producer writes the predicate value, the scheduler conditionally retires or
dispatches a clobber task, and the consumer exposes the resulting value.

Supported platforms: `a2a3sim`, `a2a3`, `a5sim`, and `a5`.

Consumers:

- `tests/st/a2a3/host_build_graph/predicated_dispatch/`
- `tests/st/a2a3/host_build_graph/dfx/dep_gen/`
- `tests/st/a2a3/tensormap_and_ringbuffer/predicated_dispatch/`
- `tests/st/a5/tensormap_and_ringbuffer/predicated_dispatch/`

The bundle is runtime-neutral: both runtimes consume the same orchestration ABI,
function IDs, signatures, and incore behavior.
