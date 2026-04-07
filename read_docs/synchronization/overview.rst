Synchronization Overview
========================

Coordinating execution between AICPU and AICORE processors.

Why Synchronization?
--------------------

AICPU and AICORE are separate processors that can run concurrently:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    Execution Timeline                       │
   │                                                             │
   │  AICPU:   [preprocess]────────────────►[postprocess]       │
   │                      │                 ▲                    │
   │                      ▼ sync            │ sync               │
   │  AICORE:              [compute]────────►                    │
   │                                                             │
   │  Without sync: AICORE may start before data ready           │
   │  Without sync: postprocess may read before compute done     │
   └─────────────────────────────────────────────────────────────┘

Synchronization Methods
-----------------------

Three approaches, with different trade-offs:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Method
     - Mechanism
     - Latency
     - Use Case
   * - Register
     - Shared register read/write
     - ~1 μs
     - Simple flag signaling
   * - Atomic
     - Atomic memory operations
     - ~10 μs
     - Complex state, counters
   * - Queue
     - Software queue in HBM
     - ~50 μs
     - Data passing, pipelines

Choosing a Method
-----------------

**Use Register when:**

- Simple producer-consumer
- Single flag/signal
- Lowest latency needed

**Use Atomic when:**

- Multiple producers or consumers
- Need counter semantics
- Compare-and-swap logic needed

**Use Queue when:**

- Passing data (not just signals)
- Multiple items to communicate
- Pipeline with buffering needed

Execution Model
---------------

.. code-block:: text

   Host
    │
    │ platform_kernel_launch(aicpu_kernel)
    ├────────────────────────────────────────► AICPU runs
    │                                               │
    │ platform_kernel_launch(aicore_kernel)         │
    ├────────────────────────────────────────► AICORE runs
    │                                               │
    │                                               ▼
    │ platform_stream_sync()                 ◄─────────
    ▼
   Host continues

Kernels on the same stream execute in order, but AICPU and AICORE
can overlap. Internal sync needed if one depends on the other.

Common Patterns
---------------

**Pattern 1: AICPU prepares, AICORE computes**

.. code-block:: text

   AICPU:
       prepare_data()
       signal_ready()      // Set flag

   AICORE:
       wait_ready()        // Poll flag
       compute()

**Pattern 2: AICORE computes, AICPU reduces**

.. code-block:: text

   AICORE (multiple blocks):
       partial = compute_local()
       atomic_add(partial_sum, partial)
       if (block_id == last_block):
           signal_compute_done()

   AICPU:
       wait_compute_done()
       final_result = reduce(partial_sums)

**Pattern 3: Pipeline**

.. code-block:: text

   AICPU Producer:
       while more_data:
           item = prepare_next()
           queue.push(item)

   AICORE Consumer:
       while not done:
           item = queue.pop()
           process(item)
           signal_item_done()

Performance Considerations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Issue
     - Impact
   * - Polling overhead
     - Wastes cycles spinning
   * - Too many syncs
     - Serializes parallel execution
   * - Coarse granularity
     - Under-utilizes hardware
   * - Fine granularity
     - Sync overhead dominates

Best practice: Batch work to amortize sync cost.
