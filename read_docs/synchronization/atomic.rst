Atomic-Based Synchronization
============================

Using atomic operations in HBM for AICPU-AICORE coordination.

Overview
--------

Atomic operations on shared memory provide flexible synchronization:

.. code-block:: text

   ┌─────────────┐                         ┌─────────────┐
   │   AICPU     │                         │   AICORE    │
   │             │    ┌──────────────┐     │             │
   │  atomic_add ────►│ HBM (shared) │◄──── atomic_add  │
   │  atomic_cas ────►│   memory     │◄──── atomic_cas  │
   │  load/store ────►│              │◄──── load/store  │
   │             │    └──────────────┘     │             │
   └─────────────┘                         └─────────────┘

   Latency: ~10 μs (slower than register, more flexible)

Why Use Atomics?
----------------

vs. Registers:
- More state (full memory vs few registers)
- Counters, arrays of flags
- CAS for complex updates

vs. Queues:
- Lower latency
- Simpler implementation
- Better for signaling (vs data passing)

Atomic Operations
-----------------

Both AICPU and AICORE support:

.. code-block:: cpp

   // Add
   atomic_fetch_add(ptr, value);

   // Compare-and-swap
   atomic_compare_exchange(ptr, expected, desired);

   // Load/Store with ordering
   atomic_load(ptr);
   atomic_store(ptr, value);

Counter Pattern
---------------

Multiple producers increment shared counter:

.. code-block:: cpp

   // Shared state in HBM
   struct SharedState {
       std::atomic<int32_t> items_produced;
       std::atomic<int32_t> items_consumed;
   };

   // AICPU Producer
   void producer(SharedState* state) {
       // Produce items
       for (int i = 0; i < N; i++) {
           produce_item(i);
           state->items_produced.fetch_add(1, std::memory_order_release);
       }
   }

   // AICORE Consumer (multiple blocks)
   void consumer(SharedState* state, Args* args) {
       while (true) {
           int consumed = state->items_consumed.load(std::memory_order_acquire);
           int produced = state->items_produced.load(std::memory_order_acquire);

           if (consumed >= produced && /* done signal */) {
               break;
           }

           if (consumed < produced) {
               // Try to claim an item
               if (state->items_consumed.compare_exchange_weak(
                       consumed, consumed + 1, std::memory_order_acq_rel)) {
                   process_item(consumed);
               }
           }
       }
   }

Barrier Pattern
---------------

Wait for all blocks to reach a point:

.. code-block:: cpp

   // Shared in HBM
   std::atomic<int32_t>* barrier_count;

   // AICORE kernel (each block)
   void kernel_with_barrier(Args* args) {
       // Phase 1: independent work
       do_phase1();

       // Barrier: wait for all blocks
       int arrived = barrier_count->fetch_add(1, std::memory_order_acq_rel);
       int num_blocks = get_block_dim();

       if (arrived == num_blocks - 1) {
           // Last to arrive: reset for next use, signal done
           barrier_count->store(0, std::memory_order_release);
       } else {
           // Wait for all
           while (barrier_count->load(std::memory_order_acquire) != 0) {
               // Spin
           }
       }

       // Phase 2: all blocks past barrier
       do_phase2();
   }

Progress Tracking
-----------------

AICPU monitors AICORE progress:

.. code-block:: cpp

   // AICORE reports progress
   void aicore_kernel(Args* args, std::atomic<int32_t>* progress) {
       for (int tile = 0; tile < num_tiles; tile++) {
           process_tile(tile);

           // Report progress every 100 tiles
           if (tile % 100 == 0) {
               progress->store(tile, std::memory_order_release);
           }
       }
       progress->store(num_tiles, std::memory_order_release);  // Done
   }

   // AICPU monitors
   void aicpu_monitor(Args* args, std::atomic<int32_t>* progress) {
       int total = args->num_tiles;
       while (true) {
           int current = progress->load(std::memory_order_acquire);
           log_progress(current, total);

           if (current >= total) break;
           sleep_us(1000);  // Check every 1ms
       }
   }

Memory Ordering
---------------

Choose appropriate ordering:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Ordering
     - Use When
   * - ``relaxed``
     - Counter only, no other data depends on it
   * - ``acquire``
     - Reading flag that guards data
   * - ``release``
     - Writing flag after data is ready
   * - ``acq_rel``
     - Both reading and writing (e.g., CAS)
   * - ``seq_cst``
     - Need total ordering (rarely needed)

Performance Tips
----------------

1. **Minimize contention** - Spread updates across addresses
2. **Batch updates** - Accumulate locally, update once
3. **Use weak CAS** - ``compare_exchange_weak`` for spin loops
4. **Avoid false sharing** - Pad atomics to cache line

.. code-block:: cpp

   // Avoid false sharing
   struct alignas(64) PaddedAtomic {
       std::atomic<int32_t> value;
   };

   PaddedAtomic per_block_counters[MAX_BLOCKS];

Example Code
------------

Full working example: :file:`examples/17-aicpu-aicore-atomic/`

Host code demonstrating atomic-based synchronization:

.. literalinclude:: ../../examples/17-aicpu-aicore-atomic/main.cpp
   :language: cpp
   :caption: 17-aicpu-aicore-atomic/main.cpp
   :lines: 1-80

AICPU monitor kernel:

.. literalinclude:: ../../examples/17-aicpu-aicore-atomic/aicpu_kernel/monitor.cpp
   :language: cpp
   :caption: 17-aicpu-aicore-atomic/aicpu_kernel/monitor.cpp

AICORE kernel with atomics (PTO-ISA):

.. literalinclude:: ../../examples/17-aicpu-aicore-atomic/aicore_kernel.pto
   :language: text
   :caption: 17-aicpu-aicore-atomic/aicore_kernel.pto
