AICORE Atomic Operations
========================

Atomic operations for coordination between AICORE blocks.

Overview
--------

When multiple AICORE blocks access the same global memory location,
atomic operations ensure correctness.

.. code-block:: text

   Block 0: atomic_add(GM[x], 1)  ──┐
   Block 1: atomic_add(GM[x], 1)  ──┼──► Result: GM[x] += 2
   ...                              │
   Block N: atomic_add(GM[x], 1)  ──┘

Supported Atomic Operations
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Operation
     - Syntax
     - Description
   * - Atomic Add
     - ``ATOMIC_ADD gm_addr, value``
     - GM[addr] += value
   * - Atomic Max
     - ``ATOMIC_MAX gm_addr, value``
     - GM[addr] = max(GM[addr], value)
   * - Atomic Min
     - ``ATOMIC_MIN gm_addr, value``
     - GM[addr] = min(GM[addr], value)
   * - Atomic CAS
     - ``ATOMIC_CAS gm_addr, cmp, val``
     - if GM[addr]==cmp: GM[addr]=val

Usage Patterns
--------------

**Pattern 1: Global Counter**

.. code-block:: text

   // Each block increments global counter
   ATOMIC_ADD gm_counter, processed_count

   // Host reads final count after kernel completes

**Pattern 2: Global Maximum**

.. code-block:: text

   // Each block finds local max, then updates global
   local_max = reduce_max(my_data)
   ATOMIC_MAX gm_max, local_max

**Pattern 3: Lock-Free Queue Index**

.. code-block:: text

   // Get unique slot in output buffer
   my_slot = ATOMIC_ADD gm_write_index, batch_size
   // Write to gm_output[my_slot : my_slot + batch_size]

Reduction Example
-----------------

Parallel sum across all blocks:

.. code-block:: text

   // Kernel: each block sums portion of data
   __kernel__ void parallel_sum(float* data, int n, float* result) {
       int block_id = get_block_idx();
       int block_size = n / get_block_dim();
       int start = block_id * block_size;

       // Local sum in UB
       float local_sum = 0;
       for (int i = start; i < start + block_size; i++) {
           local_sum += data[i];
       }

       // Atomic add to global result
       ATOMIC_ADD result, local_sum
   }

Performance Considerations
--------------------------

Atomic operations are slower than regular memory access:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Operation
     - Relative Cost
     - Notes
   * - Regular load
     - 1×
     - Baseline
   * - Regular store
     - 1×
     - Baseline
   * - Atomic add
     - 10-50×
     - Serializes conflicting accesses
   * - Atomic CAS
     - 10-100×
     - May retry on conflict

Optimization Tips
-----------------

1. **Minimize atomic operations** - Accumulate locally first

   .. code-block:: text

      // BAD: atomic per element
      for (int i = 0; i < N; i++) {
          ATOMIC_ADD global_sum, data[i]
      }

      // GOOD: local sum, one atomic
      local_sum = 0
      for (int i = 0; i < N; i++) {
          local_sum += data[i]
      }
      ATOMIC_ADD global_sum, local_sum

2. **Reduce contention** - Spread writes across addresses

   .. code-block:: text

      // Instead of all blocks writing to one location,
      // use per-block slots and reduce on host
      gm_partial_sums[block_id] = local_sum

3. **Use appropriate operation** - Don't use CAS when add suffices

Memory Ordering
---------------

Atomic operations have implicit memory ordering:

- Operations before atomic are visible after atomic completes
- Useful for signaling between blocks

.. code-block:: text

   // Producer block
   STORE gm_data, computed_result
   PIPE_BARRIER
   ATOMIC_ADD gm_ready_flag, 1  // Signal data ready

   // Consumer block
   while (ATOMIC_LOAD gm_ready_flag < expected):
       // Spin wait
   LOAD local_data, gm_data  // Safe to read

Example Code
------------

Full working example: :file:`examples/14-aicore-atomic/`

Host code demonstrating multi-block atomics:

.. literalinclude:: ../../examples/14-aicore-atomic/main.cpp
   :language: cpp
   :caption: 14-aicore-atomic/main.cpp
   :lines: 1-80

AICORE atomic operations kernel (PTO-ISA):

.. literalinclude:: ../../examples/14-aicore-atomic/kernel.pto
   :language: text
   :caption: 14-aicore-atomic/kernel.pto
