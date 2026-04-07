AICPU Atomic Operations
=======================

Atomic operations for thread-safe memory access in AICPU kernels.

Overview
--------

AICPU kernels may run on multiple ARM cores simultaneously. When multiple
cores access the same memory location, atomic operations ensure correctness.

.. code-block:: text

   Without atomics:           With atomics:
   ─────────────────          ─────────────────
   Core 0: read(x)=5          Core 0: atomic_add(x,1)
   Core 1: read(x)=5                     ↓
   Core 0: write(x)=6         Core 1: atomic_add(x,1)
   Core 1: write(x)=6                    ↓
   Result: x=6 (WRONG!)       Result: x=7 (CORRECT!)

C++ Atomics
-----------

Use C++11 atomics for portable code:

.. code-block:: cpp

   #include <atomic>

   struct AtomicArgs {
       std::atomic<int32_t>* counter;  // Device pointer to atomic
       void* data;
       int32_t count;
   };

   extern "C" void counting_kernel(AtomicArgs* args) {
       for (int i = 0; i < args->count; i++) {
           // Atomically increment counter
           args->counter->fetch_add(1, std::memory_order_relaxed);
       }
   }

Atomic Operations
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Description
   * - ``fetch_add``
     - Add and return old value
   * - ``fetch_sub``
     - Subtract and return old value
   * - ``fetch_and``
     - Bitwise AND and return old value
   * - ``fetch_or``
     - Bitwise OR and return old value
   * - ``exchange``
     - Set new value, return old value
   * - ``compare_exchange``
     - CAS (compare-and-swap)
   * - ``load``
     - Read atomically
   * - ``store``
     - Write atomically

Memory Ordering
---------------

.. code-block:: cpp

   // Relaxed - no ordering guarantees (fastest)
   counter->fetch_add(1, std::memory_order_relaxed);

   // Acquire - subsequent reads see prior writes
   int val = counter->load(std::memory_order_acquire);

   // Release - prior writes visible to acquire loads
   counter->store(val, std::memory_order_release);

   // Seq_cst - full sequential consistency (slowest)
   counter->fetch_add(1, std::memory_order_seq_cst);

For simple counters, ``memory_order_relaxed`` is usually sufficient.

Example: Parallel Histogram
---------------------------

.. code-block:: cpp

   #include <atomic>

   struct HistogramArgs {
       void* input;                    // Input data
       std::atomic<int32_t>* bins;     // Histogram bins (atomics)
       int32_t count;                  // Number of elements
       int32_t num_bins;               // Number of bins
   };

   extern "C" void histogram_kernel(HistogramArgs* args) {
       uint8_t* data = (uint8_t*)args->input;

       for (int i = 0; i < args->count; i++) {
           int bin = data[i] % args->num_bins;
           args->bins[bin].fetch_add(1, std::memory_order_relaxed);
       }
   }

Host setup:

.. code-block:: cpp

   // Allocate atomic array on device
   int num_bins = 256;
   std::atomic<int32_t>* dev_bins = (std::atomic<int32_t>*)
       platform_malloc(num_bins * sizeof(std::atomic<int32_t>));

   // Initialize to zero
   std::vector<int32_t> zeros(num_bins, 0);
   platform_memcpy_h2d(dev_bins, zeros.data(), num_bins * sizeof(int32_t));

   // Launch kernel
   HistogramArgs args = {dev_input, dev_bins, count, num_bins};
   platform_aicpu_launch(...);

Compare-And-Swap (CAS)
----------------------

For complex atomic updates:

.. code-block:: cpp

   // Atomically update max value
   void atomic_max(std::atomic<int32_t>* target, int32_t value) {
       int32_t current = target->load(std::memory_order_relaxed);
       while (value > current) {
           if (target->compare_exchange_weak(current, value,
                   std::memory_order_relaxed)) {
               break;
           }
           // current is updated on failure
       }
   }

Spin Locks
----------

.. code-block:: cpp

   // Simple spinlock using atomic flag
   struct SpinLock {
       std::atomic<int32_t> locked{0};

       void lock() {
           while (locked.exchange(1, std::memory_order_acquire) == 1) {
               // Spin
           }
       }

       void unlock() {
           locked.store(0, std::memory_order_release);
       }
   };

.. warning::

   Spin locks can cause performance issues if held too long.
   Prefer atomic operations when possible.

Performance Tips
----------------

1. **Minimize contention** - Partition work to reduce shared access
2. **Use relaxed ordering** - When strict ordering isn't needed
3. **Batch updates** - Accumulate locally, then do one atomic update
4. **Avoid false sharing** - Pad atomic variables to cache line size

.. code-block:: cpp

   // Pad to avoid false sharing
   struct alignas(64) PaddedCounter {
       std::atomic<int64_t> value;
   };

   PaddedCounter counters[NUM_THREADS];  // Each on separate cache line

Example Code
------------

Full working example: :file:`examples/07-aicpu-atomic/`

Host code demonstrating atomic patterns:

.. literalinclude:: ../../examples/07-aicpu-atomic/main.cpp
   :language: cpp
   :caption: 07-aicpu-atomic/main.cpp
   :lines: 1-80

AICPU kernel with atomic operations:

.. literalinclude:: ../../examples/07-aicpu-atomic/kernel/atomic_counter.cpp
   :language: cpp
   :caption: 07-aicpu-atomic/kernel/atomic_counter.cpp
