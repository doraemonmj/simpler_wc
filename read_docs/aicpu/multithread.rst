AICPU Multi-Core Parallel Execution
====================================

**⚠️ EDUCATIONAL EXAMPLE**: This example uses AICPU for compute to demonstrate eval model threading.

**Production Reality:**

- **AICPU role**: Controls AICore, handles dynamic shapes, complex control flow
- **AICore role**: Performs actual computation (matrix ops, SIMD operations)
- **Hardware limit**: Maximum **4 AICPU cores** for scheduling on Ascend 910
- **For production compute**: Use AICore examples (09-18)

Overview
--------

AICPU supports multi-core parallelism through the ``aicpuNum`` parameter in ``rtAicpuKernelLaunchExWithArgs()``.
This example teaches the eval model architecture used by Ascend NPUs.

Eval Model vs Pthread Model
----------------------------

AICPU uses an **eval model** for multi-threading, fundamentally different from traditional pthread:

**Pthread Model** (traditional CPU threading):

.. code-block:: text

   Parent thread on CPU:
     → pthread_create(worker1, function, args)
     → pthread_create(worker2, function, args)
     → pthread_create(worker3, function, args)
     ...

   Each worker thread:
     - Spawned dynamically by parent
     - Gets thread ID via TLS (Thread-Local Storage)
     - Can execute different code paths

**Eval Model** (AICPU hardware threading):

.. code-block:: text

   Runtime loads .so once into shared memory
   Runtime directs N physical CPU cores → execute entry point

   All cores simultaneously:
     1. Execute same entry point function
     2. Get thread ID from allocThreadIdx() (hardware mechanism)
     3. Thread ID based on physical CPU assignment + cluster membership

**Key Differences:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Pthread Model
     - AICPU Eval Model
   * - Thread creation
     - ``pthread_create()`` call
     - Hardware directs cores to entry point
   * - Thread ID source
     - Software TLS
     - Hardware CPU affinity (``sched_getcpu()``)
   * - Entry points
     - Can differ per thread
     - All cores execute same function
   * - Coordination
     - Parent/child relationship
     - Peer cores with barrier sync
   * - Identity
     - Software-assigned
     - Hardware physical core assignment

Hardware Thread Indexing: allocThreadIdx()
-------------------------------------------

The CANN runtime assigns thread IDs using the ``allocThreadIdx()`` mechanism:

.. code-block:: cpp

   // Simplified from CANN internal implementation
   // See: runtime/device/device_machine.cpp in CANN source
   int allocThreadIdx(int nrAicpu) {
       int threadIdx = -1;
       int cpu = sched_getcpu();              // Get physical CPU core ID
       cpumask.fetch_or(1 << cpu, ...);       // Register this CPU atomically

       // Barrier: wait for all nrAicpu cores to register
       while (__builtin_popcount(cpumask) != nrAicpu) {
           sched_yield();
       }

       // Assign sequential thread ID based on CPU cluster membership
       // Clusters = groups of 4 cores (CPUS_PER_CLUSTER = 4)
       auto maskval = cpumask.load(...);
       int cpuoff = 0;
       for (int i = 0; i < ...; i++) {
           int mask = (maskval >> cpuoff) & 0xF;
           if (__builtin_popcount(mask) >= MAX_SCHEDULE_AICPU_NUM) {
               threadIdx = threadIdx_++;
               break;
           }
           cpuoff += CPUS_PER_CLUSTER;
           if (cpu < cpuoff) break;
       }
       return threadIdx;
   }

**Characteristics:**

- **Barrier synchronization**: All threads wait until all ``nrAicpu`` cores have registered
- **Cluster-aware**: Groups threads by 4-core clusters (typical Ascend 910 hardware layout)
- **CPU affinity**: Thread ID tied to which physical CPU core is executing
- **Atomic coordination**: Uses atomic ``cpumask`` for lock-free registration
- **Deterministic**: Same core assignment produces same thread ID

**Hardware Layout (Ascend 910):**

.. code-block:: text

   AICPU Complex (8 ARM Cortex-A55 cores total):
     Cluster 0: [Core 0] [Core 1] [Core 2] [Core 3]
     Cluster 1: [Core 4] [Core 5] [Core 6] [Core 7]

   Maximum for scheduling: 4 cores

Thread IDs assigned based on cluster membership and registration order.

Launching Multi-core Kernels
-----------------------------

Use the ``aicpuNum`` parameter in ``rtAicpuKernelLaunchExWithArgs()``:

.. code-block:: cpp

   // Single-core launch (example 04-aicpu-kernel-launch)
   rtAicpuKernelLaunchExWithArgs(
       rtKernelType_t::KERNEL_TYPE_AICPU_KFC,
       "AST_DYN_AICPU",
       1,  // aicpuNum=1: execute on one core
       &rtArgs, nullptr, stream, 0);

   // Multi-core launch (example 05-aicpu-multithread)
   rtAicpuKernelLaunchExWithArgs(
       rtKernelType_t::KERNEL_TYPE_AICPU_KFC,
       "AST_DYN_AICPU",
       4,  // aicpuNum=4: execute on 4 cores (hardware maximum)
       &rtArgs, nullptr, stream, 0);

**Runtime Behavior (Multi-core Launch):**

1. Runtime loads backend server ``.so`` into shared memory (once)
2. Runtime directs ``aicpuNum`` physical CPU cores → execute entry point
3. Each core runs ``DynTileFwkBackendKernelServer()`` simultaneously
4. Cores coordinate via ``allocThreadIdx()`` to get unique thread IDs (0 to nrAicpu-1)
5. Each core processes its assigned portion of work

**Entry Point Execution:**

.. code-block:: text

   All cores enter: DynTileFwkBackendKernelServer(void *arg)

   Core 0 → allocThreadIdx() returns 0
   Core 1 → allocThreadIdx() returns 1
   Core 2 → allocThreadIdx() returns 2
   Core 3 → allocThreadIdx() returns 3

   Each core then partitions work based on its thread ID.

Thread ID in Kernel Code
-------------------------

**Option 1: pthread TLS Abstraction** (recommended, used in example 05):

.. code-block:: cpp

   // In kernel code - software abstraction over eval model
   static pthread_key_t g_thread_id_key;
   static int g_next_thread_id = 0;
   static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

   int get_thread_id() {
       void* id = pthread_getspecific(g_thread_id_key);
       if (id == nullptr) {
           // First time this thread calls - assign ID
           pthread_mutex_lock(&g_mutex);
           int my_id = g_next_thread_id++;
           pthread_mutex_unlock(&g_mutex);
           pthread_setspecific(g_thread_id_key, (void*)(intptr_t)my_id);
           return my_id;
       }
       return (int)(intptr_t)id;
   }

   // In init kernel:
   pthread_key_create(&g_thread_id_key, nullptr);
   g_next_thread_id = 0;

   // In main kernel:
   int thread_id = get_thread_id();  // Returns 0-3 for 4 cores

This approach abstracts the eval model for simplicity. Under the hood, the CANN runtime
has already assigned threads to specific CPU cores via ``allocThreadIdx()``.

**Why use pthread TLS?**

- ✅ Clean abstraction - hides hardware complexity
- ✅ Well-understood pattern
- ✅ Portable across different CANN versions
- ✅ Runtime has already coordinated cores via ``allocThreadIdx()``

**Option 2: Hardware Mechanism** (conceptual, educational):

.. code-block:: cpp

   // Conceptual - closer to actual hardware behavior
   int get_thread_id(int num_threads) {
       // In actual CANN runtime, allocThreadIdx() provides this
       // For demonstration, approximate with CPU affinity
       int cpu = sched_getcpu();
       return (cpu >= 0) ? (cpu % num_threads) : 0;
   }

This shows the underlying mechanism but is less portable. The pthread approach is recommended for application code.

Work Partitioning Strategies
-----------------------------

**Contiguous Chunks** (recommended for cache efficiency):

.. code-block:: cpp

   int thread_id = get_thread_id();
   int chunk_size = total_work / num_threads;
   int start = thread_id * chunk_size;
   int end = (thread_id == num_threads - 1) ?
             total_work : start + chunk_size;  // Last thread handles remainder

   for (int i = start; i < end; i++) {
       process(data[i]);  // Each core processes contiguous elements
   }

**Benefits:**

- Better cache locality (sequential access)
- Fewer cache line conflicts between cores
- Predictable memory access patterns

**Strided Access** (alternative, simpler load balancing):

.. code-block:: cpp

   for (int i = thread_id; i < total_work; i += num_threads) {
       process(data[i]);  // Core 0: 0,4,8...; Core 1: 1,5,9...
   }

**Trade-offs:**

- Simpler code (no remainder handling)
- Automatic load balancing
- Worse cache locality (strided access)
- Can cause cache line thrashing

Contiguous chunks are usually 2-3x faster due to cache effects.

Synchronization
---------------

**No synchronization needed when:**

- ✅ Each thread writes to **disjoint memory regions** (no overlapping writes)
- ✅ **Read-only access** to shared data (multiple readers OK)
- ✅ No dependencies between threads' work (embarrassingly parallel)

.. code-block:: cpp

   // Example: No sync needed - disjoint writes
   int start = thread_id * chunk_size;
   int end = start + chunk_size;

   for (int i = start; i < end; i++) {
       output[i] = input_a[i] + input_b[i];  // Each thread writes unique indices
   }

**Synchronization required when:**

- ❌ Multiple threads **modify shared state** (counters, flags, etc.)
- ❌ **Dependencies** between threads' work (producer-consumer, etc.)
- ❌ **Reduction operations** (sum, max, etc. across all threads)

For these cases:

- See `example 07-aicpu-atomic <../../examples/07-aicpu-atomic/README.md>`_ for atomic operations
- See `example 08-aicpu-queue <../../examples/08-aicpu-queue/README.md>`_ for lock-free inter-core communication

Example: 05-aicpu-multithread
------------------------------

See `examples/05-aicpu-multithread <../../examples/05-aicpu-multithread/README.md>`_
for a complete multi-core parallel execution example.

**What it demonstrates:**

- Eval model execution (4 cores execute same entry point)
- Thread ID assignment via pthread TLS abstraction
- Contiguous chunk work partitioning
- Disjoint memory access (no synchronization needed)
- Educational: AICPU compute for threading demo (production uses AICore)

**Key code locations:**

- Host launch: `main.cpp:440 <../../examples/05-aicpu-multithread/main.cpp#L440>`_ (``aicpuNum=4``)
- Thread ID: `kernel/parallel_add_kernel.cpp:56 <../../examples/05-aicpu-multithread/kernel/parallel_add_kernel.cpp#L56>`_
- Work partition: `kernel/parallel_add_kernel.cpp:131 <../../examples/05-aicpu-multithread/kernel/parallel_add_kernel.cpp#L131>`_

**Expected behavior:**

.. code-block:: bash

   $ ./05-aicpu-multithread
   === AICPU Multi-threaded Kernel Example ===
   Using 4 AICPU cores in parallel (hardware maximum)
   EDUCATIONAL: Production uses AICore for compute

   PASS: All 1024 elements correct (4 cores)
     Input_A[0]  = 0, Input_B[0]  = 0  →  Output[0]  = 0
     Input_A[N-1] = 1023, Input_B[N-1] = 511.5  →  Output[N-1] = 1534.5

   === End of AICPU Multi-threaded Example ===

Performance Considerations
--------------------------

**Parallelism Scaling:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Workload Type
     - Scaling Behavior
   * - Compute-bound
     - Near-linear scaling (3-4x with 4 cores)
   * - Memory-bound
     - Limited scaling (2-3x due to shared bandwidth)
   * - Very small datasets
     - No scaling or slowdown (overhead > benefit)

**Memory Bandwidth:**

- All cores share HBM bandwidth (~900 GB/s on Ascend 910)
- Memory-bound operations saturate bandwidth with 2-3 cores
- Compute-intensive operations scale better

**Cache Effects:**

- Each core has L1 (32KB I + 32KB D) and shared L2 caches
- Contiguous chunk assignment → better cache hit rates
- Strided access → cache line thrashing → worse performance

**When Multi-core Helps:**

- ✅ Large datasets (> 1MB per core, ideally > 10MB total)
- ✅ Compute-intensive operations (math, transformations, compression)
- ✅ Embarrassingly parallel workloads (no inter-thread communication)
- ✅ Operations with high compute-to-memory ratio

**When Single-core is Better:**

- ❌ Small datasets (< 100KB total) - overhead dominates
- ❌ Pure memory copy operations - bandwidth saturated quickly
- ❌ Complex inter-thread synchronization - contention overhead
- ❌ Irregular memory access patterns - cache conflicts

**Rule of Thumb:**

Use multi-core when: ``total_work_size > 1MB AND compute_ops_per_byte > 10``

Key Takeaways
-------------

✅ **Eval Model:**

- AICPU uses eval model, not pthread spawning
- All cores execute same entry point simultaneously
- Thread identity from hardware (``allocThreadIdx``), not just software

✅ **Hardware Threading:**

- Barrier synchronization via atomic CPU mask
- Cluster-aware assignment (4 cores per cluster)
- Physical CPU affinity determines thread ID

✅ **Work Distribution:**

- Contiguous chunks preferred (cache efficiency)
- Disjoint memory access eliminates sync overhead
- Last thread handles remainder for load balancing

✅ **Performance:**

- 4x parallelism limited by memory bandwidth
- Multi-core only beneficial for larger workloads
- Cache locality matters significantly
- **AICPU for control, AICore for compute** in production
