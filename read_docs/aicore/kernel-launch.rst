AICORE Kernel Launch
====================

Launching AICORE kernels from host code.

Overview
--------

This page covers AICORE kernel launching concepts and APIs. For a complete working example,
see :file:`examples/09-aicore-basic/` which demonstrates AICORE kernel loading and execution.

Kernel Format
-------------

AICORE kernels are compiled to ``.o`` binary format:

.. code-block:: text

   kernel.py (TIK)     →  compiler  →  kernel.o
   kernel.cpp (Ascend C)  →  compiler  →  kernel.o

The ``.o`` file contains:

- Compiled instructions for Cube, Vector, Scalar units
- Metadata (argument layout, resource requirements)
- Symbol information

Launch API
----------

.. code-block:: cpp

   PlatformKernel platform_kernel_load(
       const void* bin,      // .o file contents
       size_t size,          // Size of .o file
       const char* name      // Kernel entry point name
   );

   int platform_kernel_launch(
       PlatformKernel kernel, // Loaded kernel handle
       uint32_t blocks,       // Number of AICORE blocks to use
       void* args,            // Pointer to argument struct
       size_t args_size,      // sizeof(argument struct)
       PlatformStream stream  // Stream (NULL for default)
   );

   void platform_kernel_unload(PlatformKernel kernel);

Block Count Selection
---------------------

The ``blocks`` parameter determines parallelism:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │  Choosing Block Count                                       │
   │                                                             │
   │  blocks = 1:   One AICORE, sequential                       │
   │                Good for: small data, debugging              │
   │                                                             │
   │  blocks = 24:  All AICOREs (A3), maximum parallelism        │
   │                Good for: large data, production             │
   │                                                             │
   │  blocks = N:   Data split across N blocks                   │
   │                Each processes: total_elements / N           │
   └─────────────────────────────────────────────────────────────┘

Guidelines:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Data Size
     - Blocks
     - Reason
   * - < 10K elements
     - 1-4
     - Overhead dominates
   * - 10K - 1M elements
     - 4-16
     - Balance parallelism/overhead
   * - > 1M elements
     - 24
     - Maximum throughput

Argument Structure
------------------

Arguments are passed as a packed struct:

.. code-block:: cpp

   // Host and kernel must use identical struct
   struct MatMulArgs {
       void* A;          // GM pointer [M, K]
       void* B;          // GM pointer [K, N]
       void* C;          // GM pointer [M, N]
       int32_t M;
       int32_t N;
       int32_t K;
       int32_t _pad;     // Alignment padding
   };

Rules:

1. All pointers must be device pointers (from ``platform_malloc``)
2. Use fixed-width types (``int32_t``, not ``int``)
3. Pad to 8-byte alignment
4. Total size passed to ``platform_kernel_launch``

Kernel Inside (Conceptual)
--------------------------

What the kernel sees:

.. code-block:: cpp

   // Pseudo-code for kernel internals
   __kernel__ void vec_add_kernel(VecAddArgs* args) {
       int block_idx = get_block_idx();    // 0 to 23
       int block_dim = get_block_dim();    // 24

       // Calculate my portion
       int elements_per_block = args->n / block_dim;
       int my_start = block_idx * elements_per_block;
       int my_end = my_start + elements_per_block;

       // Process my portion
       float* a = (float*)args->a;
       float* b = (float*)args->b;
       float* c = (float*)args->c;

       for (int i = my_start; i < my_end; i += VECTOR_WIDTH) {
           // Load to UB
           // Vector add in UB
           // Store from UB
       }
   }

Error Handling
--------------

.. code-block:: cpp

   PlatformKernel kernel = platform_kernel_load(bin, size, "entry");
   if (!kernel) {
       printf("Failed to load kernel\n");
       return 1;
   }

   int ret = platform_kernel_launch(kernel, blocks, &args, sizeof(args), stream);
   if (ret != PLATFORM_SUCCESS) {
       printf("Launch failed: %d\n", ret);
       return 1;
   }

   ret = platform_stream_sync(stream);
   if (ret != PLATFORM_SUCCESS) {
       printf("Kernel execution failed: %d\n", ret);
       return 1;
   }

.. note::

   Kernel errors (out of bounds, etc.) may only be detected at sync time.
   Always check ``platform_stream_sync()`` return value.
