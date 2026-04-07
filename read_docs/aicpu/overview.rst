AICPU Overview
==============

AICPU (AI Control Processing Unit) are ARM-based processors on the Ascend NPU
that handle operations not suited for the AICORE compute units.

What is AICPU?
--------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                        AICPU                                │
   │  ┌───────────────────────────────────────────────────────┐  │
   │  │  ARM Cortex-A55 cores (8 cores typical)               │  │
   │  │                                                       │  │
   │  │  • Full C/C++ execution environment                   │  │
   │  │  • Direct HBM access                                  │  │
   │  │  • Standard library support                           │  │
   │  │  • Low launch overhead from host                      │  │
   │  └───────────────────────────────────────────────────────┘  │
   │                                                             │
   │  Best for:                                                  │
   │  • Dynamic shape operations                                 │
   │  • Complex control flow                                     │
   │  • Sparse operations                                        │
   │  • Data preprocessing                                       │
   │  • Operations with small tensor sizes                       │
   └─────────────────────────────────────────────────────────────┘

AICPU vs AICORE
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - AICPU
     - AICORE
   * - Processor Type
     - ARM Cortex cores
     - Custom DSA (Cube + Vector)
   * - Kernel Format
     - ``.so`` shared library
     - ``.o`` binary
   * - Programming
     - Standard C/C++
     - TIK / Ascend C / PTO-ISA
   * - Throughput
     - Lower (CPU-class)
     - Higher (accelerator-class)
   * - Flexibility
     - High (any algorithm)
     - Medium (data-parallel)
   * - Use Case
     - Control, dynamic shapes
     - Matrix ops, convolution

When to Use AICPU
-----------------

✅ **Good for AICPU:**

- Gather/Scatter with dynamic indices
- Non-maximum suppression (NMS)
- Unique/Sort operations
- String processing
- Custom control flow

❌ **Better on AICORE:**

- Matrix multiplication
- Convolution
- Element-wise operations (large tensors)
- Reduction operations
- Batch normalization

Kernel Structure
----------------

AICPU kernels are standard shared libraries:

.. code-block:: cpp

   // my_kernel.cpp - AICPU kernel

   extern "C" {

   struct MyKernelArgs {
       void* input;
       void* output;
       int size;
   };

   void my_kernel_entry(MyKernelArgs* args) {
       float* in = (float*)args->input;
       float* out = (float*)args->output;

       for (int i = 0; i < args->size; i++) {
           out[i] = in[i] * 2.0f;  // Simple operation
       }
   }

   }  // extern "C"

Build as shared library:

.. code-block:: bash

   aarch64-linux-gnu-g++ -shared -fPIC -o my_kernel.so my_kernel.cpp

Launch from host:

.. code-block:: cpp

   // Read .so file into memory
   void* so_data = read_file("my_kernel.so", &so_size);

   // Pack arguments
   MyKernelArgs args = {dev_input, dev_output, 1024};

   // Launch
   platform_aicpu_launch(so_data, so_size, "my_kernel_entry",
                         &args, sizeof(args), stream);

Performance Considerations
--------------------------

1. **Launch overhead**: ~3μs from host (same as AICORE)
2. **Compute speed**: ARM cores are slower than AICORE for parallel ops
3. **Memory bandwidth**: Full HBM bandwidth available
4. **Parallelism**: Can use multiple AICPU cores

.. tip::

   Use AICPU as a "fallback" for operations that don't map well to
   AICORE. Modern frameworks automatically select the best processor.

Example Code
------------

Full working example: :file:`examples/04-aicpu-kernel-launch/`

Host code (launches AICPU kernel):

.. literalinclude:: ../../examples/04-aicpu-kernel-launch/main.cpp
   :language: cpp
   :caption: 04-aicpu-kernel-launch/main.cpp
   :lines: 1-80

AICPU kernel implementation:

.. literalinclude:: ../../examples/04-aicpu-kernel-launch/kernel/scale_kernel.cpp
   :language: cpp
   :caption: 04-aicpu-kernel-launch/kernel/scale_kernel.cpp
