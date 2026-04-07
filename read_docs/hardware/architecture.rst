Architecture Overview
=====================

Hardware Components
-------------------

.. code-block:: text

   Host CPU
     ↓ PCIe/UB (~3μs)
   Device:
     ├── AICPU (control processors, ARM-based)
     │    ↓ On-chip (~0μs)
     └── AICore Blocks (24 blocks)
          Each block contains:
          - 1 Cube Core (Matrix ops)
            - Scalar Unit
            - L0A Buffer
            - L0B Buffer
            - L0C Buffer
            - L1 Buffer
            - Cube Compute Unit
          - 2 Vector Cores (SIMD ops)
            - Scalar Unit
            - UB Buffer
            - Vector Compute Unit

.. list-table:: Latency Summary
   :header-rows: 1

   * - From
     - To
     - Latency
     - Notes
   * - Host CPU
     - AICPU
     - ~3μs
     - PCI/UB transfer
   * - Host CPU
     - AICore
     - ~3μs
     - PCI/UB transfer
   * - AICPU
     - AICore
     - ~0μs
     - Tightly coupled on-chip

.. note::

   AICPU and AICore are tightly coupled on the same chip, enabling near-zero
   latency coordination. AICPU controls AICore execution through shared registers,
   atomics, and queues.

AICORE Architecture
-------------------

Each AICORE block contains:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                        AICORE Block                             │
   │                                                                 │
   │  ┌───────────────────────────┐  ┌──────────────────────────┐    │
   │  │      Cube Core            │  │   Vector Core #1         │    │
   │  │   (Matrix Operations)     │  │   (SIMD Operations)      │    │
   │  │                           │  │                          │    │
   │  │  ┌─────────────────────┐  │  │  ┌────────────────────┐  │    │
   │  │  │   Scalar Unit       │  │  │  │   Scalar Unit      │  │    │
   │  │  │   (Control flow)    │  │  │  │   (Control flow)   │  │    │
   │  │  └─────────────────────┘  │  │  └────────────────────┘  │    │
   │  │                           │  │                          │    │
   │  │  ┌─────────────────────┐  │  │  ┌────────────────────┐  │    │
   │  │  │   L0A Buffer        │  │  │  │   UB Buffer        │  │    │
   │  │  │   (Matrix input A)  │  │  │  │   (Vector data)    │  │    │
   │  │  └─────────────────────┘  │  │  └────────────────────┘  │    │
   │  │  ┌─────────────────────┐  │  │                          │    │
   │  │  │   L0B Buffer        │  │  │  ┌────────────────────┐  │    │
   │  │  │   (Matrix input B)  │  │  │  │  Vector Compute    │  │    │
   │  │  └─────────────────────┘  │  │  │  Unit (SIMD)       │  │    │
   │  │  ┌─────────────────────┐  │  │  └────────────────────┘  │    │
   │  │  │   L0C Buffer        │  │  │                          │    │
   │  │  │   (Matrix output)   │  │  └──────────────────────────┘    │
   │  │  └─────────────────────┘  │                                  │
   │  │                           │  ┌──────────────────────────┐    │
   │  │  ┌─────────────────────┐  │  │   Vector Core #2         │    │
   │  │  │   L1 Buffer         │  │  │   (SIMD Operations)      │    │
   │  │  │   (Shared memory)   │  │  │                          │    │
   │  │  └─────────────────────┘  │  │  ┌────────────────────┐  │    │
   │  │                           │  │  │   Scalar Unit      │  │    │
   │  │  ┌─────────────────────┐  │  │  │   (Control flow)   │  │    │
   │  │  │  Cube Compute Unit  │  │  │  └────────────────────┘  │    │
   │  │  │  (Matrix multiply)  │  │  │                          │    │
   │  │  └─────────────────────┘  │  │  ┌────────────────────┐  │    │
   │  │                           │  │  │   UB Buffer        │  │    │
   │  └───────────────────────────┘  │  │   (Vector data)    │  │    │
   │                                 │  └────────────────────┘  │    │
   │                                 │                          │    │
   │                                 │  ┌────────────────────┐  │    │
   │                                 │  │  Vector Compute    │  │    │
   │                                 │  │  Unit (SIMD)       │  │    │
   │                                 │  └────────────────────┘  │    │
   │                                 │                          │    │
   │                                 └──────────────────────────┘    │
   └─────────────────────────────────────────────────────────────────┘

Cube Core
^^^^^^^^^

- **Purpose**: Matrix multiplication (the "tensor core")
- **Operation**: C = A × B where A is [M,K], B is [K,N]
- **Throughput**: High TFLOPS for FP16/BF16 matrix ops
- **Components**:
  - Scalar Unit: Control flow and address calculation
  - L0A Buffer: Matrix input A
  - L0B Buffer: Matrix input B
  - L0C Buffer: Matrix output/accumulator
  - L1 Buffer: Shared memory for intermediate data
  - Cube Compute Unit: Performs the actual matrix multiplication

Vector Cores (2 per AICore block)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Purpose**: Element-wise operations, reductions, activations
- **Operation**: SIMD operations on vectors
- **Throughput**: Lower than Cube Core, but more flexible
- **Components** (each core has):
  - Scalar Unit: Control flow and address calculation
  - UB Buffer (Unified Buffer): Fast memory for vector data
  - Vector Compute Unit: Performs SIMD operations

AICPU Architecture
------------------

AICPU are ARM-based control processors:

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │             AICPU                       │
   │ ┌─────────────────────────────────────┐ │
   │ │  ARM Cortex cores                   │ │
   │ │  - Standard C/C++ execution         │ │
   │ │  - Full instruction set             │ │
   │ │  - Can access HBM directly          │ │
   │ └─────────────────────────────────────┘ │
   │                                         │
   │ Use cases:                              │
   │  - Dynamic shape operations             │
   │  - Complex control flow                 │
   │  - Sparse operations                    │
   │  - Data preprocessing                   │
   └─────────────────────────────────────────┘

Execution Model
---------------

Kernels are launched from host and execute on device:

.. code-block:: text

   Host CPU                    Device (AICPU + AICore)
   ────────                    ───────────────────────
       │
       │  1. aclInit(nullptr)
       │─────────────────────────► Initialize runtime
       │
       │  2. aclrtMalloc(&devPtr, size, policy)
       │─────────────────────────► Allocate HBM
       │
       │  3. aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)
       │─────────────────────────► DMA transfer
       │
       │  4. aclrtLaunchKernel(...)
       │─────────────────────────► Submit to queue
       │                                │
       │                                ▼
       │                          ┌──────────┐
       │                          │ AICPU or │
       │                          │ AICore   │
       │                          │ executes │
       │                          └──────────┘
       │                                │
       │  5. aclrtSynchronizeStream()   │
       │◄───────────────────────────────┘ Wait
       │
       │  6. aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_HOST)
       │─────────────────────────► DMA transfer
       │
      Done

Block Parallelism
-----------------

AICore blocks execute in parallel. When launching a kernel, you specify how many blocks to use:

.. code-block:: cpp

   // Launch on 4 AICore blocks in parallel
   platform_kernel_launch(kernel, 4, &args, sizeof(args), stream);

Hardware Execution
^^^^^^^^^^^^^^^^^^

The device contains 24 AICore blocks. When you launch with ``block_dim=4``:

.. code-block:: text

   Device (24 AICore blocks total)
   ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
   │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │ ...
   └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
      ACTIVE    ACTIVE    ACTIVE    ACTIVE     IDLE      IDLE

   Only 4 blocks are used. The other 20 blocks remain idle.

Block Properties
^^^^^^^^^^^^^^^^

Each AICore block:

- **Executes independently**: No synchronization between blocks during execution
- **Runs the same kernel code**: All blocks execute identical instructions
- **Has isolated memory**: Each block has its own L0A/L0B/L0C, L1, and UB buffers
- **Knows its identity**: Kernel code can query ``block_idx`` to determine which block it is
- **Processes a data partition**: Typically, each block handles a portion of the total workload

Work Division Pattern
^^^^^^^^^^^^^^^^^^^^^

A common pattern is dividing data equally across blocks:

.. code-block:: text

   Example: Processing 1024 elements on 4 blocks

   Block 0 (block_idx=0): elements [  0, 255]  ───┐
   Block 1 (block_idx=1): elements [256, 511]  ───┼─► Execute in parallel
   Block 2 (block_idx=2): elements [512, 767]  ───┤
   Block 3 (block_idx=3): elements [768, 1023] ───┘

Inside the kernel:

.. code-block:: cpp

   // Each block computes its data range
   uint32_t block_idx = get_block_idx();
   uint32_t block_dim = get_block_dim();  // 4 in this example

   uint32_t total_elements = 1024;
   uint32_t elements_per_block = total_elements / block_dim;  // 256

   uint32_t start = block_idx * elements_per_block;      // 0, 256, 512, 768
   uint32_t end = start + elements_per_block;            // 256, 512, 768, 1024

   // Process elements [start, end)
   for (uint32_t i = start; i < end; i++) {
       // Process element i
   }

Memory Isolation
^^^^^^^^^^^^^^^^

Each block's memory is completely isolated:

.. code-block:: text

   Block 0               Block 1               Block 2
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │ L0A/B/C  │         │ L0A/B/C  │         │ L0A/B/C  │
   │ L1 (1MB) │  ←──────┼────X─────┼──────→  │ L1 (1MB) │
   │ UB       │   No    │ UB       │   No    │ UB       │
   └──────────┘ Sharing └──────────┘ Sharing └──────────┘

Blocks cannot directly share or communicate through their local buffers.
All inter-block communication must go through global HBM.
