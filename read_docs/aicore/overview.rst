AICORE Overview
===============

AICORE is the primary compute engine of Ascend NPUs, designed for
high-throughput tensor operations.

Architecture
------------

Each AICORE block contains two specialized units:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                        AICORE Block                             │
   │                                                                 │
   │  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
   │  │    Cube Unit (AIC)      │  │     Vector Unit (AIV)       │  │
   │  │                         │  │                             │  │
   │  │  Matrix Multiplication  │  │  Element-wise Operations    │  │
   │  │  ───────────────────    │  │  ─────────────────────────  │  │
   │  │  C[M,N] = A[M,K]×B[K,N] │  │  y = f(x) element-by-element│  │
   │  │                         │  │                             │  │
   │  │  Input:  L0A, L0B       │  │  Memory: UB (256KB)         │  │
   │  │  Output: L0C            │  │                             │  │
   │  │                         │  │  Operations:                │  │
   │  │  Ops:                   │  │  • Add, Mul, Div            │  │
   │  │  • FP16 MatMul          │  │  • Exp, Log, Sqrt           │  │
   │  │  • INT8 MatMul          │  │  • ReLU, Sigmoid, Tanh      │  │
   │  │  • BF16 MatMul          │  │  • Reduce (sum, max, min)   │  │
   │  └─────────────────────────┘  └─────────────────────────────┘  │
   │                                                                 │
   │  ┌─────────────────────────────────────────────────────────┐   │
   │  │                  Scalar Unit                             │   │
   │  │  • Control flow (loops, branches)                        │   │
   │  │  • Address calculation                                   │   │
   │  │  • Data movement coordination                            │   │
   │  └─────────────────────────────────────────────────────────┘   │
   │                                                                 │
   │  ┌─────────────────────────────────────────────────────────┐   │
   │  │                  L1 Buffer (1MB)                         │   │
   │  │  Staging area for Cube and Vector data                   │   │
   │  └─────────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────────┘

Cube Unit (AIC) Details
-----------------------

The Cube unit performs matrix multiplication:

.. code-block:: text

   Operation: C = A × B

   Where:
   • A is M×K matrix (loaded to L0A)
   • B is K×N matrix (loaded to L0B)
   • C is M×N matrix (accumulated in L0C)

   Typical tile sizes (A3):
   • M = 16, K = 16, N = 16 (FP16)
   • M = 16, K = 32, N = 16 (INT8)

Throughput (A3 / 910C):

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Data Type
     - TFLOPS
     - Notes
   * - FP16
     - ~320
     - Native support
   * - BF16
     - ~320
     - Native support
   * - INT8
     - ~640
     - Double throughput
   * - FP32
     - ~160
     - Emulated via FP16

Vector Unit (AIV) Details
-------------------------

The Vector unit performs SIMD operations:

.. code-block:: text

   Operation: y[i] = f(x[i]) for all i

   Memory: Unified Buffer (UB) - 256KB
   Vector width: 256 elements (typical)

   Supported operations:
   • Arithmetic: add, sub, mul, div, sqrt, rsqrt
   • Transcendental: exp, log, pow, sin, cos
   • Activation: relu, sigmoid, tanh, gelu
   • Comparison: max, min, gt, lt, eq
   • Reduction: sum, max, min (across vector)
   • Type conversion: fp16↔fp32, int8↔fp16

Programming Models
------------------

Three ways to program AICORE:

**1. TIK (Tensor Iterator Kernel)**

- Python-based DSL
- Huawei's official high-level language
- Generates AICORE binary

.. code-block:: python

   from tik import Tik

   tik_instance = Tik()
   # ... TIK code ...
   tik_instance.BuildCCE(kernel_name="my_kernel", ...)

**2. Ascend C**

- C++ extension for AICORE
- Lower level than TIK
- More control over memory and instructions

.. code-block:: cpp

   __global__ void my_kernel(__gm__ float* input, __gm__ float* output) {
       // Ascend C code
   }

**3. PTO-ISA (Direct Instructions)**

- Lowest level
- Direct control over all units
- Used by this guide for teaching

.. code-block:: text

   # PTO-ISA pseudo-assembly
   LOAD  UB[0:1024], GM[input_addr]
   VMUL  UB[0:1024], UB[0:1024], 2.0
   STORE GM[output_addr], UB[0:1024]

Execution Model
---------------

AICORE kernels execute in blocks:

.. code-block:: cpp

   // Host launches kernel on N blocks
   platform_kernel_launch(kernel, N, &args, sizeof(args), stream);

Inside kernel:

.. code-block:: text

   block_idx = get_block_idx()  // 0 to N-1
   block_dim = get_block_dim()  // N

   // Each block processes portion of data
   my_start = block_idx * elements_per_block
   my_end = my_start + elements_per_block

Pipeline Execution
------------------

AICORE uses pipeline parallelism:

.. code-block:: text

   Time →
   ──────────────────────────────────────────────────────────
   Scalar: [addr calc] [addr calc] [addr calc] [addr calc]
   Load:         [load]      [load]      [load]      [load]
   Cube:              [mmul]      [mmul]      [mmul]
   Vector:                 [activ]     [activ]     [activ]
   Store:                       [store]     [store]    [store]

Different stages can overlap, maximizing throughput.

When to Use AICORE
------------------

✅ **Good for AICORE:**

- Matrix multiplication (GEMM)
- Convolution (as tiled GEMM)
- Large element-wise operations
- Batch normalization
- Softmax

❌ **Better on AICPU:**

- Dynamic shape operations
- Sparse operations
- Small tensors (< 1000 elements)
- Complex control flow

Example Code
------------

Full working example: :file:`examples/09-aicore-basic/`

Host code (launches AICORE kernel):

.. literalinclude:: ../../examples/09-aicore-basic/main.cpp
   :language: cpp
   :caption: 09-aicore-basic/main.cpp
   :lines: 1-80

AICORE kernel (PTO-ISA):

.. literalinclude:: ../../examples/09-aicore-basic/kernel.pto
   :language: text
   :caption: 09-aicore-basic/kernel.pto
