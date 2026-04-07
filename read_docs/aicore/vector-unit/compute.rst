Vector Compute Operations
=========================

Element-wise operations using PTO-ISA vector instructions.

Instruction Categories
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Category
     - Operations
     - Example
   * - Arithmetic
     - add, sub, mul, div
     - ``VADD dst, src1, src2``
   * - Math
     - sqrt, rsqrt, exp, log
     - ``VEXP dst, src``
   * - Activation
     - relu, sigmoid, tanh
     - ``VRELU dst, src``
   * - Comparison
     - max, min, gt, lt
     - ``VMAX dst, src1, src2``
   * - Reduction
     - sum, max, min
     - ``VREDUCE_SUM dst, src``
   * - Type Convert
     - fp16↔fp32, int8↔fp16
     - ``VCONV_F16_F32 dst, src``
   * - Memory
     - load, store, copy
     - ``VLOAD dst, addr``

Basic Vector Operations
-----------------------

**Vector Add (VADD)**

.. code-block:: text

   VADD  ub_dst[0:N], ub_src1[0:N], ub_src2[0:N]

   // C equivalent:
   for (int i = 0; i < N; i++) {
       dst[i] = src1[i] + src2[i];
   }

**Vector Multiply (VMUL)**

.. code-block:: text

   VMUL  ub_dst[0:N], ub_src1[0:N], ub_src2[0:N]

**Vector Scalar (VMULS)**

.. code-block:: text

   VMULS ub_dst[0:N], ub_src[0:N], scalar_value

   // Multiply all elements by scalar
   for (int i = 0; i < N; i++) {
       dst[i] = src[i] * scalar_value;
   }

Activation Functions
--------------------

**ReLU**

.. code-block:: text

   VRELU ub_dst[0:N], ub_src[0:N]

   // max(0, x)
   for (int i = 0; i < N; i++) {
       dst[i] = src[i] > 0 ? src[i] : 0;
   }

**Sigmoid**

.. code-block:: text

   VSIGMOID ub_dst[0:N], ub_src[0:N]

   // 1 / (1 + exp(-x))

**GELU**

.. code-block:: text

   VGELU ub_dst[0:N], ub_src[0:N]

   // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Reduction Operations
--------------------

**Sum Reduction**

.. code-block:: text

   VREDUCE_SUM ub_dst[0], ub_src[0:N]

   // Sum all elements
   dst[0] = 0;
   for (int i = 0; i < N; i++) {
       dst[0] += src[i];
   }

**Max Reduction**

.. code-block:: text

   VREDUCE_MAX ub_dst[0], ub_src[0:N]

   // Find maximum
   dst[0] = src[0];
   for (int i = 1; i < N; i++) {
       dst[0] = max(dst[0], src[i]);
   }

Type Conversion
---------------

**FP16 to FP32**

.. code-block:: text

   VCONV_F16_F32 ub_dst[0:N], ub_src[0:N]

   // Widen FP16 to FP32
   // Note: dst needs 2x space

**FP32 to FP16**

.. code-block:: text

   VCONV_F32_F16 ub_dst[0:N], ub_src[0:N]

   // Narrow FP32 to FP16

Data Movement
-------------

**Load from GM to UB**

.. code-block:: text

   VLOAD ub_dst[0:N], gm_addr, N

**Store from UB to GM**

.. code-block:: text

   VSTORE gm_addr, ub_src[0:N], N

**Copy within UB**

.. code-block:: text

   VCOPY ub_dst[0:N], ub_src[0:N]

Fused Operations
----------------

Some operations can be fused for efficiency:

**Multiply-Add (MAD)**

.. code-block:: text

   VMAD ub_dst[0:N], ub_a[0:N], ub_b[0:N], ub_c[0:N]

   // dst = a * b + c

**Add-ReLU**

.. code-block:: text

   VADD_RELU ub_dst[0:N], ub_src1[0:N], ub_src2[0:N]

   // dst = relu(src1 + src2)

Softmax Example
---------------

Complete softmax implementation:

.. code-block:: text

   // Input: ub_x[0:N]
   // Output: ub_y[0:N]
   // Temp: ub_tmp[0:N]

   // 1. Find max for numerical stability
   VREDUCE_MAX ub_max[0], ub_x[0:N]

   // 2. Subtract max and exp
   VSUBS ub_tmp[0:N], ub_x[0:N], ub_max[0]
   VEXP ub_tmp[0:N], ub_tmp[0:N]

   // 3. Sum of exp
   VREDUCE_SUM ub_sum[0], ub_tmp[0:N]

   // 4. Reciprocal
   VRECIP ub_sum[0], ub_sum[0]

   // 5. Normalize
   VMULS ub_y[0:N], ub_tmp[0:N], ub_sum[0]

Performance Tips
----------------

1. **Maximize vector width** - Process 256 elements (FP16) per instruction
2. **Fuse operations** - Use combined instructions when available
3. **Minimize data movement** - Keep data in UB for multiple operations
4. **Use appropriate precision** - FP16 has 2x throughput of FP32

Example Code
------------

Full working example: :file:`examples/14-vector-simd/`

Host code demonstrating vector operations:

.. literalinclude:: ../../../examples/14-vector-simd/main.cpp
   :language: cpp
   :caption: 14-vector-simd/main.cpp
   :lines: 1-80

AICORE vector kernel (PTO-ISA):

.. literalinclude:: ../../../examples/14-vector-simd/kernel.pto
   :language: text
   :caption: 14-vector-simd/kernel.pto

Pipeline Synchronization
=========================

Managing pipeline dependencies in the Vector unit.

Pipeline Overview
-----------------

The Vector unit has multiple pipeline stages:

.. code-block:: text

   Time →
   ────────────────────────────────────────────────────────
   Load Pipe:    [LD1] [LD2] [LD3] [LD4]
   Compute Pipe:       [C1 ] [C2 ] [C3 ] [C4 ]
   Store Pipe:              [ST1] [ST2] [ST3] [ST4]

   Without sync: pipes run independently (overlap = good)
   With sync:    ensure data is ready before next stage

Why Synchronization?
--------------------

.. code-block:: text

   // Problem: compute before data is loaded

   VLOAD ub_x[0:N], gm_addr      // Takes ~100 cycles
   VMUL ub_y[0:N], ub_x[0:N], 2  // Starts immediately!
                                  // But ub_x not ready yet!

   // Solution: sync after load

   VLOAD ub_x[0:N], gm_addr
   PIPE_BARRIER                   // Wait for load to complete
   VMUL ub_y[0:N], ub_x[0:N], 2  // Now ub_x is valid

Synchronization Primitives
---------------------------

**PIPE_BARRIER**

Wait for all pending operations on a pipe:

.. code-block:: text

   VLOAD ub_a, gm_addr_a
   VLOAD ub_b, gm_addr_b
   PIPE_BARRIER           // Wait for both loads
   VADD ub_c, ub_a, ub_b  // Safe to use ub_a, ub_b

**SET_FLAG / WAIT_FLAG**

Fine-grained synchronization with flags:

.. code-block:: text

   VLOAD ub_a, gm_addr_a
   SET_FLAG flag_a        // Signal when ub_a is ready

   VLOAD ub_b, gm_addr_b
   SET_FLAG flag_b        // Signal when ub_b is ready

   // Can do other work here...

   WAIT_FLAG flag_a       // Wait for ub_a only
   // Use ub_a

   WAIT_FLAG flag_b       // Wait for ub_b
   // Use ub_a and ub_b

Common Synchronization Patterns
--------------------------------

**Pattern 1: Load-Compute-Store**

.. code-block:: text

   for each tile:
       VLOAD ub_in, gm_in + offset
       PIPE_BARRIER              // Wait for load

       VMUL ub_out, ub_in, 2.0
       // More compute...
       PIPE_BARRIER              // Wait for compute

       VSTORE gm_out + offset, ub_out
       PIPE_BARRIER              // Wait for store (if reusing buffer)

**Pattern 2: Double Buffering**

.. code-block:: text

   // Tile 0: Load to A
   VLOAD ub_a, gm_in + 0
   SET_FLAG load_a

   for tile = 1 to N-1:
       // Load next tile to B
       VLOAD ub_b, gm_in + tile * TILE_SIZE
       SET_FLAG load_b

       // Wait for A and compute
       WAIT_FLAG load_a
       VMUL ub_a, ub_a, 2.0
       PIPE_BARRIER

       // Store A
       VSTORE gm_out + (tile-1) * TILE_SIZE, ub_a
       SET_FLAG store_a

       // Swap A and B
       WAIT_FLAG store_a
       WAIT_FLAG load_b
       swap(ub_a, ub_b)
       swap(load_a, load_b)

   // Final tile
   WAIT_FLAG load_a
   VMUL ub_a, ub_a, 2.0
   PIPE_BARRIER
   VSTORE gm_out + (N-1) * TILE_SIZE, ub_a

**Pattern 3: Reduction**

.. code-block:: text

   // Sum all elements
   VLOAD ub_data, gm_in, N
   PIPE_BARRIER

   VREDUCE_SUM ub_sum, ub_data
   PIPE_BARRIER            // Wait for reduction

   VSTORE gm_out, ub_sum, 1
   PIPE_BARRIER            // Ensure store completes

Synchronization Performance
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Issue
     - Impact
   * - Too many barriers
     - Serializes execution, loses pipeline benefit
   * - Too few barriers
     - Data hazards, incorrect results
   * - Barrier placement
     - At tile boundaries, not per-instruction

Rules of Thumb
--------------

1. **Barrier after loads** - Before using loaded data
2. **Barrier after compute** - Before storing results
3. **Double buffer** - Overlap load/compute/store
4. **Use flags** - For fine-grained control
5. **Profile** - Find actual bottlenecks

Debugging Sync Issues
---------------------

Symptoms of missing synchronization:

- Random incorrect results
- Results change between runs
- Works with small data, fails with large
- Works with 1 block, fails with many

To debug:

1. Add barriers after every operation (will be slow but correct)
2. Remove barriers one at a time
3. Find minimum barriers needed
