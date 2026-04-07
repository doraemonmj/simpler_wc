UB Allocation
=============

Managing Unified Buffer (UB) memory in AICORE kernels.

UB Overview
-----------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                 Unified Buffer (UB)                         │
   │                      256 KB                                 │
   │                                                             │
   │  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
   │  │  Buffer A   │  Buffer B   │  Buffer C   │  Workspace  │ │
   │  │   64 KB     │   64 KB     │   64 KB     │    64 KB    │ │
   │  └─────────────┴─────────────┴─────────────┴─────────────┘ │
   │                                                             │
   │  You decide the layout. Above is just one example.         │
   └─────────────────────────────────────────────────────────────┘

Allocation Strategies
---------------------

**Strategy 1: Static Allocation**

Divide UB at compile time:

.. code-block:: cpp

   // Define buffer layout (compile-time constants)
   #define UB_SIZE         (256 * 1024)
   #define INPUT_BUF_SIZE  (64 * 1024)
   #define OUTPUT_BUF_SIZE (64 * 1024)
   #define TEMP_BUF_SIZE   (128 * 1024)

   // Offsets in UB
   #define INPUT_OFFSET    0
   #define OUTPUT_OFFSET   (INPUT_BUF_SIZE)
   #define TEMP_OFFSET     (INPUT_BUF_SIZE + OUTPUT_BUF_SIZE)

**Strategy 2: Double Buffering**

Overlap load and compute:

.. code-block:: cpp

   // Two buffers for ping-pong
   #define BUF_A_OFFSET  0
   #define BUF_B_OFFSET  (128 * 1024)

   // While computing on A, load next data to B
   // While computing on B, load next data to A

**Strategy 3: Dynamic Layout**

Calculate at runtime based on tensor sizes:

.. code-block:: cpp

   // Calculate buffer sizes based on tile dimensions
   int tile_m = 64;
   int tile_n = 128;
   int input_size = tile_m * tile_n * sizeof(float);
   int output_size = tile_m * tile_n * sizeof(float);

   // Verify fits in UB
   assert(input_size + output_size <= UB_SIZE);

Alignment Requirements
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Data Type
     - Alignment
     - Notes
   * - FP32
     - 32 bytes
     - 8 elements
   * - FP16
     - 32 bytes
     - 16 elements
   * - INT8
     - 32 bytes
     - 32 elements

Always align buffer addresses:

.. code-block:: cpp

   #define ALIGN_32(x) (((x) + 31) & ~31)

   int input_size = ALIGN_32(tile_size * sizeof(float));
   int output_offset = ALIGN_32(input_size);

Memory Layout Example
---------------------

For a vector add kernel:

.. code-block:: text

   UB Layout (256KB):
   ┌────────────────────────────────────────────────────────────┐
   │ Offset   │ Size    │ Purpose                               │
   ├──────────┼─────────┼───────────────────────────────────────┤
   │ 0        │ 64KB    │ Input A tile                          │
   │ 64KB     │ 64KB    │ Input B tile                          │
   │ 128KB    │ 64KB    │ Output C tile                         │
   │ 192KB    │ 64KB    │ Temporary / workspace                 │
   └──────────┴─────────┴───────────────────────────────────────┘

.. code-block:: cpp

   // Buffer definitions
   __ub__ float* ub_a = (__ub__ float*)(UB_BASE + 0);
   __ub__ float* ub_b = (__ub__ float*)(UB_BASE + 64 * 1024);
   __ub__ float* ub_c = (__ub__ float*)(UB_BASE + 128 * 1024);

   // Process one tile
   load(ub_a, gm_a + offset, tile_size);
   load(ub_b, gm_b + offset, tile_size);

   vec_add(ub_c, ub_a, ub_b, tile_size);

   store(gm_c + offset, ub_c, tile_size);

Tile Size Calculation
---------------------

Choose tile size to maximize UB utilization:

.. code-block:: cpp

   // Available UB for data (reserve some for workspace)
   const int USABLE_UB = 240 * 1024;  // 240KB of 256KB

   // For vec_add: need 3 buffers (A, B, C)
   const int NUM_BUFFERS = 3;
   const int MAX_TILE_BYTES = USABLE_UB / NUM_BUFFERS;  // 80KB each
   const int MAX_TILE_ELEMENTS = MAX_TILE_BYTES / sizeof(float);  // 20K

   // Round down to alignment
   const int TILE_SIZE = (MAX_TILE_ELEMENTS / 8) * 8;  // ~20K elements

Common Mistakes
---------------

1. **Exceeding UB size** → Silent corruption or crash
2. **Misalignment** → Performance penalty or error
3. **Forgetting workspace** → Temp buffers for reduction, etc.
4. **Not double-buffering** → Missing optimization opportunity

Best Practices
--------------

1. **Plan layout first** - Sketch UB usage before coding
2. **Use constants** - Define sizes as compile-time constants
3. **Verify at compile time** - Static assert total <= 256KB
4. **Leave headroom** - Keep 5-10% for runtime needs
5. **Align everything** - 32-byte minimum for all buffers

Example Code
------------

Full working example: :file:`examples/15-vector-ub/`

Host code demonstrating UB allocation patterns:

.. literalinclude:: ../../../examples/15-vector-ub/main.cpp
   :language: cpp
   :caption: 15-vector-ub/main.cpp
   :lines: 1-80

AICORE UB management kernel (PTO-ISA):

.. literalinclude:: ../../../examples/15-vector-ub/kernel.pto
   :language: text
   :caption: 15-vector-ub/kernel.pto
