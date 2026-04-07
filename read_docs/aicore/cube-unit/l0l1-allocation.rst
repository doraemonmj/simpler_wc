L1/L0 Allocation
================

Managing L1, L0A, L0B, and L0C buffers for Cube operations.

Buffer Overview
---------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                  Cube Unit Memory                           │
   │                                                             │
   │  L1 Buffer (1 MB):                                          │
   │  ┌─────────────────────────────────────────────────────┐   │
   │  │  Staging area for data from HBM                      │   │
   │  │  Shared between Cube and Vector                      │   │
   │  │  Software managed (you allocate regions)             │   │
   │  └─────────────────────────────────────────────────────┘   │
   │                                                             │
   │  L0A (64 KB):         L0B (64 KB):         L0C (256 KB):   │
   │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐ │
   │  │ Matrix A tile │   │ Matrix B tile │   │ Output/Accum  │ │
   │  │ Input buffer  │   │ Input buffer  │   │ Result buffer │ │
   │  └───────────────┘   └───────────────┘   └───────────────┘ │
   └─────────────────────────────────────────────────────────────┘

L1 Buffer Allocation
--------------------

L1 is shared between Cube and Vector operations:

.. code-block:: cpp

   // L1 layout example
   #define L1_SIZE          (1024 * 1024)  // 1 MB

   // Cube data staging
   #define L1_A_OFFSET      0
   #define L1_A_SIZE        (256 * 1024)   // 256 KB for A tiles

   #define L1_B_OFFSET      (256 * 1024)
   #define L1_B_SIZE        (256 * 1024)   // 256 KB for B tiles

   // Vector/output staging
   #define L1_C_OFFSET      (512 * 1024)
   #define L1_C_SIZE        (512 * 1024)   // 512 KB for C/output

L0 Buffer Constraints
---------------------

L0 buffers have fixed sizes and specific purposes:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Buffer
     - Size
     - Purpose
   * - L0A
     - 64 KB
     - Left matrix (A) tile for Cube
   * - L0B
     - 64 KB
     - Right matrix (B) tile for Cube
   * - L0C
     - 256 KB
     - Output accumulator

Tile Size Calculations
----------------------

Given L0 sizes, calculate maximum tile dimensions:

**For FP16:**

.. code-block:: cpp

   // L0A: 64KB, FP16 = 2 bytes
   // Max elements in L0A: 64 * 1024 / 2 = 32K elements
   // For M×K tile: M * K = 32K

   // Common choices:
   // M=16, K=2048  (tall skinny)
   // M=128, K=256  (square-ish)
   // M=256, K=128

   // L0B: 64KB
   // K×N tile: K * N = 32K

   // L0C: 256KB
   // M×N tile: M * N = 128K (FP16)
   // Max: 128K / (16*16) = 500 basic tiles

Double Buffering in L1
----------------------

To overlap compute and memory transfers:

.. code-block:: text

   L1 Layout with Double Buffering:
   ┌────────────────────────────────────────────────────────────┐
   │ Region      │ Size    │ Purpose                            │
   ├─────────────┼─────────┼────────────────────────────────────┤
   │ A_buf_0     │ 128 KB  │ A tile, buffer 0                   │
   │ A_buf_1     │ 128 KB  │ A tile, buffer 1 (ping-pong)       │
   │ B_buf_0     │ 128 KB  │ B tile, buffer 0                   │
   │ B_buf_1     │ 128 KB  │ B tile, buffer 1 (ping-pong)       │
   │ C_staging   │ 512 KB  │ Output staging (to UB or GM)       │
   └─────────────┴─────────┴────────────────────────────────────┘

.. code-block:: text

   // Pipeline:
   // While computing with buf_0, load into buf_1

   Time →
   ────────────────────────────────────────────────────────
   Load A:  [LD A0] [LD A1] [LD A0] [LD A1]
   Load B:  [LD B0] [LD B1] [LD B0] [LD B1]
   Compute:        [A0×B0] [A1×B1] [A0×B0]
   Store:                 [ST C0] [ST C1]

Data Movement Example
---------------------

Full matrix multiply with tiling:

.. code-block:: text

   // C[M,N] = A[M,K] × B[K,N]
   // Tile sizes: Tm=128, Tn=128, Tk=256

   for m = 0 to M step Tm:
       for n = 0 to N step Tn:
           // Initialize accumulator
           CLEAR L0C

           for k = 0 to K step Tk:
               // Load A[m:m+Tm, k:k+Tk] to L1
               LOAD L1_A, GM_A[m, k], Tm * Tk

               // Load B[k:k+Tk, n:n+Tn] to L1
               LOAD L1_B, GM_B[k, n], Tk * Tn

               // Move to L0
               MOVE L0A, L1_A
               MOVE L0B, L1_B

               // Matrix multiply (accumulate)
               MMAD L0C, L0A, L0B, L0C

           // Store result
           MOVE L1_C, L0C
           STORE GM_C[m, n], L1_C, Tm * Tn

Memory Bandwidth Optimization
-----------------------------

Tips for maximizing bandwidth:

1. **Tile for L0** - Size tiles to fit L0A/L0B exactly
2. **Reuse in L1** - Load once, use multiple times
3. **Double buffer** - Overlap load and compute
4. **Align to 512B** - Memory transfers are most efficient aligned

.. code-block:: cpp

   // Alignment macro
   #define ALIGN_512(x) (((x) + 511) & ~511)

   int tile_size = ALIGN_512(M * K * sizeof(half));

Example Code
------------

Full working example: :file:`examples/13-cube-memory/`

Host code demonstrating L1/L0 buffer management:

.. literalinclude:: ../../examples/13-cube-memory/main.cpp
   :language: cpp
   :caption: 13-cube-memory/main.cpp
   :lines: 1-80

AICORE kernel with double buffering (PTO-ISA):

.. literalinclude:: ../../examples/13-cube-memory/kernel.pto
   :language: text
   :caption: 13-cube-memory/kernel.pto
