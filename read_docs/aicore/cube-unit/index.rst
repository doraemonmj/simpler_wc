Cube Unit (AIC) Programming
===========================

.. toctree::
   :maxdepth: 2

   l0l1-allocation
   compute

Overview
--------

The Cube Unit (AIC) performs matrix multiplication - the core operation
for deep learning (convolution, attention, linear layers).

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                      Cube Unit (AIC)                            │
   │                                                                 │
   │                    C = A × B                                    │
   │                                                                 │
   │  ┌─────────────┐     ┌───────────┐     ┌─────────────────────┐ │
   │  │    L0A      │     │   CUBE    │     │       L0C           │ │
   │  │   64 KB     │────►│  COMPUTE  │────►│      256 KB         │ │
   │  │  Matrix A   │     │           │     │  Accumulator        │ │
   │  └─────────────┘     └───────────┘     └─────────────────────┘ │
   │         ▲                  ▲                                    │
   │         │            ┌─────┴─────┐                             │
   │  ┌─────────────┐     │    L0B    │                             │
   │  │     L1      │     │   64 KB   │                             │
   │  │    1 MB     │────►│  Matrix B │                             │
   │  │   Staging   │     └───────────┘                             │
   │  └─────────────┘                                               │
   │         ▲                                                       │
   │         │                                                       │
   │  ┌─────────────┐                                               │
   │  │   HBM/GM    │                                               │
   │  │  32-64 GB   │                                               │
   │  └─────────────┘                                               │
   └─────────────────────────────────────────────────────────────────┘

Matrix Multiply Operation
-------------------------

The Cube performs:

.. code-block:: text

   C[M, N] = A[M, K] × B[K, N]

   Where:
   - A is the left matrix (M rows, K columns)
   - B is the right matrix (K rows, N columns)
   - C is the output matrix (M rows, N columns)

Tile sizes depend on data type:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Type
     - M
     - N
     - K
   * - FP16
     - 16
     - 16
     - 16
   * - BF16
     - 16
     - 16
     - 16
   * - INT8
     - 16
     - 16
     - 32
   * - FP32
     - 8
     - 8
     - 16

Data Flow
---------

.. code-block:: text

   1. Load A: GM → L1 → L0A
   2. Load B: GM → L1 → L0B
   3. Compute: L0A × L0B → L0C (accumulate)
   4. Repeat steps 1-3 for all K tiles
   5. Store: L0C → L1 → UB (for post-processing)
            or L0C → L1 → GM (if no post-processing)

Accumulation
------------

L0C serves as an accumulator:

.. code-block:: text

   // Multiply large matrices by tiling

   for k_tile = 0 to K/TILE_K:
       Load A[m_tile, k_tile] to L0A
       Load B[k_tile, n_tile] to L0B

       if k_tile == 0:
           MMAD L0C, L0A, L0B        // C = A × B
       else:
           MMAD L0C, L0A, L0B, L0C   // C += A × B (accumulate)

   // After all k_tiles, L0C contains full result
   Store L0C to output

Integration with Vector Unit
----------------------------

Common pattern: Cube computes, Vector post-processes:

.. code-block:: text

   // Matrix multiply + ReLU activation

   Cube: C = A × B  (result in L0C)

   // Move to UB for vector processing
   Move L0C → L1 → UB

   // Apply activation
   Vector: out = ReLU(UB)

   // Store final result
   Store UB → GM
