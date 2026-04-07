Vector Unit (AIV) Programming
=============================

.. toctree::
   :maxdepth: 2

   ub-allocation
   compute

Overview
--------

The Vector Unit (AIV) in each AICORE handles element-wise operations
using SIMD (Single Instruction, Multiple Data) execution.

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    Vector Unit (AIV)                        │
   │                                                             │
   │  ┌─────────────────────────────────────────────────────┐   │
   │  │              Unified Buffer (UB)                     │   │
   │  │                   256 KB                             │   │
   │  │                                                      │   │
   │  │  Fast on-chip memory for vector operations           │   │
   │  │  Software-managed (you control what's loaded)        │   │
   │  └─────────────────────────────────────────────────────┘   │
   │                         │                                   │
   │                         ▼                                   │
   │  ┌─────────────────────────────────────────────────────┐   │
   │  │              Vector ALU                              │   │
   │  │                                                      │   │
   │  │  256 elements per cycle (FP16)                       │   │
   │  │  128 elements per cycle (FP32)                       │   │
   │  └─────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────┘

Key Concepts
------------

1. **UB (Unified Buffer)**: 256KB fast memory per AICORE
2. **Vector Width**: 256 FP16 elements or 128 FP32 elements
3. **Software Managed**: You explicitly load/store data
4. **Tiling**: Process data in chunks that fit in UB

Programming Pattern
-------------------

.. code-block:: text

   for each tile:
       1. Load: GM → UB
       2. Compute: vector operations in UB
       3. Store: UB → GM

Example (conceptual):

.. code-block:: cpp

   // Process 1M elements in tiles of 64K
   for (int tile = 0; tile < num_tiles; tile++) {
       int offset = tile * TILE_SIZE;

       // Load tile from GM to UB
       data_copy(ub_buffer, gm_input + offset, TILE_SIZE);

       // Vector operations (in UB)
       vec_mul(ub_buffer, ub_buffer, 2.0f, TILE_SIZE);
       vec_add(ub_buffer, ub_buffer, bias, TILE_SIZE);
       vec_relu(ub_buffer, ub_buffer, TILE_SIZE);

       // Store result from UB to GM
       data_copy(gm_output + offset, ub_buffer, TILE_SIZE);
   }
