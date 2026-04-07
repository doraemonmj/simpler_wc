AICORE PMU (Performance Monitoring)
===================================

Using Performance Monitoring Unit for kernel optimization.

Overview
--------

The PMU provides hardware counters to measure kernel performance:

- Cycle counts
- Instruction counts
- Memory bandwidth utilization
- Pipeline stalls

PMU Counters
------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Counter
     - Description
   * - ``PMU_CYCLES``
     - Total cycles elapsed
   * - ``PMU_CUBE_CYCLES``
     - Cycles Cube unit was active
   * - ``PMU_VEC_CYCLES``
     - Cycles Vector unit was active
   * - ``PMU_MTE_STALL``
     - Cycles stalled waiting for memory
   * - ``PMU_PIPE_STALL``
     - Pipeline stall cycles
   * - ``PMU_L0A_LOAD``
     - L0A loads
   * - ``PMU_L0B_LOAD``
     - L0B loads
   * - ``PMU_UB_ACCESS``
     - UB accesses

Basic Usage
-----------

.. code-block:: text

   // Start profiling
   PMU_START

   // ... kernel code ...

   // Stop and read counters
   PMU_STOP
   cycles = PMU_READ(PMU_CYCLES)
   cube_cycles = PMU_READ(PMU_CUBE_CYCLES)

Profiling a Kernel
------------------

.. code-block:: text

   __kernel__ void profiled_matmul(Args* args) {
       PMU_START

       // Matrix multiply
       for (int k = 0; k < K; k += TILE_K) {
           LOAD L0A, ...
           LOAD L0B, ...
           MMAD L0C, L0A, L0B, L0C
       }

       PMU_STOP

       // Write counters to output (for host to read)
       if (get_block_idx() == 0) {
           args->perf_cycles = PMU_READ(PMU_CYCLES)
           args->perf_cube = PMU_READ(PMU_CUBE_CYCLES)
           args->perf_stall = PMU_READ(PMU_MTE_STALL)
       }
   }

Interpreting Results
--------------------

**Cube Utilization**

.. code-block:: text

   cube_util = cube_cycles / total_cycles

   > 80%: Excellent - compute bound
   50-80%: Good - some overhead
   < 50%: Memory bound or poor tiling

**Memory Stall Analysis**

.. code-block:: text

   stall_ratio = mte_stall_cycles / total_cycles

   > 30%: Memory bandwidth limited
   10-30%: Moderate stalls
   < 10%: Compute bound

**Pipeline Efficiency**

.. code-block:: text

   efficiency = (cube_cycles + vec_cycles) / total_cycles

   Low efficiency indicates:
   - Too many barriers
   - Poor double buffering
   - Unbalanced load/compute

Optimization Workflow
---------------------

1. **Baseline measurement**

   .. code-block:: text

      Run kernel, collect PMU data:
      - Total cycles: 1,000,000
      - Cube cycles: 400,000 (40%)
      - MTE stalls: 350,000 (35%)

2. **Identify bottleneck**

   .. code-block:: text

      35% MTE stalls → Memory bound
      Action: Improve data reuse, increase tile size

3. **Apply optimization**

   .. code-block:: text

      - Increase tile size in L1
      - Add double buffering
      - Reorder loops for better locality

4. **Measure again**

   .. code-block:: text

      After optimization:
      - Total cycles: 700,000 (30% faster)
      - Cube cycles: 500,000 (71%)
      - MTE stalls: 100,000 (14%)

Roofline Model
--------------

Use PMU data for roofline analysis:

.. code-block:: text

   FLOPS = cube_operations * 2  (multiply-add)
   Bytes = bytes_loaded + bytes_stored

   Operational Intensity = FLOPS / Bytes

   If OI < machine_balance:
       Memory bound → Optimize data movement
   Else:
       Compute bound → Optimize Cube usage

MindStudio Integration
----------------------

For visual profiling, use MindStudio:

1. Enable profiling in launch parameters
2. Run kernel
3. Load profile in MindStudio
4. View timeline, counters, bottlenecks

.. note::

   PMU access may require specific kernel configurations.
   Consult CANN documentation for your version.

Example Code
------------

Full working example: :file:`examples/15-aicore-pmu/`

Host code demonstrating PMU profiling:

.. literalinclude:: ../../examples/15-aicore-pmu/main.cpp
   :language: cpp
   :caption: 15-aicore-pmu/main.cpp
   :lines: 1-80

AICORE kernel with PMU instrumentation (PTO-ISA):

.. literalinclude:: ../../examples/15-aicore-pmu/kernel.pto
   :language: text
   :caption: 15-aicore-pmu/kernel.pto
