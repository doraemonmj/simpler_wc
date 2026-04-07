Hardware Specifications
=======================

This page lists hardware specifications for Ascend NPU generations.

Ascend 910 Series
-----------------

.. list-table:: Ascend 910 Variants
   :header-rows: 1
   :widths: 25 25 25 25

   * - Specification
     - 910B (A2)
     - 910C (A3)
   * - Architecture
     - DAV_2201
     - DAV_3510
   * - AICORE Blocks
     - 24
     - 24
   * - AIC (Cube) per block
     - 1
     - 1
   * - AIV (Vector) per block
     - 2
     - 2
   * - AICPU Count
     - 8
     - 8
   * - HBM Capacity
     - 64 GB
     - 64 GB
   * - HBM Bandwidth
     - ~1.6 TB/s
     - ~2.0 TB/s
   * - FP16 TFLOPS
     - 320
     - 400+

Per-Core Memory Sizes
---------------------

.. list-table:: AICORE Memory (A3 / 910C)
   :header-rows: 1
   :widths: 30 30 40

   * - Buffer
     - Size
     - Purpose
   * - L2 Cache
     - 192 MB (shared)
     - Hardware-managed cache
   * - L1 Buffer
     - 512 KB
     - Staging for Cube operations
   * - Unified Buffer (UB)
     - 192 KB
     - Vector unit scratch space
   * - L0A Buffer
     - 64 KB
     - Cube input A
   * - L0B Buffer
     - 64 KB
     - Cube input B
   * - L0C Buffer
     - 128 KB
     - Cube output accumulator

Querying Specifications at Runtime
----------------------------------

Use ACL APIs to query device info:

.. code-block:: cpp

   #include <acl/acl.h>

   aclInit(nullptr);
   aclrtSetDevice(0);

   const char* soc = aclrtGetSocName();
   printf("SoC: %s\n", soc);

   int64_t aicore_cnt;
   aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicore_cnt);
   printf("AICORE count: %ld\n", aicore_cnt);

   size_t free, total;
   aclrtGetMemInfo(ACL_HBM_MEM, &free, &total);
   printf("HBM: %lu / %lu bytes\n", free, total);
