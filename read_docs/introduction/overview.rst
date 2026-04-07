Overview
========

What is Ascend NPU?
-------------------

Ascend is Huawei's AI processor architecture, designed for deep learning training
and inference workloads. The architecture features a heterogeneous design combining
control processors (AICPU) with specialized compute blocks (AICORE) for efficient
AI acceleration.

This guide teaches Ascend NPU programming through 18 progressive examples, from basic
device queries to advanced AICPU-AICORE coordination patterns. Each example demonstrates
one hardware concept using direct CANN ACL (Ascend Computing Language) APIs.

Hardware Architecture
---------------------

The Ascend programming model can be simplified as:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                           Host                              │
   │                     CPU + Host Memory                       │
   │                                                             │
   │  Your program runs here. Launches kernels to device.        │
   └─────────────────────────────────────────────────────────────┘
                              │
                              │ PCIe
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                      Device (NPU)                           │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
   │  │   AICPU     │  │   AICPU     │  │   ...       │          │
   │  └─────────────┘  └─────────────┘  └─────────────┘          │
   │         │                │                                  │
   │         ▼                ▼                                  │
   │  ┌─────────────────────────────────────────────────────┐    │
   │  │  AICORE 0                                           │    │
   │  │  ┌───────┐  ┌───────┐  ┌───────┐                    │    │
   │  │  │  AIC  │  │  AIV  │  │  AIV  │                    │    │
   │  │  └───────┘  └───────┘  └───────┘                    │    │
   │  └─────────────────────────────────────────────────────┘    │
   │  ┌─────────────────────────────────────────────────────┐    │
   │  │  AICORE 1                                           │    │
   │  │  ┌───────┐  ┌───────┐  ┌───────┐                    │    │
   │  │  │  AIC  │  │  AIV  │  │  AIV  │                    │    │
   │  │  └───────┘  └───────┘  └───────┘                    │    │
   │  └─────────────────────────────────────────────────────┘    │
   │  ...                                                        │
   │                                                             │
   │  ┌─────────────────────────────────────────────────────┐    │
   │  │                         HBM                         │    │
   │  └─────────────────────────────────────────────────────┘    │
   └─────────────────────────────────────────────────────────────┘

**Key Components:**

- **Host CPU**: Runs your application code, manages device via ACL APIs
- **AICPU** (ARM-based processors): Handle dynamic operations, control flow, task orchestration
- **AICORE** (compute blocks): Execute high-throughput matrix and vector operations

  - **Cube Unit (AIC)**: Matrix multiplication accelerator (1 per AICORE)
  - **Vector Unit (AIV)**: SIMD processor for element-wise operations (2 per AICORE)

- **HBM**: High Bandwidth Memory, device DRAM

Programming Model
-----------------

**Two types of device code:**

1. **AICORE Kernels** (high throughput)

   - Run on Cube (matrix) and Vector units
   - Written in TIK, Ascend C, or PTO-ISA
   - Compiled to ``.o`` binary format
   - Best for: matrix multiply, convolution, element-wise ops

2. **AICPU Kernels** (flexible)

   - Run on ARM control processors
   - Written in standard C/C++
   - Compiled to ``.so`` shared library
   - Best for: dynamic shapes, control flow, complex indexing

**Execution flow:**

1. Host allocates device memory (``aclrtMalloc``)
2. Host copies input data to device (``aclrtMemcpy`` with ``ACL_MEMCPY_HOST_TO_DEVICE``)
3. Host launches kernel(s) on a stream (``aclrtLaunchKernel``)
4. Host synchronizes (``aclrtSynchronizeStream``)
5. Host copies results back (``aclrtMemcpy`` with ``ACL_MEMCPY_DEVICE_TO_HOST``)

Learning Path
-------------

This guide covers 18 progressive examples organized into four sections:

**Getting Started (Examples 01-03)**

- **01-device-query**: Enumerate NPU devices and query hardware properties
- **02-memory**: HBM allocation and host-device transfers
- **03-stream**: Asynchronous execution with streams

**AICPU Programming (Examples 04-08)**

- **04-aicpu-kernel-launch**: AICPU kernel execution with Runtime API
- **05-aicpu-multithread**: Multi-core parallel execution (8 AICPU cores)
- **06-aicpu-logging**: Debug output and error handling
- **07-aicpu-atomic**: Multi-core synchronization with atomics
- **08-aicpu-queue**: Lock-free queues (SPSC/MPMC)

**AICORE Programming (Examples 09-15)**

- **09-aicore-basic**: AICORE architecture and execution model
- **10-cube-matmul**: Cube unit matrix multiplication
- **11-cube-memory**: L1/L0A/B/C buffer management
- **12-vector-simd**: Vector unit SIMD operations
- **13-vector-ub**: UB (Unified Buffer) memory management
- **14-aicore-atomic**: Multi-block atomic operations
- **15-aicore-pmu**: Performance monitoring unit

**AICPU-AICORE Coordination (Examples 16-18)**

- **16-aicpu-aicore-register**: Register-based coordination
- **17-aicpu-aicore-atomic**: HBM atomic-based coordination
- **18-aicpu-aicore-queue**: Queue-based data passing

Who This Guide Is For
---------------------

- Developers who want to understand NPU hardware
- Framework developers (PyTorch, TensorFlow backend)
- Kernel optimization engineers
- Students learning parallel computing

This guide does **NOT** cover:

- Using high-level frameworks (use official Huawei docs)
- Model training workflows
- Distributed computing
