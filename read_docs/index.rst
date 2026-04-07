Ascend NPU Programming Guide
=============================

A practical guide to programming Huawei Ascend NPUs, focusing on hardware concepts
and low-level interfaces.

.. note::

   This guide focuses on **understanding hardware**, not on using frameworks.
   We teach how kernels work, not how to use PyTorch or TensorFlow.

Quick Start
-----------

.. code-block:: cpp

   #include <acl/acl.h>

   int main() {
       // Initialize ACL runtime
       aclInit(nullptr);
       aclrtSetDevice(0);

       // Allocate device memory (HBM)
       void* dev_ptr = nullptr;
       aclrtMalloc(&dev_ptr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);

       // Copy data to device
       aclrtMemcpy(dev_ptr, 1024, host_data, 1024,
                   ACL_MEMCPY_HOST_TO_DEVICE);

       // Launch kernel (covered in later examples)...

       // Cleanup
       aclrtFree(dev_ptr);
       aclrtResetDevice(0);
       aclFinalize();
   }

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   introduction/overview
   introduction/official-resources

.. toctree::
   :maxdepth: 2
   :caption: Hardware Architecture

   hardware/architecture
   hardware/memory-hierarchy
   hardware/specifications

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/device-query
   getting-started/memory
   getting-started/streams

.. toctree::
   :maxdepth: 2
   :caption: AICPU Programming

   aicpu/overview
   aicpu/kernel-launch
   aicpu/multithread
   aicpu/logging
   aicpu/atomic
   aicpu/queue

.. toctree::
   :maxdepth: 2
   :caption: AICORE Programming

   aicore/overview
   aicore/kernel-launch
   aicore/vector-unit/index
   aicore/cube-unit/index
   aicore/atomic
   aicore/pmu

.. toctree::
   :maxdepth: 2
   :caption: Synchronization

   synchronization/overview
   synchronization/register
   synchronization/atomic
   synchronization/queue

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
