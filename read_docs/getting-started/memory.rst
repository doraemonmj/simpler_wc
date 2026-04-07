Memory Management
=================

Device memory (HBM) is physically separate from host RAM and must be
explicitly allocated before data can be processed on the NPU. This page
demonstrates HBM allocation, synchronous host-device transfers, and proper
memory management using direct CANN ACL APIs.

Hardware Concepts
-----------------

Ascend NPUs have their own High Bandwidth Memory (HBM) physically located on
the accelerator card. This memory is distinct from host system RAM and requires
explicit data transfers.

**Memory Hierarchy:**

- **Host RAM**: System memory managed by the CPU
- **PCIe Bus**: ~32 GB/s bandwidth (PCIe 4.0 x16) connecting host and device
- **HBM** (High Bandwidth Memory): Device DRAM, 32-64 GB capacity, ~1.2 TB/s bandwidth
- **L2 Cache**: Shared cache across all AICOREs, 192-256 MB
- **L1/UB** (Unified Buffer): Per-AICORE scratchpad memory, 192-512 KB
- **L0A/B/C**: Cube engine local buffers for matrix operations

**Transfer Mechanism:**

Data moves between host and device via DMA (Direct Memory Access) over the PCIe
bus. CANN provides two transfer modes:

- **Synchronous**: ``aclrtMemcpy()`` blocks until transfer completes (covered in this example)
- **Asynchronous**: ``aclrtMemcpyAsync()`` returns immediately, transfer happens in background (see example 03-stream)

The synchronous mode is simpler but blocks the calling thread. It's appropriate
for initialization, small transfers, or when you need guaranteed completion
before proceeding.

Example Code
------------

Full source: :file:`examples/02-memory/main.cpp`

.. literalinclude:: ../../examples/02-memory/main.cpp
   :language: cpp
   :caption: Memory Operations Example (02-memory/main.cpp)

Implementation Walkthrough
--------------------------

The example demonstrates a complete memory lifecycle in 7 steps:

**Step 1: Initialize ACL runtime** `(main.cpp:27-52) <../../examples/02-memory/main.cpp#L27>`_

Initialize ACL and create a device context. Required before any memory operations.

**Step 2: Prepare host data** `(main.cpp:54-75) <../../examples/02-memory/main.cpp#L54>`_

Allocate host memory using standard ``malloc()`` and initialize with test data.
The example uses a simple pattern (0.0, 0.5, 1.0, ...) for easy verification.

**Step 3: Allocate device memory** `(main.cpp:77-96) <../../examples/02-memory/main.cpp#L77>`_

Use ``aclrtMalloc()`` to allocate HBM on the device. The ``ACL_MEM_MALLOC_HUGE_FIRST``
flag requests 2MB huge pages for better TLB efficiency. Device pointers cannot
be dereferenced on the host - they're only valid on the NPU.

**Step 4: Host-to-Device transfer** `(main.cpp:98-119) <../../examples/02-memory/main.cpp#L98>`_

Copy data from host RAM to device HBM using ``aclrtMemcpy()`` with
``ACL_MEMCPY_HOST_TO_DEVICE``. This performs a synchronous DMA transfer over PCIe,
blocking until all data reaches HBM.

**Step 5: Device-to-Host transfer** `(main.cpp:121-141) <../../examples/02-memory/main.cpp#L121>`_

Copy data back from device to host using ``aclrtMemcpy()`` with
``ACL_MEMCPY_DEVICE_TO_HOST``. Also synchronous, also blocks until complete.

**Step 6: Verify data** `(main.cpp:143-158) <../../examples/02-memory/main.cpp#L143>`_

Compare source and result data to confirm round-trip transfer correctness.
This validation pattern is common in NPU programming.

**Step 7: Cleanup** `(main.cpp:160-168) <../../examples/02-memory/main.cpp#L160>`_

Free resources in reverse allocation order: device memory, host memory, context, device, runtime.

ACL API Reference
-----------------

Memory Allocation
^^^^^^^^^^^^^^^^^

``aclrtMalloc``
"""""""""""""""

.. code-block:: cpp

   aclError aclrtMalloc(void** devPtr,
                        size_t size,
                        aclrtMallocPolicy policy);

Allocate device memory (HBM).

**Parameters:**

- ``devPtr``: Output parameter for device pointer
- ``size``: Number of bytes to allocate
- ``policy``: Allocation policy:

  - ``ACL_MEM_MALLOC_HUGE_FIRST``: Prefer 2MB huge pages (better TLB efficiency)
  - ``ACL_MEM_MALLOC_NORMAL_ONLY``: Use standard 4KB pages
  - ``ACL_MEM_MALLOC_HUGE_ONLY``: Require 2MB pages, fail if unavailable

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure (e.g., out of memory)

**Important:** Device pointers are not accessible from host code. Attempting to
dereference them will crash or return garbage data.

**Line Reference:** `main.cpp:85 <../../examples/02-memory/main.cpp#L85>`_

``aclrtFree``
"""""""""""""

.. code-block:: cpp

   aclError aclrtFree(void* devPtr);

Free previously allocated device memory.

**Parameters:**

- ``devPtr``: Device pointer from ``aclrtMalloc()``

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:162 <../../examples/02-memory/main.cpp#L162>`_

Data Transfers
^^^^^^^^^^^^^^

``aclrtMemcpy``
"""""""""""""""

.. code-block:: cpp

   aclError aclrtMemcpy(void* dst,
                        size_t destMax,
                        const void* src,
                        size_t count,
                        aclrtMemcpyKind kind);

**Synchronously** copy data between host and device. Blocks calling thread until
the DMA transfer completes.

**Parameters:**

- ``dst``: Destination pointer (host or device, depending on direction)
- ``destMax``: Maximum size of destination buffer (safety check)
- ``src``: Source pointer (host or device, depending on direction)
- ``count``: Number of bytes to copy
- ``kind``: Transfer direction:

  - ``ACL_MEMCPY_HOST_TO_DEVICE``: Copy from host RAM to device HBM
  - ``ACL_MEMCPY_DEVICE_TO_HOST``: Copy from device HBM to host RAM
  - ``ACL_MEMCPY_DEVICE_TO_DEVICE``: Copy within device HBM

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Performance Notes:**

- PCIe bandwidth: ~32 GB/s (PCIe 4.0 x16)
- Synchronous - blocks until transfer completes
- For async transfers, use ``aclrtMemcpyAsync()`` (see example 03-stream)

**Line References:**

- H2D transfer: `main.cpp:106 <../../examples/02-memory/main.cpp#L106>`_
- D2H transfer: `main.cpp:128 <../../examples/02-memory/main.cpp#L128>`_

Memory Management Patterns
---------------------------

**Round-Trip Verification:**

.. code-block:: cpp

   // Allocate host and device memory
   float* host_src = (float*)malloc(size);
   float* host_dst = (float*)malloc(size);
   void* dev_buf = nullptr;
   aclrtMalloc(&dev_buf, size, ACL_MEM_MALLOC_HUGE_FIRST);

   // Initialize source data
   for (int i = 0; i < count; i++) {
       host_src[i] = (float)i;
   }

   // Transfer: host -> device -> host
   aclrtMemcpy(dev_buf, size, host_src, size, ACL_MEMCPY_HOST_TO_DEVICE);
   aclrtMemcpy(host_dst, size, dev_buf, size, ACL_MEMCPY_DEVICE_TO_HOST);

   // Verify data integrity
   for (int i = 0; i < count; i++) {
       assert(host_src[i] == host_dst[i]);
   }

This pattern validates that memory allocation and transfers work correctly.

**Proper Cleanup Order:**

.. code-block:: cpp

   // Cleanup in reverse allocation order
   aclrtFree(dev_buf);              // 1. Free device memory
   free(host_dst);                   // 2. Free host memory
   free(host_src);
   aclrtDestroyContext(context);     // 3. Destroy context
   aclrtResetDevice(deviceId);       // 4. Reset device
   aclFinalize();                    // 5. Finalize runtime

Always clean up in reverse order of allocation to avoid resource leaks.

**Error Handling:**

.. code-block:: cpp

   void* dev_buf = nullptr;
   aclError ret = aclrtMalloc(&dev_buf, size, ACL_MEM_MALLOC_HUGE_FIRST);
   if (ret != ACL_SUCCESS) {
       printf("Failed to allocate device memory (error: %d)\n", ret);
       // Clean up any previously allocated resources
       return 1;
   }

Always check return codes. HBM is limited (32-64 GB), and allocation can fail.

Best Practices
--------------

1. **Minimize transfers** - H2D/D2H transfers are slow compared to compute (~32 GB/s vs ~1.2 TB/s HBM bandwidth)
2. **Batch transfers** - One large copy is faster than many small ones due to DMA setup overhead
3. **Reuse allocations** - Avoid ``malloc/free`` in tight loops, allocate once and reuse
4. **Check allocation failures** - HBM is limited, handle out-of-memory gracefully
5. **Use huge pages** - ``ACL_MEM_MALLOC_HUGE_FIRST`` improves TLB efficiency for large buffers
6. **Align data** - Device memory is automatically 512-byte aligned by ``aclrtMalloc()``
7. **Prefer async transfers** - For better performance, use ``aclrtMemcpyAsync()`` when possible (see 03-stream)

Common Errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Cause/Solution
   * - ``aclrtMalloc`` returns error
     - Out of HBM memory. Check available memory with ``aclrtGetMemInfo()``, or reduce allocation size.
   * - ``aclrtMemcpy`` returns error
     - Platform not initialized, invalid pointer, or size mismatch. Verify ``aclInit()`` was called and pointers are valid.
   * - Data corruption after transfer
     - Passing host pointer as device pointer (or vice versa). Check transfer direction flag matches pointer types.
   * - Crash when accessing ``devPtr``
     - Attempting to dereference device pointer on host. Use ``aclrtMemcpy()`` to transfer data to host first.
   * - Performance slower than expected
     - Small transfers have high overhead. Batch multiple small copies into one large transfer.

Sample Output
-------------

.. code-block:: text

   === Ascend NPU Memory Operations ===

   Step 1: Initialize ACL runtime...
     ACL initialized on device 0

   Step 2: Prepare host data...
     Allocated 4096 bytes on host
     Source data: [0.0, 0.5, 1.0, ... 511.5]

   Step 3: Allocate device memory...
     Allocated 4096 bytes on device (HBM)
     Device pointer: 0x12c0c0014000

   Step 4: Copy host -> device (H2D)...
     Copied 4096 bytes to device

   Step 5: Copy device -> host (D2H)...
     Copied 4096 bytes from device

   Step 6: Verify data...
     Result data: [0.0, 0.5, 1.0, ... 511.5]
     PASS: All 1024 elements match!

   Step 7: Cleanup...
     Resources freed

   === End of Memory Operations ===

.. note::

   Device pointer addresses (e.g., ``0x12c0c0014000``) will vary between runs
   and systems. The address shown is in the device's HBM address space, not
   the host's virtual memory.

Building and Running
--------------------

.. code-block:: bash

   cd examples/02-memory
   mkdir build && cd build
   cmake ..
   make
   ./02-memory

**Prerequisites:**

- CANN toolkit installed (version 5.0+)
- ``ASCEND_HOME_PATH`` environment variable set
- At least one Ascend NPU device in the system

Key Takeaways
-------------

After completing this example, you should understand:

1. **Memory Separation**: HBM is physically separate from host RAM, requiring explicit transfers
2. **Allocation**: Use ``aclrtMalloc()`` to allocate device memory with appropriate policy flags
3. **Transfers**: Use ``aclrtMemcpy()`` with direction flags for synchronous H2D/D2H transfers
4. **Synchronous Behavior**: ``aclrtMemcpy()`` blocks until transfer completes
5. **Verification Pattern**: Round-trip transfers (H2D then D2H) validate memory operations
6. **Resource Management**: Always free resources in reverse allocation order
7. **Performance**: PCIe bandwidth (~32 GB/s) is much slower than HBM bandwidth (~1.2 TB/s)

Next Steps
----------

See :doc:`streams` to learn about asynchronous memory transfers using
streams, which allow overlapping transfers with computation for better performance.
