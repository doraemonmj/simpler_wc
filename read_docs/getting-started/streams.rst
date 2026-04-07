Streams
=======

Streams are command queues that enable asynchronous execution on the NPU.
Operations submitted to a stream execute in order, but different streams
can execute concurrently.

Example Code
------------

Full source: :file:`examples/03-stream/main.cpp`

.. literalinclude:: ../../examples/03-stream/main.cpp
   :language: cpp
   :caption: Stream Operations Example (03-stream/main.cpp)

Concepts
--------

.. code-block:: text

   ┌────────────────────────────────────────────────────────────┐
   │                     Stream Model                           │
   │                                                            │
   │  Stream A:  [memcpy] → [kernel1] → [kernel2] → [memcpy]   │
   │                                                            │
   │  Stream B:  [memcpy] → [kernel3] ─────────────→ [memcpy]   │
   │                                                            │
   │  Operations within a stream: sequential                    │
   │  Operations across streams: potentially parallel           │
   └────────────────────────────────────────────────────────────┘

Default Stream
--------------

Every context can have multiple streams. You create streams explicitly
with ``aclrtCreateStream()``:

.. code-block:: cpp

   // Initialize ACL and create context
   aclInit(nullptr);
   aclrtSetDevice(0);
   aclrtContext context;
   aclrtCreateContext(&context, 0);

   // Create a stream for this context
   aclrtStream stream;
   aclrtCreateStream(&stream);

   // Use the stream for async operations (covered in later examples)
   // aclrtMemcpyAsync(dev_dst, host_src, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);

   // Synchronize and cleanup
   aclrtSynchronizeStream(stream);
   aclrtDestroyStream(stream);

Custom Streams
--------------

Create multiple streams for concurrent execution:

.. code-block:: cpp

   aclrtStream stream1;
   aclrtStream stream2;
   aclrtCreateStream(&stream1);
   aclrtCreateStream(&stream2);

   // Launch work on different streams (can run in parallel)
   // These would be actual async operations in a real program:
   // aclrtMemcpyAsync(..., stream1);
   // aclrtLaunchKernel(..., stream2);  // (kernel launch covered in later examples)

   // Wait for both streams to complete
   aclrtSynchronizeStream(stream1);
   aclrtSynchronizeStream(stream2);

   // Cleanup
   aclrtDestroyStream(stream1);
   aclrtDestroyStream(stream2);

API Reference
-------------

``aclrtCreateStream``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   aclError aclrtCreateStream(aclrtStream *stream);

Create a new stream for asynchronous operations.

**Parameters:**

- ``stream``: Pointer to receive the stream handle

**Returns:**

- ``ACL_SUCCESS`` (0) on success
- Error code on failure

**Hardware behavior:** Allocates a command queue on the device. Operations
submitted to this stream will execute in FIFO order.

**Example:** See `main.cpp:61 <../../examples/03-stream/main.cpp#L61>`_,
`main.cpp:83 <../../examples/03-stream/main.cpp#L83>`_,
`main.cpp:93 <../../examples/03-stream/main.cpp#L93>`_

``aclrtSynchronizeStream``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   aclError aclrtSynchronizeStream(aclrtStream stream);

Block the host thread until all operations in the stream complete.

**Parameters:**

- ``stream``: Stream to synchronize

**Returns:**

- ``ACL_SUCCESS`` when all operations complete
- Error code on failure

**Hardware behavior:** Host CPU waits until the device finishes all queued
operations in this stream. Other streams are unaffected and continue executing.

**Example:** See `main.cpp:164 <../../examples/03-stream/main.cpp#L164>`_,
`main.cpp:167 <../../examples/03-stream/main.cpp#L167>`_,
`main.cpp:170 <../../examples/03-stream/main.cpp#L170>`_

``aclrtDestroyStream``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   aclError aclrtDestroyStream(aclrtStream stream);

Destroy a stream and free its resources.

**Parameters:**

- ``stream``: Stream to destroy

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Hardware behavior:** Frees the command queue resources on the device. All
operations in the stream must complete before destroying (use synchronize first).

**Example:** See `main.cpp:178 <../../examples/03-stream/main.cpp#L178>`_

Synchronization Patterns
------------------------

.. note::

   The examples below show conceptual patterns. The 03-stream example focuses
   on stream lifecycle (creation, synchronization, destruction). Actual async
   operations (``aclrtMemcpyAsync``, kernel launches) are demonstrated in
   later examples.

**Pattern 1: Simple Sequential**

.. code-block:: cpp

   // All operations on one stream, execute sequentially
   aclrtStream stream;
   aclrtCreateStream(&stream);

   aclrtMemcpyAsync(dev_input, host_input, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
   // Launch kernel on stream (kernel launching covered in later examples)
   aclrtSynchronizeStream(stream);  // Wait for kernel to complete
   aclrtMemcpyAsync(host_output, dev_output, size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
   aclrtSynchronizeStream(stream);  // Wait for D2H transfer

   aclrtDestroyStream(stream);

**Pattern 2: Overlapping Compute and Transfer**

.. code-block:: cpp

   aclrtStream compute_stream;
   aclrtStream transfer_stream;
   aclrtCreateStream(&compute_stream);
   aclrtCreateStream(&transfer_stream);

   // Overlap: compute on batch N while transferring batch N+1
   for (int batch = 0; batch < num_batches; batch++) {
       // Transfer next batch (if not last)
       if (batch + 1 < num_batches) {
           aclrtMemcpyAsync(dev_next, host_next, size,
                           ACL_MEMCPY_HOST_TO_DEVICE, transfer_stream);
       }

       // Compute current batch (can overlap with transfer)
       // Kernel launch would go here

       // Sync before next iteration to ensure operations complete
       aclrtSynchronizeStream(compute_stream);
       aclrtSynchronizeStream(transfer_stream);

       // Swap buffers
       swap(dev_current, dev_next);
   }

   aclrtDestroyStream(compute_stream);
   aclrtDestroyStream(transfer_stream);

**Pattern 3: Multiple Independent Kernels**

.. code-block:: cpp

   aclrtStream stream1, stream2, stream3;
   aclrtCreateStream(&stream1);
   aclrtCreateStream(&stream2);
   aclrtCreateStream(&stream3);

   // Launch independent kernels on separate streams (can execute in parallel)
   // Kernel launches would go here on stream1, stream2, stream3

   // Wait for all to complete
   aclrtSynchronizeStream(stream1);
   aclrtSynchronizeStream(stream2);
   aclrtSynchronizeStream(stream3);

   aclrtDestroyStream(stream1);
   aclrtDestroyStream(stream2);
   aclrtDestroyStream(stream3);

When to Use Multiple Streams
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Use Multiple Streams
     - Use Single Stream
   * - Independent operations
     - Sequential dependencies
   * - Overlap compute and transfer
     - Simple kernel sequences
   * - Multiple small kernels
     - One large kernel
   * - Pipeline processing
     - Batch processing

Best Practices
--------------

1. **Start simple** - Use default stream until you need more
2. **Profile first** - Don't add streams without measuring benefit
3. **Limit stream count** - 2-4 streams usually sufficient
4. **Always sync** - Ensure completion before reading results
5. **Destroy streams** - Avoid resource leaks
