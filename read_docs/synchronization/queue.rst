AICPU-AICORE Queue
==================

Queue data structures for passing data between AICPU and AICORE.

Overview
--------

Queues enable producer-consumer patterns between processors:

.. code-block:: text

   ┌─────────────┐     ┌─────────────────────────┐     ┌─────────────┐
   │   AICPU     │     │    Queue (in HBM)       │     │   AICORE    │
   │             │     │  ┌───┬───┬───┬───┬───┐  │     │             │
   │  push() ─────────►│  │ 0 │ 1 │ 2 │ 3 │...│  │◄───── pop()     │
   │             │     │  └───┴───┴───┴───┴───┘  │     │             │
   │             │     │   head ──────►  tail    │     │             │
   └─────────────┘     └─────────────────────────┘     └─────────────┘

   Use when:
   - Passing data (not just signals)
   - Buffering between producer and consumer
   - Decoupling production and consumption rates

Queue Structure
---------------

Lock-free single-producer single-consumer queue:

.. code-block:: cpp

   template <typename T, int CAPACITY>
   struct SPSCQueue {
       T data[CAPACITY];
       std::atomic<uint32_t> head{0};  // Write position
       std::atomic<uint32_t> tail{0};  // Read position

       bool push(const T& item) {
           uint32_t h = head.load(std::memory_order_relaxed);
           uint32_t next = (h + 1) % CAPACITY;

           if (next == tail.load(std::memory_order_acquire)) {
               return false;  // Full
           }

           data[h] = item;
           head.store(next, std::memory_order_release);
           return true;
       }

       bool pop(T* item) {
           uint32_t t = tail.load(std::memory_order_relaxed);

           if (t == head.load(std::memory_order_acquire)) {
               return false;  // Empty
           }

           *item = data[t];
           tail.store((t + 1) % CAPACITY, std::memory_order_release);
           return true;
       }
   };

AICPU Producer Example
----------------------

AICPU prepares work items for AICORE:

.. code-block:: cpp

   // Work item structure
   struct WorkItem {
       void* input_ptr;
       void* output_ptr;
       int32_t size;
       int32_t flags;
   };

   using WorkQueue = SPSCQueue<WorkItem, 64>;

   // AICPU kernel
   extern "C" void producer_kernel(ProducerArgs* args) {
       WorkQueue* queue = args->queue;

       for (int batch = 0; batch < args->num_batches; batch++) {
           // Prepare work item
           WorkItem item;
           item.input_ptr = args->inputs[batch];
           item.output_ptr = args->outputs[batch];
           item.size = args->batch_size;
           item.flags = (batch == args->num_batches - 1) ? FLAG_LAST : 0;

           // Push to queue (spin if full)
           while (!queue->push(item)) {
               // Queue full, wait
           }
       }
   }

AICORE Consumer Example
-----------------------

AICORE processes work items:

.. code-block:: cpp

   // AICORE kernel
   __kernel__ void consumer_kernel(ConsumerArgs* args) {
       WorkQueue* queue = args->queue;

       while (true) {
           WorkItem item;

           // Try to get work
           if (queue->pop(&item)) {
               // Process item
               compute(item.input_ptr, item.output_ptr, item.size);

               // Check for last item
               if (item.flags & FLAG_LAST) {
                   break;
               }
           }
           // If empty, keep polling
       }
   }

Multi-Consumer Queue
--------------------

For multiple AICORE blocks consuming from one queue:

.. code-block:: cpp

   // Claim-based queue for multiple consumers
   template <typename T, int CAPACITY>
   struct MPMCQueue {
       T data[CAPACITY];
       std::atomic<uint32_t> head{0};
       std::atomic<uint32_t> tail{0};
       std::atomic<uint32_t> claim{0};  // For consumers to claim slots

       bool push(const T& item) {
           // Same as SPSC (single producer)
           uint32_t h = head.load(std::memory_order_relaxed);
           if ((h + 1) % CAPACITY == tail.load(std::memory_order_acquire)) {
               return false;
           }
           data[h] = item;
           head.store((h + 1) % CAPACITY, std::memory_order_release);
           return true;
       }

       bool pop(T* item) {
           while (true) {
               uint32_t c = claim.load(std::memory_order_relaxed);
               uint32_t h = head.load(std::memory_order_acquire);

               if (c >= h) return false;  // Empty

               // Try to claim slot
               if (claim.compare_exchange_weak(c, c + 1,
                       std::memory_order_acq_rel)) {
                   uint32_t slot = c % CAPACITY;
                   *item = data[slot];

                   // Update tail (last consumer to process updates it)
                   // This is simplified - full impl needs more care
                   return true;
               }
           }
       }
   };

Host Setup
----------

.. code-block:: cpp

   // Allocate queue in device memory
   WorkQueue* dev_queue = (WorkQueue*)platform_malloc(sizeof(WorkQueue));

   // Initialize (clear head/tail)
   WorkQueue init;
   init.head.store(0);
   init.tail.store(0);
   platform_memcpy_h2d(dev_queue, &init, sizeof(WorkQueue));

   // Launch producer and consumer on different streams
   PlatformStream prod_stream = platform_stream_create();
   PlatformStream cons_stream = platform_stream_create();

   ProducerArgs prod_args = { .queue = dev_queue, ... };
   ConsumerArgs cons_args = { .queue = dev_queue, ... };

   platform_aicpu_launch(..., &prod_args, ..., prod_stream);
   platform_kernel_launch(..., &cons_args, ..., cons_stream);

   // Wait for both
   platform_stream_sync(cons_stream);

Performance Considerations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Factor
     - Guidance
   * - Queue capacity
     - Larger = more buffering, less contention
   * - Item size
     - Smaller = better cache efficiency
   * - Polling overhead
     - Consider backoff if consumer faster than producer
   * - Memory ordering
     - Use minimal ordering for performance

Best Practices
--------------

1. **Size queue appropriately** - Balance memory vs contention
2. **Signal completion** - Use flag or poison item for termination
3. **Handle backpressure** - Decide policy when queue full
4. **Align items** - Cache-line align for better performance
5. **Test thoroughly** - Race conditions are subtle

Example Code
------------

Full working example: :file:`examples/18-aicpu-aicore-queue/`

Host code demonstrating queue-based data passing:

.. literalinclude:: ../../examples/18-aicpu-aicore-queue/main.cpp
   :language: cpp
   :caption: 18-aicpu-aicore-queue/main.cpp
   :lines: 1-80

AICPU producer kernel:

.. literalinclude:: ../../examples/18-aicpu-aicore-queue/aicpu_kernel/producer.cpp
   :language: cpp
   :caption: 18-aicpu-aicore-queue/aicpu_kernel/producer.cpp

AICORE consumer kernel (PTO-ISA):

.. literalinclude:: ../../examples/18-aicpu-aicore-queue/aicore_kernel.pto
   :language: text
   :caption: 18-aicpu-aicore-queue/aicore_kernel.pto
