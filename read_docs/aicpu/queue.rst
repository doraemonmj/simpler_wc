AICPU Queue
===========

Queue data structures for producer-consumer patterns in AICPU kernels.

Overview
--------

Queues enable communication between different parts of a pipeline:

.. code-block:: text

   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │   Producer   │  ────►  │    Queue     │  ────►  │   Consumer   │
   │   (AICPU)    │  push   │   (in HBM)   │  pop    │   (AICPU)    │
   └──────────────┘         └──────────────┘         └──────────────┘

Lock-Free Queue Structure
-------------------------

A simple ring buffer implementation:

.. code-block:: cpp

   #include <atomic>

   template <typename T, int CAPACITY>
   struct LockFreeQueue {
       T buffer[CAPACITY];
       std::atomic<int32_t> head{0};  // Write position
       std::atomic<int32_t> tail{0};  // Read position

       bool push(const T& item) {
           int32_t current_head = head.load(std::memory_order_relaxed);
           int32_t next_head = (current_head + 1) % CAPACITY;

           if (next_head == tail.load(std::memory_order_acquire)) {
               return false;  // Queue full
           }

           buffer[current_head] = item;
           head.store(next_head, std::memory_order_release);
           return true;
       }

       bool pop(T* item) {
           int32_t current_tail = tail.load(std::memory_order_relaxed);

           if (current_tail == head.load(std::memory_order_acquire)) {
               return false;  // Queue empty
           }

           *item = buffer[current_tail];
           tail.store((current_tail + 1) % CAPACITY, std::memory_order_release);
           return true;
       }

       bool empty() const {
           return head.load(std::memory_order_relaxed) ==
                  tail.load(std::memory_order_relaxed);
       }

       int size() const {
           int h = head.load(std::memory_order_relaxed);
           int t = tail.load(std::memory_order_relaxed);
           return (h - t + CAPACITY) % CAPACITY;
       }
   };

Using Queue in Kernel
---------------------

Define queue in arguments:

.. code-block:: cpp

   // Shared definitions (host and kernel)
   struct WorkItem {
       int32_t id;
       float data[16];
   };

   using WorkQueue = LockFreeQueue<WorkItem, 1024>;

   struct ProducerArgs {
       WorkQueue* queue;
       void* source_data;
       int32_t count;
   };

   struct ConsumerArgs {
       WorkQueue* queue;
       void* result_data;
       int32_t* done_flag;
   };

Producer kernel:

.. code-block:: cpp

   extern "C" void producer_kernel(ProducerArgs* args) {
       float* source = (float*)args->source_data;

       for (int i = 0; i < args->count; i++) {
           WorkItem item;
           item.id = i;
           memcpy(item.data, &source[i * 16], 16 * sizeof(float));

           // Spin until queue has space
           while (!args->queue->push(item)) {
               // Optionally yield or backoff
           }
       }
   }

Consumer kernel:

.. code-block:: cpp

   extern "C" void consumer_kernel(ConsumerArgs* args) {
       float* result = (float*)args->result_data;
       WorkItem item;

       while (true) {
           if (args->queue->pop(&item)) {
               // Process item
               for (int j = 0; j < 16; j++) {
                   result[item.id * 16 + j] = item.data[j] * 2.0f;
               }
           } else if (*args->done_flag) {
               break;  // Producer done and queue empty
           }
       }
   }

Host Setup
----------

.. code-block:: cpp

   // Allocate queue in device memory
   WorkQueue* dev_queue = (WorkQueue*)platform_malloc(sizeof(WorkQueue));

   // Initialize queue (must initialize atomics!)
   WorkQueue init_queue;
   platform_memcpy_h2d(dev_queue, &init_queue, sizeof(WorkQueue));

   // Allocate done flag
   int32_t* dev_done = (int32_t*)platform_malloc(sizeof(int32_t));
   int32_t zero = 0;
   platform_memcpy_h2d(dev_done, &zero, sizeof(int32_t));

   // Create streams for parallel execution
   PlatformStream producer_stream = platform_stream_create();
   PlatformStream consumer_stream = platform_stream_create();

   // Launch producer and consumer
   ProducerArgs prod_args = {dev_queue, dev_source, count};
   ConsumerArgs cons_args = {dev_queue, dev_result, dev_done};

   platform_aicpu_launch(prod_so, prod_size, "producer_kernel",
                         &prod_args, sizeof(prod_args), producer_stream);
   platform_aicpu_launch(cons_so, cons_size, "consumer_kernel",
                         &cons_args, sizeof(cons_args), consumer_stream);

   // Wait for producer, signal done
   platform_stream_sync(producer_stream);
   int32_t one = 1;
   platform_memcpy_h2d(dev_done, &one, sizeof(int32_t));

   // Wait for consumer
   platform_stream_sync(consumer_stream);

Multi-Producer/Multi-Consumer
-----------------------------

For multiple producers or consumers, use more sophisticated structures:

.. code-block:: cpp

   // Multi-producer single-consumer (MPSC)
   template <typename T, int CAPACITY>
   struct MPSCQueue {
       T buffer[CAPACITY];
       std::atomic<int32_t> head{0};
       std::atomic<int32_t> tail{0};
       std::atomic<int32_t> pending{0};

       bool push(const T& item) {
           int32_t pos = pending.fetch_add(1, std::memory_order_relaxed);
           if (pos >= CAPACITY) {
               pending.fetch_sub(1, std::memory_order_relaxed);
               return false;
           }

           int32_t slot = pos % CAPACITY;
           buffer[slot] = item;

           // Wait for previous writers
           while (head.load(std::memory_order_relaxed) != pos) {}
           head.store(pos + 1, std::memory_order_release);
           return true;
       }

       // pop() same as before (single consumer)
   };

Best Practices
--------------

1. **Size appropriately** - Queue too small causes spinning
2. **Initialize on device** - Don't forget atomic initialization
3. **Signal completion** - Use flag to tell consumer when done
4. **Handle backpressure** - Decide what to do when queue is full
5. **Avoid starvation** - Balance producer and consumer rates

Performance Considerations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Factor
     - Impact
   * - Queue capacity
     - Larger = less contention, more memory
   * - Item size
     - Smaller = better cache efficiency
   * - Contention
     - Many producers = more spinning
   * - Memory ordering
     - Stricter = slower but safer

Example Code
------------

Full working example: :file:`examples/08-aicpu-queue/`

Host code demonstrating queue patterns:

.. literalinclude:: ../../examples/08-aicpu-queue/main.cpp
   :language: cpp
   :caption: 08-aicpu-queue/main.cpp
   :lines: 1-80

AICPU producer/consumer kernels:

.. literalinclude:: ../../examples/08-aicpu-queue/kernel/queue_kernels.cpp
   :language: cpp
   :caption: 08-aicpu-queue/kernel/queue_kernels.cpp
