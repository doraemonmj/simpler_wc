Register-Based Synchronization
==============================

Using shared registers for fast signaling between AICPU and AICORE.

Overview
--------

Ascend provides shared registers accessible from both AICPU and AICORE:

.. code-block:: text

   ┌─────────────┐                    ┌─────────────┐
   │   AICPU     │                    │   AICORE    │
   │             │    ┌─────────┐     │             │
   │  write ────────► │ SHARED  │ ◄────── read     │
   │  read  ◄──────── │ REGISTER│ ────────► write  │
   │             │    └─────────┘     │             │
   └─────────────┘                    └─────────────┘

   Latency: ~1 μs (very fast)

Register Operations
-------------------

**Write to register:**

.. code-block:: cpp

   // AICPU
   WRITE_SHARED_REG(reg_id, value);

   // AICORE
   WRITE_SHARED_REG(reg_id, value);

**Read from register:**

.. code-block:: cpp

   // Both AICPU and AICORE
   value = READ_SHARED_REG(reg_id);

Signal Pattern
--------------

Producer signals when data is ready:

.. code-block:: cpp

   // AICPU Producer
   void producer_kernel(Args* args) {
       // Prepare data
       process_input(args->input, args->output);

       // Memory barrier to ensure writes are visible
       __sync_synchronize();

       // Signal ready
       WRITE_SHARED_REG(0, 1);  // reg 0 = ready flag
   }

   // AICORE Consumer
   void consumer_kernel(Args* args) {
       // Wait for signal
       while (READ_SHARED_REG(0) == 0) {
           // Spin
       }

       // Data is ready, process
       compute(args->data);
   }

Bidirectional Signaling
-----------------------

For round-trip communication:

.. code-block:: cpp

   // Registers:
   // REG 0: AICPU → AICORE (data ready)
   // REG 1: AICORE → AICPU (compute done)

   // AICPU
   void controller_kernel(Args* args) {
       for (int batch = 0; batch < num_batches; batch++) {
           // Prepare batch
           prepare_batch(batch);
           __sync_synchronize();

           // Signal AICORE
           WRITE_SHARED_REG(0, batch + 1);

           // Wait for AICORE
           while (READ_SHARED_REG(1) < batch + 1) {
               // Spin
           }

           // AICORE done, continue
       }
   }

   // AICORE
   void compute_kernel(Args* args) {
       for (int batch = 0; batch < num_batches; batch++) {
           // Wait for data
           while (READ_SHARED_REG(0) < batch + 1) {
               // Spin
           }

           // Compute
           process_batch(batch);
           __sync_synchronize();

           // Signal done
           WRITE_SHARED_REG(1, batch + 1);
       }
   }

Multiple Registers
------------------

Use multiple registers for richer state:

.. code-block:: cpp

   // Register assignments
   #define REG_STATE        0   // Current state machine state
   #define REG_BATCH_ID     1   // Current batch being processed
   #define REG_ERROR_CODE   2   // Error signaling
   #define REG_PROGRESS     3   // Progress counter

   // AICPU sets state
   WRITE_SHARED_REG(REG_STATE, STATE_PROCESSING);
   WRITE_SHARED_REG(REG_BATCH_ID, current_batch);

   // AICORE reads state
   int state = READ_SHARED_REG(REG_STATE);
   int batch = READ_SHARED_REG(REG_BATCH_ID);

Memory Ordering
---------------

Register writes don't guarantee memory writes are visible:

.. code-block:: cpp

   // WRONG - data may not be visible
   gm_data[0] = computed_value;
   WRITE_SHARED_REG(0, 1);  // Signal may arrive before data

   // CORRECT - barrier ensures ordering
   gm_data[0] = computed_value;
   __sync_synchronize();      // Memory barrier
   WRITE_SHARED_REG(0, 1);    // Now safe

Limitations
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Limitation
     - Description
   * - Limited count
     - Only a few registers available (4-8 typical)
   * - Single value
     - Each register holds one 32/64-bit value
   * - Polling required
     - Reader must poll (no interrupt)
   * - All blocks see same
     - Can't have per-block registers

Best Practices
--------------

1. **Use memory barriers** - Before signaling
2. **Minimize polling** - Poll with backoff if needed
3. **Clear after use** - Reset registers for next round
4. **Document assignments** - Track which register means what

Example Code
------------

Full working example: :file:`examples/16-aicpu-aicore-register/`

Host code demonstrating register-based synchronization:

.. literalinclude:: ../../examples/16-aicpu-aicore-register/main.cpp
   :language: cpp
   :caption: 16-aicpu-aicore-register/main.cpp
   :lines: 1-80

AICPU producer kernel:

.. literalinclude:: ../../examples/16-aicpu-aicore-register/aicpu_kernel/producer.cpp
   :language: cpp
   :caption: 16-aicpu-aicore-register/aicpu_kernel/producer.cpp

AICORE consumer kernel (PTO-ISA):

.. literalinclude:: ../../examples/16-aicpu-aicore-register/aicore_kernel.pto
   :language: text
   :caption: 16-aicpu-aicore-register/aicore_kernel.pto
