AICPU Logging
=============

Debug output from AICPU kernels for development and troubleshooting.

Basic Logging
-------------

AICPU kernels can use standard C I/O for debug output:

.. code-block:: cpp

   // Inside AICPU kernel
   #include <cstdio>

   extern "C" void my_kernel(Args* args) {
       printf("[AICPU] Kernel started, count=%d\n", args->count);

       for (int i = 0; i < args->count; i++) {
           if (i % 1000 == 0) {
               printf("[AICPU] Processing element %d\n", i);
           }
           // ... processing ...
       }

       printf("[AICPU] Kernel finished\n");
   }

.. note::

   Output appears in the system log or stdout depending on runtime
   configuration. Performance impact is significant - use sparingly.

Conditional Logging
-------------------

Use compile-time or runtime flags to control logging:

.. code-block:: cpp

   // Compile-time control
   #ifdef DEBUG_KERNEL
   #define KLOG(fmt, ...) printf("[AICPU] " fmt "\n", ##__VA_ARGS__)
   #else
   #define KLOG(fmt, ...) ((void)0)
   #endif

   extern "C" void my_kernel(Args* args) {
       KLOG("Starting with count=%d", args->count);
       // ...
   }

Build with debug:

.. code-block:: bash

   # Debug build
   g++ -shared -fPIC -DDEBUG_KERNEL -o kernel_debug.so kernel.cpp

   # Release build (no logging)
   g++ -shared -fPIC -O2 -o kernel.so kernel.cpp

Runtime-Controlled Logging
--------------------------

Pass log level through arguments:

.. code-block:: cpp

   struct KernelArgs {
       void* input;
       void* output;
       int32_t count;
       int32_t log_level;  // 0=none, 1=errors, 2=info, 3=debug
   };

   extern "C" void my_kernel(KernelArgs* args) {
       if (args->log_level >= 2) {
           printf("[INFO] Processing %d elements\n", args->count);
       }

       for (int i = 0; i < args->count; i++) {
           if (args->log_level >= 3 && i % 100 == 0) {
               printf("[DEBUG] Element %d\n", i);
           }
           // ... processing ...
       }
   }

Error Reporting
---------------

For production code, return error codes instead of logging:

.. code-block:: cpp

   struct KernelArgs {
       void* input;
       void* output;
       int32_t count;
       int32_t* error_code;  // Device pointer for error output
   };

   extern "C" void my_kernel(KernelArgs* args) {
       if (args->input == nullptr) {
           *(args->error_code) = 1;  // ERROR_NULL_INPUT
           return;
       }

       if (args->count <= 0) {
           *(args->error_code) = 2;  // ERROR_INVALID_COUNT
           return;
       }

       // ... processing ...

       *(args->error_code) = 0;  // SUCCESS
   }

Host side:

.. code-block:: cpp

   int32_t error_code = 0;
   int32_t* dev_error = (int32_t*)platform_malloc(sizeof(int32_t));
   platform_memcpy_h2d(dev_error, &error_code, sizeof(int32_t));

   args.error_code = dev_error;
   platform_aicpu_launch(...);
   platform_stream_sync(NULL);

   platform_memcpy_d2h(&error_code, dev_error, sizeof(int32_t));
   if (error_code != 0) {
       printf("Kernel error: %d\n", error_code);
   }

Performance Considerations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Logging Method
     - Impact
   * - ``printf`` in loop
     - Very slow, avoid in production
   * - Conditional compile
     - Zero cost when disabled
   * - Runtime flag check
     - Small overhead, acceptable
   * - Error code return
     - Minimal, recommended for production

Best Practices
--------------

1. **Remove debug logs in production** - Use conditional compilation
2. **Log at boundaries** - Entry, exit, major phases only
3. **Use error codes** - Machine-readable, low overhead
4. **Include context** - Block ID, element index, etc.
5. **Timestamp if needed** - Use ``clock_gettime()`` for profiling

Example Code
------------

Full working example: :file:`examples/06-aicpu-logging/`

Host code demonstrating logging patterns:

.. literalinclude:: ../../examples/06-aicpu-logging/main.cpp
   :language: cpp
   :caption: 06-aicpu-logging/main.cpp
   :lines: 1-80

AICPU kernel with debug logging:

.. literalinclude:: ../../examples/06-aicpu-logging/kernel/debug_kernel.cpp
   :language: cpp
   :caption: 06-aicpu-logging/kernel/debug_kernel.cpp
