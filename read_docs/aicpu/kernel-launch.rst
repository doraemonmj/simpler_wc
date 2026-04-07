AICPU Kernel Launch
===================

This page covers how to load and launch AICPU kernels from host code using CANN Runtime API.

.. note::

   See example ``examples/04-aicpu-kernel-launch/`` for a complete working implementation
   using Runtime API ``rtAicpuKernelLaunchExWithArgs()`` with the backend server pattern.

Backend Server Pattern
----------------------

AICPU kernel execution requires the **backend server pattern** - a multi-layer structure:

1. **System Kernel**: ``libaicpu_extend_kernels.so`` (CANN-provided)
2. **Backend Server**: Your compiled ``.so`` (e.g., ``libtilefwk_backend_server.so``)
3. **Entry Points**: Specific function names expected by system kernel

The system kernel acts as an intermediary, loading your backend server and calling entry points by name.

**Required Entry Points:**

.. code-block:: cpp

   extern "C" {

   // Called during init phase
   int DynTileFwkBackendKernelServerInit(void *arg);

   // Called during execution phase
   int DynTileFwkBackendKernelServer(void *arg);

   // Optional static variant
   int StaticTileFwkBackendKernelServer(void *arg);

   }

**Required Library Name:**

Your backend server must be named ``libtilefwk_backend_server.so`` for the system kernel to find it.

Writing an AICPU Kernel
-----------------------

Example kernel (see `examples/04-aicpu-kernel-launch/kernel/scale_kernel.cpp <../../examples/04-aicpu-kernel-launch/kernel/scale_kernel.cpp>`_):

.. code-block:: cpp

   #include <cstdint>

   extern "C" {

   // Your kernel arguments (must match host side exactly)
   struct ScaleArgs {
       void* input;      // HBM pointer
       void* output;     // HBM pointer
       int32_t count;
       float scale;
   };

   // DeviceArgs for custom argument passing
   struct DeviceArgs {
       uint64_t unused[12];
       uint64_t aicpuSoBin;
       uint64_t aicpuSoLen;
       uint64_t customArgsPtr;  // Pointer to ScaleArgs
   };

   static ScaleArgs* g_scaleArgs = nullptr;

   __attribute__((visibility("default")))
   int DynTileFwkBackendKernelServerInit(void *arg) {
       if (arg == nullptr) return -1;

       // Extract DeviceArgs from system kernel's internal structure
       DeviceArgs* devArgs = reinterpret_cast<DeviceArgs*>(
           *reinterpret_cast<uint64_t**>(reinterpret_cast<char*>(arg) + 40));

       if (devArgs != nullptr && devArgs->customArgsPtr != 0) {
           g_scaleArgs = reinterpret_cast<ScaleArgs*>(devArgs->customArgsPtr);
       }
       return 0;
   }

   __attribute__((visibility("default")))
   int DynTileFwkBackendKernelServer(void *arg) {
       if (arg == nullptr || g_scaleArgs == nullptr) return -1;

       ScaleArgs* args = g_scaleArgs;
       float* in = reinterpret_cast<float*>(args->input);
       float* out = reinterpret_cast<float*>(args->output);

       // Full C++ support - can use loops, STL, libm, etc.
       for (int32_t i = 0; i < args->count; i++) {
           out[i] = in[i] * args->scale;
       }
       return 0;
   }

   __attribute__((visibility("default")))
   int StaticTileFwkBackendKernelServer(void *arg) {
       return 0;  // Not used in this example
   }

   }

Compiling
---------

CMake configuration (see `examples/04-aicpu-kernel-launch/kernel/CMakeLists.txt <../../examples/04-aicpu-kernel-launch/kernel/CMakeLists.txt>`_):

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.10)
   project(tilefwk_backend_server LANGUAGES CXX)

   set(CMAKE_CXX_STANDARD 17)
   add_library(tilefwk_backend_server SHARED scale_kernel.cpp)

   target_compile_options(tilefwk_backend_server PRIVATE
       -fPIC           # Position independent code
       -O2             # Optimization
       -fno-exceptions # Optional: smaller code
   )

   set_target_properties(tilefwk_backend_server PROPERTIES
       OUTPUT_NAME "tilefwk_backend_server"
       PREFIX "lib"
       SUFFIX ".so"
   )

Build:

.. code-block:: bash

   # Native compile on Ascend device
   mkdir build && cd build
   cmake ..
   make  # Produces libtilefwk_backend_server.so

Launch API
----------

Host-side launch requires several steps:

1. Load ``.so`` binary to device HBM
2. Allocate and prepare argument structures in HBM
3. Launch init kernel (``DynTileFwkKernelServerInit``)
4. Launch main kernel (``DynTileFwkKernelServer``)

**Key Runtime APIs:**

.. code-block:: cpp

   // Load .so to HBM
   void* dev_so = nullptr;
   rtMalloc(&dev_so, so_size, RT_MEMORY_HBM, 0);
   rtMemcpy(dev_so, so_size, so_data, so_size, RT_MEMCPY_HOST_TO_DEVICE);

   // Launch kernel
   int rtAicpuKernelLaunchExWithArgs(
       rtKernelType_t kernel_type,    // KERNEL_TYPE_AICPU_KFC
       const char* stub_func,         // "AST_DYN_AICPU"
       uint32_t aicpu_num,            // Number of AICPU cores (usually 1)
       rtAicpuArgsEx_t* args,         // Extended args structure
       void* sm_desc,                 // nullptr
       rtStream_t stream,             // Stream handle
       uint32_t flags                 // 0
   );

Complete Example
----------------

See `examples/04-aicpu-kernel-launch/main.cpp <../../examples/04-aicpu-kernel-launch/main.cpp>`_ for full implementation.

Key steps:

1. **Initialize** (`main.cpp:199 <../../examples/04-aicpu-kernel-launch/main.cpp#L199>`_):

   .. code-block:: cpp

      rtSetDevice(0);
      rtStream_t stream;
      rtStreamCreate(&stream, 0);

2. **Allocate memory** (`main.cpp:232 <../../examples/04-aicpu-kernel-launch/main.cpp#L232>`_):

   .. code-block:: cpp

      void* dev_input = nullptr;
      rtMalloc(&dev_input, size, RT_MEMORY_HBM, 0);
      rtMemcpy(dev_input, size, host_input, size, RT_MEMCPY_HOST_TO_DEVICE);

3. **Load .so** (`main.cpp:107 <../../examples/04-aicpu-kernel-launch/main.cpp#L107>`_):

   .. code-block:: cpp

      void* dev_so = nullptr;
      rtMalloc(&dev_so, so_size, RT_MEMORY_HBM, 0);
      rtMemcpy(dev_so, so_size, so_data, so_size, RT_MEMCPY_HOST_TO_DEVICE);

4. **Prepare arguments** (`main.cpp:290 <../../examples/04-aicpu-kernel-launch/main.cpp#L290>`_):

   .. code-block:: cpp

      // Allocate ScaleArgs in HBM
      ScaleArgs* dev_args = nullptr;
      rtMalloc(&dev_args, sizeof(ScaleArgs), RT_MEMORY_HBM, 0);

      ScaleArgs args = {dev_input, dev_output, count, scale};
      rtMemcpy(dev_args, sizeof(args), &args, sizeof(args), RT_MEMCPY_HOST_TO_DEVICE);

      // Configure DeviceArgs
      DeviceArgs device_args;
      device_args.aicpuSoBin = (uint64_t)dev_so;
      device_args.aicpuSoLen = so_size;
      device_args.customArgsPtr = (uint64_t)dev_args;

5. **Launch kernels** (`main.cpp:353 <../../examples/04-aicpu-kernel-launch/main.cpp#L353>`_):

   .. code-block:: cpp

      // Init kernel
      LaunchAiCpuKernel(stream, &kernel_args, "DynTileFwkKernelServerInit", 1);

      // Main kernel
      LaunchAiCpuKernel(stream, &kernel_args, "DynTileFwkKernelServer", 1);

6. **Synchronize** (`main.cpp:390 <../../examples/04-aicpu-kernel-launch/main.cpp#L390>`_):

   .. code-block:: cpp

      rtStreamSynchronize(stream);
      rtMemcpy(host_output, size, dev_output, size, RT_MEMCPY_DEVICE_TO_HOST);

Argument Passing
----------------

The backend server pattern requires custom argument workaround:

**Problem**: System kernel doesn't directly pass custom arguments to backend server.

**Solution**: Use ``DeviceArgs.customArgsPtr`` indirection:

1. Host allocates custom args (``ScaleArgs``) in HBM
2. Host sets ``DeviceArgs.customArgsPtr`` to point to custom args
3. Init kernel extracts ``customArgsPtr`` from system kernel's internal structure
4. Main kernel uses saved pointer to access custom args

**DeviceArgs Layout:**

.. code-block:: cpp

   struct DeviceArgs {
       uint64_t unused[12];       // Padding for system kernel
       uint64_t aicpuSoBin;       // HBM address of .so
       uint64_t aicpuSoLen;       // Size of .so
       uint64_t customArgsPtr;    // Our custom args pointer
   };

**Extract in Init Kernel:**

.. code-block:: cpp

   DeviceArgs* devArgs = reinterpret_cast<DeviceArgs*>(
       *reinterpret_cast<uint64_t**>(reinterpret_cast<char*>(arg) + 40));

The offset (+40 bytes) is where the system kernel stores the ``DeviceArgs`` pointer in its internal structure.

AICPU Capabilities
------------------

AICPU runs on ARM Cortex-A55 cores with full C++ support:

✅ **Supported:**

- Full C/C++ standard library (libc/libstdc++)
- STL containers, algorithms, iterators
- Exceptions, RTTI
- Math library (libm) - sin, cos, sqrt, etc.
- Dynamic memory allocation (malloc/new)
- Floating-point operations
- Standard control flow (loops, conditionals, function calls)

❌ **Not Available:**

- Specialized compute units (no Cube/Vector like AICore)
- Ascend C tensor operators (those are AICore-only)
- Direct L1 buffer management (AICPU uses standard caches)

**When to use AICPU:**

- Dynamic shapes (size unknown at compile time)
- Complex control flow (data-dependent branches)
- Sparse or irregular memory access
- Standard algorithms (sorting, hashing, etc.)
- Operations not well-suited for SIMD/matrix units

Error Handling
--------------

Always check return codes:

.. code-block:: cpp

   int rc = rtAicpuKernelLaunchExWithArgs(...);
   if (rc != 0) {
       std::cerr << "Kernel launch failed: " << rc << '\n';
       return rc;
   }

   // Kernel errors may not surface until sync
   rc = rtStreamSynchronize(stream);
   if (rc != 0) {
       std::cerr << "Kernel execution failed: " << rc << '\n';
   }

.. warning::

   Kernel runtime errors may not be detected until ``rtStreamSynchronize()``.
   Always check both launch and sync return values.

Next Steps
----------

- For multi-core parallel execution on AICPU, see :doc:`multithread`
- For logging and debugging, see :doc:`logging`
- For synchronization between cores, see :doc:`atomic` and :doc:`queue`
