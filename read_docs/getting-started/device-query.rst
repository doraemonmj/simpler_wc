Device Query
============

The first step in NPU programming is discovering available devices
and their capabilities. Example 01 demonstrates how to enumerate
Ascend NPU devices and query their hardware properties using direct
CANN ACL (Ascend Computing Language) APIs.

Hardware Concepts
-----------------

Ascend NPU devices have a hierarchical compute and memory architecture:

**Compute Units:**

- **AICORE**: Primary compute units with dual-engine architecture:

  - **Cube Engine**: Matrix multiplication accelerator
  - **Vector Engine**: SIMD processor for element-wise operations

- **AICPU**: Control processors for task orchestration and scalar operations

**Memory Hierarchy:**

- **HBM** (High Bandwidth Memory): Device DRAM, 32-64 GB typical
- **L2 Cache**: Shared cache across all AICOREs, ~192-256 MB
- **L1/UB** (Unified Buffer): Per-AICORE scratchpad, ~192-512 KB
- **L0A/B/C**: Cube engine local buffers, 64-256 KB each

Example Code
------------

Full source: :file:`examples/01-device-query/main.cpp`

.. literalinclude:: ../../examples/01-device-query/main.cpp
   :language: cpp
   :caption: Device Query Example (01-device-query/main.cpp)

Two Query Methods
-----------------

This example demonstrates two complementary approaches to querying hardware properties:

1. **Runtime Queries via ACL APIs**

   Properties that can be queried at runtime using CANN ACL functions:

   - Device count (``aclrtGetDeviceCount``)
   - SoC version string (``aclrtGetSocName``)
   - Core counts: AICORE, Vector, AICPU (``aclrtGetDeviceInfo``)
   - HBM capacity and usage (``aclrtGetMemInfo``)

2. **Platform Config File Parsing**

   Architectural constants that are fixed at chip design time and cannot be
   queried via APIs. These are read from CANN platform INI files:

   Path: ``$ASCEND_HOME_PATH/aarch64-linux/data/platform_config/<SoC>.ini``

   Properties:

   - Core counts (verification/cross-check with runtime queries)
   - L2 cache size
   - Per-AICORE buffer sizes: UB, L1, L0A, L0B, L0C

The example explicitly labels each output section with its data source
(``via aclrtGetSocName``, ``via aclrtGetDeviceInfo``, ``from platform_config``)
to teach the distinction between these two query methods.

ACL API Reference
-----------------

Runtime Initialization
^^^^^^^^^^^^^^^^^^^^^^

``aclInit``
"""""""""""

.. code-block:: cpp

   aclError aclInit(const char* configPath);

Initialize the ACL runtime. Must be called before any other ACL functions.

**Parameters:**

- ``configPath``: Optional configuration file path (typically ``nullptr``)

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:140 <../../examples/01-device-query/main.cpp#L140>`_

``aclFinalize``
"""""""""""""""

.. code-block:: cpp

   aclError aclFinalize();

Clean up ACL runtime resources. Should be called before program exit.

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:259 <../../examples/01-device-query/main.cpp#L259>`_

Device Enumeration
^^^^^^^^^^^^^^^^^^

``aclrtGetDeviceCount``
"""""""""""""""""""""""

.. code-block:: cpp

   aclError aclrtGetDeviceCount(uint32_t* count);

Query the number of NPU devices on the PCIe bus.

**Parameters:**

- ``count``: Output parameter for device count

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:149 <../../examples/01-device-query/main.cpp#L149>`_

Device Context Management
^^^^^^^^^^^^^^^^^^^^^^^^^

``aclrtSetDevice``
""""""""""""""""""

.. code-block:: cpp

   aclError aclrtSetDevice(int32_t deviceId);

Set the current device context. Required before querying device-specific properties.

**Parameters:**

- ``deviceId``: Device index (0-based)

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:169 <../../examples/01-device-query/main.cpp#L169>`_

``aclrtResetDevice``
""""""""""""""""""""

.. code-block:: cpp

   aclError aclrtResetDevice(int32_t deviceId);

Reset and release device context.

**Parameters:**

- ``deviceId``: Device index (0-based)

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:252 <../../examples/01-device-query/main.cpp#L252>`_

Device Properties
^^^^^^^^^^^^^^^^^

``aclrtGetSocName``
"""""""""""""""""""

.. code-block:: cpp

   const char* aclrtGetSocName();

Get the SoC version string (e.g., "Ascend910_9392").

**Returns:**

- Pointer to SoC name string, or ``nullptr`` on error

**Line Reference:** `main.cpp:177 <../../examples/01-device-query/main.cpp#L177>`_

``aclrtGetDeviceInfo``
""""""""""""""""""""""

.. code-block:: cpp

   aclError aclrtGetDeviceInfo(int32_t deviceId,
                               aclrtDeviceAttr attr,
                               int64_t* value);

Query device attributes such as core counts.

**Parameters:**

- ``deviceId``: Device index (0-based)
- ``attr``: Attribute to query (see below)
- ``value``: Output parameter for attribute value

**Attribute Constants:**

- ``ACL_DEV_ATTR_AICORE_CORE_NUM``: Number of AICORE blocks
- ``ACL_DEV_ATTR_VECTOR_CORE_NUM``: Number of Vector cores
- ``ACL_DEV_ATTR_AICPU_CORE_NUM``: Number of AICPU cores

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:190-199 <../../examples/01-device-query/main.cpp#L190>`_

``aclrtGetMemInfo``
"""""""""""""""""""

.. code-block:: cpp

   aclError aclrtGetMemInfo(aclrtMemAttr attr,
                           size_t* free,
                           size_t* total);

Query memory capacity and usage.

**Parameters:**

- ``attr``: Memory type (``ACL_HBM_MEM`` for device memory)
- ``free``: Output parameter for free memory in bytes
- ``total``: Output parameter for total memory in bytes

**Returns:**

- ``ACL_SUCCESS`` on success
- Error code on failure

**Line Reference:** `main.cpp:210 <../../examples/01-device-query/main.cpp#L210>`_

Platform Config Parsing
^^^^^^^^^^^^^^^^^^^^^^^

The example includes a custom ``read_soc_config()`` function that parses
CANN platform INI files to read architectural constants:

**Function:** `read_soc_config() <../../examples/01-device-query/main.cpp#L61>`_

**Config File Sections:**

- ``[SoCInfo]``: Core counts, L2 cache size
- ``[AICoreSpec]``: Per-core buffer sizes (UB, L1, L0A/B/C)

**Why this is needed:** Many hardware parameters are architectural constants
(baked into hardware design) and cannot be queried via ACL APIs. CANN stores
these in SoC-specific INI files.

Sample Output
-------------

.. code-block:: text

   === Ascend NPU Device Query ===

   Found 16 NPU device(s)

   --- Device 0 ---
     SoC Version                       : Ascend910_9392 (via aclrtGetSocName)

     Core Configuration (via aclrtGetDeviceInfo):
       AICORE cores                    : 24
       Vector cores                    : 48
       AICPU cores                     : 6

     Memory Hierarchy (via aclrtGetMemInfo):
       HBM Total                       : 61.27 GB
       HBM Free                        : 60.87 GB

     Hardware Configuration (from platform_config/Ascend910_9392.ini):
       AICORE cores                    : 24
       Cube cores                      : 24
       Vector cores                    : 48
       AICPU cores                     : 6

       L2 Cache                        : 192.00 MB

       AICORE Buffers (per core):
         Unified Buffer (UB)           : 192.00 KB
         L1 Buffer                     : 512.00 KB
         L0A Buffer                    : 64.00 KB
         L0B Buffer                    : 64.00 KB
         L0C Buffer                    : 128.00 KB

.. note::

   Output varies by hardware. The example shows Ascend910_9392 with 16 devices.
   Your system may show different SoC versions (910A, 910B, 910C, 310P, etc.)
   and device counts (1, 4, 8, or 16 devices typical).

   All output colons align at column 36 for improved readability.

Building and Running
--------------------

.. code-block:: bash

   cd examples/01-device-query
   mkdir build && cd build
   cmake ..
   make
   ./01-device-query

**Prerequisites:**

- CANN toolkit installed (version 5.0+)
- ``ASCEND_HOME_PATH`` environment variable set
- At least one Ascend NPU device in the system

Key Takeaways
-------------

After completing this example, you should understand:

1. **Device Enumeration**: How to query the number of NPU devices using ``aclrtGetDeviceCount()``
2. **Hardware Architecture**: The distinction between AICORE (compute), AICPU (control), and memory hierarchy
3. **Two Query Methods**: Runtime ACL APIs vs platform config file parsing for architectural constants
4. **Device Context**: The need to set device context with ``aclrtSetDevice()`` before querying properties
5. **Data Source Transparency**: Understanding which properties come from runtime queries vs static config files

Best Practices
--------------

1. **Always check device count** before assuming devices exist
2. **Handle multi-device systems** gracefully - servers often have 8-16 NPUs
3. **Validate all ACL return codes** - check for ``ACL_SUCCESS`` after every API call
4. **Set device context** before querying device-specific properties
5. **Read platform config when needed** - for architectural constants not exposed via ACL APIs
6. **Log device info at startup** - helps with debugging and performance analysis

Next Steps
----------

See :doc:`memory` to learn about HBM allocation, host-device memory
transfers, and the memory hierarchy in practice.
