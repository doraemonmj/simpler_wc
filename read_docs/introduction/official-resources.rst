Official Resources
==================

Huawei provides extensive documentation for Ascend development.
This guide complements (not replaces) the official resources.

CANN Documentation
------------------

CANN (Compute Architecture for Neural Networks) is Huawei's software stack for Ascend.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Resource
     - URL
   * - CANN Documentation Portal
     - https://www.hiascend.com/document
   * - CANN Developer Guide
     - https://www.hiascend.com/document/detail/en/canncommercial/
   * - Ascend C Programming Guide
     - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
   * - ACL API Reference
     - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/appdevg/acldevg/acldevg_0001.html

Development Tools
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - Description
   * - MindStudio
     - IDE for Ascend development (profiling, debugging)
   * - ATC (Ascend Tensor Compiler)
     - Model conversion and optimization
   * - TIK
     - DSL for writing AICORE kernels
   * - Ascend C
     - C++ extension for AICORE kernels

Community Resources
-------------------

- **Ascend**: https://gitcode.com/ascend/
- **CANN**: https://gitcode.com/cann

Version Compatibility
---------------------

This guide is written for:

- **CANN Version**: 8.5+
- **Hardware**: Ascend 910B (A2), Ascend 910C (A3)
- **Driver**: Matching CANN version

.. note::

   API signatures may change between CANN versions. This guide uses
   high-level ACL APIs which are more stable than internal ``rt*`` APIs.
