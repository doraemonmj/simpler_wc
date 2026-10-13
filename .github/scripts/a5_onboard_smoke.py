# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Initialize the allocated A5 through AscendCL without project dependencies."""

import ctypes
import os


def check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with AscendCL error {result}")


device = int(os.environ["TASK_DEVICE"].split(",", maxsplit=1)[0])
acl = ctypes.CDLL("libascendcl.so")
acl.aclInit.argtypes = [ctypes.c_char_p]
acl.aclInit.restype = ctypes.c_int
acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
acl.aclrtGetDeviceCount.restype = ctypes.c_int
acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
acl.aclrtSetDevice.restype = ctypes.c_int
acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
acl.aclrtResetDevice.restype = ctypes.c_int
acl.aclFinalize.restype = ctypes.c_int

check(acl.aclInit(None), "aclInit")
try:
    count = ctypes.c_uint32()
    check(acl.aclrtGetDeviceCount(ctypes.byref(count)), "aclrtGetDeviceCount")
    check(acl.aclrtSetDevice(device), "aclrtSetDevice")
    print(f"A5 onboard smoke passed: allocated device {device}, visible devices {count.value}")
    check(acl.aclrtResetDevice(device), "aclrtResetDevice")
finally:
    check(acl.aclFinalize(), "aclFinalize")
