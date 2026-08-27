/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "async_kernel_api.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    volatile int32_t spin = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(args[1]); ++i) {
        ++spin;
    }
    (void)spin;
    volatile __gm__ uint32_t *counter = reinterpret_cast<volatile __gm__ uint32_t *>(static_cast<uint64_t>(args[0]));
    *counter = 1;
    pto2::detail::defer_flush_range(counter, sizeof(*counter));
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
}
