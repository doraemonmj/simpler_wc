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
#pragma once

#include <cstdint>

namespace simpler::a5::l0 {

inline constexpr uint32_t kControlMagic = 0x905A5B01;
inline constexpr uint32_t kBufferBytes = 1u << 20;
inline constexpr uint32_t kPhysicalCoreCount = 108;  // 36 clusters x (AIC, AIV0, AIV1)
inline constexpr uint32_t kCluster = 27;
inline constexpr uint32_t kSubCoreType = 1;  // AIV0, start trigger on core 90
inline constexpr uint32_t kChannel = 17;

struct Registers {
    uint32_t en, gen, len, glitch, base_l, base_h, wptr, crt, wrt;
};

struct ControlResult {
    uint32_t magic;
    int32_t status;
    uint32_t operation;
    uint32_t reserved;
    Registers before;
    Registers after;
};
static_assert(sizeof(ControlResult) == 88);

struct ControlArgs {
    uint64_t reg_table;
    uint64_t buffer_addr;
    uint64_t result_addr;
    uint64_t restore_glitch;
    uint32_t physical_core;
    uint32_t operation;  // 0=configure, 1=disarm, 2=snapshot
};
static_assert(sizeof(ControlArgs) == 40);

}  // namespace simpler::a5::l0
