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
#include "profiling_config.h"

#if SIMPLER_A5_ONBOARD_L0

#include "common/l0_control.h"

namespace {
using simpler::a5::l0::ControlArgs;
using simpler::a5::l0::ControlResult;
using simpler::a5::l0::Registers;

uint32_t read_reg(uint64_t base, uint32_t offset) {
    return *reinterpret_cast<volatile uint32_t *>(base + offset);
}
void write_reg(uint64_t base, uint32_t offset, uint32_t value) {
    *reinterpret_cast<volatile uint32_t *>(base + offset) = value;
}
void barrier() { __asm__ __volatile__("dsb sy" ::: "memory"); }

Registers snapshot(uint64_t base) {
    return {read_reg(base, 0xC4),   read_reg(base, 0xB000), read_reg(base, 0xB004),
            read_reg(base, 0xB008), read_reg(base, 0xB00C), read_reg(base, 0xB010),
            read_reg(base, 0xB01C), read_reg(base, 0xB020), read_reg(base, 0xB028)};
}

void disarm(uint64_t base, uint32_t restore_glitch) {
    write_reg(base, 0xC4, 0);
    write_reg(base, 0xB000, 0);
    barrier();
    write_reg(base, 0xB00C, 0);
    write_reg(base, 0xB010, 0);
    write_reg(base, 0xB004, 0);
    write_reg(base, 0xB01C, 0);
    write_reg(base, 0xB008, restore_glitch);
    barrier();
}
}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l0_control(void *arg) {
    if (arg == nullptr) return -1;
    const auto &a = *reinterpret_cast<const ControlArgs *>(arg);
    if (a.reg_table == 0 || a.result_addr == 0 || a.physical_core >= simpler::a5::l0::kPhysicalCoreCount) return -1;
    auto *result = reinterpret_cast<ControlResult *>(a.result_addr);
    result->magic = simpler::a5::l0::kControlMagic;
    result->status = 0;
    result->operation = a.operation;
    result->reserved = 0;
    const uint64_t base = reinterpret_cast<const uint64_t *>(a.reg_table)[a.physical_core];
    if (base == 0) {
        result->status = -1;
        return 0;
    }
    result->before = snapshot(base);
    if (a.operation == 2) {
        result->after = result->before;
    } else if (a.operation == 0) {
        if (a.buffer_addr == 0 || (a.buffer_addr & 63) != 0 || (a.buffer_addr >> 48) != 0 ||
            (result->before.gen & 1) || result->before.len || result->before.base_l || result->before.base_h) {
            result->status = -2;
            result->after = result->before;
            return 0;
        }
        write_reg(base, 0xC4, 0);
        barrier();
        write_reg(base, 0xB00C, static_cast<uint32_t>(a.buffer_addr));
        write_reg(base, 0xB010, static_cast<uint32_t>(a.buffer_addr >> 32));
        write_reg(base, 0xB004, simpler::a5::l0::kBufferBytes);
        write_reg(base, 0xB01C, 0);
        write_reg(base, 0xB008, 0);
        write_reg(base, 0xB024, 1);
        write_reg(base, 0xB028, 0);
        barrier();
        write_reg(base, 0xB000, 1);
        barrier();
        result->after = snapshot(base);
        if (result->after.gen != 1 || result->after.len != simpler::a5::l0::kBufferBytes ||
            result->after.base_l != static_cast<uint32_t>(a.buffer_addr) ||
            result->after.base_h != static_cast<uint32_t>(a.buffer_addr >> 32)) {
            result->status = -3;
            disarm(base, result->before.glitch);
            result->after = snapshot(base);
        }
    } else if (a.operation == 1) {
        disarm(base, static_cast<uint32_t>(a.restore_glitch));
        result->after = snapshot(base);
        if ((result->after.en & 1) || (result->after.gen & 1) || result->after.len ||
            result->after.base_l || result->after.base_h ||
            result->after.glitch != static_cast<uint32_t>(a.restore_glitch)) {
            result->status = -4;
        }
    } else {
        result->status = -5;
    }
    barrier();
    return 0;
}

#endif
