
#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>

#include "device_log.h"

constexpr uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
constexpr uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
constexpr uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;

constexpr uint32_t REG_SPR_DATA_MAIN_BASE = 0xA0;
constexpr uint32_t REG_SPR_COND = 0x4C8;

void WriteToAicore(uint64_t* regAddrs, int coreid, uint32_t val);

void EnableToWritting(uint64_t* regs, uint32_t coreid);

void CloseToWritting(uint64_t* regs, uint32_t coreid);
