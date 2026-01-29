#ifndef RUNTIME_HOSTREGS_H
#define RUNTIME_HOSTREGS_H

#include <dlfcn.h>
#include <runtime/rt.h>
#include <sched.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

const int32_t MODULE_TYPE_AI_CORE = 4;
const int32_t INFO_TYPE_OCCUPY = 8;
const uint8_t AICORE_MAP_BUFF_LEN = 2;

constexpr int ADDR_MAP_TYPE_REG_AIC_CTRL = 2;
constexpr uint32_t SUB_CORE_PER_AICORE = 3;

namespace DAV_2201 {
constexpr uint32_t MAX_CORE = 25;
}

struct AddrMapInPara {
    unsigned int addr_type;
    unsigned int devid;
};

struct AddrMapOutPara {
    unsigned long long ptr;
    unsigned long long len;
};

bool GetPgMask(uint64_t &valid, int64_t deviceId);

int GetAicoreRegInfo(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addrType, int64_t deviceId);

void GetAicoreRegs(std::vector<int64_t> &regs, uint64_t deviceId);

#endif