#include "regs.h"

#include <cstdint>

void WriteToAicore(uint64_t* regs, int coreid, uint32_t val) {
    void* base = reinterpret_cast<void*>(regs[coreid]);
    if (base != nullptr) {
        volatile uint32_t* targetReg = reinterpret_cast<volatile uint32_t*>(regs[coreid] + REG_SPR_DATA_MAIN_BASE);
        *targetReg = val;
        DEV_INFO("[AICPU->AICORE] Wrote 0x%lx to offset 0x%x", val, REG_SPR_DATA_MAIN_BASE);
    }
    // 写入寄存器的值，哪怕程序结束也会在寄存器里，影响下一次读入，注意！！！
    __sync_synchronize();
}

void EnableToWritting(uint64_t* regs, uint32_t coreid) {
    volatile uint32_t* fastPathReg = reinterpret_cast<volatile uint32_t*>(regs[coreid] + REG_SPR_FAST_PATH_ENABLE);
    *fastPathReg = REG_SPR_FAST_PATH_OPEN;

    __sync_synchronize();

    volatile uint32_t* targetReg = reinterpret_cast<volatile uint32_t*>(regs[coreid] + REG_SPR_DATA_MAIN_BASE);
    *targetReg = 0;
}

void CloseToWritting(uint64_t* regs, uint32_t coreid) {
    volatile uint32_t* fastPathReg = reinterpret_cast<volatile uint32_t*>(regs[coreid] + REG_SPR_FAST_PATH_ENABLE);
    *fastPathReg = REG_SPR_FAST_PATH_CLOSE;

    __sync_synchronize();
}