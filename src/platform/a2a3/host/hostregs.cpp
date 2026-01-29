#include "hostregs.h"

bool GetPgMask(uint64_t &valid, int64_t deviceId) {
    uint64_t aicore_bitmap[AICORE_MAP_BUFF_LEN] = {0};
    int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * AICORE_MAP_BUFF_LEN;
    auto halFuncDevInfo =
        (int (*)(uint64_t deviceId, int32_t moduleType, int32_t infoType, void *buf, int32_t *size))dlsym(
            nullptr, "halGetDeviceInfoByBuff");
    if (halFuncDevInfo == nullptr) {
        return false;
    }
    auto ret = halFuncDevInfo(static_cast<uint32_t>(deviceId),
        MODULE_TYPE_AI_CORE,
        INFO_TYPE_OCCUPY,
        reinterpret_cast<void *>(&aicore_bitmap[0]),
        &size_n);
    if (ret != 0) {
        return false;
    }
    valid = aicore_bitmap[0];
    return true;
}

int GetAicoreRegInfo(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addrType, int64_t deviceId) {
    uint64_t valid = 0;
    if (!GetPgMask(valid, deviceId)) {
        valid = 0xFFFFFFFF;
        return -1;
    }

    uint64_t coreStride = 8 * 1024 * 1024;  // 8M
    uint64_t subCoreStride = 0x100000ULL;

    auto isValid = [&valid](int id) {
        const uint64_t mask = (1ULL << 25) - 1;
        return ((static_cast<uint64_t>(valid) ^ mask) & (1ULL << id)) == 0;
    };

    auto halFunc =
        (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue, size_t *outSizeRet))dlsym(
            nullptr, "halMemCtl");

    if (halFunc == nullptr) {
        return -1;
    }

    struct AddrMapInPara inMapPara;
    struct AddrMapOutPara outMapPara;
    inMapPara.devid = deviceId;
    inMapPara.addr_type = addrType;
    auto ret = halFunc(0,
        reinterpret_cast<void *>(&inMapPara),
        sizeof(struct AddrMapInPara),
        reinterpret_cast<void *>(&outMapPara),
        nullptr);

    if (ret != 0) {
        return ret;
    }

    for (uint32_t i = 0; i < DAV_2201::MAX_CORE; i++) {
        for (uint32_t j = 0; j < SUB_CORE_PER_AICORE; j++) {
            uint64_t vaddr = 0UL;
            if (isValid(i)) {
                vaddr = outMapPara.ptr + (i * coreStride + j * subCoreStride);
            }
            if (j == 0) {
                aic.push_back(vaddr);
            } else {
                aiv.push_back(vaddr);
            }
        }
    }
    return 0;
}

void GetAicoreRegs(std::vector<int64_t> &regs, uint64_t deviceId) {
    std::vector<int64_t> aiv;
    std::vector<int64_t> aic;
    int rt = GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL, deviceId);  //
    // 通过 IOMMU/SMMU 映射后的用户态虚拟地址

    if (rt != 0) {
        std::cout << " get aicore regs fail" << std::endl;
        return;
    }
    regs.insert(regs.end(), aic.begin(), aic.end());
    regs.insert(regs.end(), aiv.begin(), aiv.end());
    return;
}