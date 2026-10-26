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

#include <cstddef>

#include "profiling_config.h"

#if SIMPLER_A5_ONBOARD_L0

#include <array>
#include <functional>
#include <string>

#include "common/l0_control.h"
#include "runtime/rt.h"

namespace simpler::a5::l0 {

class Capture {
public:
    using Launch = std::function<int(rtStream_t, void *, size_t, int, const std::string &)>;
    int begin(int device, uint64_t reg_table, rtStream_t stream, const Launch &launch,
              const std::string &prefix, uint64_t run_epoch);
    int finish(rtStream_t stream, const Launch &launch);
    void abandon() noexcept;
    bool active() const { return active_; }
    bool unsafe() const { return unsafe_; }
    bool ever_started() const { return ever_started_; }

private:
    int control(rtStream_t stream, const Launch &launch, uint32_t operation, uint32_t physical_core,
                ControlResult *out);
    bool disarm_all(rtStream_t stream, const Launch &launch);
    int release_after_disarm();
    void record_unavailable(const char *stage) const noexcept;

    int device_ = -1;
    uint64_t reg_table_ = 0;
    void *buffer_ = nullptr;
    void *result_ = nullptr;
    std::array<uint32_t, kPhysicalCoreCount> original_glitch_{};
    std::array<uint32_t, kPhysicalCoreCount> start_bytes_{};
    std::array<uint32_t, kPhysicalCoreCount> valid_bytes_{};
    std::array<bool, kPhysicalCoreCount> configured_{};
    std::array<bool, kPhysicalCoreCount> occupied_{};
    uint64_t run_epoch_ = 0;
    std::string prefix_;
    bool channel_started_ = false;
    bool ever_started_ = false;
    bool active_ = false;
    bool unsafe_ = false;
};

}  // namespace simpler::a5::l0

#endif
