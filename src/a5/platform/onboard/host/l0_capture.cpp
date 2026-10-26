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

#include "host/l0_capture.h"

#include <acl/acl.h>
#include <driver/ascend_hal_base.h>
#include <profiling/aprof_pub.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/l0_trace_decode.h"
#include "common/unified_log.h"

namespace simpler::a5::l0 {
namespace {
constexpr size_t kTotalBufferBytes = size_t{kPhysicalCoreCount} * kBufferBytes;
}

void Capture::record_unavailable(const char *stage) const noexcept {
    try {
        std::filesystem::create_directories(prefix_);
        std::ofstream json(prefix_ + "/l0_status.json");
        json << "{\"schema\":\"l0_onboard_status_v1\",\"source\":\"live\",\"capture\":\"unavailable\","
                "\"stage\":\"" << stage << "\",\"channel\":" << kChannel << ",\"run_epoch\":" << run_epoch_
             << "}\n";
    } catch (...) {
        std::fprintf(stderr, "L0: unable to save failure status for stage %s\n", stage);
    }
}

int Capture::control(rtStream_t stream, const Launch &launch, uint32_t operation, uint32_t physical_core,
                     ControlResult *out) {
    const uint64_t target = reinterpret_cast<uint64_t>(buffer_) + size_t{physical_core} * kBufferBytes;
    ControlArgs args{reg_table_, target, reinterpret_cast<uint64_t>(result_), original_glitch_[physical_core],
                     physical_core, operation};
    int rc = -1;
    try {
        rc = launch(stream, &args, sizeof(args), 1, "simpler_aicpu_l0_control");
    } catch (...) {
        std::fprintf(stderr, "L0 control launch threw operation=%u core=%u\n", operation, physical_core);
        return -1;
    }
    if (rc != 0) {
        std::fprintf(stderr, "L0 control launch operation=%u core=%u rc=%d\n", operation, physical_core, rc);
        return rc;
    }
    rc = aclrtSynchronizeStreamWithTimeout(stream, 5000);
    if (rc != 0) {
        std::fprintf(stderr, "L0 control sync operation=%u core=%u rc=%d\n", operation, physical_core, rc);
        return rc;
    }
    rc = aclrtMemcpy(out, sizeof(*out), result_, sizeof(*out), ACL_MEMCPY_DEVICE_TO_HOST);
    if (rc != 0) {
        std::fprintf(stderr, "L0 control result copy operation=%u core=%u rc=%d\n", operation, physical_core, rc);
        return rc;
    }
    const int status = out->magic == kControlMagic && out->operation == operation ? out->status : -1;
    if (status != 0) {
        std::fprintf(stderr, "L0 control operation=%u core=%u status=%d magic=%08x\n",
                     operation, physical_core, status, out->magic);
    }
    return status;
}

bool Capture::disarm_all(rtStream_t stream, const Launch &launch) {
    bool confirmed = true;
    for (uint32_t core = 0; core < kPhysicalCoreCount; ++core) {
        if (!configured_[core]) continue;
        ControlResult stopped{};
        if (control(stream, launch, 1, core, &stopped) != 0) {
            confirmed = false;
            break;
        }
        valid_bytes_[core] = stopped.before.wptr;
        configured_[core] = false;
    }
    return confirmed;
}

int Capture::begin(int device, uint64_t reg_table, rtStream_t stream, const Launch &launch,
                   const std::string &prefix, uint64_t run_epoch) {
    if (active_ || unsafe_ || ever_started_ || device != 0 || reg_table == 0 || stream == nullptr || prefix.empty()) return -1;
    device_ = device;
    reg_table_ = reg_table;
    prefix_ = prefix;
    run_epoch_ = run_epoch;
    if (aclrtMalloc(&buffer_, kTotalBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&result_, sizeof(ControlResult), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMemset(buffer_, kTotalBufferBytes, 0xA5, kTotalBufferBytes) != 0) {
        std::fprintf(stderr, "L0 begin: GM allocation or initialization failed\n");
        record_unavailable("gm_allocation");
        release_after_disarm();
        return -1;
    }
    for (uint32_t core = 0; core < kPhysicalCoreCount; ++core) {
        ControlResult preflight{};
        if (control(stream, launch, 2, core, &preflight) != 0 || preflight.before.gen || preflight.before.len ||
            preflight.before.base_l || preflight.before.base_h) {
            std::fprintf(stderr, "L0 begin: preflight failed core=%u gen=%u len=%u base=%08x:%08x\n",
                         core, preflight.before.gen, preflight.before.len, preflight.before.base_h,
                         preflight.before.base_l);
            record_unavailable("preflight");
            unsafe_ = true;
            release_after_disarm();
            return -1;
        }
        original_glitch_[core] = preflight.before.glitch;
    }

    channel_list_t channels{};
    if (prof_drv_get_channels(device_, &channels) != 0) {
        record_unavailable("channel_query");
        release_after_disarm();
        return -1;
    }
    bool available = false;
    for (unsigned i = 0; i < channels.channel_num; ++i) available |= channels.channel[i].channel_id == kChannel;
    if (!available) {
        record_unavailable("channel_unavailable");
        release_after_disarm();
        return -1;
    }
    for (uint32_t core = 0; core < kPhysicalCoreCount; ++core) {
        ControlResult setup{};
        if (control(stream, launch, 0, core, &setup) != 0) {
            std::fprintf(stderr, "L0 begin: GM configuration failed core=%u; trying safe disarm\n", core);
            // A failed operation may have touched this core before returning.
            // Include it in the disarm check; the snapshot above proved it empty.
            configured_[core] = true;
            const bool disarmed = disarm_all(stream, launch);
            record_unavailable(disarmed ? "gm_configuration" : "disarm_unconfirmed");
            if (disarmed) release_after_disarm();
            else abandon();
            return -1;
        }
        configured_[core] = true;
    }
    struct BiuUserData { uint32_t size, mode, sub_core_type, physical_core_id; } user{16, 0, kSubCoreType, kCluster};
    prof_start_para start{};
    start.channel_type = PROF_TS_TYPE;
    start.real_time = PROF_REAL;
    start.user_data = &user;
    start.user_data_size = sizeof(user);
    ever_started_ = true;
    if (prof_drv_start(device_, kChannel, &start) != 0) {
        std::fprintf(stderr, "L0 begin: prof_drv_start channel %u failed\n", kChannel);
        const bool disarmed = disarm_all(stream, launch);
        record_unavailable(disarmed ? "prof_drv_start" : "disarm_unconfirmed");
        if (disarmed) release_after_disarm();
        else abandon();
        return -1;
    }
    channel_started_ = true;
    for (uint32_t core = 0; core < kPhysicalCoreCount; ++core) {
        ControlResult current{};
        const uint64_t target = reinterpret_cast<uint64_t>(buffer_) + size_t{core} * kBufferBytes;
        if (control(stream, launch, 2, core, &current) != 0 || current.before.gen != 1 ||
            current.before.len != kBufferBytes || current.before.base_l != static_cast<uint32_t>(target) ||
            current.before.base_h != static_cast<uint32_t>(target >> 32) ||
            current.before.wptr > kBufferBytes || (current.before.wptr & 3) != 0) {
            std::fprintf(stderr, "L0 begin: channel start changed GM config core=%u gen=%u len=%u base=%08x:%08x wptr=%u\n",
                         core, current.before.gen, current.before.len, current.before.base_h, current.before.base_l,
                         current.before.wptr);
            const bool disarmed = disarm_all(stream, launch);
            record_unavailable(disarmed ? "post_start_config" : "disarm_unconfirmed");
            if (disarmed) release_after_disarm();
            else abandon();
            return -1;
        }
        start_bytes_[core] = current.before.wptr;
    }
    active_ = true;
    return 0;
}

int Capture::release_after_disarm() {
    int rc = 0;
    if (channel_started_) {
        rc = prof_stop(device_, kChannel);
        channel_started_ = false;
    }
    if (result_ != nullptr) {
        const int free_rc = aclrtFree(result_);
        if (rc == 0) rc = free_rc;
        result_ = nullptr;
    }
    if (buffer_ != nullptr) {
        const int free_rc = aclrtFree(buffer_);
        if (rc == 0) rc = free_rc;
        buffer_ = nullptr;
    }
    active_ = false;
    return rc;
}

int Capture::finish(rtStream_t stream, const Launch &launch) {
    if (!active_) return -1;
    if (!disarm_all(stream, launch)) {
        LOG_ERROR("L0: unable to confirm all GM buffers disarmed; retaining GM");
        record_unavailable("disarm_unconfirmed");
        abandon();
        return -1;
    }
    int artifact_rc = 0;
    try {
        std::filesystem::create_directories(prefix_);
        std::ofstream survey(prefix_ + "/l0_survey.json");
        uint32_t configured_count = 0;
        for (uint32_t core = 0; core < kPhysicalCoreCount; ++core) configured_count += !occupied_[core];
        survey << "{\"schema\":\"l0_onboard_survey_v1\",\"source\":\"live\",\"device\":" << device_
               << ",\"channel\":" << kChannel << ",\"run_epoch\":" << run_epoch_
               << ",\"target_cores\":" << kPhysicalCoreCount << ",\"configured_cores\":" << configured_count
               << ",\"cores\":[";
        for (uint32_t core = 0; core < kPhysicalCoreCount; ++core) {
            if (core) survey << ',';
            const uint32_t bytes = valid_bytes_[core];
            const uint32_t start = start_bytes_[core];
            const char *status = occupied_[core] ? "occupied_after_start" : "empty";
            size_t events = 0;
            if (bytes > kBufferBytes || bytes < start || (bytes & 3) != 0) {
                status = "invalid_pointer";
                artifact_rc = -1;
            } else if (bytes > start) {
                std::vector<uint8_t> raw(bytes);
                void *source = static_cast<uint8_t *>(buffer_) + size_t{core} * kBufferBytes;
                if (aclrtMemcpy(raw.data(), bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                    status = "copy_failed";
                    artifact_rc = -1;
                } else {
                    const std::string stem = prefix_ + "/l0_core" + std::to_string(core);
                    std::ofstream bin(stem + ".bin", std::ios::binary);
                    bin.write(reinterpret_cast<const char *>(raw.data()), raw.size());
                    bin.close();
                    std::ofstream json(stem + ".json");
                    json << "{\"schema\":\"l0_onboard_trace_v1\",\"device\":" << device_
                         << ",\"channel\":" << kChannel << ",\"physical_core\":" << core
                         << ",\"run_epoch\":" << run_epoch_ << ",\"source\":\"live\""
                         << ",\"disarmed\":true,\"valid_bytes\":" << bytes
                         << ",\"capture_start_offset\":" << start
                         << ",\"captured_bytes\":" << bytes - start << ",\"events\":[";
                    bool first = true;
                    decode_contiguous(raw.data() + start, raw.size() - start, [&](const Stamp &stamp) {
                        if (!first) json << ',';
                        first = false;
                        ++events;
                        json << "{\"pipe\":" << unsigned(stamp.pipe) << ",\"region\":" << stamp.region_id
                             << ",\"block\":" << stamp.block_id << ",\"cycle\":" << stamp.cycle
                             << ",\"byte_offset\":" << start + stamp.byte_offset << '}';
                    });
                    json << "]}\n";
                    json.close();
                    if (!bin || !json) {
                        status = "write_failed";
                        artifact_rc = -1;
                    } else {
                        status = "captured";
                    }
                }
            }
            survey << "{\"physical_core\":" << core << ",\"valid_bytes\":" << bytes
                   << ",\"capture_start_offset\":" << start
                   << ",\"captured_bytes\":" << (bytes >= start ? bytes - start : 0)
                   << ",\"events\":" << events << ",\"status\":\"" << status << "\"}";
        }
        survey << "]}\n";
        survey.close();
        if (!survey) artifact_rc = -1;
    } catch (...) {
        artifact_rc = -1;
    }
    const int release_rc = release_after_disarm();
    return artifact_rc != 0 ? artifact_rc : release_rc;
}

void Capture::abandon() noexcept {
    // A device failure gives no proof that the DMA engine stopped. Keep the
    // allocations and channel registration; no unsafe rtFree or force reset.
    active_ = false;
    unsafe_ = true;
    channel_started_ = false;
    buffer_ = nullptr;
    result_ = nullptr;
}

}  // namespace simpler::a5::l0

#endif
