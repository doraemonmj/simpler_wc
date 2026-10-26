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
#include <cstdint>

namespace simpler::a5::l0 {

// DAV_3510 BIU/perfmon stores little-endian 4-byte chunks. START_STAMP is a
// four-chunk absolute cycle/block anchor. Pipe chunks (ctrl 0..6) carry a
// 16-bit cycle delta and a 12-bit user region. STATE chunks (ctrl 15) only
// advance the running cycle. This parser handles one contiguous, non-wrapped
// buffer image; a ring collector must linearize wrapped data first.
struct Stamp {
    uint64_t cycle;
    std::size_t byte_offset;
    uint16_t block_id;
    uint16_t region_id;
    uint8_t pipe;
};

inline uint32_t read_le32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

template <typename Emit>
std::size_t decode_contiguous(const uint8_t *data, std::size_t size, Emit &&emit) {
    if (data == nullptr) return 0;
    uint64_t cycle = 0;
    uint16_t block_id = 0;
    bool anchored = false;
    std::size_t count = 0;
    for (std::size_t pos = 0; pos + 4 <= size;) {
        const std::size_t offset = pos;
        const uint32_t word = read_le32(data + pos);
        const uint8_t ctrl = static_cast<uint8_t>(word >> 28);
        if (ctrl == 14) {
            if (pos + 16 > size) break;
            bool group = true;
            for (int c = 1; c < 4; ++c) group &= (read_le32(data + pos + c * 4) >> 28) == 14;
            if (!group) {
                // A damaged anchor cannot define absolute cycles. Wait for a
                // later complete START_STAMP group rather than mislabel data.
                anchored = false;
                pos += 4;
                continue;
            }
            cycle = 0;
            for (int c = 3; c >= 0; --c) cycle = (cycle << 16) | (read_le32(data + pos + c * 4) & 0xffff);
            block_id = static_cast<uint16_t>(
                (((read_le32(data + pos + 4) >> 16) & 0xff) << 8) | ((word >> 16) & 0xff)
            );
            anchored = true;
            pos += 16;
            continue;
        }
        pos += 4;
        if (!anchored || word == 0xefffffff || word == 0xdddddddd) continue;
        cycle += word & 0xffff;
        if (ctrl > 6) continue;
        emit(Stamp{cycle, offset, block_id, static_cast<uint16_t>((word >> 16) & 0xfff), ctrl});
        ++count;
    }
    return count;
}

}  // namespace simpler::a5::l0
