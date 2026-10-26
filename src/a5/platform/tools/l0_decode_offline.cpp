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
// Decode a preserved, contiguous onboard L0 GM image without accessing the device.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "common/l0_trace_decode.h"

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: l0_decode_offline RAW.bin EVENTS.json [first_task_bytes]\n";
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 2;
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (bytes.empty() || bytes.size() % 4 != 0) {
        std::cerr << "L0 raw data must be nonempty and four-byte aligned\n";
        return 2;
    }
    const std::size_t boundary = argc == 4 ? std::strtoull(argv[3], nullptr, 10) : 0;
    if (argc == 4 && (boundary == 0 || boundary >= bytes.size() || boundary % 4 != 0)) return 2;
    std::ofstream json(argv[2]);
    if (!json) return 2;
    json << "{\"schema\":\"l0_onboard_trace_v1\",\"source\":\"archive\",\"valid_bytes\":"
         << bytes.size() << ",\"events\":[";
    bool first = true;
    const std::size_t count = simpler::a5::l0::decode_contiguous(
        bytes.data(), bytes.size(), [&](const simpler::a5::l0::Stamp &stamp) {
            if (!first) json << ',';
            first = false;
            json << "{\"pipe\":" << unsigned(stamp.pipe) << ",\"region\":" << stamp.region_id
                 << ",\"block\":" << stamp.block_id << ",\"cycle\":" << stamp.cycle
                 << ",\"byte_offset\":" << stamp.byte_offset;
            if (boundary) json << ",\"task_window\":" << (stamp.byte_offset < boundary ? 1 : 2);
            json << '}';
        }
    );
    json << "]}\n";
    if (!json || count == 0) return 1;
    std::cout << "decoded " << count << " events from " << bytes.size() << " archived bytes\n";
    return 0;
}
