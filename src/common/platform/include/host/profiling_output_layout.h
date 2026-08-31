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

#ifndef SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_PROFILING_OUTPUT_LAYOUT_H
#define SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_PROFILING_OUTPUT_LAYOUT_H

#include <cstddef>
#include <string_view>

namespace simpler::dfx {

inline bool is_index_component(std::string_view component, std::string_view prefix) noexcept {
    if (component.size() <= prefix.size() || component.substr(0, prefix.size()) != prefix) return false;
    for (const char value : component.substr(prefix.size())) {
        if (value < '0' || value > '9') return false;
    }
    return true;
}

/**
 * Identify the per-Rank output layout that requires Host/Device clock alignment.
 *
 * The producer is ``python/simpler/worker.py::_read_config_from_mailbox``. Keep
 * its ``rankN/dN`` construction and this parser in sync; changing only one side
 * silently disables clock anchors and makes directory conversion fail later.
 */
inline bool is_rank_dispatch_output_prefix(std::string_view path) noexcept {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\'))
        path.remove_suffix(1);
    if (path.empty()) return false;

    const std::size_t dispatch_separator = path.find_last_of("/\\");
    if (dispatch_separator == std::string_view::npos) return false;
    const std::string_view dispatch = path.substr(dispatch_separator + 1);

    const std::string_view parent = path.substr(0, dispatch_separator);
    const std::size_t rank_separator = parent.find_last_of("/\\");
    const std::size_t rank_start = (rank_separator == std::string_view::npos) ? 0 : rank_separator + 1;
    const std::string_view rank = parent.substr(rank_start);
    return is_index_component(rank, "rank") && is_index_component(dispatch, "d");
}

}  // namespace simpler::dfx

#endif  // SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_PROFILING_OUTPUT_LAYOUT_H
