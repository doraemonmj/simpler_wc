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

#include <gtest/gtest.h>

#include "host/profiling_output_layout.h"

TEST(ProfilingOutputLayoutTest, AcceptsRankDispatchSuffix) {
    EXPECT_TRUE(simpler::dfx::is_rank_dispatch_output_prefix("out/rank0/d0"));
    EXPECT_TRUE(simpler::dfx::is_rank_dispatch_output_prefix("out/rank12/d34/"));
    EXPECT_TRUE(simpler::dfx::is_rank_dispatch_output_prefix("rank2/d7"));
    EXPECT_TRUE(simpler::dfx::is_rank_dispatch_output_prefix(R"(out\rank3\d9\)"));
}

TEST(ProfilingOutputLayoutTest, RejectsOtherOutputPrefixes) {
    EXPECT_FALSE(simpler::dfx::is_rank_dispatch_output_prefix(""));
    EXPECT_FALSE(simpler::dfx::is_rank_dispatch_output_prefix("d0"));
    EXPECT_FALSE(simpler::dfx::is_rank_dispatch_output_prefix("/d0"));
    EXPECT_FALSE(simpler::dfx::is_rank_dispatch_output_prefix("out/rank/d0"));
    EXPECT_FALSE(simpler::dfx::is_rank_dispatch_output_prefix("out/rank0/dispatch0"));
    EXPECT_FALSE(simpler::dfx::is_rank_dispatch_output_prefix("out/rank0/d"));
}
