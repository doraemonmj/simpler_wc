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
// L0-only wrapper around the existing batch paged-attention online-update kernel.
#define SIMPLER_A5_ONBOARD_L0 1
#define SIMPLER_A5_L0_START_REGION 106
#define SIMPLER_A5_L0_END_REGION 107
#define SIMPLER_A5_L0_MARK_PIPE PIPE_MTE3
#include "../../../../../batch_paged_attention/kernels/aiv/aiv_online_update.cpp"
