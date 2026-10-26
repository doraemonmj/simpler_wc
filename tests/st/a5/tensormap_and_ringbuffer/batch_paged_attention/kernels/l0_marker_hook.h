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

// The regular PA kernels compile unchanged. Only the dedicated L0 wrapper
// defines these values before including the kernel source.
#if defined(SIMPLER_A5_ONBOARD_L0) && SIMPLER_A5_ONBOARD_L0
#ifndef SIMPLER_A5_L0_START_REGION
#error "L0 start region is required"
#endif
#ifndef SIMPLER_A5_L0_END_REGION
#error "L0 end region is required"
#endif
#ifndef SIMPLER_A5_L0_MARK_PIPE
#error "L0 marker pipe is required"
#endif
#define SIMPLER_PA_L0_MARK_BEGIN() bisheng::cce::mark_stamp<SIMPLER_A5_L0_MARK_PIPE, SIMPLER_A5_L0_START_REGION>()
#define SIMPLER_PA_L0_MARK_END() bisheng::cce::mark_stamp<SIMPLER_A5_L0_MARK_PIPE, SIMPLER_A5_L0_END_REGION>()
#else
#define SIMPLER_PA_L0_MARK_BEGIN() ((void)0)
#define SIMPLER_PA_L0_MARK_END() ((void)0)
#endif
