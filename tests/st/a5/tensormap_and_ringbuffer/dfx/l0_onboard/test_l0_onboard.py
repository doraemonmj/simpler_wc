#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One MIX cohort exercises all 36 AIC, 36 AIV0, and 36 AIV1 cores."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestL0Onboard(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/l0_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {"func_id": 0, "name": "L0_MARK_AIC", "source": "kernels/aic/l0_mark.cpp", "core_type": "aic"},
            {"func_id": 1, "name": "L0_MARK_AIV0", "source": "kernels/aiv/l0_mark.cpp", "core_type": "aiv"},
            {"func_id": 2, "name": "L0_MARK_AIV1", "source": "kernels/aiv/l0_mark.cpp", "core_type": "aiv"},
        ],
    }

    CASES = [
        {
            "name": "all_physical_cores_one_mix_task",
            "platforms": ["a5"],
            "manual": ["a5"],
            "params": {},
            "config": {"enable_l0": True, "enable_chip_swimlane": 1},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(TensorArg("output", torch.zeros(108 * 16, dtype=torch.float32)))

    def compute_golden(self, args, params):
        for block in range(36):
            for slot in range(3):
                args.output[(block * 3 + slot) * 16] = float(block)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
