# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Moderate 16-task paged attention with optional onboard L0 collection."""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as pa_generate_inputs
from simpler_setup.scene_test import _outputs_dir

_PARAMS = {
    "batch": 4,
    "num_heads": 16,
    "kv_head_num": 1,
    "head_dim": 128,
    "block_size": 128,
    "context_len": 512,
    "max_model_len": 512,
    "dtype": "bfloat16",
}


def assert_complete_l0_capture(capture_dir):
    """The PA demo passes only when capture and task attribution also pass."""
    status = capture_dir / "l0_status.json"
    assert not status.exists(), f"L0 capture unavailable: {json.loads(status.read_text())}"
    assert (capture_dir / "l0_survey.json").is_file(), "L0 survey missing"
    joined = capture_dir / "l0_task_join.json"
    assert joined.is_file(), "L0 task join missing"
    result = json.loads(joined.read_text())
    assert result["counts"].get("ordered_marker_pair", 0) > 0, result["counts"]
    assert all(event["task_id"] is not None for event in result["events"]), result["counts"]
    assert len({event["task_id"] for event in result["events"]}) > 1, "expected multiple runtime task IDs"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestL0PagedAttention(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "../../batch_paged_attention/kernels/orchestration/paged_attention_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {"func_id": 0, "name": "PA_QK", "source": "paged_attention/kernels/aic/qk.cpp",
             "core_type": "aic", "signature": [D.IN, D.IN, D.OUT]},
            {"func_id": 1, "name": "PA_SF", "source": "paged_attention/kernels/aiv/sf.cpp",
             "core_type": "aiv", "signature": [D.IN, D.OUT, D.OUT, D.OUT]},
            {"func_id": 2, "name": "PA_PV", "source": "paged_attention/kernels/aic/pv.cpp",
             "core_type": "aic", "signature": [D.IN, D.IN, D.OUT]},
            {"func_id": 3, "name": "PA_UP", "source": "paged_attention/kernels/aiv/up.cpp",
             "core_type": "aiv", "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT]},
        ],
    }

    CASES = [
        {"name": "baseline", "platforms": ["a5"], "manual": True, "params": _PARAMS,
         "config": {"enable_chip_swimlane": 1}},
        {"name": "capture", "platforms": ["a5"], "manual": True, "params": _PARAMS,
         "config": {"enable_chip_swimlane": 1, "enable_l0": True}},
    ]

    def test_run(self, st_platform, st_worker, request):
        capture_selected = any(case["name"] == "capture" for case in self._matching_cases(st_platform, request))
        old = set(_outputs_dir().glob("TestL0PagedAttention_capture_*")) if capture_selected else set()
        super().test_run(st_platform, st_worker, request)
        if not capture_selected:
            return
        created = set(_outputs_dir().glob("TestL0PagedAttention_capture_*")) - old
        assert len(created) == 1, f"expected one new L0 capture directory, got {len(created)}"
        assert_complete_l0_capture(created.pop())

    def generate_args(self, params):
        return TaskArgsBuilder(*[
            TensorArg(name, value) if isinstance(value, torch.Tensor) else Scalar(name, value)
            for name, value in pa_generate_inputs(params)
        ])

    def compute_golden(self, args, params):
        tensors = {spec.name: spec.value for spec in args.specs if isinstance(spec, TensorArg)}
        pa_compute_golden(tensors, params)
        for spec in args.specs:
            if isinstance(spec, TensorArg) and spec.name in tensors:
                getattr(args, spec.name)[:] = tensors[spec.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
