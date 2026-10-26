# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L0 task attribution must not invent an ID between two runtime tasks."""

import json

from simpler_setup.tools.l0_task_join import join, pa_ordered_assignments


def test_two_tasks_on_one_core_only_match_their_own_cycle_windows(tmp_path):
    (tmp_path / "chip_swimlane_records.json").write_text(json.dumps({
        "metadata": {"clock_freq_hz": 1_000_000_000, "core_types": ["aic"]},
        "aicore_tasks": [[0, 42, 1, 100, 200, 0, 1], [0, 43, 2, 300, 400, 0, 1]],
    }))
    (tmp_path / "l0_survey.json").write_text(json.dumps({"run_epoch": 1, "channel": 17}))
    raw = b"\x00\x00\x00\xe0" * 4 + b"\x00\x00\x0a\x00" * 2
    (tmp_path / "l0_core0.bin").write_bytes(raw)
    (tmp_path / "l0_core0.json").write_text(json.dumps({
        "physical_core": 0, "run_epoch": 1, "capture_start_offset": 0,
        "captured_bytes": len(raw), "valid_bytes": len(raw),
        "events": [
            {"pipe": 0, "region": 10, "block": 0, "cycle": 150, "byte_offset": 16},
            {"pipe": 0, "region": 10, "block": 0, "cycle": 250, "byte_offset": 20},
        ],
    }))

    result = join(tmp_path)
    assert result["events"][0]["task_id"] == 42
    assert result["events"][0]["join_status"] == "matched_interval"
    assert result["events"][1]["task_id"] is None
    assert result["events"][1]["join_status"] == "no_task_interval"


def test_paged_attention_marker_pairs_match_task_order_without_clock_alignment(tmp_path):
    (tmp_path / "chip_swimlane_records.json").write_text(json.dumps({
        "metadata": {"clock_freq_hz": 1_000_000_000, "core_types": ["aic"]},
        "aicore_tasks": [[0, 42, 1, 100, 200, 0, 1], [0, 43, 2, 300, 400, 0, 1]],
    }))
    (tmp_path / "l0_survey.json").write_text(json.dumps({"run_epoch": 1, "channel": 17}))
    regions = [100, 101, 104, 105]
    raw = b"\x00\x00\x00\xe0" * 4 + b"".join((region << 16).to_bytes(4, "little") for region in regions)
    (tmp_path / "l0_core0.bin").write_bytes(raw)
    (tmp_path / "l0_core0.json").write_text(json.dumps({
        "physical_core": 0, "run_epoch": 1, "capture_start_offset": 0,
        "captured_bytes": len(raw), "valid_bytes": len(raw),
        "events": [
            {"pipe": 0, "region": region, "block": 7 if index < 2 else 9,
             "cycle": 1000 + index * 10, "byte_offset": 16 + index * 4}
            for index, region in enumerate(regions)
        ],
    }))

    result = join(tmp_path)
    assert [event["task_id"] for event in result["events"]] == [42, 42, 43, 43]
    assert result["counts"] == {"ordered_marker_pair": 4}
    assert pa_ordered_assignments(result["events"][:-1], result["runtime_tasks"]) is None
