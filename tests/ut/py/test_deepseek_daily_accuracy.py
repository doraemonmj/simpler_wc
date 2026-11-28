# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the daily decode accuracy checks using a two-rank CPU fixture."""

import pytest
import torch

from examples.a2a3.tensormap_and_ringbuffer.deepseek_v4_flash_decode import accuracy
from simpler_setup.goldens import deepseek_v4_flash_decode as model


@pytest.fixture
def fixture_dir(tmp_path, monkeypatch):
    shapes = {"hidden_out": (2, 2), "logits": (2, 4), "sampled_ids": (2, 3)}
    for name, shape in shapes.items():
        spec = model.PARAM_SPEC_BY_NAME[name]
        monkeypatch.setitem(model.PARAM_SPEC_BY_NAME, name, spec._replace(shape=shape))
    weights = torch.tensor([[1, 0], [0, 1], [0, -2], [-2, 0]], dtype=torch.bfloat16)
    outputs = {}
    for rank, shard in enumerate(weights.chunk(2)):
        # Every rank projects its own rows against both vocabulary shards.
        hidden = torch.tensor([[1, 2], [-3, 0]], dtype=torch.bfloat16) * (rank + 1)
        rows = torch.tensor([1, 0], dtype=torch.int32)
        logits = hidden[rows.long()].float() @ weights.float().T
        ids = torch.zeros(shapes["sampled_ids"], dtype=torch.int32)
        ids[:, 0] = logits.argmax(dim=1)
        torch.save(rows, tmp_path / f"logit_row_indices_r{rank}.pt")
        torch.save(shard, tmp_path / f"lm_head_weight_r{rank}.pt")
        outputs.update({f"hidden_out_r{rank}": hidden, f"logits_r{rank}": logits, f"sampled_ids_r{rank}": ids})
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer"):
        torch.save(outputs, tmp_path / f"{runtime}.pt")
    return tmp_path


def test_valid_outputs(fixture_dir):
    accuracy.compare(fixture_dir)


@pytest.mark.parametrize(
    "error", ["common_logits", "common_token", "different_hidden", "zero_row", "nan", "shape", "dtype"]
)
def test_rejects_bad_outputs(fixture_dir, error):
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer"):
        path = fixture_dir / f"{runtime}.pt"
        outputs = torch.load(path, weights_only=True)
        if error == "common_logits":
            outputs["logits_r1"][0, 0] += 0.1
        elif error == "common_token":
            outputs["sampled_ids_r1"][0, 0] = 0
        elif error == "different_hidden" and runtime == "tensormap_and_ringbuffer":
            outputs["hidden_out_r1"][0, 0] += 1
        elif error == "zero_row":
            # A missed write can otherwise satisfy the projection and greedy oracle together.
            outputs["hidden_out_r1"][1].zero_()
            outputs["logits_r1"][0].zero_()
            outputs["sampled_ids_r1"][0].zero_()
        elif error == "nan":
            outputs["hidden_out_r0"][0, 0] = float("nan")
        elif error == "shape":
            outputs["logits_r1"] = outputs["logits_r1"][:1]
        elif error == "dtype":
            outputs["sampled_ids_r0"] = outputs["sampled_ids_r0"].to(torch.int64)
        torch.save(outputs, path)
    with pytest.raises(AssertionError):
        accuracy.compare(fixture_dir)
