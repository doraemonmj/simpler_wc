# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare shared decode inputs and verify both runtime readbacks outside the device allocation."""

import argparse
from pathlib import Path

import torch

from simpler_setup.goldens import deepseek_v4_flash_decode as model


def prepare(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, tensor in model.param_tensors(seed=1234):
        torch.save(tensor, directory / f"{name}.pt")
        del tensor


def _cpu_logits(directory: Path, rank: int, hidden: torch.Tensor) -> torch.Tensor:
    indices = torch.load(directory / f"logit_row_indices_r{rank}.pt", weights_only=True)
    selected = hidden.index_select(0, indices.long()).float()
    tiles = []
    for shard in range(model.LM_HEAD_TP_SIZE):
        weight = torch.load(directory / f"lm_head_weight_r{shard}.pt", mmap=True, weights_only=True)
        # Expand one vocabulary tile to FP32 at a time to bound temporary host memory.
        tiles.extend(selected @ tile.float().T for tile in weight.split(2048))
    return torch.cat(tiles, dim=1)


def compare(directory: Path) -> None:
    hbg = torch.load(directory / "host_build_graph.pt", weights_only=True)
    tmr = torch.load(directory / "tensormap_and_ringbuffer.pt", weights_only=True)
    for rank in range(model.N_RANKS):
        for field in ("hidden_out", "logits", "sampled_ids"):
            key = f"{field}_r{rank}"
            spec = model.PARAM_SPEC_BY_NAME[field]
            tensor = hbg[key]
            if tuple(tensor.shape) != spec.shape or tensor.dtype != getattr(torch, spec.dtype.lower()):
                raise AssertionError(
                    f"{key}: expected {spec.shape} {spec.dtype}, got {tuple(tensor.shape)} {tensor.dtype}"
                )
            if not tensor.isfinite().all():
                raise AssertionError(f"{key}: non-finite output")
            # The two runtimes schedule the same kernels over identical inputs.
            # Compare every element, including all logits and the padded ID buffer.
            torch.testing.assert_close(tmr[key], tensor, rtol=0, atol=0, msg=lambda msg: f"{key}: {msg}")

        hidden = hbg[f"hidden_out_r{rank}"]
        if (hidden == 0).all(dim=1).any():
            raise AssertionError(f"rank {rank}: an entire hidden row is zero for the seeded decode input")
        reference = _cpu_logits(directory, rank, hidden)
        logits = hbg[f"logits_r{rank}"]
        # Independent FP32 accumulation of a 4096-element dot product reached
        # 9.1e-6 absolute error on A3. Allow a small absolute floor near zero.
        torch.testing.assert_close(
            logits, reference, atol=2e-5, rtol=1e-5, msg=lambda msg: f"rank {rank} LM head: {msg}"
        )
        actual_ids = hbg[f"sampled_ids_r{rank}"][:, 0].long()
        torch.testing.assert_close(
            actual_ids, reference.argmax(dim=1), msg=lambda msg: f"rank {rank} greedy IDs: {msg}"
        )
        print(
            f"rank {rank}: both runtimes agree, CPU LM head max_abs={(logits - reference).abs().max().item():.3g}, "
            f"{actual_ids.numel()} greedy tokens match",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "compare"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.directory)
    else:
        compare(args.directory)
