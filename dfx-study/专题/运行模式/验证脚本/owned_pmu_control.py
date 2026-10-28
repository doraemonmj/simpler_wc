# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Control: run the same AIC callable and PMU config in owned Program mode."""

import argparse
import csv
import struct
from pathlib import Path

from probe import compile_callable
from simpler.task_interface import CallConfig, DataType, TaskArgs, Tensor, TensorArgType
from simpler.worker import Worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--args-then-pmu", action="store_true")
    args = parser.parse_args()
    target = compile_callable("a2a3", "tensormap_and_ringbuffer")
    owner = Worker(level=2, device_id=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    handle = owner.register(target)
    owner.init()
    state = None
    try:
        state = owner.malloc(64)
        owner.copy_to(state, bytearray(struct.pack("16i", *([7] * 16))))
        task = TaskArgs()
        task.add_tensor(Tensor(state, shapes=(16,), dtype=DataType.INT32), TensorArgType.INOUT)
        task.add_scalar(1)
        task.add_scalar(123)
        if args.args_then_pmu:
            args_prefix = args.output / "args"
            args_prefix.mkdir(parents=True, exist_ok=False)
            args_config = CallConfig()
            args_config.enable_dump_args = "full"
            args_config.output_prefix = str(args_prefix)
            owner.run(handle, task, args_config)
            assert (args_prefix / "args_dump/args_dump.json").is_file()
        cfg = CallConfig()
        cfg.enable_pmu = 2
        pmu_prefix = args.output / "pmu" if args.args_then_pmu else args.output
        cfg.output_prefix = str(pmu_prefix)
        pmu_prefix.mkdir(parents=True, exist_ok=False)
        owner.run(handle, task, cfg)
        result = bytearray(64)
        owner.copy_from(result, state)
        assert struct.unpack("16i", result) == (123,) + (7,) * 15
        with (pmu_prefix / "pmu.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(row["run_epoch"] != "0" for row in rows), rows
        print(f"owned_pmu PASS rows={len(rows)} epoch={rows[0]['run_epoch']}", flush=True)
    finally:
        if state is not None:
            owner.free(state)
        owner.close()


if __name__ == "__main__":
    main()
