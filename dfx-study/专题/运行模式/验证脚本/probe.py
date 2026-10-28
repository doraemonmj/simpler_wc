# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise borrowed Kernel execution through the documented public Worker API."""

import argparse
import csv
import ctypes
import json
from pathlib import Path

from simpler.buffer import Buffer
from simpler.task_interface import ArgDirection, CallConfig, DataType, TaskArgs, Tensor, TensorArgType
from simpler.worker import Worker

from simpler_setup.runtime_builder import RuntimeBuilder
from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key

ROOT = Path(__file__).resolve().parents[4]


def compile_callable(arch, runtime):
    spec = {
        "orchestration": {
            "source": str(ROOT / "tests/st/common/kernel_eager_state/state_orchestration.cpp"),
            "function_name": "state_from_scalars",
            "signature": [ArgDirection.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(
                    ROOT
                    / "tests/st"
                    / arch
                    / "tensormap_and_ringbuffer/predicated_dispatch/kernels/aic/kernel_write_gate.cpp"
                ),
                "core_type": "aic",
                "signature": [ArgDirection.INOUT],
            }
        ],
    }
    key = l3_compile_cache_key(__name__, "KernelDfxUserProbe", "state_from_scalars", arch, runtime)
    return compile_chip_callable_spec(spec, arch, runtime, key)


def bind_rts(arch, runtime):
    lib = ctypes.CDLL(str(RuntimeBuilder(arch).get_binaries(runtime).host_path))
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.rtCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


def check_artifacts(name, prefix, expected_epoch):
    if name.startswith("chip_dep"):
        chip = json.loads((prefix / "chip_swimlane_records.json").read_text())
        deps = json.loads((prefix / "deps.json").read_text())
        assert chip["aicore_tasks"], chip
        assert deps["tasks"], deps
        assert {row[-1] for row in chip["aicore_tasks"]} == {expected_epoch}, chip["aicore_tasks"]
        return {"aicore_tasks": len(chip["aicore_tasks"]), "dep_tasks": len(deps["tasks"])}
    if name == "args":
        manifest = json.loads((prefix / "args_dump/args_dump.json").read_text())
        assert manifest["args"] and manifest["dump_args_level"] == 3, manifest
        assert {record["run_epoch"] for record in manifest["args"]} == {expected_epoch}, manifest["args"]
        payload = prefix / "args_dump" / manifest["bin_file"]
        assert payload.stat().st_size > 0
        return {"args": len(manifest["args"]), "payload_bytes": payload.stat().st_size}
    if name == "pmu":
        with (prefix / "pmu.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all("run_epoch" in row for row in rows), rows
        assert {int(row["run_epoch"]) for row in rows} == {expected_epoch}, rows
        return {"pmu_rows": len(rows), "epochs": sorted({row["run_epoch"] for row in rows})}
    if name == "scope":
        path = prefix / "scope_stats/scope_stats.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows, rows
        return {"scope_rows": len(rows)}
    assert name == "off"
    assert not any(prefix.rglob("chip_swimlane_records.json"))
    assert not any(prefix.rglob("deps.json"))
    assert not any(prefix.rglob("args_dump.json"))
    assert not any(prefix.rglob("pmu.csv"))
    assert not any(prefix.rglob("scope_stats.jsonl"))
    return {"diagnostic_files": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="a2a3")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-pmu", action="store_true")
    parser.add_argument("--only-pmu", action="store_true")
    parser.add_argument("--cases", help="comma-separated case names, replacing the default sequence")
    args = parser.parse_args()
    target = compile_callable(args.arch, args.runtime)
    lib = bind_rts(args.arch, args.runtime)
    assert lib.rtSetDevice(args.device) == 0
    caller_context = ctypes.c_void_p()
    assert lib.rtCtxGetCurrent(ctypes.byref(caller_context)) == 0
    assert caller_context.value

    owner = Worker(level=2, device_id=args.device, platform=args.arch, runtime=args.runtime)
    handle = owner.register(target)
    owner.init(borrowed_context_generation=20261022)
    stream, allocation, host = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    backing = None
    nbytes = 16 * ctypes.sizeof(ctypes.c_int32)
    cases = ["off", "chip_dep_1", "args", "pmu"]
    if args.skip_pmu:
        cases.remove("pmu")
    if args.runtime == "tensormap_and_ringbuffer":
        cases.append("scope")
    cases.extend(["off_again", "chip_dep_2"])
    if args.only_pmu:
        cases = ["pmu"]
    if args.cases:
        cases = args.cases.split(",")
    try:
        assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        assert lib.aclrtMalloc(ctypes.byref(allocation), nbytes, 0) == 0
        assert lib.aclrtMallocHost(ctypes.byref(host), nbytes) == 0
        assert allocation.value is not None
        assert host.value is not None
        backing = Buffer.wrap(address=allocation.value, nbytes=nbytes, location=owner.device_location)
        view = Tensor(backing, shapes=(16,), dtype=DataType.INT32)
        values = (ctypes.c_int32 * 16).from_address(host.value)
        for number, name in enumerate(cases, 1):
            prefix = args.output / name
            prefix.mkdir(parents=True, exist_ok=False)
            cfg = CallConfig()
            cfg.output_prefix = str(prefix)
            if name.startswith("chip_dep"):
                cfg.enable_chip_swimlane = 1
                cfg.enable_dep_gen = True
            elif name == "args":
                cfg.enable_dump_args = "full"
            elif name == "pmu":
                cfg.enable_pmu = 2
            elif name == "scope":
                cfg.enable_scope_stats = True
            values[:] = [7] * 16
            assert lib.aclrtMemcpyAsync(allocation, nbytes, host, nbytes, 1, stream) == 0
            task = TaskArgs()
            task.add_tensor(view, TensorArgType.INOUT)
            task.add_scalar(1)
            task.add_scalar(100 + number)
            run = owner.submit(handle, task, config=cfg, caller_stream=stream.value)
            assert lib.aclrtMemcpyAsync(host, nbytes, allocation, nbytes, 2, stream) == 0
            sync_rc = lib.aclrtSynchronizeStreamWithTimeout(stream, 60000)
            assert sync_rc == 0, f"{name}: caller stream synchronize failed: {sync_rc}"
            run.wait()
            assert list(values) == [100 + number] + [7] * 15, (name, list(values))
            stats = check_artifacts(name if not name.startswith("off") else "off", prefix, number)
            print(json.dumps({"case": name, "result": list(values)[:2], **stats}, sort_keys=True), flush=True)
        backing.close()
        backing = None
        owner.close()
        current = ctypes.c_void_p()
        assert lib.rtCtxGetCurrent(ctypes.byref(current)) == 0
        assert current.value == caller_context.value
        assert lib.aclrtMemcpyAsync(host, nbytes, allocation, nbytes, 2, stream) == 0
        assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        assert list(values) == [100 + len(cases)] + [7] * 15
        print(
            json.dumps(
                {"case": "after_close", "caller_context_retained": True, "caller_stream_and_allocation_usable": True}
            ),
            flush=True,
        )
    finally:
        if backing is not None:
            backing.close()
        owner.close()
        if stream.value:
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        if allocation.value:
            assert lib.aclrtFree(allocation) == 0
        if host.value:
            assert lib.aclrtFreeHost(host) == 0
        if stream.value:
            assert lib.aclrtDestroyStream(stream) == 0


if __name__ == "__main__":
    main()
