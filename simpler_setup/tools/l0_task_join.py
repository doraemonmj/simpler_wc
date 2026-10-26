#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Join onboard L0 stamps to the real runtime task token and emit a trace.

Usage: python -m simpler_setup.tools.l0_task_join OUTPUT_DIRECTORY
The directory must contain chip_swimlane_records.json and l0_core*.json from
the same run. A unique task on the logical core can be linked without a common
clock; this evidence is labeled separately from timestamp interval matching.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

_PA_END_REGION = {100: 101, 102: 103, 104: 105, 106: 107}


def pa_ordered_assignments(events, core_tasks):
    """Pair PA entry/exit stamps, then map complete spans by per-core order.

    Return None unless every runtime task and every L0 span has a one-to-one
    position. This is an order join, not a device-clock alignment claim.
    """
    if not events or not core_tasks:
        return None
    groups = []
    for index, event in enumerate(events):
        region = int(event["region"])
        block = int(event["block"])
        if not groups or groups[-1][0] != region:
            groups.append((region, block, [index]))
        elif groups[-1][1] == block:
            groups[-1][2].append(index)
        else:
            return None
    ordered = sorted(core_tasks, key=lambda task: task["start_cycles"])
    if len(groups) != 2 * len(ordered):
        return None
    if any(a["end_cycles"] > b["start_cycles"] for a, b in zip(ordered, ordered[1:])):
        return None
    assigned = {}
    for task, begin, end in zip(ordered, groups[::2], groups[1::2]):
        if _PA_END_REGION.get(begin[0]) != end[0] or begin[1] != end[1]:
            return None
        if max(int(events[i]["cycle"]) for i in begin[2]) > min(int(events[i]["cycle"]) for i in end[2]):
            return None
        for index in begin[2] + end[2]:
            assigned[index] = task
    return assigned


def logical_core(physical):
    """A5 36-cluster MIX layout: 36 AIC followed by 72 interleaved AIV."""
    if not 0 <= physical < 108:
        raise ValueError(f"invalid A5 physical subcore {physical}")
    if physical < 18:
        return physical, "aic", physical
    if physical < 54:
        cluster, sub = divmod(physical - 18, 2)
        return 36 + 2 * cluster + sub, "aiv", cluster
    if physical < 72:
        return physical - 36, "aic", physical - 36
    cluster, sub = divmod(physical - 72, 2)
    cluster += 18
    return 36 + 2 * cluster + sub, "aiv", cluster


def join(directory):
    swimlane = json.loads((directory / "chip_swimlane_records.json").read_text())
    survey = json.loads((directory / "l0_survey.json").read_text())
    epoch = int(survey["run_epoch"])
    freq = int(swimlane["metadata"]["clock_freq_hz"])
    if freq <= 0:
        raise ValueError("missing device clock frequency")
    core_types = swimlane["metadata"]["core_types"]
    tasks = defaultdict(list)
    for row in swimlane["aicore_tasks"]:
        if len(row) != 7:
            raise ValueError("expected seven-column aicore_tasks rows")
        core, token, reg_id, start, end, _, task_epoch = map(int, row)
        if task_epoch == epoch:
            tasks[core].append({"task_id": token, "reg_task_id": reg_id,
                                "start_cycles": start, "end_cycles": end})
    if not tasks:
        raise ValueError(f"no runtime tasks for L0 run_epoch={epoch}")
    runtime_tasks = [dict(logical_core=core, **task)
                     for core, group in sorted(tasks.items()) for task in group]
    origin = min(t["start_cycles"] for group in tasks.values() for t in group)
    us = lambda cycle: (cycle - origin) * 1_000_000 / freq
    trace = [
        {"name": "process_name", "ph": "M", "pid": 0,
         "args": {"name": "runtime task clock"}},
        {"name": "process_name", "ph": "M", "pid": 1,
         "args": {"name": "L0 BIU clock (independent origin)"}},
    ]
    for core, group in tasks.items():
        for task in group:
            trace.append({"name": f"task {task['task_id']}", "ph": "X", "pid": 0,
                          "tid": core, "ts": us(task["start_cycles"]),
                          "dur": (task["end_cycles"] - task["start_cycles"]) * 1_000_000 / freq,
                          "args": {"task_id": task["task_id"], "reg_task_id": task["reg_task_id"],
                                   "logical_core": core}})
    joined = []
    counts = Counter()
    core_files = sorted(directory.glob("l0_core*.json"), key=lambda p: int(p.stem[7:]))
    if not core_files:
        raise ValueError("no L0 core JSON files")
    all_cycles = [int(event["cycle"]) for path in core_files
                  for event in json.loads(path.read_text())["events"]]
    if not all_cycles:
        raise ValueError("no decoded L0 events")
    first_l0_cycle = min(all_cycles)
    for path in core_files:
        data = json.loads(path.read_text())
        physical = int(data["physical_core"])
        if int(data["run_epoch"]) != epoch:
            raise ValueError(f"run_epoch mismatch in {path}")
        raw_path = path.with_suffix(".bin")
        raw = raw_path.read_bytes()
        start_offset = int(data["capture_start_offset"])
        if len(raw) != int(data["valid_bytes"]) or len(raw) - start_offset != int(data["captured_bytes"]):
            raise ValueError(f"raw capture window mismatch in {raw_path}")
        if start_offset < 0 or start_offset + 16 > len(raw) or any(
            raw[offset + 3] >> 4 != 14 for offset in range(start_offset, start_offset + 16, 4)
        ):
            raise ValueError(f"missing L0 START_STAMP in {raw_path}")
        logical, kind, cluster = logical_core(physical)
        if logical >= len(core_types) or core_types[logical] != kind:
            raise ValueError(f"core type mismatch: physical={physical} logical={logical}")
        pa_markers = any(int(event["region"]) in (*_PA_END_REGION, *_PA_END_REGION.values())
                         for event in data["events"])
        pa_assignments = pa_ordered_assignments(data["events"], tasks[logical]) if pa_markers else None
        for event_index, event in enumerate(data["events"]):
            cycle = int(event["cycle"])
            offset = int(event["byte_offset"])
            if offset < start_offset + 16 or offset + 4 > len(raw):
                raise ValueError(f"L0 event outside raw capture window in {raw_path}")
            word = int.from_bytes(raw[offset:offset + 4], "little")
            if word >> 28 != int(event["pipe"]) or (word >> 16) & 0xFFF != int(event["region"]):
                raise ValueError(f"L0 event differs from raw bytes in {raw_path} at {offset}")
            if pa_markers:
                candidate = pa_assignments.get(event_index) if pa_assignments else None
                candidates = [candidate] if candidate else []
                status = "ordered_marker_pair" if candidate else "unmatched_pa_markers"
            else:
                candidates = [t for t in tasks[logical]
                              if t["start_cycles"] <= cycle <= t["end_cycles"]]
                block_matches = int(event["block"]) == cluster
                if len(candidates) == 1 and block_matches:
                    status = "matched_interval"
                elif len(candidates) == 0 and len(tasks[logical]) == 1 and block_matches:
                    candidates = tasks[logical]
                    status = "unique_core_task"
                elif not block_matches:
                    status = "block_mismatch"
                else:
                    status = "ambiguous" if len(candidates) > 1 else "no_task_interval"
            counts[status] += 1
            record = {"run_epoch": epoch, "physical_core": physical,
                      "logical_core": logical, "core_type": kind, "cluster": cluster,
                      **event, "join_status": status,
                      "task_id": candidates[0]["task_id"] if status in
                      ("matched_interval", "unique_core_task", "ordered_marker_pair") else None,
                      "reg_task_id": candidates[0]["reg_task_id"] if status in
                      ("matched_interval", "unique_core_task", "ordered_marker_pair") else None}
            joined.append(record)
            if record["task_id"] is not None:
                trace.append({"name": f"L0 pipe {event['pipe']} region {event['region']}",
                              "ph": "i", "s": "t", "pid": 1, "tid": logical,
                              "ts": (cycle - first_l0_cycle) * 1_000_000 / freq,
                              "args": record})
    output = {"schema": "l0_runtime_task_join_v1", "source": "live",
              "run_epoch": epoch, "channel": survey["channel"],
              "clock_freq_hz": freq, "counts": dict(counts),
              "clock_note": "Raw runtime and L0 timestamps are not directly aligned in this capture; trace process tracks use independent origins and cannot be compared horizontally.",
              "task_count": sum(map(len, tasks.values())), "l0_core_files": len(core_files),
              "runtime_tasks": runtime_tasks, "events": joined}
    (directory / "l0_task_join.json").write_text(json.dumps(output, indent=2) + "\n")
    (directory / "l0_task_trace.json").write_text(json.dumps({"traceEvents": trace}) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = join(args.directory)
    print(json.dumps({key: result[key] for key in
                      ("run_epoch", "channel", "task_count", "l0_core_files", "counts")}, indent=2))
    if sum(value for status, value in result["counts"].items()
           if status not in ("matched_interval", "unique_core_task", "ordered_marker_pair")):
        raise SystemExit("some L0 stamps could not be joined; see l0_task_join.json")


if __name__ == "__main__":
    main()
