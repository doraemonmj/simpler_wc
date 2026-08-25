# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _packaging_script() -> str:
    workflow: dict[str, Any] = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "_packaging.yml").read_text())
    matches = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("name") == "Run packaging matrix"
    ]
    assert len(matches) == 1
    return matches[0]


def _compiler_setup() -> str:
    script = _packaging_script()
    return script.split(". .venv/bin/activate\n", 1)[1].split('if [[ "${{ runner.os }}" == "macOS" ]]', 1)[0]


def _run_setup(script: str, env: dict[str, str]) -> list[str]:
    command = f'{script}\nprintf "%s\\n%s\\n" "${{CC-unset}}" "${{CXX-unset}}"'
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", command],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return result.stdout.splitlines()


def test_packaging_modes_share_one_compiler_identity_without_overriding_callers(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for compiler in ("gcc", "g++"):
        path = bin_dir / compiler
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)

    env = os.environ.copy()
    env.pop("CC", None)
    env.pop("CXX", None)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    script = _compiler_setup()

    assert _run_setup(script, env) == [str(bin_dir / "gcc"), str(bin_dir / "g++")]

    env["CC"] = "/opt/custom/clang"
    env["CXX"] = "/opt/custom/clang++"
    assert _run_setup(script, env) == [env["CC"], env["CXX"]]

    env.pop("CXX")
    assert _run_setup(script, env) == [env["CC"], "unset"]
