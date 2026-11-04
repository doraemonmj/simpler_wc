#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

source /usr/local/Ascend/cann/set_env.sh

python3 - <<'PY'
import os
import site
import subprocess
import sys
from urllib.parse import urlsplit

print(f"runner user: {os.geteuid()} ({os.environ.get('USER', 'unknown')})")
print(f"python: {sys.executable} {sys.version.split()[0]}")
print(f"user site: {site.getusersitepackages()} (enabled={site.ENABLE_USER_SITE})")
if not site.ENABLE_USER_SITE:
    raise SystemExit("Python user site is disabled")
for name in ("HTTP_PROXY", "HTTPS_PROXY", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
    value = os.environ.get(name, "")
    print(f"{name} host: {urlsplit(value).hostname if value else '(unset)'}")
for name in ("global.index-url", "global.extra-index-url"):
    result = subprocess.run([sys.executable, "-m", "pip", "config", "get", name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"pip {name} host: {urlsplit(result.stdout.strip()).hostname}")
PY

# Pin the CPU build so PyPI cannot resolve a CUDA wheel and its large NVIDIA dependencies.
timeout --signal=TERM --kill-after=10s 600s \
  python3 -m pip install --user --timeout 60 --retries 2 \
  'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu

# Match the onboard job's build packages, project dependency, and [test] extra.
timeout --signal=TERM --kill-after=10s 180s \
  python3 -m pip install --user --timeout 30 --retries 2 \
  'scikit-build-core>=0.10.0' 'nanobind>=2.0.0,<3' 'cmake>=3.20' 'ninja>=1.11' \
  'cloudpickle>=2.2' 'pytest>=6.0' 'pytest-timeout>=2.3' 'pytest-xdist>=3.0' \
  'PyYAML>=6.0' 'safetensors>=0.4' 'numpy>=1.24'

# GitHub Actions shells do not include the user bin directory on PATH. Expose
# the user-installed build tools through the existing system PATH so this host
# can run the same CI workflow after its runner token is changed.
user_bin="$(python3 -m site --user-base)/bin"
for tool in cmake ctest cpack ccmake ninja; do
  source_path="$user_bin/$tool"
  target_path="/usr/local/bin/$tool"
  if [ ! -x "$source_path" ]; then
    continue
  fi
  if [ ! -e "$target_path" ] && [ ! -L "$target_path" ]; then
    ln -s "$source_path" "$target_path"
  fi
done
hash -r
for tool in gcc g++ git cmake ninja; do
  command -v "$tool"
done
cmake --version
ninja --version

check_venv=$(mktemp -d "${RUNNER_TEMP:-/tmp}/simpler-user-site-check.XXXXXX")
trap 'rm -rf "$check_venv"' EXIT
python3 -m venv --system-site-packages "$check_venv"
"$check_venv/bin/python" - <<'PY'
import cmake
import cloudpickle
import nanobind
import ninja
import numpy
import pytest
import pytest_timeout
import safetensors
import scikit_build_core
import site
import torch
import xdist
import yaml

assert torch.__version__ == "2.7.1+cpu", torch.__version__
assert torch.version.cuda is None
assert hasattr(torch, "uint16") and hasattr(torch, "uint32")
assert torch.__file__.startswith(site.getusersitepackages()), torch.__file__
print(f"venv inherited CPU torch {torch.__version__} from {torch.__file__}")
print(f"user site: {site.getusersitepackages()}")
PY
"$check_venv/bin/python" -m pip install --dry-run --no-index --no-build-isolation \
  --config-settings=build.targets=build_package_a2a3 '.[test]'
