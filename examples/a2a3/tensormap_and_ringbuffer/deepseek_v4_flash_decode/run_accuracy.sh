#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Both runtimes consume the same prepared fixture on the same allocated device pair.
set -euo pipefail
workdir=$1
devices=$2
bash .claude/skills/onboard-arch-precheck/check.sh a2a3
for runtime in host_build_graph tensormap_and_ringbuffer; do
    .venv/bin/python "examples/a2a3/$runtime/deepseek_v4_flash_decode/main.py" \
        --platform a2a3 --device "$devices" --compiled-chip "$workdir/$runtime.bin" \
        --fixture-dir "$workdir" --output-file "$workdir/$runtime.pt"
done
