#!/usr/bin/env bash

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Check if A5-related code changed relative to a base ref.
# Exit 0 = A5 changed (should run A5 CI), Exit 1 = no A5 changes (skip).
#
# Usage:
#   tools/check_a5_changes.sh                  # compare against origin/main
#   tools/check_a5_changes.sh origin/main      # explicit base
#   tools/check_a5_changes.sh <base_sha>       # GitHub Actions base SHA

set -euo pipefail

BASE="${1:-origin/main}"

git diff --name-only "$BASE"...HEAD \
    | grep -qE '^(src/a5/|examples/a5/|tests/(st|device_tests)/a5/)'
