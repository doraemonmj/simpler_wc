# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source-level lifetime contracts for the A5 HCCL communication backend."""

from pathlib import Path

_A5_COMM_HCCL = Path(__file__).resolve().parents[3] / "src" / "a5" / "platform" / "onboard" / "host" / "comm_hccl.cpp"


def _function_body(source: str, declaration: str) -> str:
    declaration_start = source.index(declaration)
    body_start = source.index("{", declaration_start)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start : index + 1]
    raise AssertionError(f"unterminated function body for {declaration!r}")


def test_a5_comm_destroy_releases_hccl_before_urma_workspace():
    source = _A5_COMM_HCCL.read_text(encoding="utf-8")
    body = _function_body(source, 'extern "C" int comm_destroy')

    destroy_call = body.index("hccl_comm_destroy(h->hccl_comm)")
    urma_reset = body.index("reset_base_urma_workspace(h)")

    assert destroy_call < urma_reset, "URMA Finalize must run after HcclCommDestroy"


def test_a5_failed_urma_init_keeps_manager_until_comm_destroy():
    source = _A5_COMM_HCCL.read_text(encoding="utf-8")
    body = _function_body(source, "static bool init_urma_workspace")

    retain_manager = "workspace = std::make_unique<pto::comm::urma::UrmaWorkspaceManager>()"
    init_call = "const bool initialized = workspace->Init"
    check_failure = "if (!initialized)"
    for statement in (retain_manager, init_call, check_failure):
        assert statement in body

    assert body.index(retain_manager) < body.index(init_call) < body.index(check_failure)
