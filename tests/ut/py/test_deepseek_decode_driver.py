# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Daily standalone DeepSeek model driver contracts."""

import pytest
from simpler.task_interface import ChipCallable

from examples.a2a3.tensormap_and_ringbuffer.deepseek_v4_flash_decode import main as driver


def test_compile_only_persists_chip_callable(monkeypatch, tmp_path):
    chip = ChipCallable.build(signature=[], func_name="decode_test", binary=b"image", children=[])
    monkeypatch.setattr(driver, "compile_chip_callable_spec", lambda *args: chip)
    path = tmp_path / "model.bin"
    assert driver.main(["-p", "a2a3", "--compile-only", "--compiled-chip", str(path)]) == 0
    assert ChipCallable.from_bytes(path.read_bytes()).buffer_size() == chip.buffer_size()


def test_run_requires_requested_precompiled_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        driver.run([0, 1], "a2a3", compiled_chip=tmp_path / "missing.bin")


def test_prepared_fixture_cannot_be_skipped(tmp_path):
    with pytest.raises(ValueError, match="skip-golden"):
        driver.run([0, 1], "a2a3", fixture_dir=tmp_path, skip_golden=True)
