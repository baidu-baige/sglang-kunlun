# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registration must report the failures ``batch_register`` silently swallows."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_kunlun.hooks.disaggregation.mooncake_register import (
    ALIGNMENT_2M,
    _reg_one,
    register_buffer_to_engine_kunlun,
)

LOGGER = "sglang_kunlun.hooks.disaggregation.mooncake_register"


def engine_with(register_memory=None, batch_register=None):
    """Engine exposing the single-buffer API used for status reporting."""
    return SimpleNamespace(
        engine=SimpleNamespace(register_memory=register_memory),
        batch_register=batch_register or Mock(return_value=0),
    )


def manager(engine, *, kv=1, aux=0):
    ptrs = [0x10000000 + i * ALIGNMENT_2M for i in range(kv)]
    lens = [1024] * kv
    aux_ptrs = [0x20000000 + i * ALIGNMENT_2M for i in range(aux)]
    return SimpleNamespace(
        engine=engine,
        kv_args=SimpleNamespace(
            kv_data_ptrs=ptrs,
            kv_data_lens=lens,
            aux_data_ptrs=aux_ptrs,
            aux_data_lens=[64 * 1024] * aux,
            state_data_ptrs=[],
            state_data_lens=[],
        ),
    )


def test_single_buffer_api_surfaces_the_return_code():
    assert _reg_one(engine_with(register_memory=Mock(return_value=0)), 0x1000, 100) == (
        True,
        "rc=0",
    )
    ok, detail = _reg_one(engine_with(register_memory=Mock(return_value=12)), 0x1000, 100)
    assert not ok and detail == "rc=12"


def test_length_is_rounded_to_2mb():
    single = Mock(return_value=0)
    _reg_one(engine_with(register_memory=single), 0x1000, ALIGNMENT_2M + 1)
    single.assert_called_once_with(0x1000, 2 * ALIGNMENT_2M)


def test_exception_is_a_failure():
    ok, detail = _reg_one(
        engine_with(register_memory=Mock(side_effect=RuntimeError("boom"))), 0x1000, 100
    )
    assert not ok and "boom" in detail


def test_batch_fallback_keeps_old_semantics():
    ok, _ = _reg_one(engine_with(batch_register=Mock(return_value=0)), 0x1000, 100)
    assert ok
    ok, _ = _reg_one(engine_with(batch_register=Mock(return_value=None)), 0x1000, 100)
    assert ok
    ok, _ = _reg_one(engine_with(batch_register=Mock(return_value=1)), 0x1000, 100)
    assert not ok


@pytest.mark.parametrize("failing_index", [0, 3])
def test_hook_counts_and_names_each_failed_buffer(caplog, failing_index):
    total = 4

    def register_memory(ptr, length):
        # One aux buffer hits ENOMEM; batch_register would have reported 0.
        return 12 if ptr == 0x20000000 + failing_index * ALIGNMENT_2M else 0

    caplog.set_level(logging.INFO, logger=LOGGER)
    register_buffer_to_engine_kunlun(
        manager(engine_with(register_memory=register_memory), kv=2, aux=total)
    )

    assert "per-buffer registration ok=5 fail=1" in caplog.text
    assert f"aux[{failing_index}] ptr=0x{0x20000000 + failing_index * ALIGNMENT_2M:x}" in (
        caplog.text
    )
    assert "missing from RDMA metadata" in caplog.text


def test_hook_stays_quiet_when_everything_registers(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    register_buffer_to_engine_kunlun(
        manager(engine_with(register_memory=Mock(return_value=0)), kv=2, aux=3)
    )

    assert "per-buffer registration ok=5 fail=0" in caplog.text
    assert "missing from RDMA metadata" not in caplog.text
