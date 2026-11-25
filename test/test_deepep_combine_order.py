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
"""Check consumer ordering without initializing torch or a distributed runtime."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "sglang_kunlun/hooks/layers/moe/token_dispatcher/deepep.py"
)


@pytest.mark.parametrize("recv_hook", [False, True])
@pytest.mark.parametrize("params_dtype", ["float16", "bfloat16", None])
def test_combine_result_is_not_read_before_communication_completes(
    recv_hook, params_dtype
):
    events = []
    completed = False

    class Tensor:
        def __init__(self, dtype, is_result=False):
            self.dtype = dtype
            self.is_result = is_result

        def contiguous(self):
            return self

        def to(self, dtype):
            if self.is_result:
                assert completed, "combine output read before completion wait"
                events.append("convert")
            return Tensor(dtype, self.is_result)

    def finish(*args):
        nonlocal completed
        completed = True
        events.append("wait")

    native_event = object()
    result = Tensor("bfloat16", is_result=True)
    def native_combine(**kwargs):
        assert kwargs["async_finish"] == (not recv_hook)
        assert kwargs["return_recv_hook"] == recv_hook
        return result, native_event, finish

    buffer = SimpleNamespace(low_latency_combine=native_combine)
    tree = ast.parse(SOURCE.read_text())
    owner_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_DeepEPDispatcherImplLowLatency"
    )
    methods = [
        node for node in owner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_combine_core", "combine_b")
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *methods], type_ignores=[])
    )
    namespace = dict(
        torch=SimpleNamespace(Tensor=Tensor, bfloat16="bfloat16"),
        nullcontext=nullcontext,
        _kunlun_wait_event=finish,
        _kunlun_release_event=lambda *args: events.append("release"),
        _deepep_combine_sync_enabled=lambda: False,
    )
    exec(compile(module, str(SOURCE), "exec"), namespace)
    owner = SimpleNamespace(
        _get_buffer=lambda: buffer,
        overlap_args=None,
        meta_overlap_args=None,
        handle=object(),
        packed_recv_count=None,
        return_recv_hook=recv_hook,
        params_dtype=params_dtype,
    )
    raw, hook = namespace["_combine_core"](
        owner, Tensor("bfloat16"), object(), object()
    )
    assert raw is result
    assert events == []
    final = namespace["combine_b"](owner, raw, hook)
    assert final.dtype == (params_dtype or "bfloat16")
    assert events[0] == "wait"
    assert events[-1] == "release"
    if params_dtype == "float16":
        assert events == ["wait", "convert", "release"]
