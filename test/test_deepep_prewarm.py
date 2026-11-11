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
"""Regress runner ownership without importing the device/distributed runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "sglang_kunlun/hooks/layers/attn_tp_gather.py"
)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
def test_prewarm_only_uses_runner_owned_buffer_and_geometry(wrapped, is_draft):
    mode = Mock()
    mode.enable_low_latency.return_value = True
    group = object()

    def implementation(experts, topk):
        buffer = SimpleNamespace(low_latency_mode=True, low_latency_dispatch=Mock())
        return SimpleNamespace(
            group=group,
            hidden_size=5120,
            params_bytes=2,
            deepep_mode=mode,
            num_max_dispatch_tokens_per_rank=384,
            num_experts=experts,
            router_topk=topk,
            _get_buffer=Mock(return_value=buffer),
        )

    target = implementation(384, 6)
    draft = implementation(128, 3)
    selected, foreign = (draft, target) if is_draft else (target, draft)
    dispatcher = SimpleNamespace(_low_latency_dispatcher=selected)
    if wrapped:
        dispatcher = SimpleNamespace(_inners=[dispatcher, dispatcher])
    # Multiple layers sharing the same native buffer need only one prewarm.
    modules = [
        SimpleNamespace(),
        SimpleNamespace(dispatcher=dispatcher),
        SimpleNamespace(dispatcher=dispatcher),
    ]
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(model=SimpleNamespace(modules=lambda: modules))
    )
    zeros = Mock(side_effect=lambda shape, **kw: (shape, kw))
    synchronize = Mock()
    namespace = dict(
        torch=SimpleNamespace(
            zeros=zeros,
            bfloat16="bf16",
            int64="i64",
            cuda=SimpleNamespace(synchronize=synchronize),
        ),
        logger=SimpleNamespace(info=Mock()),
    )
    functions = [
        node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in (
            "_model_low_latency_dispatchers", "_prewarm_deepep_low_latency"
        )
    ]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
         namespace)
    namespace["_prewarm_deepep_low_latency"](runner, 2)
    selected._get_buffer.assert_called_once_with()
    foreign._get_buffer.assert_not_called()
    buffer = selected._get_buffer.return_value
    args, kwargs = buffer.low_latency_dispatch.call_args
    assert args[0][0] == (1, 5120)
    assert args[1][0] == (1, selected.router_topk)
    assert args[2:] == (384, selected.num_experts)
    assert kwargs == dict(use_fp8=True, async_finish=False, return_recv_hook=False)
    synchronize.assert_called_once_with()
