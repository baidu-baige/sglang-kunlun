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
from types import SimpleNamespace

import pytest
import torch

from sglang_kunlun.kernels.dspark_greedy_torch import sharded_greedy_step


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Accelerator unavailable")
    return request.param


def test_padding_ties_and_nan(device):
    # Padded vocab logits must never compete with real entries. NaNs and ties
    # both select the first real ID, matching the upstream reduction contract.
    bias = torch.zeros((3, 4), device=device, dtype=torch.bfloat16)
    base = torch.tensor(
        [[2, 9, 9, 1, 999], [0, float("nan"), 8, float("nan"), 999],
         [-float("inf")] * 4 + [999]],
        device=device,
    )
    result = sharded_greedy_step(
        bias, base, group=SimpleNamespace(world_size=1), vocab_start=100
    )
    assert result.cpu().tolist() == [101, 101, 100]
    assert result.dtype == torch.int64


def test_distributed_ties_nan_and_empty_shard(device):
    class PairGather:
        world_size = 2

        def __init__(self):
            self.responses = iter((
                torch.tensor([[9, float("nan"), 7]], device=device),
                torch.tensor([[4, 3, 8]], dtype=torch.int64, device=device),
            ))

        def all_gather(self, local, dim):
            return torch.cat((local, next(self.responses)), dim=dim)

    bias = torch.zeros((3, 2), device=device)
    base = torch.tensor([[9, 0], [float("nan"), 0], [1, 2]], device=device)
    result = sharded_greedy_step(
        bias, base, group=PairGather(), vocab_start=100
    )
    assert result.cpu().tolist() == [4, 3, 8]

    empty = torch.empty((3, 0), device=device)
    result = sharded_greedy_step(
        empty, empty, group=PairGather(), vocab_start=100
    )
    assert result.cpu().tolist() == [4, 3, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Accelerator unavailable")
def test_graph_replay_reads_changed_logits():
    # A selector that runs eagerly during capture passes a static-input test.
    # Change the winner after capture to detect missing recorded operations.
    bias = torch.zeros((2, 129280), dtype=torch.bfloat16, device="cuda")
    base = torch.zeros_like(bias, dtype=torch.float32)
    base[:, 12345] = 10
    group = SimpleNamespace(world_size=1)
    for _ in range(2):
        sharded_greedy_step(bias, base, group=group, vocab_start=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = sharded_greedy_step(bias, base, group=group, vocab_start=0)
    for winner in (12346, 42, 129279):
        base.zero_()
        base[:, winner] = 10
        graph.replay()
        torch.cuda.synchronize()
        assert result.cpu().tolist() == [winner, winner]
