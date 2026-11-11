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
import pytest
import torch

from sglang_kunlun.hooks.speculative.dspark_accept_torch import (
    cap_correct_len_kunlun,
    gather_row_bonus_kunlun,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Accelerator unavailable")
    return request.param


def test_accept_caps_and_bonus_include_inactive_rows(device):
    correct = torch.tensor([4, 2, 0], dtype=torch.int32, device=device)
    lengths = torch.tensor([2, 6, 0], dtype=torch.int64, device=device)
    table = torch.arange(18, dtype=torch.int64, device=device).reshape(3, 6)
    capped, trimmed = cap_correct_len_kunlun(
        None, correct_len=correct, verify_lens=lengths
    )
    bonus = gather_row_bonus_kunlun(table=table, idx=capped)
    assert capped.cpu().tolist() == [1, 2, -1]
    assert trimmed.cpu().tolist() == [3, 0, 1]
    assert bonus.cpu().tolist() == [1, 8, 17]
    assert capped.dtype == correct.dtype
    assert bonus.dtype == torch.int64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Accelerator unavailable")
def test_accept_graph_replay_reads_changed_caps_and_bonus():
    correct = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([6, 6], dtype=torch.int64, device="cuda")
    table = torch.arange(12, dtype=torch.int64, device="cuda").reshape(2, 6)

    def select():
        capped, trimmed = cap_correct_len_kunlun(
            None, correct_len=correct, verify_lens=lengths
        )
        return capped, trimmed, gather_row_bonus_kunlun(table=table, idx=capped)

    for _ in range(2):
        select()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = select()
    for length, offset, expected in (
        (2, 100, ([1, 1], [3, 1], [101, 107])),
        (0, 200, ([-1, -1], [5, 3], [305, 311])),
    ):
        lengths.fill_(length)
        table.add_(offset)
        graph.replay()
        torch.cuda.synchronize()
        assert tuple(value.cpu().tolist() for value in outputs) == expected
