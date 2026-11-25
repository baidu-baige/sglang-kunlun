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
"""Host metadata buffers must be 2MB-aligned before RDMA registration."""

import logging
from types import SimpleNamespace

import pytest
import torch

from sglang_kunlun.hooks.disaggregation.metadata_align import (
    _HOST_BUFFERS as HOST_BUFFERS,
    align_host_buffers,
    align_metadata_buffers,
)
from sglang_kunlun.hooks.mem_cache._align import ALIGNMENT_2M

LOGGER = "sglang_kunlun.hooks.disaggregation.metadata_align"


def owner(**overrides):
    """Minimal stand-in for MetadataBuffers: same names, CPU tensors."""
    size, hidden = 8, 128
    buffers = dict(
        output_ids=torch.ones((size, 16), dtype=torch.int32),
        cached_tokens=torch.arange(size * 16, dtype=torch.int32).reshape(size, 16),
        output_token_logprobs_val=torch.zeros((size, 16), dtype=torch.float32),
        output_token_logprobs_idx=torch.zeros((size, 16), dtype=torch.int32),
        output_top_logprobs_val=torch.zeros((size, 128), dtype=torch.float32),
        output_top_logprobs_idx=torch.zeros((size, 128), dtype=torch.int32),
        output_token_sampling_mask_len=None,
        output_token_sampling_mask_idx=None,
        output_token_sampling_logprobs=None,
        output_topk_p=torch.zeros((size, 16), dtype=torch.float32),
        output_topk_index=torch.zeros((size, 16), dtype=torch.int64),
        output_hidden_states=torch.zeros((size, hidden), dtype=torch.float16),
        output_dsa_topk_indices=None,
        bootstrap_room=torch.zeros((size, 8), dtype=torch.uint64),
        kv_checksum=None,
    )
    buffers.update(overrides)
    return SimpleNamespace(custom_mem_pool=None, **buffers)


def test_every_host_buffer_ends_up_2mb_aligned():
    buf_owner = owner()
    names = ("output_ids", "cached_tokens", "output_top_logprobs_val",
             "output_hidden_states", "bootstrap_room")
    before = {
        name: (getattr(buf_owner, name).shape, getattr(buf_owner, name).dtype,
               getattr(buf_owner, name).clone())
        for name in names
    }
    expected = sum(
        1
        for name in HOST_BUFFERS
        if isinstance(getattr(buf_owner, name, None), torch.Tensor)
        and getattr(buf_owner, name).data_ptr() % ALIGNMENT_2M != 0
    )

    moved = align_host_buffers(buf_owner)

    assert moved == expected and moved > 0
    for name, (shape, dtype, data) in before.items():
        buf = getattr(buf_owner, name)
        assert buf.data_ptr() % ALIGNMENT_2M == 0
        assert buf.shape == shape and buf.dtype == dtype
        assert buf.is_contiguous()
        torch.testing.assert_close(buf, data)


def test_item_lengths_are_preserved():
    buf_owner = owner()
    row_bytes = {name: getattr(buf_owner, name)[0].nbytes
                 for name in ("output_top_logprobs_val", "bootstrap_room", "output_ids")}

    align_host_buffers(buf_owner)

    assert {name: getattr(buf_owner, name)[0].nbytes for name in row_bytes} == row_bytes


def test_already_aligned_buffers_are_left_alone():
    buf_owner = owner()
    align_host_buffers(buf_owner)
    pointer = buf_owner.output_ids.data_ptr()

    assert align_host_buffers(buf_owner) == 0
    assert buf_owner.output_ids.data_ptr() == pointer


def test_custom_mem_pool_buffers_are_not_reallocated(caplog):
    buf_owner = owner()
    buf_owner.custom_mem_pool = object()
    pointer = buf_owner.output_ids.data_ptr()

    caplog.set_level(logging.INFO, logger=LOGGER)
    align_metadata_buffers(None, buf_owner)

    assert buf_owner.output_ids.data_ptr() == pointer
    assert "2MB-aligned" not in caplog.text


def test_hook_logs_how_many_moved(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    align_metadata_buffers(None, owner())

    assert "host metadata buffer(s) for RDMA registration" in caplog.text


def test_none_and_non_tensor_attributes_are_skipped():
    buf_owner = owner(output_topk_p=object())

    align_host_buffers(buf_owner)

    assert isinstance(buf_owner.output_topk_p, object)
    assert buf_owner.output_top_logprobs_val.data_ptr() % ALIGNMENT_2M == 0


def test_real_metadata_buffers_become_registrable():
    """End-to-end against upstream's class: every advertised ptr is aligned."""
    try:
        from sglang.srt.disaggregation.utils import MetadataBuffers
    except Exception:  # pragma: no cover - upstream not installed
        pytest.skip("Upstream SGLang not installed")

    buffers = MetadataBuffers(
        size=4,
        hidden_size=5120,
        hidden_states_dtype=torch.float16,
        max_sampling_mask_tokens=8192,
        output_dsa_topk_indices_dim=128,
    )
    _, lens_before, items_before = buffers.get_buf_infos()

    align_metadata_buffers(None, buffers)

    ptrs, lens_after, items_after = buffers.get_buf_infos()
    assert ptrs and all(ptr % ALIGNMENT_2M == 0 for ptr in ptrs)
    # Prefill indexes into these buffers with its own item lengths, so the
    # per-request stride and the total sizes must not move.
    assert lens_after == lens_before
    assert items_after == items_before
