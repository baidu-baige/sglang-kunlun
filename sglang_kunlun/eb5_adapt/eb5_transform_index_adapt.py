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
"""``dsa/transform_index`` 的纯 torch 实现（昆仑 P800 无 Triton）。

替换的三个符号（fork 里都是 ``kernel[grid](...)`` 形式，Triton 被 stub 后不可用）：

* ``prepare_trtllm_nope_sparse_metadata`` —— per-query 有效 top-k 长度；
* ``transform_index_page_table_decode_fast`` —— decode 行对齐 gather；
* ``transform_index_page_table_prefill_fast`` —— prefill 按 request 行 gather。

语义与原 kernel 逐元素一致（填充值 ``-1``、dtype ``int32``）。**不要**在这里引入
D2H（``.item()`` / ``bool(tensor.any())``）：昆仑设备半坏时会永久挂住。
"""

from __future__ import annotations

import importlib
from itertools import accumulate
from typing import List, Optional

import torch


def _import_sibling(name: str):
    """导入同目录模块，兼容包内（相对）与 ``EB5_KUNLUN_ADAPT_DIR`` 平铺两种方式。"""

    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ModuleNotFoundError as exc:
            if exc.name not in (f"{__package__}.{name}", name):
                raise
    return importlib.import_module(name)


def prepare_trtllm_nope_sparse_metadata(page_table: torch.Tensor) -> torch.Tensor:
    """逐查询统计有效 top-k 长度；空行（整行 -1）把第 0 列改成 dummy token 0。

    原 Triton kernel 的 torch 复刻。注意这里是**就地**修改 ``page_table``，
    与原实现一致；判空用 ``torch.where`` 而不是 ``bool(...any())``，避免 D2H。
    """

    assert page_table.ndim == 2
    assert page_table.dtype == torch.int32
    assert page_table.is_contiguous()
    topk_lens = (page_table >= 0).sum(dim=1, dtype=torch.int32)
    if page_table.shape[1] > 0:
        is_empty = topk_lens == 0
        first = page_table[:, 0]
        page_table[:, 0] = torch.where(is_empty, torch.zeros_like(first), first)
        topk_lens = torch.where(is_empty, torch.ones_like(topk_lens), topk_lens)
    return topk_lens


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """decode：行对齐 gather，负下标/越界写 -1。"""

    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    qo_len = topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)

    src = page_table[:qo_len].to(torch.int32)
    sel = topk_indices.to(torch.int64)
    n_cols = src.shape[1]
    valid = (sel >= 0) & (sel < n_cols)
    # clamp 用 out-of-place：``sel`` 可能和调用方的 topk_indices 是同一对象。
    gathered = torch.gather(src, 1, sel.clamp(min=0, max=max(n_cols - 1, 0)))
    torch.where(
        valid,
        gathered,
        torch.full((), -1, dtype=torch.int32, device=result.device),
        out=result,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    output_num_tokens: Optional[int] = None,
    page_table_is_expanded: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """prefill：token 行 → 请求行由 ``cu_seqlens_q`` 定位，负值/越界/padding 写 -1。"""

    from sglang.kernels.ops.attention.dsa.transform_index import (
        _allocate_prefill_result,
    )

    assert page_size == 1
    assert topk_indices.shape[1] == 2048
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(topk_indices, real_num_tokens, output_num_tokens)
    if real_num_tokens == 0:
        return result

    device = topk_indices.device
    rows = result.shape[0]
    if rows == 0:
        return result

    if cu_seqlens_q is None:
        cu = torch.tensor(
            [0, *accumulate(extend_lens_cpu)], dtype=torch.int64, device=device
        )
    else:
        cu = cu_seqlens_q.to(device=device, dtype=torch.int64)
    row_idx = torch.arange(rows, device=device, dtype=torch.int64)
    n_req = cu.numel() - 1
    req = (torch.searchsorted(cu, row_idx, right=True) - 1).clamp(
        min=0, max=max(n_req - 1, 0)
    )
    src_row = row_idx if page_table_is_expanded else req
    # padding 行可能超出 page_table 行数（expanded 时表里只有 real 行），先夹住再 mask。
    src_row = src_row.clamp(max=max(page_table.shape[0] - 1, 0))
    src = page_table.index_select(0, src_row).to(torch.int32)
    n_cols = src.shape[1]

    sel_src = topk_indices[:rows]
    if sel_src.shape[0] < rows:
        # output_num_tokens 可以大于 topk_indices 行数（spec padding）。
        sel_src = torch.cat(
            [
                sel_src,
                sel_src.new_full((rows - sel_src.shape[0], sel_src.shape[1]), -1),
            ],
            dim=0,
        )
    sel = sel_src.to(torch.int64)
    valid = (row_idx < real_num_tokens).unsqueeze(1) & (sel >= 0) & (sel < n_cols)
    gathered = torch.gather(src, 1, sel.clamp(min=0, max=max(n_cols - 1, 0)))
    torch.where(
        valid,
        gathered,
        torch.full((), -1, dtype=torch.int32, device=device),
        out=result,
    )
    return result


_TARGETS = (
    (
        "sglang.kernels.ops.attention.dsa.transform_index",
        "prepare_trtllm_nope_sparse_metadata",
        prepare_trtllm_nope_sparse_metadata,
    ),
    (
        "sglang.kernels.ops.attention.dsa.transform_index",
        "transform_index_page_table_decode_fast",
        transform_index_page_table_decode_fast,
    ),
    (
        "sglang.kernels.ops.attention.dsa.transform_index",
        "transform_index_page_table_prefill_fast",
        transform_index_page_table_prefill_fast,
    ),
)


def apply() -> list:
    """把三个 torch 实现替换到 fork 的调用点（含 import 期绑死的别名）。"""

    mla = _import_sibling("eb5_mla_adapt")
    patched = []
    for module_name, attr, impl in _TARGETS:
        patched.extend(mla.patch_everywhere(module_name, attr, impl))
    return patched
