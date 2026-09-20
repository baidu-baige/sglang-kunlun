"""W8A8-INT8 engram embedding table for DeepSeek-V4.1 on Kunlun/XPU.

Upstream `EngramEmbedding` stores the table as an fp8_e4m3 payload with e8m0
scales, one per 32-element block (the FP8 release). The W8A8-INT8 release stores

    layers.N.engram.embed.weight        I8   [num_embeddings, dim]
    layers.N.engram.embed.weight_scale  BF16 [num_embeddings, 1]

i.e. compressed-tensors channel-wise int8: one scale per row. The row sharding,
host-table machinery and the all-reduce contract are identical; only the payload
dtype, the scale layout and the dequant arithmetic differ, so this class keeps the
upstream base for the fp8 format and adds the int8 layout alongside it.

Dequant: ``values = weight[ids].float() * weight_scale[ids].float()`` -> bf16,
which is exactly what the compressed-tensors channel-wise scale means.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from torch import nn

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers import engram as engram_mod
from sglang.srt.layers.engram import EngramEmbedding


_INT8_CACHE: Optional[bool] = None


def _index_has_int8_engram(model_path: str) -> bool:
    """Whether the checkpoint index names `engram.embed.weight_scale`."""
    try:
        for fn in os.listdir(model_path):
            if not fn.endswith(".safetensors.index.json"):
                continue
            with open(os.path.join(model_path, fn)) as f:
                weight_map = json.load(f).get("weight_map", {})
            return any(k.endswith("engram.embed.weight_scale") for k in weight_map)
    except Exception:  # noqa: BLE001
        return False
    return False


def engram_embed_weight_is_int8() -> bool:
    """Whether the engram table is channel-wise int8 (vs the fp8 block layout).

    Decided from, in order: the ``DSV4_ENGRAM_INT8`` override, the checkpoint's
    safetensors index (``MODEL_PATH``), and the HF quantization config. The
    index path matters because ``EngramEmbedding`` is constructed while the model
    is still being built -- ``get_model()`` is not published yet, so a
    config-only probe would silently fall back to the fp8 layout.
    """
    global _INT8_CACHE
    if _INT8_CACHE is not None:
        return _INT8_CACHE

    override = os.environ.get("DSV4_ENGRAM_INT8")
    if override is not None and override.strip() != "":
        _INT8_CACHE = override.strip().lower() not in ("0", "false", "no")
        return _INT8_CACHE

    model_path = os.environ.get("MODEL_PATH") or os.environ.get("DSV4_MODEL_PATH") or ""
    if model_path and _index_has_int8_engram(model_path):
        _INT8_CACHE = True
        return True

    try:
        hf = get_model().model_config.hf_config
        qc = getattr(hf, "quantization_config", None)
        if qc is None:
            _INT8_CACHE = False
            return False
        if not isinstance(qc, dict):
            qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
        for group in (qc.get("config_groups") or {}).values():
            targets = group.get("targets") or []
            if not any("engram" in t and "embed" in t for t in targets):
                continue
            weights = group.get("weights") or {}
            _INT8_CACHE = weights.get("type") == "int" and weights.get("num_bits") == 8
            return _INT8_CACHE
    except Exception:  # noqa: BLE001 - detection must never break model build
        pass
    _INT8_CACHE = False
    return False


@plugin_hook("sglang.srt.layers.engram.EngramEmbedding", type=HookType.REPLACE)
class EngramEmbeddingKunlun(EngramEmbedding):
    """`EngramEmbedding` that also serves the W8A8-INT8 table layout."""

    def __init__(self, num_embeddings: int, dim: int, layer_id: int):
        self._int8 = engram_embed_weight_is_int8()
        if not self._int8:
            # FP8 checkpoint: keep the upstream fp8_e4m3 + e8m0 layout verbatim.
            super().__init__(num_embeddings, dim, layer_id)
            return
        nn.Module.__init__(self)
        self.dim = dim
        self.tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.row_start = num_embeddings * tp_rank // self.tp_size
        row_end = num_embeddings * (tp_rank + 1) // self.tp_size
        self.rows = row_end - self.row_start
        self.host_table: Optional[object] = None
        if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            self._init_host_table_int8(num_embeddings, dim, layer_id)
        else:
            self.weight = nn.Parameter(
                torch.empty(self.rows, dim, dtype=torch.int8), requires_grad=False
            )
            self.weight_scale = nn.Parameter(
                torch.empty(self.rows, 1, dtype=torch.bfloat16), requires_grad=False
            )
        # `_load_rows` is dtype/name agnostic: it copies this rank's row slice.
        self.weight.weight_loader = self._load_rows
        self.weight_scale.weight_loader = self._load_rows

    def _init_host_table_int8(self, num_embeddings: int, dim: int, layer_id: int):
        # Upstream dropped `_HostTable.choose_layout` and the per-call `pin`
        # flag: the layout is taken straight from the env var ("shared" or
        # "per_rank", the old "private"), and the mapping is always registered
        # with cudaHostRegister. Mirror `EngramEmbedding._init_host_table`.
        layout = envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get()
        n = num_embeddings if layout == "shared" else self.rows
        w_bytes = n * dim  # int8 payload
        s_bytes = n * 2  # one bf16 scale per row
        self.host_table = engram_mod._HostTable(
            layout,
            max(1, w_bytes + s_bytes),  # mmap needs storage even for an empty shard
            f"sglang_engram_{layer_id}",
            get_tp_group(),
        )
        raw = self.host_table.bytes[: w_bytes + s_bytes]
        weight = raw[:w_bytes].view(torch.int8).view(n, dim)
        scale = raw[w_bytes : w_bytes + s_bytes].view(torch.bfloat16).view(n, 1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)

    def forward(
        self,
        indices: torch.Tensor,
        forward_batch=None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        """forward"""
        if not self._int8:
            return super().forward(
                indices, forward_batch, cp_all_tokens=cp_all_tokens
            )
        if os.environ.get("DSV4_KUNLUN_ZERO_ENGRAM") == "1":
            # Accuracy A/B: make the engram lookup a well-defined no-op (the
            # residual then carries `x + gate * 0`). Every rank takes this branch,
            # so the TP all-reduce contract stays consistent.
            return self._empty(indices).zero_()
        if self._shared:
            if indices.shape[0] == 0:
                return self._empty(indices)
            out = self._empty(indices)
            from sglang.kernels.ops.embeddings.engram_gather import engram_gather

            # block_size == dim selects the one-scale-per-row (int8) decode.
            engram_gather(
                self.weight.data_ptr(),
                self.weight_scale.data_ptr(),
                indices.reshape(-1),
                out.view(-1, self.dim),
                self.dim,
                self.dim,
            )
            return out
        return super().forward(indices, forward_batch, cp_all_tokens=cp_all_tokens)

    def _owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
        if not self._int8:
            return super()._owned_rows(indices)
        if self.rows == 0:
            return self._empty(indices).zero_()
        if self.host_table is None:
            # Device-resident shard: plain tensor gather, no raw pointers.
            local = (indices - self.row_start).to(torch.int64)
            owned = (local >= 0) & (local < self.rows)
            local = local.masked_fill(~owned, 0)
            values = self.weight[local].float() * self.weight_scale[local].float()
            return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        out = self._empty(indices)
        engram_gather(
            self.weight.data_ptr(),
            self.weight_scale.data_ptr(),
            indices.reshape(-1),
            out.view(-1, self.dim),
            self.dim,
            self.dim,
            row_lo=self.row_start,
            row_hi=self.row_start + self.rows,
        )
        return out
