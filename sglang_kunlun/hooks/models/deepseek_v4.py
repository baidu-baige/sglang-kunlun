# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Hooks for DeepseekV4DecoderLayer methods (hc_post, hc_pre)."""
from __future__ import annotations

from typing import Literal

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
import kunlun_ops


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post",
    type=HookType.REPLACE,
)
def hc_post_kunlun(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """Kunlun-native hc_post via ``kunlun_ops.hc_post_kunlun_impl``.

    Computes ``out = post * x + comb @ residual`` (fused).
    Only N == 4 (hc_mult) is supported on XPU.
    """
    if x.shape[0] == 0:
        return torch.empty(
            (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
        )

    batch = x.shape[0]
    N = self.hc_mult
    D = x.shape[-1]
    out = torch.empty((batch, N, D), dtype=x.dtype, device=x.device)
    kunlun_ops.hc_post_kunlun_impl(
        post.contiguous(),
        x.contiguous(),
        comb.contiguous(),
        residual.contiguous(),
        out,
        batch, N, D,
    )
    return out


# --------------------------- INT4 checkpoint support ---------------------
# The compressed-tensors targets/ignore lists in the INT4 checkpoint use the
# original HF module names (`attn`/`ffn`/`w1..w3`), while the weights
# themselves are remapped by `remap_weight_name_to_dpsk_hf_format`. The mapper
# is picked up by the model loader to remap the quantization config. The keys
# look like regexes because the checkpoint targets are regex strings; 0.5.14
# WeightsMapper still matches them as plain substrings.
DeepseekV4ForCausalLM.packed_modules_mapping = {
    "wqkv_a": ["wq_a", "wkv"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

DeepseekV4ForCausalLM.hf_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={
        r"\.attn\.": r"\.self_attn\.",
        r"\.ffn\.": r"\.mlp\.",
        r"\.attn_norm": r"\.input_layernorm",
        r"\.ffn_norm": r"\.post_attention_layernorm",
    },
    orig_to_new_suffix={
        r"\.w1$": r"\.gate_proj$",
        r"\.w2$": r"\.down_proj$",
        r"\.w3$": r"\.up_proj$",
    },
)


def _fuse_wqkv_a_weight_scale(weights):
    """Concat the ``wq_a``/``wkv`` per-channel scales for the fused ``wqkv_a``.

    Upstream `load_weights` already fuses ``.weight`` and ``.weight_scale_inv``;
    the int4 checkpoint instead carries ``.weight_scale``, which is fused here
    so the generic loader path picks it up as a plain ``wqkv_a.weight_scale``.
    """
    pending = {}
    for name, tensor in weights:
        is_q = name.endswith(".wq_a.weight_scale")
        if not is_q and not name.endswith(".wkv.weight_scale"):
            yield name, tensor
            continue
        fused_name = name.replace(".wq_a." if is_q else ".wkv.", ".wqkv_a.")
        bucket = pending.setdefault(fused_name, {})
        shard: Literal["q", "kv"] = "q" if is_q else "kv"
        assert shard not in bucket, f"duplicate shard {shard} for {fused_name}"
        bucket[shard] = tensor
        if len(bucket) == 2:
            yield fused_name, torch.cat([bucket["q"], bucket["kv"]], dim=0)
            pending.pop(fused_name)
    assert not pending, f"unpaired wqkv_a weight scales: {sorted(pending)}"


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
    type=HookType.AROUND,
)
def load_weights_kunlun(original_fn, self, weights, is_nextn=False):
    """Pre-process the int4/int8 weight stream, then defer to upstream."""
    if envs.SGLANG_OPT_FUSE_WQA_WKV.get():
        weights = _fuse_wqkv_a_weight_scale(weights)
    return original_fn(self, weights, is_nextn)
