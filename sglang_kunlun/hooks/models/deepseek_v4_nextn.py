"""Hooks for DeepseekV4 NextN / MTP INT4 checkpoint mapping."""

from __future__ import annotations

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ForCausalLMNextN
from sglang.srt.models.utils import WeightsMapper

# Install the base mapper first so NextN can reuse its substr/suffix maps.
from . import deepseek_v4 as _deepseek_v4  # noqa: F401

_base_mapper = DeepseekV4ForCausalLM.hf_to_sglang_mapper

# The checkpoint names the MTP weights `mtp.<i>.*`, but this module tree is
# `model.{e_proj,h_proj,enorm,hnorm,...}` for the layer-external weights and
# `model.decoder.*` for the decoder layer. The compressed-tensors
# targets/ignore lists use the checkpoint names, so they need to be
# rewritten on top of the base DeepseekV4 remapping.
DeepseekV4ForCausalLMNextN.hf_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={
        **_base_mapper.orig_to_new_substr,
        r"mtp\.\d+\.attn_norm": r"decoder\.input_layernorm",
        r"mtp\.\d+\.ffn_norm": r"decoder\.post_attention_layernorm",
        r"mtp\.\d+\.attn\.": r"decoder\.self_attn\.",
        r"mtp\.\d+\.ffn\.": r"decoder\.mlp\.",
        r"mtp\.\d+\.": r"model\.",
    },
    orig_to_new_prefix=_base_mapper.orig_to_new_prefix,
    orig_to_new_suffix=_base_mapper.orig_to_new_suffix,
)

DeepseekV4ForCausalLMNextN.packed_modules_mapping = {
    "wqkv_a": ["wq_a", "wkv"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
