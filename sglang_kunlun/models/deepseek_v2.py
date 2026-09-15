"""DeepseekV2-family (DeepSeek-V3 / GLM-5.x DSA) model registration.

The checkpoint's compressed-tensors config quantizes the MLP gate/up
projections under their unfused names (``mlp.gate_proj`` / ``mlp.up_proj``),
but the runtime fuses them into a single ``gate_up_proj`` column-parallel
layer. ``DeepseekV2ForCausalLM`` carries an empty class-level
``packed_modules_mapping`` (only ``fused_qkv_a_proj_with_mqa`` is appended in
``__init__``), so without this entry ``find_matched_target`` cannot resolve
``model.layers.N.mlp.gate_up_proj`` and model load crashes with
"Unable to find matching target ...".

This mirrors the DeepSeek-V4 hook in ``hooks/models/deepseek_v4.py``, which
injects ``gate_up_proj`` into ``DeepseekV4ForCausalLM.packed_modules_mapping``.
Subclasses (``GlmMoeDsaForCausalLM``, ``DeepseekV3ForCausalLMNextN``,
``GlmMoeDsaForCausalLMNextN``) inherit the mapping, so the base model and the
EAGLE/NextN draft both resolve the fused MLP.
"""

from __future__ import annotations

from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

DeepseekV2ForCausalLM.packed_modules_mapping.setdefault(
    "gate_up_proj", ["gate_proj", "up_proj"]
)
# The base model appends this fused MLA Q/KV projection in __init__ before
# calling update_packed_modules_mapping, but the NextN/EAGLE draft
# (DeepseekV3ForCausalLMNextN) bypasses that __init__ entirely, so the
# compressed-tensors config would otherwise never learn to expand the fused
# name into q_a_proj / kv_a_proj_with_mqa for the decoder layer.
DeepseekV2ForCausalLM.packed_modules_mapping.setdefault(
    "fused_qkv_a_proj_with_mqa", ["q_a_proj", "kv_a_proj_with_mqa"]
)

# ---------------------------------------------------------------------------
# transformers GlmMoeDsaConfig.head_dim remap fix
#
# ``GlmMoeDsaConfig.attribute_map`` declares ``head_dim -> qk_rope_head_dim``,
# so ``AutoConfig.from_pretrained`` overwrites the checkpoint's explicit
# ``qk_rope_head_dim: 64`` with its ``head_dim: 192`` (which is the qk-nope
# dim, not the rope dim). The model then computes ``qk_head_dim =
# qk_nope_head_dim + qk_rope_head_dim = 192 + 192 = 384`` and allocates
# ``q_b_proj`` with ``64 * 384 = 24576`` output rows, while the checkpoint
# stores ``64 * 256 = 16384`` — weight load fails with
# "start (15360) + length (3072) exceeds dimension size (16384)".
#
# GLM-5.x checkpoints always carry explicit ``qk_nope_head_dim`` /
# ``qk_rope_head_dim`` / ``qk_head_dim``, so the checkpoint values are
# authoritative; drop the spurious ``head_dim`` mapping.
try:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )

    _attribute_map = dict(GlmMoeDsaConfig.attribute_map or {})
    for _key, _val in list(_attribute_map.items()):
        if _val == "qk_rope_head_dim":
            _attribute_map.pop(_key)
    GlmMoeDsaConfig.attribute_map = _attribute_map
except Exception:
    # transformers without the GLM-5.x config class: nothing to patch.
    pass

