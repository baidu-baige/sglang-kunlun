"""Consolidate DeepSeek-family quantization/config injection into a loader hook instead of import side effects.

Previously these injections were scattered at import time across the various hook modules (assigning attributes
to upstream classes at module level):

* ``hooks/models/deepseek_v4.py``       -- DSV4's ``hf_to_sglang_mapper``
* ``hooks/models/deepseek_v4_nextn.py`` -- NextN's mapper / packed_modules_mapping
* ``hooks/models/deepseek_v2.py``       -- V2-family packed + GLM's attribute_map

The problem with import side effects: timing drifts with import order and the blast radius is uncontrolled
(it previously caused the old mapper to grab the slot first, broke the new conditional installation, and every
wqkv_a.weight_scale mismatched and was dropped). It is now moved
to just before the two upstream consumption points (BEFORE hooks), positioned uniformly by HookRegistry at startup:

* ``sglang.srt.utils.hf_transformers.config.get_config``
    -- fix GLM's ``attribute_map`` (head_dim redirect) before config parsing.
* ``sglang.srt.model_loader.loader._get_quantization_config``
    -- inject before the quant config is built and before mapper / packed_modules_mapping are read.

Installation is idempotent (``setdefault`` / "don't overwrite if already present"); the injected class
attributes are only read by the loader of the corresponding family, and are harmless no-ops for other models;
it is the same pattern as upstream ``install_shared_experts_fusion_decision`` (which makes a class-level
decision on model_class at the loader's instantiation point).
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_logged: set[str] = set()


def _log_once(key: str, message: str, *args) -> None:
    """Log only once per process, and only on rank0 (each of the 8 TP ranks logging one line is pure noise)."""
    if key in _logged:
        return
    _logged.add(key)
    from sglang.srt.utils import log_info_on_rank0

    log_info_on_rank0(logger, message % args if args else message)


# ---------------------------------------------------------------------------
# GLM-5.x: fix the head_dim redirect in GlmMoeDsaConfig.attribute_map
#
# ``attribute_map`` declares ``head_dim -> qk_rope_head_dim``, and the base
# ``PreTrainedConfig.__setattr__``/``__getattribute__`` redirects via that table, so
# the checkpoint's explicit ``qk_rope_head_dim: 64`` gets overwritten by ``head_dim: 192`` (the qk-nope dim),
# the model builds q_b_proj with ``192 + 192 = 384`` (24576 rows) while the checkpoint has
# 16384 rows, and loading reports "exceeds dimension size". GLM-5.x checkpoints always carry explicit
# qk_nope/qk_rope/qk_head_dim, so drop this redundant mapping.
# ---------------------------------------------------------------------------
def _ensure_glm_attribute_map() -> None:
    try:
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
            GlmMoeDsaConfig,
        )
    except Exception:
        # transformers does not ship the GLM-5.x config class: nothing to handle.
        return
    amap = dict(getattr(GlmMoeDsaConfig, "attribute_map", None) or {})
    dropped = [key for key, value in amap.items() if value == "qk_rope_head_dim"]
    if not dropped:
        return
    for key in dropped:
        amap.pop(key)
    GlmMoeDsaConfig.attribute_map = amap
    _log_once(
        "glm_attr_map",
        "sglang-kunlun: dropped GlmMoeDsaConfig.attribute_map entries %s "
        "(head_dim must not remap to qk_rope_head_dim)",
        dropped,
    )


@plugin_hook(
    "sglang.srt.utils.hf_transformers.config.get_config",
    type=HookType.BEFORE,
)
def ensure_glm_attribute_map_kunlun(*args, **kwargs):
    """Fix GLM's attribute_map before config parsing (idempotent)."""
    _ensure_glm_attribute_map()
    return None


# ---------------------------------------------------------------------------
# DSV4 family: name alignment for compressed-tensors quantization config
#
# W4A8/W8A8 checkpoints' quantization_config uses the checkpoint's original naming
# (``layers.N.attn.*`` / ``layers.N.ffn.*``, wq_a and wkv not fused), while sglang's module
# paths are ``layers.N.self_attn.*`` / ``layers.N.mlp.*`` and are fused into wqkv_a by default.
# Without a mapper, find_matched_target throws
# "Unable to find matching target for model.layers.0.self_attn.wqkv_a".
#
# The rule is substring replacement over regex pattern strings (WeightsMapper matches by key length in
# descending order and stops at the first hit), so the longer wq_a/wkv->wqkv_a rules take effect before the
# generic attn->self_attn. Checkpoints already exported with sglang
# naming match nothing, making this a harmless no-op.
# ---------------------------------------------------------------------------
_DSV4_QUANT_NAME_MAP_BASE = {
    # DSpark draft uses fuse_wqa_wkv=False; the mtp side is not fused anyway.
    r"mtp\.\d+\.attn\.": r"mtp\.\d+\.self_attn\.",
    r"layers\.\d+\.attn\.": r"layers\.\d+\.self_attn\.",
    r"layers\.\d+\.ffn\.": r"layers\.\d+\.mlp\.",
    r"mtp\.\d+\.ffn\.": r"mtp\.\d+\.mlp\.",
}

# Only fold wq_a/wkv onto wqkv_a when fusion is enabled (the two share the same scheme, and map into a single key).
# Must not fold when fusion is disabled, otherwise the model contains two separate wq_a / wkv modules and won't match.
_DSV4_QUANT_NAME_MAP_FUSED = {
    r"layers\.\d+\.attn\.wq_a$": r"layers\.\d+\.self_attn\.wqkv_a$",
    r"layers\.\d+\.attn\.wkv$": r"layers\.\d+\.self_attn\.wqkv_a$",
}


def _build_dsv4_quant_name_map(fuse_wqa_wkv: bool) -> dict:
    mapping = dict(_DSV4_QUANT_NAME_MAP_BASE)
    if fuse_wqa_wkv:
        mapping.update(_DSV4_QUANT_NAME_MAP_FUSED)
    return mapping


def _dsv4_family_classes() -> list:
    """The DSV4-family classes that need their mapper before the loader (in dependency order)."""
    classes = []
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    classes.append(DeepseekV4ForCausalLM)
    try:
        from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

        classes.append(DeepseekV4ForCausalLMDSpark)
    except ImportError:
        pass
    return classes


def _install_dsv4_quant_name_mapper() -> None:
    """Install the quant name mapper for the DSV4 family (idempotent, does not overwrite upstream's own)."""
    from sglang.srt import environ as _environ
    from sglang.srt.models.utils import WeightsMapper

    fuse = bool(_environ.envs.SGLANG_OPT_FUSE_WQA_WKV.get())
    mapper = WeightsMapper(orig_to_new_substr=_build_dsv4_quant_name_map(fuse))
    installed = []
    for cls in _dsv4_family_classes():
        # Don't overwrite if a mapper is already present (if upstream adds one later, upstream wins).
        if getattr(cls, "hf_to_sglang_mapper", None) is None:
            cls.hf_to_sglang_mapper = mapper
            installed.append(cls.__name__)
    if installed:
        _log_once(
            "dsv4_mapper",
            "sglang-kunlun: installed DSV4 quant name mapper on %s (fuse_wqa_wkv=%s)",
            installed,
            fuse,
        )


def _install_dsv4_nextn_mapper() -> None:
    """NextN/MTP mapper and fused-name expansion (layered on top of the DSV4 base mapper)."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ForCausalLMNextN
    from sglang.srt.models.utils import WeightsMapper

    if getattr(DeepseekV4ForCausalLMNextN, "packed_modules_mapping", None) is None:
        DeepseekV4ForCausalLMNextN.packed_modules_mapping = {
            "wqkv_a": ["wq_a", "wkv"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }

    base_mapper = getattr(DeepseekV4ForCausalLM, "hf_to_sglang_mapper", None)
    if base_mapper is None:
        return
    if getattr(DeepseekV4ForCausalLMNextN, "hf_to_sglang_mapper", None) is not None:
        return
    # The checkpoint uses ``mtp.<i>.*`` while this module tree is ``model.*`` / ``model.decoder.*``;
    # compressed-tensors targets/ignore are written in checkpoint naming, so they need an additional rewrite.
    DeepseekV4ForCausalLMNextN.hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            **base_mapper.orig_to_new_substr,
            r"mtp\.\d+\.attn_norm": r"decoder\.input_layernorm",
            r"mtp\.\d+\.ffn_norm": r"decoder\.post_attention_layernorm",
            r"mtp\.\d+\.attn\.": r"decoder\.self_attn\.",
            r"mtp\.\d+\.ffn\.": r"decoder\.mlp\.",
            r"mtp\.\d+\.": r"model\.",
        },
        orig_to_new_prefix=base_mapper.orig_to_new_prefix,
        orig_to_new_suffix=base_mapper.orig_to_new_suffix,
    )
    _log_once("nextn_mapper", "sglang-kunlun: installed NextN/MTP weight-name mapper")


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.get_hf_to_sglang_mapper",
    type=HookType.AROUND,
)
@plugin_hook(
    "sglang.srt.models.deepseek_v4_nextn.DeepseekV4ForCausalLMNextN.get_hf_to_sglang_mapper",
    type=HookType.AROUND,
)
def get_dsv4_quant_name_mapper_kunlun(original_fn, cls, config):
    """Expose compressed-tensors mappings through the 0.5.22 loader API.

    The new loader prefers this getter over the legacy class attribute. Upstream
    implements it only for Quark, so our existing compressed-tensors mapping
    would otherwise be silently skipped.

    Args:
        original_fn: Upstream class-level quantization mapper getter.
        config: Model configuration containing the quantization method.

    Returns:
        The class's compressed-tensors mapper, or the upstream getter's result.
    """
    quant = getattr(config, "quantization_config", None) or {}
    if quant.get("quant_method") == "compressed-tensors":
        mapper = getattr(cls, "hf_to_sglang_mapper", None)
        if mapper is not None:
            return mapper
    return original_fn(cls, config)


def _install_v2_packed_mapping() -> None:
    """Fused MLP name expansion for the V2 family (including GLM-5.x DSA subclasses) (idempotent)."""
    from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

    mapping = getattr(DeepseekV2ForCausalLM, "packed_modules_mapping", None)
    if mapping is None:
        DeepseekV2ForCausalLM.packed_modules_mapping = mapping = {}
    before = len(mapping)
    mapping.setdefault("gate_up_proj", ["gate_proj", "up_proj"])
    # The base model appends this entry in __init__, but NextN/EAGLE draft bypasses that
    # __init__; the compressed-tensors config needs it to expand the fused name to q_a_proj /
    # kv_a_proj_with_mqa.
    mapping.setdefault(
        "fused_qkv_a_proj_with_mqa", ["q_a_proj", "kv_a_proj_with_mqa"]
    )
    if len(mapping) != before:
        _log_once(
            "v2_packed",
            "sglang-kunlun: filled DeepseekV2ForCausalLM.packed_modules_mapping (%s)",
            sorted(mapping),
        )


@plugin_hook(
    "sglang.srt.model_loader.loader._get_quantization_config",
    type=HookType.BEFORE,
)
def inject_quant_names_kunlun(*args, **kwargs):
    """Inject packed_modules_mapping / mapper before the quant config is built (idempotent, on demand).

    This hook point is before model instantiation and before the loader reads
    ``packed_modules_mapping`` (loader.py:172) and ``hf_to_sglang_mapper``
    (loader.py:262), equivalent to the injection timing chosen by upstream
    ``install_shared_experts_fusion_decision``.
    """
    del args, kwargs  # only seize the timing; don't modify the arguments
    try:
        _install_v2_packed_mapping()
        _install_dsv4_quant_name_mapper()
        _install_dsv4_nextn_mapper()
    except Exception:
        logger.exception("sglang-kunlun: failed to inject quant name mappings")
    return None
