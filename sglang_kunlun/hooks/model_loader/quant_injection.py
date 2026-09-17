"""把 DeepSeek 家族的量化/配置注入从 import 副作用收敛到 loader hook。

原先这些注入散落在各 hook 模块 import 时（模块级给上游类赋属性）：

* ``hooks/models/deepseek_v4.py``       —— DSV4 的 ``hf_to_sglang_mapper``
* ``hooks/models/deepseek_v4_nextn.py`` —— NextN 的 mapper / packed_modules_mapping
* ``hooks/models/deepseek_v2.py``       —— V2 家族 packed + GLM 的 attribute_map

import 副作用的问题：时机随 import 顺序漂移、影响面不受控（曾导致旧 mapper
抢先占位、新版条件安装失效、wqkv_a.weight_scale 全部对不上而丢失）。现在改为
挂在两个上游消费点之前（BEFORE hook），由 HookRegistry 统一在启动时就位：

* ``sglang.srt.utils.hf_transformers.config.get_config``
    —— config 解析之前修 GLM 的 ``attribute_map``（head_dim 重定向）。
* ``sglang.srt.model_loader.loader._get_quantization_config``
    —— quant config 构建、读取 mapper / packed_modules_mapping 之前注入。

安装幂等（``setdefault`` / "已有则不覆盖"），注入的类属性只被对应家族的
loader 读取，对其它模型是无害的空操作；与上游
``install_shared_experts_fusion_decision``（在 loader 的实例化点对 model_class
做类级决策）是同一模式。
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_logged: set[str] = set()


def _log_once(key: str, message: str, *args) -> None:
    """每个进程只打一次，且只在 rank0（8 个 TP rank 各打一条纯属噪声）。"""
    if key in _logged:
        return
    _logged.add(key)
    from sglang.srt.utils import log_info_on_rank0

    log_info_on_rank0(logger, message % args if args else message)


# ---------------------------------------------------------------------------
# GLM-5.x: 修正 GlmMoeDsaConfig.attribute_map 的 head_dim 重定向
#
# ``attribute_map`` 声明 ``head_dim -> qk_rope_head_dim``，而基类
# ``PreTrainedConfig.__setattr__``/``__getattribute__`` 会按该表重定向，于是
# checkpoint 显式的 ``qk_rope_head_dim: 64`` 被 ``head_dim: 192``（qk-nope 维）
# 覆盖，模型按 ``192 + 192 = 384`` 建 q_b_proj（24576 行）而 checkpoint 是
# 16384 行，加载报 "exceeds dimension size"。GLM-5.x checkpoint 总是显式带
# qk_nope/qk_rope/qk_head_dim，因此删掉这条多余映射。
# ---------------------------------------------------------------------------
def _ensure_glm_attribute_map() -> None:
    try:
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
            GlmMoeDsaConfig,
        )
    except Exception:
        # transformers 不带 GLM-5.x config 类：无需处理。
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
    """config 解析前修正 GLM 的 attribute_map（幂等）。"""
    _ensure_glm_attribute_map()
    return None


# ---------------------------------------------------------------------------
# DSV4 家族: compressed-tensors 量化配置的命名对齐
#
# W4A8/W8A8 checkpoint 的 quantization_config 用 checkpoint 原始命名
# （``layers.N.attn.*`` / ``layers.N.ffn.*``、wq_a 与 wkv 未融合），sglang 的模块
# 路径是 ``layers.N.self_attn.*`` / ``layers.N.mlp.*`` 且默认融合成 wqkv_a。
# 不注入 mapper 时 find_matched_target 会抛
# "Unable to find matching target for model.layers.0.self_attn.wqkv_a"。
#
# 规则是正则模式串上的子串替换（WeightsMapper 按 key 长度倒序匹配、命中即停），
# 更长的 wq_a/wkv→wqkv_a 规则先于通用的 attn→self_attn 生效。对已用 sglang
# 命名导出的 checkpoint 全部不命中，是无害空操作。
# ---------------------------------------------------------------------------
_DSV4_QUANT_NAME_MAP_BASE = {
    # DSpark draft 用 fuse_wqa_wkv=False，mtp 侧本来就不融合。
    r"mtp\.\d+\.attn\.": r"mtp\.\d+\.self_attn\.",
    r"layers\.\d+\.attn\.": r"layers\.\d+\.self_attn\.",
    r"layers\.\d+\.ffn\.": r"layers\.\d+\.mlp\.",
    r"mtp\.\d+\.ffn\.": r"mtp\.\d+\.mlp\.",
}

# 开融合时才把 wq_a/wkv 折到 wqkv_a 上（两者 scheme 相同，映射后合并成一个 key）。
# 关融合时必须不折，否则模型里是分开的 wq_a / wkv 两个模块、反而匹配不上。
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
    """DSV4 家族里需要在 loader 前拿到 mapper 的类（按依赖顺序）。"""
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
    """给 DSV4 家族装量化 name mapper（幂等，不覆盖上游自带）。"""
    from sglang.srt import environ as _environ
    from sglang.srt.models.utils import WeightsMapper

    fuse = bool(_environ.envs.SGLANG_OPT_FUSE_WQA_WKV.get())
    mapper = WeightsMapper(orig_to_new_substr=_build_dsv4_quant_name_map(fuse))
    installed = []
    for cls in _dsv4_family_classes():
        # 已经自带 mapper 的话不覆盖（上游若后续补上，以上游为准）。
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
    """NextN/MTP 的 mapper 与 fused 名称展开（在 DSV4 base mapper 之上叠加）。"""
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
    # checkpoint 用 ``mtp.<i>.*``，而这棵模块树是 ``model.*`` / ``model.decoder.*``，
    # compressed-tensors 的 targets/ignore 按 checkpoint 命名书写，需要叠加改写。
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


def _install_v2_packed_mapping() -> None:
    """V2 家族（含 GLM-5.x DSA 子类）的 fused MLP 名称展开（幂等）。"""
    from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

    mapping = getattr(DeepseekV2ForCausalLM, "packed_modules_mapping", None)
    if mapping is None:
        DeepseekV2ForCausalLM.packed_modules_mapping = mapping = {}
    before = len(mapping)
    mapping.setdefault("gate_up_proj", ["gate_proj", "up_proj"])
    # base 模型在 __init__ 里 append 这一项，但 NextN/EAGLE draft 会绕过该
    # __init__，compressed-tensors 配置需要它才能把融合名展开到 q_a_proj /
    # kv_a_proj_with_mqa。
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
    """quant config 构建前注入 packed_modules_mapping / mapper（幂等、按需）。

    这一挂点在模型实例化之前、以及 loader 读取
    ``packed_modules_mapping``（loader.py:172）与 ``hf_to_sglang_mapper``
    （loader.py:262）之前，等价于上游 ``install_shared_experts_fusion_decision``
    选择的注入时机。
    """
    del args, kwargs  # 只是抢占时机，不改参数
    try:
        _install_v2_packed_mapping()
        _install_dsv4_quant_name_mapper()
        _install_dsv4_nextn_mapper()
    except Exception:
        logger.exception("sglang-kunlun: failed to inject quant name mappings")
    return None
