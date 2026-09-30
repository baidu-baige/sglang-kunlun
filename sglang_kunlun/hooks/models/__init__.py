"""General plugin hooks for Kunlun-specific SGLang behavior.

本目录只保留 **方法级** hook（DeepSeek-V4/NextN 的 forward 路径）。
权重命名 mapper / packed_modules_mapping / GLM attribute_map 的注入已收敛到
``hooks/model_loader/quant_injection.py``（挂在 loader 消费点之前的 BEFORE
hook），不再有 import 时的全局类属性副作用。
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

# DeepSeek-V4 / NextN 方法级 hook —— DSV4-Flash 运行必需，直接导入。
from . import deepseek_v4  # noqa: F401
from . import deepseek_v4_precision  # noqa: F401
from . import deepseek_v4_nextn  # noqa: F401

# 以下为从 dev-kunlun-0.5.19-dsv4.1 合入的 DSV4.1 / Kimi-K3 模型 hook。
# 它们依赖更新版本的上游 sglang（如 ``sglang.srt.layers.engram``），在当前
# pinned 的 v0.5.19 base 上可能不存在。DSV4-Flash 本身用不到这些 hook，
# 故做可选导入：缺依赖时跳过（base 升级后自动生效），避免拖垮整个插件加载。
for _opt in (
    "dsv41_engram_int8",
    "dsv41_dspark_weights",
    "kimi_k3_gemm",
    "kimi_k3_text_only",
    "kimi_k3_w4a16_loader",
):
    try:
        __import__(f"{__name__}.{_opt}", fromlist=[_opt])
    except ImportError as _e:
        _logger.warning("skip optional dsv4.1/K3 model hook %s: %s", _opt, _e)
