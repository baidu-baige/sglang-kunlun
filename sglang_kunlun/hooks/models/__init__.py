"""General plugin hooks for Kunlun-specific SGLang behavior.

本目录只保留 **方法级** hook（DeepSeek-V4/NextN 的 forward 路径）。
权重命名 mapper / packed_modules_mapping / GLM attribute_map 的注入已收敛到
``hooks/model_loader/quant_injection.py``（挂在 loader 消费点之前的 BEFORE
hook），不再有 import 时的全局类属性副作用。
"""

from . import deepseek_v4  # noqa: F401
from . import deepseek_v4_precision  # noqa: F401
from . import deepseek_v4_nextn  # noqa: F401
from . import dsv41_engram_int8  # noqa: F401
from . import dsv41_dspark_weights  # noqa: F401

# Kimi-K3 model hook registrations.
from . import kimi_k3_gemm  # noqa: F401
from . import kimi_k3_text_only  # noqa: F401
from . import kimi_k3_w4a16_loader  # noqa: F401
