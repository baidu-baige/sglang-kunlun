"""General plugin hooks for Kunlun-specific SGLang behavior.

本目录只保留 **方法级** hook（DeepSeek-V4/NextN 的 forward 路径）。
权重命名 mapper / packed_modules_mapping / GLM attribute_map 的注入已收敛到
``hooks/model_loader/quant_injection.py``（挂在 loader 消费点之前的 BEFORE
hook），不再有 import 时的全局类属性副作用。
"""

from . import deepseek_v4  # noqa: F401
from . import deepseek_v4_precision  # noqa: F401
from . import deepseek_v4_nextn  # noqa: F401
