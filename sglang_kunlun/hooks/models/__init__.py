"""General plugin hooks for Kunlun-specific SGLang behavior."""

from . import deepseek_v4  # noqa: F401
from . import deepseek_v4_precision  # noqa: F401
from . import deepseek_v4_nextn  # noqa: F401

# Kimi-K3 model hook registrations.
from . import kimi_k3_gemm  # noqa: F401
from . import kimi_k3_text_only  # noqa: F401
from . import kimi_k3_w4a16_loader  # noqa: F401
