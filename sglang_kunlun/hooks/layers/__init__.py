"""Kunlun layer hooks.

Importing this package registers all layer-level ``@plugin_hook`` targets.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding."
    "_use_triton_embedding",
    type=HookType.REPLACE,
)
def _disable_cuda_triton_vocab_embedding(self, input_):
    """Kunlun tensors use the eager shard embedding path, not CUDA Triton."""
    return False


from . import linear  # noqa: F401,E402
from . import logits_processor  # noqa: F401,E402
from . import mhc  # noqa: F401
from . import quantization  # noqa: F401
from . import moe  # noqa: F401
from . import cp  # noqa: F401,E402
from . import rotary_embedding  # noqa: F401,E402

# Must be imported before upstream resolves attention backend modules.
from . import attention  # noqa: F401,E402

# Kimi-K3 layer hooks. ``embedding`` is deliberately absent: the
# ``_use_triton_embedding`` REPLACE above is the inlined equivalent, and a second
# registration on the same target would conflict.
from . import kda  # noqa: F401,E402
from . import activation  # noqa: F401,E402
from . import attn_residual  # noqa: F401,E402
