"""DSA (formerly NSA) related Kunlun hooks.

``kunlun_dsa_indexer`` is the single patch point for the indexer: it registers
``plugin_hook`` REPLACE hooks on ``Indexer.forward_cuda`` / ``forward_xpu`` (plus
``_weights_proj_bf16_in_fp32_out``). It supersedes the old ``nsa_indexer`` module
which replaced 7 individual ``Indexer._get_*`` methods and could therefore silently
mix Kunlun code with upstream control flow. ``nsa_indexer`` is intentionally NOT
imported any more - it is dead code and can be deleted once this path is validated.
"""
from . import index_buf_accessor  # noqa: F401
from . import index_buf_accessor_v4  # noqa: F401
from . import quant_k_cache_v4  # noqa: F401
from . import triton_kernel  # noqa: F401
from . import kunlun_dsa_indexer  # noqa: F401
