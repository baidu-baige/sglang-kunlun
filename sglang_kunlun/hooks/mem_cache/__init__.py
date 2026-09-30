"""mem_cache hooks."""
from sglang.srt.mem_cache import memory_pool_host as _memory_pool_host

from . import common  # noqa: F401
from . import kunlun_hicache  # noqa: F401

# ``shared_host_kv_hooks`` targets ``memory_pool_host.HostPoolGroup``, which only
# exists in newer SGLang. Importing it against an older host registers hooks that
# ``HookRegistry.apply_hooks()`` can never resolve, and it logs a full traceback
# per target per TP rank at every startup. Register it only when the class is
# actually there.
if hasattr(_memory_pool_host, "HostPoolGroup"):
    from . import shared_host_kv_hooks  # noqa: F401
