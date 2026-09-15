"""Kunlun MiniMax-M3 integration — single self-contained subpackage.

Everything MiniMax-M3-specific lives under this package, so porting the model
to a newer sglang tree is "copy this directory + one import line" in
``sglang_kunlun/hooks/registry.py`` (see the IMPORTANT note there about the
import order relative to ``kernel_ops.install()``).

    compat.py          import-order-sensitive Triton patches for the upstream
                       sparse modules (applied at package import)
    m3.py              MiniMaxM3MoE hook: Kunlun bf16 router GEMM
                       (dual stream is native upstream since 0.5.19)
    vl.py              MiniMaxVLVisionTransformer hook: bf16 pixel values
    sparse_backend.py  MiniMaxSparseAttnBackend TARGET_VERIFY metadata hooks
    sparse_ops/        REPLACE hooks for the two upstream sparse dispatchers
    kernels.py         kernel_ops JIT replacements (fused qknorm rope, fused
                       KV+index store, MiniMax MoE fused gate)
    kv_pools.py        Kunlun paged sub-pools for MiniMaxSparseKVPool
    kv_cache.py        KVCacheConfigurator._build_oot_mha_kv_pool AROUND hook
                       that routes Minimax-sparse models to the Kunlun pool
"""

from __future__ import annotations

from .compat import apply_compat_patches

# Must run before the submodules below (and before kernel_ops.install()) import
# the upstream MiniMax sparse modules.
apply_compat_patches()

from . import kernels  # noqa: E402,F401  (registers JIT replacements)
from . import kv_cache  # noqa: E402,F401
from . import kv_pools  # noqa: E402,F401
from . import m3  # noqa: E402,F401
from . import sparse_backend  # noqa: E402,F401
from . import sparse_ops  # noqa: E402,F401
from . import vl  # noqa: E402,F401
