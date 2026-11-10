"""Re-open the in-tree KV-pool paths Kunlun has a validated implementation for.

0.5.22 added ``reject_out_of_tree_path`` guards to ``KVCacheConfigurator``
(``sglang/srt/platforms/interface.py``): any out-of-tree platform that reaches a
pool family with "no platform seam yet" now raises ``NotImplementedError``
instead of building in-tree classes. Kunlun is ``PlatformEnum.OOT``, so a plain
DSV4 boot dies in ``_build_token_to_kv_pool`` before a single weight is loaded::

    NotImplementedError: Out-of-tree platform KunlunSRTPlatform (device
    'kunlun') reached the DeepSeek-V4 KV pool, which has no platform seam yet
    and would otherwise build in-tree classes that assume CUDA.

For the DeepSeek-V4 pool the premise does not hold on Kunlun: ``DeepSeekV4TokenToKVPool``
allocates through ``torch`` on ``self.device`` and the Kunlun-specific parts of
it (buffer carving, the C4/C128 state pools, HiSparse host pools) are already
owned by this plugin's hooks -- this is the pool 0.5.17/0.5.19 ran in
production. So the guard is relaxed for exactly the subsystems Kunlun has a
validated path for, and left raising for every other one: a pool family that
really has no Kunlun implementation must keep failing loudly at boot rather than
building CUDA-assuming classes.

Scoped to the configurator's own module binding (it does
``from sglang.srt.platforms.interface import reject_out_of_tree_path`` at import
time) so every other caller of the guard keeps the upstream behaviour.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# Subsystems whose in-tree implementation is device-agnostic enough for Kunlun,
# with the plugin owning the device-specific parts. Keep this list short: every
# entry is a promise that the plugin covers that pool family.
_KUNLUN_SUPPORTED_SUBSYSTEMS = frozenset(
    {
        "the DeepSeek-V4 KV pool",
        # hooks/models/minimax/kv_cache.py swaps in KunlunMiniMaxSparseKVPool.
        "the MiniMax sparse-index KV pool",
    }
)

_announced: set[str] = set()


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.reject_out_of_tree_path",
    type=HookType.REPLACE,
)
def reject_out_of_tree_path_kunlun(platform, *, subsystem: str) -> None:
    """Allow the pool families Kunlun implements; keep rejecting the rest."""
    if subsystem in _KUNLUN_SUPPORTED_SUBSYSTEMS:
        if subsystem not in _announced:
            _announced.add(subsystem)
            logger.info(
                "sglang-kunlun: %s has a Kunlun implementation; keeping the "
                "in-tree pool instead of rejecting the out-of-tree platform",
                subsystem,
            )
        return

    from sglang.srt.platforms.interface import reject_out_of_tree_path

    reject_out_of_tree_path(platform, subsystem=subsystem)


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool._scatter_mla_rows",
    type=HookType.REPLACE,
)
def dsa_scatter_mla_rows_kunlun(self, dst_buffer, loc, cache_k_nope, cache_k_rope):
    """Scatter MLA latent rows with the Kunlun in-place kernel.

    0.5.23's in-tree ``_scatter_mla_rows`` now short-circuits to a torch-native
    scatter when ``supports_triton`` is False (Kunlun), which copies the whole
    latent buffer on XPU and OOMs. Call ``kunlun_ops.set_mla_kv_buffer_triton``
    directly instead -- same kernel ``KunlunMLATokenToKVPool.set_mla_kv_buffer``
    uses. Tensors arrive already cast to ``store_dtype`` by ``_write_mla_kv_buffer``.
    KUNLUN_TODO: no DCP-sharded path yet; refuse loudly rather than mis-write.
    """
    import kunlun_ops

    dcp_span = self._write_loc_dcp_span
    if dcp_span != 1:
        raise NotImplementedError(
            "KUNLUN_TODO: DCP-sharded MLA latent scatter (span="
            f"{dcp_span}) has no Kunlun kernel path yet"
        )
    nope_2d = cache_k_nope.contiguous().view(cache_k_nope.shape[0], -1)
    if cache_k_rope is not None and cache_k_rope.numel() > 0:
        rope_2d = cache_k_rope.contiguous().view(cache_k_rope.shape[0], -1)
    else:
        rope_2d = cache_k_nope.new_empty((cache_k_nope.shape[0], 0))
    kunlun_ops.set_mla_kv_buffer_triton(dst_buffer, loc, nope_2d, rope_2d)


def _install_kunlun_ernie_mla_subpool() -> None:
    """ERNIE 5.5's latent MLA sub-pool needs an int8 index-K buffer on Kunlun.

    ``Ernie5_5TokenToKVPool`` builds its ``mla_kv_pool`` from the module-global
    ``_LayerSubsetDSATokenToKVPool``, whose ``index_k_with_scale_buffer_dtype``
    defaults to uint8; the Kunlun indexer kernels (``gather_paged_scale``)
    require int8 (Char). Rebind the global to a thin int8 subclass -- rebinding,
    not editing base, which is read-only. Base's subset-aware allocation (0-row
    placeholders for HCA layers) is kept.
    """
    try:
        import torch

        import sglang.srt.mem_cache.ernie55_memory_pool as ernie_mp
    except Exception as exc:  # pragma: no cover - ERNIE module not present
        logger.debug("ERNIE MLA sub-pool seam not installed: %s", exc)
        return

    base_cls = getattr(ernie_mp, "_LayerSubsetDSATokenToKVPool", None)
    if base_cls is None or getattr(base_cls, "_kunlun_int8_index", False):
        return

    class _KunlunLayerSubsetDSATokenToKVPool(base_cls):
        _kunlun_int8_index = True
        index_k_with_scale_buffer_dtype = torch.int8

        def __init__(self, *args, **kwargs):
            # Honor --kv-cache-dtype for the MLA latent sub-pool. Base hardcodes
            # bf16 here (it only branches fp8-vs-bf16 and ignores fp16), so under
            # ``--kv-cache-dtype fp16`` the latent silently stayed bf16. The
            # DSA/MLA kernels register both bf16 and fp16 (fwd_kvcache_mla_v2,
            # sparse_prefill_fwd_opt: out.dtype==q.dtype), so follow the request:
            # fp16 -> fp16, bfloat16 -> bf16, fp8/auto -> unchanged.
            import torch

            if kwargs.get("dtype") == torch.bfloat16:
                try:
                    from sglang.srt.runtime_context import get_server_args

                    kv = str(
                        getattr(get_server_args(), "kv_cache_dtype", "") or ""
                    ).lower()
                except Exception:
                    kv = ""
                if kv in ("fp16", "float16", "half"):
                    kwargs["dtype"] = torch.float16
            super().__init__(*args, **kwargs)

    ernie_mp._LayerSubsetDSATokenToKVPool = _KunlunLayerSubsetDSATokenToKVPool
    logger.info(
        "sglang-kunlun: ERNIE 5.5 MLA sub-pool uses an int8 index-K buffer "
        "(gather_paged_scale requires Char)"
    )


_install_kunlun_ernie_mla_subpool()
