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

Hooked on the configurator's own module binding, which is where the guard is
called from (``kv_cache_configurator`` does ``from sglang.srt.platforms.interface
import reject_out_of_tree_path`` at import time, and its call sites read that
module global). Note that the override is *not* confined to that module:
``HookRegistry._apply_target`` follows the ``setattr`` with ``_propagate_patch``,
which rebinds every module-level alias of the same function object -- including
``sglang.srt.platforms.interface.reject_out_of_tree_path`` itself. So the
delegation below must use the ``original_fn`` the AROUND hook is handed; a
re-import of the upstream name resolves to this wrapper again and recurses.
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
    type=HookType.AROUND,
)
def reject_out_of_tree_path_kunlun(
    original_fn, platform, *, subsystem: str
) -> None:
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

    # Carry the upstream guard through the call: re-importing it by name
    # resolves to this wrapper (see the module docstring) and recurses.
    return original_fn(platform, subsystem=subsystem)
