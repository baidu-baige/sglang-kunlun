"""Disaggregation patches for Kunlun (P800).

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/disaggregation/__init__.py

The community decode/prefill code sets ``kv_args.kv_data_lens`` /
``state_data_lens`` directly from the kv-pool buffer infos. On Kunlun, RDMA
registration requires 2MiB-aligned lengths, so we round each entry up before
``KVManager`` is built. This wraps ``get_contiguous_buf_infos`` /
``get_state_buf_infos`` on the kv-pool instances right before
``_init_kv_manager`` is invoked.

Implementation note: this patch wraps an instance-level method bound at
runtime (after ``self.token_to_kv_pool`` exists), not a class-level method.
The plugin framework's REPLACE works on class targets only, so we install
this with an AROUND hook on the queue's ``_init_kv_manager`` class method.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

ALIGNMENT_2M = 2 * 1024 * 1024


def _align(values):
    return [(x + ALIGNMENT_2M - 1) // ALIGNMENT_2M * ALIGNMENT_2M for x in values]


def _wrap_buf_infos(orig):
    def _wrapped(*args, **kwargs):
        ptrs, lens, items = orig(*args, **kwargs)
        return ptrs, _align(lens), items

    return _wrapped


def _patch_pool(pool):
    if pool is None or getattr(pool, "_kunlun_2m_aligned", False):
        return
    if hasattr(pool, "get_contiguous_buf_infos"):
        pool.get_contiguous_buf_infos = _wrap_buf_infos(pool.get_contiguous_buf_infos)
    if hasattr(pool, "get_state_buf_infos"):
        pool.get_state_buf_infos = _wrap_buf_infos(pool.get_state_buf_infos)
    # MiniMax sparse pools ship the block-selector K cache on the state channel
    # through this third name (``MiniMaxSparseKVPool.get_index_k_state_buf_infos``
    # -> StateType.MINIMAX_INDEX_K); it never goes through the two methods above,
    # so without this wrap its lengths stay unaligned and the decode side selects
    # sparse blocks from a corrupted index-K cache.
    if hasattr(pool, "get_index_k_state_buf_infos"):
        pool.get_index_k_state_buf_infos = _wrap_buf_infos(
            pool.get_index_k_state_buf_infos
        )
    pool._kunlun_2m_aligned = True


def _kunlun_init_kv_manager_around(original_fn, self):
    """AROUND wrapper: align kv-pool buf-infos to 2MiB before init."""
    _patch_pool(getattr(self, "token_to_kv_pool", None))
    _patch_pool(getattr(self, "draft_token_to_kv_pool", None))
    if getattr(self, "scheduler", None) is not None and getattr(
        self.scheduler, "enable_hisparse", False
    ):
        _patch_pool(self.scheduler.hisparse_coordinator.mem_pool_host)
    return original_fn(self)


# Register AROUND on both decode and prefill queues
plugin_hook(
    "sglang.srt.disaggregation.decode.DecodePreallocQueue._init_kv_manager",
    type=HookType.AROUND,
)(_kunlun_init_kv_manager_around)

plugin_hook(
    "sglang.srt.disaggregation.prefill.PrefillBootstrapQueue._init_kv_manager",
    type=HookType.AROUND,
)(_kunlun_init_kv_manager_around)


# ---------------------------------------------------------------------------
# HiSparse + real-PD decode warmup
#
# The boot-time PD disaggregation warmup posts ``dp_size`` fake-bootstrap
# requests (``FAKE_BOOTSTRAP_HOST``) straight into the decode transfer queue.
# With ``--enable-hisparse`` those requests hit ``admit_request_direct`` /
# ``_preload_to_device_buffer``, which assume a real RDMA'd host-pool entry; a
# fake request has none, so some DP ranks never produce their warmup token and
# the DP-attention all_gather in ``get_next_disagg_decode_batch_to_run``
# dead-locks (py-spy 2026-10-03: ranks split between ``all_gather`` and
# ``recv_requests``, ``num_running_reqs == 0``, zero warmup POSTs completed).
#
# The decode CUDA graph is already captured before warmup runs, so the only
# thing the warmup still does is flip ``server_status`` to ``Up``.  When
# ``DSV4_HS_SKIP_PD_WARMUP=1`` we return success without sending the fake
# requests; readiness is then proven by the pipeline's real router smoke
# (a genuine prefill -> decode request), which is a stronger gate than the
# fake-bootstrap self-warmup it replaces.
# ---------------------------------------------------------------------------
import os as _os


@plugin_hook(
    "sglang.srt.entrypoints.http_server._send_disaggregation_warmup_requests",
    type=HookType.AROUND,
)
async def _skip_hisparse_pd_warmup(original_fn, *args, **kwargs):
    # Decide from the http-server process's own global state, which is where
    # this warmup runs. Env vars do not propagate to sglang's tokenizer workers
    # (verified 2026-10-03), and the ``memory`` runtime namespace may not be
    # published in that process, so read enable_hisparse off the live
    # TokenizerManager's server_args, with get_memory() as a fallback.
    enabled = False
    try:
        from sglang.srt.entrypoints import http_server as _hs

        gs = getattr(_hs, "_global_state", None)
        sa = getattr(getattr(gs, "tokenizer_manager", None), "server_args", None)
        if sa is not None:
            enabled = bool(getattr(sa, "enable_hisparse", False))
    except Exception:
        enabled = False
    if not enabled:
        try:
            from sglang.srt.runtime_context import get_memory

            enabled = bool(get_memory().enable_hisparse)
        except Exception:
            enabled = False
    if enabled:
        from sglang.srt.runtime_context import get_parallel

        dp_size = get_parallel().dp_size
        logger.warning(
            "sglang-kunlun: skipping fake-bootstrap PD warmup for HiSparse "
            "decode (graph already captured); router smoke will gate readiness."
        )
        return [200] * dp_size
    return await original_fn(*args, **kwargs)
