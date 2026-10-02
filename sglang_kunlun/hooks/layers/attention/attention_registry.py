"""Register the ``kunlun`` attention backend factory.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/attention_registry.py
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


try:
    from sglang.srt.layers.attention.attention_registry import register_attention_backend
except Exception as e:  # pragma: no cover - upstream layout drift
    register_attention_backend = None
    logger.warning("Could not import register_attention_backend: %s", e)


if register_attention_backend is not None:
    @register_attention_backend("kunlun")
    def create_kunlun_attention_backend(runner):
        """Lazy-construct the generic Kunlun attention backend."""
        from .kunlun_backend import KunlunAttentionBackend

        return KunlunAttentionBackend(runner)

    @register_attention_backend("kunlun_compressed")
    def create_kunlun_compressed_attention_backend(runner):
        """Lazy-construct the Kunlun DeepSeek-V4 attention backend."""
        from .kunlun_deepseek_v4_backend import KunlunDeepseekV4AttnBackend

        return KunlunDeepseekV4AttnBackend(runner)

    @register_attention_backend("kunlun_nsa")
    def create_kunlun_nsa_attention_backend(runner):
        """Lazy-construct the Kunlun DeepSeek sparse attention (DSA/NSA) backend."""
        from .kunlun_nsa_backend import KunlunDSAAttnBackend

        return KunlunDSAAttnBackend(runner)

    # Upstream keys a lot of DSA-specific plumbing on the *name* "dsa"/"nsa"
    register_attention_backend("dsa")(create_kunlun_nsa_attention_backend)
    register_attention_backend("nsa")(create_kunlun_nsa_attention_backend)

    logger.info(
        "Registered 'kunlun', 'kunlun_compressed' and 'kunlun_nsa' attention "
        "backends; 'dsa'/'nsa' now resolve to KunlunDSAAttnBackend"
    )


def install_kunlun_mla_forward_dispatch() -> None:
    """Teach the DeepSeek/K3 forward-method dispatch about the ``kunlun`` name.

    ``AttentionBackendRegistry.get_handler`` falls back to the *triton* handler
    for unregistered names
    (``models/deepseek_common/attention_backend_handler.py:49``), and that
    handler only leaves absorbed MLA when there is no prefix at all
    (``:224-228`` ``sum(extend_prefix_lens_cpu) == 0``). So the second chunk of
    any prompt longer than --chunked-prefill-size arrived as absorbed MLA with
    paged prefix history, which ``KunlunK3MLAAttnBackend._forward_extend_mla``
    correctly refuses -- the "query LOD total != KV LOD total" raise.

    Registering an explicit handler routes that case to ``MHA_CHUNKED_KV``,
    which is what sets ``ForwardBatch.attn_attend_prefix_cache`` and reaches
    ``_forward_extend_mla_chunked_prefix``.

    Two deliberate deviations from upstream's ``_handle_attention_backend``:

    * The zero-prefix case stays on ``AttnForwardMethod.MHA``, matching what the
      triton fallback already did and what has been verified working, rather
      than upstream's ``MHA_CHUNKED_KV``.
    * ``chunked_prefix_cache_threshold`` is not consulted. Upstream sends
      short-prefix batches to absorbed MLA as an optimisation; on Kunlun that
      path cannot read paged prefix history *at any length*, so every non-zero
      prefix must go chunked.

    Called from the backend-build hook rather than at import time: this module is
    imported by ``launch.py`` before the model packages are safe to pull in, and
    ``deepseek_common`` would be a circular import there.
    """
    try:
        # Everything comes from the handler module's own namespace on purpose:
        # ``is_in_*_cuda_graph`` live under
        # ``model_executor/runner_backend_utils/`` and ``AttnForwardMethod`` under
        # ``attention_forward_methods/forward_methods``, so importing them from
        # their defining modules would be one more set of paths to keep in sync.
        from sglang.srt.models.deepseek_common.attention_backend_handler import (
            AttentionBackendRegistry,
            AttnForwardMethod,
            _dispatch_mla_subtype,
            is_in_breakable_cuda_graph,
            is_in_tc_piecewise_cuda_graph,
            mla_use_prefill_cp,
        )
    except Exception as exc:  # pragma: no cover - upstream layout drift
        logger.warning("Could not install Kunlun MLA forward dispatch: %s", exc)
        return

    if AttentionBackendRegistry._handlers.get("kunlun") is not None:
        return

    def handle_attention_kunlun(attn, forward_batch):
        """Forward-method choice for the Kunlun MLA backends."""
        if is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph():
            return AttnForwardMethod.MLA
        if mla_use_prefill_cp(forward_batch):
            return _dispatch_mla_subtype(attn, forward_batch)
        if forward_batch.forward_mode.is_extend_without_speculative():
            prefix_lens = forward_batch.extend_prefix_lens_cpu or []
            if sum(prefix_lens) == 0:
                return AttnForwardMethod.MHA
            if not attn.disable_chunked_prefix_cache:
                return AttnForwardMethod.MHA_CHUNKED_KV
        return _dispatch_mla_subtype(attn, forward_batch)

    for name in ("kunlun", "kunlun_compressed"):
        AttentionBackendRegistry.register(name, handle_attention_kunlun)
    logger.info("Registered Kunlun MLA forward-method dispatch for %s", "kunlun")


@plugin_hook(
    "sglang.srt.layers.attention.attention_registry.attn_backend_wrapper",
    type=HookType.AROUND,
)
def kunlun_attn_backend_wrapper(original_fn, runner, full_attn_backend):
    """Select the Kunlun KDA backend for Kimi-K3 / Kimi-Linear hybrids.

    Upstream picks the linear-attention backend inline
    (``attention_registry.py:496-506``) with hardware branches for NPU and XPU;
    Kunlun cannot use either because it reports ``device_type="cuda"``, so
    ``is_xpu()`` is False and it would fall through to the CUDA
    ``KDAAttnBackend``.  This is the single integration point that redirects it
    -- the operator differences live in ``KunlunKDAAttnBackend``, not in hooks.

    Everything else (GDN, Mamba2, MiniMax, Dots3, NSA, plain MLA) is left to
    upstream.
    """
    from sglang.srt.configs.hybrid_arch import kimi_linear_config

    # Safe to do here (unlike at import time): the model packages are loaded by
    # the time a backend is being built. Idempotent.
    install_kunlun_mla_forward_dispatch()

    if kimi_linear_config(runner.model_config) is None:
        return original_fn(runner, full_attn_backend)

    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    from .kunlun_k3_mla_backend import KunlunK3MLAAttnBackend
    from .kunlun_kda_backend import KunlunKDAAttnBackend

    cfg = kimi_linear_config(runner.model_config)
    if runner.is_draft_worker:
        # Mirrors upstream: the draft worker is assumed full-attention only.
        full_attn_layers = [0]
    else:
        full_attn_layers = cfg.full_attention_layer_ids

    # Upstream publishes the per-phase linear-attn kernel choice onto the runner
    # *inside* this same function, before constructing the KDA backend
    # (``attention_registry.py:429-433``). Because this hook never delegates to
    # ``original_fn`` for Kimi-Linear/K3, that assignment has to be reproduced
    # here: ``KDAAttnBackend.__init__`` reads
    # ``model_runner.linear_attn_backends`` at ``linear/kda_backend.py:400`` and
    # otherwise raises AttributeError. ``prefill_default`` stays None -- upstream
    # only computes a non-None one for hybrid-GDN models
    # (``flashinfer_gdn_prefill_default``), and K3 is not one.
    from sglang.kernels.ops.attention.fla.utils import check_environments
    from sglang.srt.layers.attention.linear.utils import resolve_linear_attn_backends

    check_environments()
    runner.linear_attn_backends = resolve_linear_attn_backends(prefill_default=None)

    # The full-attention half arrives as the generic KunlunAttentionBackend,
    # whose MLA branches still target the FA3/FA4 kernels. K3's latent cache
    # needs the native kunlun_ops MLA path, so swap in the K3 subclass rather
    # than editing the shared backend (DeepSeek-V4 rides on those same
    # branches).
    k3_full_attn_backend = KunlunK3MLAAttnBackend(runner)

    logger.info(
        "Kimi-Linear/K3 detected: KunlunKDAAttnBackend for %d linear layers, "
        "KunlunK3MLAAttnBackend for %d full-attention layers (was %s)",
        len(cfg.linear_layer_ids),
        len(full_attn_layers),
        type(full_attn_backend).__name__,
    )
    return HybridLinearAttnBackend(
        k3_full_attn_backend, KunlunKDAAttnBackend(runner), full_attn_layers
    )

