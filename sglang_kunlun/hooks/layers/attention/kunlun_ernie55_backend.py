"""Kunlun ERNIE 5.5 attention backend.

ERNIE 5.5 is DeepSeek-V4 compressed MLA plus ERNIE deltas, so this mirrors base
``Ernie5_5AttnBackend(DeepseekV4AttnBackend)`` one level down the Kunlun tree:
subclass ``KunlunDeepseekV4AttnBackend`` and override only the ERNIE deltas.

- HCA layers (``compress_ratio == 128``) use the compressed backend unchanged.
- MLA layers (``compress_ratio == -2``) run a DSA sub-backend over the latent
  sub-pool; the model enters them inside ``mla_layer_context()``, which publishes
  that sub-backend as the forward-context attn backend.
- ``index_topk=2048`` widened in ``KunlunErnie55AttnMetadata``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DEFAULT_INDEX_TOPK
from sglang.srt.mem_cache.ernie55_memory_pool import DSA_PAGE_SIZE
from sglang.srt.model_executor.forward_context import (
    forward_context,
    get_forward_context,
)

from .kunlun_deepseek_v4_backend import (
    KunlunDeepseekV4AttnBackend,
    KunlunDeepseekV4MultiStepBackend,
    KunlunDSV4AttnMetadata,
)
from .kunlun_nsa_backend import KunlunDSAAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class _MlaSubBackendView:
    """Reads through to the wrapped object except for the replaced fields.

    Lets the MLA sub-backend see the latent sub-pool, its 64-token page size and
    the ``hybrid_mla_*`` model config, with everything else falling through to
    the real ``model_runner`` / ``model_config``.
    """

    def __init__(self, base, **replaced):
        self._base = base
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._base, name)


class KunlunErnie55AttnMetadata(KunlunDSV4AttnMetadata):
    """Kunlun DSV4 core-attn metadata widened for ERNIE 5.5's index_topk=2048."""

    def _supported_index_topks(self) -> tuple:
        return (512, 1024, 2048)


class KunlunErnie55MlaAttnBackend(KunlunDSAAttnBackend):
    """DSA sub-backend for ERNIE 5.5's MLA layers, on the Kunlun sparse kernels.

    KUNLUN_TODO(ERNIE55): phase-1 runs plain DSA top-k over the full context.
    Base's sliding-window top-k (``top-k(prefix) + last window``) needs a Kunlun
    equivalent of FlashMLA's ``append_window_to_topk``; it only matters for
    sequences longer than ``index_topk`` (shorter ones select every token, so
    the result is already exact).
    """

    def __init__(self, model_runner: "ModelRunner", *, window: int, **kwargs):
        self.window = window  # reserved for the future windowed top-k
        super().__init__(model_runner, **kwargs)

    @contextlib.contextmanager
    def read_rows(self, rows, num_rows):
        """Enter the row-read scope without pruning any row.

        Every row is always computed here, so unlike base this is a no-op.
        KUNLUN_TODO(ERNIE55): prune rows whose output is unread, as base does.
        """
        yield

    _SINK_WARNED = False

    def _prepare_attn_sink(self, attn_sink):
        """Pass ERNIE's learnable MLA sink through as fp32.

        The parent folds it into the output post-hoc from the kernel's
        natural-log normalizer (prefill lse / decode max_logits+p_sums), so the
        sink is no longer dropped. Cast to fp32 to match the lse dtype.
        """
        if attn_sink is None:
            return None
        if not KunlunErnie55MlaAttnBackend._SINK_WARNED:
            KunlunErnie55MlaAttnBackend._SINK_WARNED = True
            logger.info(
                "ERNIE55: folding the MLA learnable attention sink post-hoc "
                "(sigmoid(lse - sink))."
            )
        return attn_sink.to(torch.float32)

    def _cast_to_cache_dtype(self, args, kwargs):
        """Cast float attention inputs to the latent cache dtype.

        The MLA latent cache is bf16 (base pins it); the Kunlun DSA kernels
        reject a mixed q/kv dtype, so under an fp16 model the attention inputs
        (q, k, v, q_rope, k_rope) are cast to the cache dtype. Returns the
        rewritten (args, kwargs) and the original input dtype to restore on out.
        """
        cache_dtype = self.token_to_kv_pool.dtype
        orig_dtype = (
            args[0].dtype if args and isinstance(args[0], torch.Tensor) else None
        )

        def cast(t):
            """Cast a float tensor to the cache dtype, else pass it through."""
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.dtype != cache_dtype
            ):
                return t.to(cache_dtype)
            return t

        args = tuple(cast(a) for a in args)
        for name in ("q_rope", "k_rope"):
            if name in kwargs:
                kwargs[name] = cast(kwargs[name])
        return args, kwargs, orig_dtype

    def forward_extend(self, *args, attn_sink=None, **kwargs):
        """Run the DSA extend kernel on cache-dtype inputs, then restore out dtype."""
        args, kwargs, orig_dtype = self._cast_to_cache_dtype(args, kwargs)
        out = super().forward_extend(
            *args, attn_sink=self._prepare_attn_sink(attn_sink), **kwargs
        )
        if orig_dtype is not None and out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out

    def forward_decode(self, *args, attn_sink=None, **kwargs):
        """Run the DSA decode kernel on cache-dtype inputs, then restore out dtype."""
        args, kwargs, orig_dtype = self._cast_to_cache_dtype(args, kwargs)
        out = super().forward_decode(
            *args, attn_sink=self._prepare_attn_sink(attn_sink), **kwargs
        )
        if orig_dtype is not None and out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out


class KunlunErnie55AttnBackend(KunlunDeepseekV4AttnBackend):
    """ERNIE 5.5: reuse the Kunlun DSV4 compressed-MLA backend for the HCA
    layers, and drive the MLA layers through a DSA sub-backend selected per
    layer by ``mla_layer_context`` -- mirroring base ``Ernie5_5AttnBackend``."""

    core_attn_metadata_cls = KunlunErnie55AttnMetadata

    def __init__(self, model_runner: "ModelRunner", **kwargs):
        super().__init__(model_runner, **kwargs)
        # No MLA layers in range (e.g. a layer-reduced model that is pure HCA)
        # means a plain DSV4 pool with no latent sub-pool; the model never enters
        # mla_layer_context, so run without the sub-backend.
        mla_kv_pool = getattr(self.token_to_kv_pool, "mla_kv_pool", None)
        if mla_kv_pool is None:
            self.mla_attn_backend = None
            return
        # Base keeps HCA metadata on a width it accepts (no C4 layer here).
        self.index_topk = DEFAULT_INDEX_TOPK

        hf_config = model_runner.model_config.hf_config
        mla_model_config = _MlaSubBackendView(
            model_runner.model_config,
            num_attention_heads=hf_config.hybrid_mla_num_attention_heads,
            qk_nope_head_dim=hf_config.hybrid_mla_qk_nope_head_dim,
            qk_rope_head_dim=hf_config.hybrid_mla_qk_rope_head_dim,
            kv_lora_rank=hf_config.hybrid_mla_kv_lora_rank,
        )
        self.mla_attn_backend = KunlunErnie55MlaAttnBackend(
            _MlaSubBackendView(
                model_runner,
                model_config=mla_model_config,
                token_to_kv_pool=mla_kv_pool,
                page_size=DSA_PAGE_SIZE,
            ),
            window=hf_config.sliding_window,
            # A multi-step draft step reads its own slice of the shared KV locs.
            speculative_step_id=kwargs.get("speculative_step_id", 0),
            topk=kwargs.get("topk", 0),
            speculative_num_steps=kwargs.get("speculative_num_steps", 0),
        )
        self.mla_attn_backend.prefill_attention_backend_str = "dsa"
        self.mla_attn_backend.decode_attention_backend_str = "dsa"

    @contextlib.contextmanager
    def mla_layer_context(self):
        """Run an MLA layer: publish the MLA sub-backend as the forward-context
        backend, which the DSA attention and indexer read directly."""
        if self.mla_attn_backend is None:
            yield
            return
        with forward_context(
            dataclasses.replace(
                get_forward_context(), attn_backend=self.mla_attn_backend
            )
        ):
            yield

    @contextlib.contextmanager
    def read_rows(self, rows, num_rows):
        """Delegate the row-read scope to the MLA sub-backend, if any."""
        if self.mla_attn_backend is None:
            yield
            return
        with self.mla_attn_backend.read_rows(rows, num_rows):
            yield

    def _dispatch_sub(self, name: str, *args, **kwargs) -> None:
        """Forward a metadata/graph-lifecycle call to the MLA sub-backend if it
        exists and exposes the method (its surface tracks the Kunlun DSA backend)."""
        if self.mla_attn_backend is None:
            return
        fn = getattr(self.mla_attn_backend, name, None)
        if fn is not None:
            fn(*args, **kwargs)

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Build HCA metadata, then the MLA sub-backend's metadata."""
        super().init_forward_metadata(forward_batch)
        self._dispatch_sub("init_forward_metadata", forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate graph state for both the HCA and MLA sub-backends."""
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._dispatch_sub("init_cuda_graph_state", max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self, forward_batch: "ForwardBatch", in_capture: bool = False
    ) -> None:
        """Refresh out-graph metadata on both the HCA and MLA sub-backends."""
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self._dispatch_sub(
            "init_forward_metadata_out_graph", forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch") -> None:
        """Refresh in-graph metadata on both the HCA and MLA sub-backends."""
        super().init_forward_metadata_in_graph(forward_batch)
        self._dispatch_sub("init_forward_metadata_in_graph", forward_batch)

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self, forward_batch: "ForwardBatch"
    ):
        """Reject breakable prefill CUDA graphs.

        The MLA layers' metadata would not be refreshed on replay (base
        ``Ernie5_5AttnBackend`` raises the same way).
        """
        raise NotImplementedError(
            "ERNIE 5.5 Flash does not support breakable prefill CUDA graphs"
        )

    def shared_read_ends(self, fm):
        """Return the later read end of the HCA path and the MLA sub-backend."""
        from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds

        ends = [super().shared_read_ends(fm)]
        sub = getattr(self.mla_attn_backend, "shared_read_ends", None)
        if sub is not None:
            ends.append(sub(fm))
        return SharedReadEnds.max_of(tuple(ends))


class KunlunErnie55MultiStepBackend(KunlunDeepseekV4MultiStepBackend):
    """EAGLE self-draft multi-step backend for ERNIE 5.5.

    Same Kunlun DSV4 multi-step orchestration (per-step out/in-graph metadata),
    but each draft step is a ``KunlunErnie55AttnBackend`` so the MTP/NextN
    draft's MLA layer runs through the HCA + MLA/DSA dual-backend. Mirrors base
    ``Ernie5_5MultiStepBackend``.
    """

    def __init__(
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int
    ):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        # Same container init as KunlunDeepseekV4MultiStepBackend, then build
        # ERNIE step backends instead of DSV4 ones.
        DeepseekV4AttnBackend.__init__(self, model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            KunlunErnie55AttnBackend(
                model_runner,
                speculative_step_id=i,
                topk=topk,
                speculative_num_steps=speculative_num_steps,
            )
            for i in range(speculative_num_steps)
        ]
        self._mtp_probe_replay_calls = 0
