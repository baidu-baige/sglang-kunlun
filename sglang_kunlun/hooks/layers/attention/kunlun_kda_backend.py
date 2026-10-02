"""Kunlun (P800/XPU) implementation of Kimi Delta Attention.

Why a backend subclass instead of monkey-patching ``TritonKDAKernel``:
the kimi-k3 branch REPLACE-hooked ``TritonKDAKernel.decode`` / ``.extend`` and
``kda_fused_decode.covered``.  That works but it is invisible at the call site,
it fights the dispatcher's own backend selection, and it cannot express the
things a hardware backend legitimately needs to change (cuda-graph buffers,
which fast paths exist at all, what happens under speculative decode).  Upstream
already models "this hardware runs KDA differently" as a ``KDAAttnBackend``
subclass -- see ``srt/hardware_backend/npu/attention/ascend_kda_backend.py``.
This file follows that precedent.

What stays in the base class: metadata construction, the mamba cache pool, the
prefix-cache track plumbing, speculative rollback bookkeeping, and every
non-operator control-flow decision.  What lives here: the conv1d and recurrent
operator calls, and the fast paths that must not be attempted on Kunlun.

Kunlun advertises ``device_type="cuda"`` (``platform/device.py``), so every
upstream ``is_cuda()`` gate answers True.  That means the nvcc-JIT K3 fast paths
(``kda_fused_decode``, ``kimi_k3.mla_output_gate``, ...) are *offered* to us and
must be declined explicitly rather than filtered out by the platform check.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def _is_capturing() -> bool:
    """True while a cuda-graph capture is recording on the current stream."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # pragma: no cover - no accelerator in unit tests
        return False


class KunlunKDAKernel:
    """KDA recurrence backed by the XSpeedGate wheel.

    Presents the same call surface as ``TritonKDAKernel`` so the base
    dispatcher can hold it without knowing the difference.  Only the three
    entry points KDA actually reaches are implemented; the rest raise rather
    than silently returning something plausible.
    """

    # No packed (one-launch, T==1) decode op exists on XPU yet, so the base
    # backend must take the split q/k/v path.
    supports_packed_decode: bool = False
    # The fused conv+gating chain-verify kernel is CUDA-only; declining it here
    # keeps ``KDAAttnBackend.__init__`` from wiring ``_fused_chain_verify_fn``.
    supports_fused_chain_verify: bool = False
    # Reported to the base class' graph capability probe.
    supports_safe_gate: bool = False

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound=None,
        **kwargs,
    ) -> torch.Tensor:
        """One recurrent step per request."""
        del kwargs
        from sglang_kunlun.kernels.k3_recurrent import fused_recurrent_kda_decode

        return fused_recurrent_kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            lower_bound=lower_bound,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log=None,
        dt_bias=None,
        lower_bound=None,
        return_intermediate_states: bool = False,
        **kwargs,
    ):
        """Chunked/varlen prefill recurrence."""
        if kwargs.get("is_spec_decode"):
            # draft_extend_v2 must stay rollback-able (the caller may discard
            # the result), and the fused op commits ssm_states in place.
            raise NotImplementedError(
                "Kunlun KDA extend does not support draft_extend_v2 yet: the "
                "fused recurrence commits ssm_states in place, which is not "
                "rollback-able. Run without --speculative-algorithm."
            )

        from sglang_kunlun.kernels.k3_recurrent import (
            fused_recurrent_kda_extend,
            recurrent_kda_extend,
        )

        if return_intermediate_states:
            # 0.5.19 moved this out of ServerArgs into the runtime-context
            # helpers (srt/runtime_context.py:1912).
            from sglang.srt.runtime_context import mamba_cache_chunk_size

            return recurrent_kda_extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                return_intermediate_states=True,
                checkpoint_every=mamba_cache_chunk_size(),
            )

        return fused_recurrent_kda_extend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            sequence_lengths=kwargs.get("extend_seq_lens_cpu"),
        )

    def packed_decode(self, *args, **kwargs):
        """Reject packed decode explicitly.

        Kunlun has no packed KDA decode op and ``supports_packed_decode`` is
        False, so the dispatcher must never route here; this raises rather than
        silently dropping the call if it does.
        """
        raise NotImplementedError(
            "Kunlun KDA has no packed decode op; supports_packed_decode is False "
            "so the backend should never call this."
        )

    def target_verify(self, *args, **kwargs):
        """Reject speculative target verification.

        Not implemented: it needs gamma+1 sequential recurrent steps writing
        each post-state into
        ``intermediate_states_buffer[intermediate_state_indices[b], step]``,
        with the *extend* beta convention (beta pre-activated). Until then, run
        K3 without ``--speculative-algorithm``.
        """
        raise NotImplementedError(
            "Kunlun KDA target_verify is not implemented. It needs gamma+1 "
            "sequential recurrent steps writing each post-state into "
            "intermediate_states_buffer[intermediate_state_indices[b], step], "
            "with the *extend* beta convention (beta pre-activated). Until then "
            "run K3 without --speculative-algorithm."
        )


class KunlunKDAAttnBackend(KDAAttnBackend):
    """KDA on Kunlun XPU."""

    # The verify kernels do not exist on Kunlun yet, so ragged verify graphs
    # must not be advertised.
    supports_ragged_verify_graph: bool = False
    # Base class reads this on the extra-buffer track path; KDA stores conv
    # states as [kernel-1, dim] and the base __init__ already exposes the
    # transposed shape, so nothing to change here.

    def __init__(self, model_runner):
        super().__init__(model_runner)

        kernel = KunlunKDAKernel()
        dispatcher = self.kernel_dispatcher
        dispatcher.triton_kernel = kernel
        dispatcher.decode_kernel = kernel
        dispatcher.extend_kernel = kernel
        dispatcher.verify_kernel = kernel
        dispatcher.supports_packed_decode = False
        if getattr(self, "_fused_chain_verify_fn", None) is not None:
            logger.warning(
                "Disabling the fused KDA chain-verify kernel: no Kunlun "
                "implementation. Unset SGLANG_OPT_FUSED_KDA_VERIFY."
            )
            self._fused_chain_verify_fn = None

        self._fused_override_notice = None

        logger.info(
            "KunlunKDAAttnBackend active: conv1d + recurrence on kunlun_ops / "
            "xspeedgate_ops, packed decode off, verify unimplemented."
        )

    # ------------------------------------------------------------------ decode

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """Decode step: conv1d update, then one recurrent step."""
        del kwargs
        from sglang_kunlun.kernels.causal_conv1d import (
            kda_causal_conv1d_update_kunlun,
        )

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        if layer_cache.replayssm_d is not None:
            raise NotImplementedError(
                "Kunlun KDA does not support ReplaySSM (--enable-linear-replayssm): "
                "the ring commit path has no Kunlun kernel."
            )

        # The pool stores [slots, state_len, dim]; the op contract wants
        # [slots, dim, state_len], matching the upstream triton call.
        qkv = kda_causal_conv1d_update_kunlun(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        core_attn_out = self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            lower_bound=layer.lower_bound,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )
        return core_attn_out

    # ------------------------------------------------------------------ extend

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """Prefill / extend, including chunked prefill."""
        del kwargs
        if forward_batch.forward_mode.is_target_verify():
            return self._forward_target_verify(layer, forward_batch, mixed_qkv, a, b)

        from sglang_kunlun.kernels.causal_conv1d import (
            kda_causal_conv1d_prefill_kunlun,
        )

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)
        ssm_states = mamba_cache_params.temporal

        if forward_batch.extend_prefix_lens is None:
            raise RuntimeError(
                "extend_prefix_lens cannot be None in non-TARGET_VERIFY mode."
            )
        has_initial_state = forward_batch.extend_prefix_lens > 0

        if self.forward_metadata.has_mamba_track_mask:
            # Prefix-cache track snapshot. KDA's pool is [kernel-1, dim] so raw
            # rows of the pre-conv [tokens, dim] input index straight in.
            mamba_cache_params.conv[0][
                self.forward_metadata.conv_states_mask_indices
            ] = mixed_qkv[self.forward_metadata.track_conv_indices]

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
        q_w, k_w, v_w = layer.conv_weights.split(splits, dim=0)
        q_state, k_state, v_state = conv_states.split(splits, dim=-2)
        if layer.bias is not None:
            q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
        else:
            q_bias, k_bias, v_bias = None, None, None

        conv_args = dict(
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
        )
        q = kda_causal_conv1d_prefill_kunlun(
            q, q_w, q_bias, q_state, **conv_args
        ).transpose(0, 1)
        k = kda_causal_conv1d_prefill_kunlun(
            k, k_w, k_bias, k_state, **conv_args
        ).transpose(0, 1)
        v = kda_causal_conv1d_prefill_kunlun(
            v, v_w, v_bias, v_state, **conv_args
        ).transpose(0, 1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        track_ssm = self.forward_metadata.has_mamba_track_mask
        core_attn_out = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.lower_bound,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            is_spec_decode=forward_batch.forward_mode.is_draft_extend_v2(),
            return_intermediate_states=track_ssm,
            track_ssm_h_src=(
                self.forward_metadata.track_ssm_h_src if track_ssm else None
            ),
        )
        if track_ssm:
            core_attn_out, h = core_attn_out
            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, self.forward_metadata
            )

        if (
            getattr(self, "accept_lens_pool", None) is not None
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # Fused-accept staging, mirrored from the base class. Only reachable
            # with a flashinfer verify backend, which Kunlun does not select --
            # kept so the branch does not silently diverge if that changes.
            slots = cache_indices.to(torch.int64)
            intermediate_ssm = mamba_cache_params.intermediate_ssm
            intermediate_ssm[slots, 0] = ssm_states[slots].to(intermediate_ssm.dtype)
            self.accept_lens_pool[slots] = 1

        return core_attn_out

    # -------------------------------------------------------------- speculative

    def _forward_target_verify(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        raise NotImplementedError(
            "Kunlun KDA target_verify is not implemented. Three pieces are "
            "missing: (1) intermediate_conv_window / intermediate_state_indices "
            "support in causal_conv1d_update_kunlun, (2) a gamma+1 step verify "
            "recurrence, (3) torch scatters for "
            "scatter_mamba_states_after_mtp_verify and "
            "fused_conv_window_scatter_with_mask. Run K3 without "
            "--speculative-algorithm until those land."
        )