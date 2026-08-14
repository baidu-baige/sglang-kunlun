"""Kunlun-owned DeepSeek V4 production precision hooks without diagnostics."""

from __future__ import annotations

import torch
from kunlun_ops import hc_post_kunlun_impl
from typing import Optional
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.kernel_ops import dsv4_mqa_wo_a_einsum_kunlun


def _dsv4_dump(self, name, value):
    callback = getattr(self, "_dsv4_tensor_dump_callback", None)
    if callback is not None and isinstance(value, torch.Tensor):
        callback(name, value)


_FP16_DTYPE_NAMES = frozenset(("fp16", "float16", "half"))


# Golden keeps this hook disabled so the upstream Torch hc_pre runs and the
# mHC post/comb stay fp32; the Kunlun fused hc_pre returns fp16.
# @plugin_hook(
#     "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_pre",
#     type=HookType.REPLACE,
# )
def hc_pre_reference_sinkhorn_kunlun(
    self,
    x,
    hc_fn,
    hc_scale,
    hc_base,
    norm=None,
    forward_batch=None,
):
    """Use the Golden Kunlun fused MHC-pre contract."""
    del norm, forward_batch
    dtype = x.dtype
    if self.hc_mult != 4:
        raise ValueError("Kunlun hc_pre supports hc_mult=4 only")

    if x.shape[0] == 0:
        y = torch.empty((0, x.shape[-1]), dtype=dtype, device=x.device)
        post = torch.empty((0, self.hc_mult), dtype=torch.float32, device=x.device)
        comb = torch.empty(
            (0, self.hc_mult, self.hc_mult), dtype=torch.float32, device=x.device
        )
        return y, post, comb, False

    y, post, comb = torch.ops.xspeedgate_ops.hc_pre(
        x.contiguous(),
        hc_fn.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        rms_eps=self.rms_norm_eps,
        hc_pre_eps=self.hc_eps,
        hc_sinkhorn_eps=self.hc_eps,
        mhc_post_mult_value=2.0,
        sinkhorn_iters=self.hc_sinkhorn_iters,
    )
    return y, post, comb, False


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post",
    type=HookType.REPLACE,
)
def hc_post_kunlun(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """Use the Golden Kunlun fused MHC-post contract."""
    if x.shape[0] == 0:
        return torch.empty(
            (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
        )

    assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
    assert post.shape == (x.shape[0], self.hc_mult)
    assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)
    out = torch.empty_like(residual)
    hc_post_kunlun_impl(
        post.contiguous(),
        x.contiguous(),
        comb.contiguous(),
        residual.contiguous(),
        out,
        x.shape[0],
        self.hc_mult,
        x.shape[-1],
    )
    return out


@plugin_hook(
    "sglang.srt.models.deepseek_v2.MoEGate.forward",
    type=HookType.AROUND,
)
def moe_gate_forward_half_precision_kunlun(
    original_fn,
    self,
    hidden_states,
    gemm_output_zero_allocator=None,
    forward_batch=None,
):
    """Match the direct half-precision DSV4 router GEMM contract."""
    if (
        self.is_deepseek_v4
        and hidden_states.dtype == self.weight.dtype
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
    ):
        return hidden_states @ self.weight.T
    return original_fn(
        self,
        hidden_states,
        gemm_output_zero_allocator,
        forward_batch,
    )


def _restore_requested_fp16_parameter_dtype(*linears):
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.server_args import get_global_server_args

    if get_global_server_args().dtype not in _FP16_DTYPE_NAMES:
        return

    for linear in linears:
        if not isinstance(
            getattr(linear, "quant_method", None), UnquantizedLinearMethod
        ):
            continue
        linear.params_dtype = torch.float16
        weight = getattr(linear, "weight", None)
        if weight is not None and weight.dtype != torch.float16:
            weight.data = weight.data.to(dtype=torch.float16)


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.indexer.C4Indexer.__init__",
    type=HookType.AFTER,
)
def initialize_c4_indexer_parameter_dtype_kunlun(result, self, *args, **kwargs):
    """Restore the requested dtype for C4 indexer projections."""
    _restore_requested_fp16_parameter_dtype(self.wq_b, self.weights_proj)
    return result


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.compressor.Compressor.__init__",
    type=HookType.AFTER,
)
def initialize_compressor_parameter_dtype_kunlun(result, self, *args, **kwargs):
    """Restore the requested dtype for the compressor gate projection."""
    _restore_requested_fp16_parameter_dtype(self.wkv_gate)
    return result


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer.__init__",
    type=HookType.AFTER,
)
def initialize_mqa_rope_policy_kunlun(result, self, config, *args, **kwargs):
    """Restore the FP16 wo_a and dense-layer RoPE contracts."""
    _restore_requested_fp16_parameter_dtype(self.wo_a)

    if self.compress_ratio:
        return result

    from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
    from sglang.srt.models.deepseek_v4 import get_rope_config

    rope_theta, rope_scaling = get_rope_config(config)
    if not rope_scaling:
        return result
    self.freqs_cis = precompute_freqs_cis(
        dim=self.qk_rope_head_dim,
        seqlen=config.max_position_embeddings,
        original_seq_len=0,
        base=rope_theta,
        factor=rope_scaling["factor"],
        beta_fast=rope_scaling["beta_fast"],
        beta_slow=rope_scaling["beta_slow"],
    )
    return result


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer._compute_q_b",
    type=HookType.REPLACE,
)
def compute_q_b_kunlun(self, q, positions, q_out=None):
    """Write Q norm/RoPE into contiguous local storage before a TP-slice copy."""
    from sglang.srt.models.deepseek_v4 import fused_q_norm_rope

    if not getattr(torch, "_dsv4_qnorm_resolution_probed", False):
        import sys as _sys

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

        torch._dsv4_qnorm_resolution_probed = True
        print(
            "DSV4_QNORM_RESOLUTION module=%s qualname=%s file=%s"
            % (
                getattr(fused_q_norm_rope, "__module__", "?"),
                getattr(fused_q_norm_rope, "__qualname__", "?"),
                getattr(getattr(fused_q_norm_rope, "__code__", None), "co_filename", "?"),
            ),
            file=_sys.stderr,
            flush=True,
        )

    _dsv4_dump(self, "self_attn.wq_b.input.q_norm", q)
    q, _ = self.wq_b(q)
    _dsv4_dump(self, "self_attn.wq_b.output", q)
    q = q.view(-1, self.n_local_heads, self.head_dim)
    local_q_out = torch.empty_like(q)
    fused_q_norm_rope(q, local_q_out, self.eps, self.freqs_cis, positions)
    _dsv4_dump(self, "self_attn.q_local_after_norm_rope", local_q_out)
    if q_out is None:
        return local_q_out
    q_out.copy_(local_q_out)
    return q_out


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer._compute_kv_bf16",
    type=HookType.REPLACE,
)
def compute_kv_bf16_kunlun(self, x, positions, qkv_a=None):
    """Normalize KV and apply RoPE before it is handed to attention/cache code."""
    if qkv_a is not None:
        kv = qkv_a[..., self.q_lora_rank :]
    else:
        kv, _ = self.wkv(x)
    kv = self.kv_norm(kv)

    from sglang.srt.models.deepseek_v4 import fused_rope_inplace

    fused_rope_inplace(
        kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
        None,
        self.freqs_cis,
        positions,
    )
    return kv


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer.forward",
    type=HookType.REPLACE,
)
def mqa_forward_global_head_layout_kunlun(
    self, x, positions, forward_batch, x_quant=None
):
    """Version-pinned MQA forward with deterministic TP-global Q head slots.

    This is intentionally a full method hook: the upstream implementation reads
    the attention backend through a module global, so wrapping it would require
    request-hot-path global rebinding. All non-local Q and sink slots are zeroed,
    and only this rank's local output slice is returned.
    """
    import sglang.srt.models.deepseek_v4 as upstream
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if not upstream.get_attn_tp_context().input_scattered and x.shape[0] == 0:
        return x

    attn_backend = get_attn_backend()
    enable_multi_stream = (
        upstream.envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
        and self.alt_streams is not None
        and upstream.get_is_capture_mode()
        and x.shape[0] <= self._multi_stream_bs_limit
        and not (
            self.dsa_enable_prefill_cp and upstream.dsa_use_prefill_cp(forward_batch)
        )
        and not (upstream._is_hip and self.compressor is None)
    )

    tp_slice = slice(None)
    q_global = None
    q_out = None
    local_sink = self.attn_sink
    if self.attn_tp_size > 1:
        # Golden drives the operator with TP-global head slots: this rank owns
        # [rank * n_local_heads, (rank + 1) * n_local_heads) and every other slot
        # stays zero, for both Q and the attention sink. Per-rank dumps of the
        # Golden operator boundary confirm this placement.
        start = self.attn_tp_rank * self.n_local_heads
        stop = start + self.n_local_heads
        tp_slice = slice(start, stop)
        q_global = x.new_zeros(x.shape[0], self.n_heads, self.head_dim)
        q_out = q_global[:, tp_slice, :]

        local_sink = getattr(self, "_attn_sink_local", None)
        if (
            local_sink is None
            or local_sink.shape[0] != self.n_heads
            or local_sink.device != self.attn_sink.device
        ):
            local_sink = self.attn_sink.new_zeros(self.n_heads)
            local_sink[tp_slice].copy_(self.attn_sink[tp_slice])
            self._attn_sink_local = local_sink

    if enable_multi_stream:
        if upstream._is_hip:
            q = self._forward_prepare_multi_stream_hip(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
            )
        else:
            q = self._forward_prepare_multi_stream(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
            )
        kv = None
    else:
        q, kv = self._forward_prepare(
            x,
            positions,
            forward_batch,
            attn_backend,
            q_out,
            x_quant=x_quant,
        )

    attn_k = kv if kv is not None else q
    _dsv4_dump(self, "self_attn.q_local_after_prepare", q)
    if q_global is not None:
        _dsv4_dump(self, "self_attn.q_global_after_prepare", q_global)
    if is_unified_kv_triton():
        attn_q = q_out if q_out is not None else q
        _dsv4_dump(self, "self_attn.q_backend", attn_q)
        o = attn_backend.forward(
            q=attn_q,
            k=attn_k,
            v=attn_k,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=kv is not None,
        )
    else:
        attn_q = q_global if q_global is not None else q
        _dsv4_dump(self, "self_attn.q_backend", attn_q)
        save_kv_cache = False
        if (
            forward_batch.forward_mode.is_extend()
            and upstream.is_in_breakable_cuda_graph()
        ):
            o = attn_q.new_empty(
                (*attn_q.shape[:-1], self.attn_mqa.v_head_dim),
            )
            upstream.bcg_deepseek_v4_attention_with_output(
                attn_q,
                attn_k,
                o,
                self.attn_mqa.layer_id,
                self.compress_ratio,
                local_sink,
                save_kv_cache,
            )
        else:
            o = attn_backend.forward(
                q=attn_q,
                k=attn_k,
                v=attn_k,
                layer=self.attn_mqa,
                forward_batch=forward_batch,
                compress_ratio=self.compress_ratio,
                attn_sink=local_sink,
                save_kv_cache=save_kv_cache,
            )
        if q_global is not None:
            o = o[:, tp_slice, :]

    if upstream._is_npu:
        upstream.v4_rope_inplace_npu(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions,
            inverse=True,
        )
    else:
        upstream.fused_rope_inplace(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

    _dsv4_dump(
        self,
        "self_attn.compressed_attention.output.out_local_post_rope",
        o,
    )
    o = o.view(o.shape[0], self.n_local_groups, -1)
    if upstream._FP8_WO_A_GEMM:
        import deep_gemm

        token_count, group_count, group_dim = o.shape
        rank_dim = self.o_lora_rank
        o_fp8, o_scale = upstream.sglang_per_token_group_quant_fp8(
            o.reshape(token_count * group_count, group_dim).contiguous(),
            group_size=128,
            scale_ue8m0=True,
        )
        output = torch.empty(
            token_count,
            group_count,
            rank_dim,
            device=o.device,
            dtype=torch.bfloat16,
        )
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (
                o_fp8.view(token_count, group_count, group_dim),
                o_scale.view(token_count, group_count, -1),
            ),
            (
                self.wo_a.weight.view(group_count, rank_dim, group_dim),
                self.wo_a.weight_scale_inv.data,
            ),
            output,
            recipe=(1, 1, 128),
        )
        o = output
    else:
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        _dsv4_dump(self, "self_attn.wo_a.input.o", o)
        _dsv4_dump(self, "self_attn.wo_a.input.weight", wo_a)
        o = dsv4_mqa_wo_a_einsum_kunlun(o, wo_a)
        _dsv4_dump(self, "self_attn.wo_a.output", o)

    o, _ = self.wo_b(o.flatten(1))
    _dsv4_dump(self, "self_attn.wo_b.output", o)
    if self.attn_tp_size > 1 and self.attn_tp_size < upstream.get_parallel().tp_size:
        o = upstream.attn_tp_all_reduce(o)
    _dsv4_dump(self, "self_attn.output", o)
    return o
