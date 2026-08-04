"""Kunlun-owned DeepSeek V4 production precision hooks without diagnostics."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.kernel_ops import dsv4_mqa_wo_a_einsum_kunlun


_FP16_DTYPE_NAMES = frozenset(("fp16", "float16", "half"))


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

    from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis
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

    q, _ = self.wq_b(q)
    q = q.view(-1, self.n_local_heads, self.head_dim)
    local_q_out = torch.empty_like(q)
    fused_q_norm_rope(q, local_q_out, self.eps, self.freqs_cis, positions)
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
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
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
    if self.tp_size > 1:
        start = self.tp_rank * self.n_local_heads
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
    if is_unified_kv_triton():
        attn_q = q_out if q_out is not None else q
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
        o = dsv4_mqa_wo_a_einsum_kunlun(o, wo_a)

    o, _ = self.wo_b(o.flatten(1))
    if (
        self.tp_size > 1
        and self.tp_size < upstream.get_tensor_model_parallel_world_size()
    ):
        o = upstream.attn_tp_all_reduce(o)
    return o
