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
@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_pre",
    type=HookType.REPLACE,
)
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
    if out.dtype is torch.float16:
        # The fused MHC-post overflows to inf/NaN under long context, so fp16 must be guarded.
        out = torch.nan_to_num(out)
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

    from sglang_kunlun.compat import server_arg

    if server_arg("dtype") not in _FP16_DTYPE_NAMES:
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
    "sglang.srt.models.deepseek_v4.MqaAttentionBase.__init__",
    type=HookType.AFTER,
)
def initialize_mqa_attention_parameter_dtype_kunlun(result, self, *args, **kwargs):
    """Restore wo_a to the requested dtype.

    Upstream hardcodes wo_a's params_dtype to bf16 in the shared base class, and
    both target's MQALayer and DSpark draft's DSparkAttention are its subclasses,
    so this must be attached to the base class.
    """
    _restore_requested_fp16_parameter_dtype(self.wo_a)
    return result


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer.__init__",
    type=HookType.AFTER,
)
def initialize_mqa_rope_policy_kunlun(result, self, config, *args, **kwargs):
    """Restore the dense-layer RoPE contract."""
    if self.compress_ratio:
        return result

    from sglang.kernels.ops.attention.dsv4.rope import precompute_freqs_cis
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
    """Write Q norm/RoPE into contiguous local storage before a TP-slice copy.

    0.5.23 added an early `return q` for `use_mega_attention` (the mega path
    norms and ropes Q inside its own kernel). That backend is rejected in
    `mqa_forward_global_head_layout_kunlun`, so this body always runs the
    norm+RoPE; the assert keeps the two hooks from drifting apart.
    """
    assert not getattr(self, "use_mega_attention", False), (
        "mega attention is not implemented on Kunlun; see "
        "mqa_forward_global_head_layout_kunlun"
    )
    from sglang.srt.models.deepseek_v4 import fused_q_norm_rope

    _dsv4_dump(self, "self_attn.wq_b.input.q_norm", q)
    q, _ = self.wq_b(q)
    _dsv4_dump(self, "self_attn.wq_b.output", q)
    q = q.view(-1, self.n_local_heads, self.head_dim)
    local_q_out = torch.empty_like(q)
    # Q's per-head weight-free RMSNorm is controlled by `config.q_head_norm`: it
    # is True for V4 and **False for V4.1** (`configs/deepseek_v41.py:48`).
    # Upstream `_compute_q_b` uses exactly this to decide whether to pass eps or
    # None to `fused_q_norm_rope` (None = RoPE only,
    # `kernels/ops/attention/dsv4/elementwise.py:194`). This code previously
    # passed `self.eps` unconditionally, so V4.1's Q was normalized one extra
    # time per layer: measured q's RMS was pinned to 1.000000 (the same on every
    # layer), while the old stack was 1.75~2.12 and varied per layer; attention
    # logits were therefore systematically too small, dropping wq_b's per-layer
    # magnitude information, so under long prompts the model could not focus
    # precisely and only repeated nearby content (short prompts still answer
    # correctly, so it is easily mistaken for "weak long-context capability").
    # The same decision already exists in the fast path of
    # `hooks/models/deepseek_v4.py`.
    norm_eps = self.eps if getattr(self, "q_head_norm", False) else None
    fused_q_norm_rope(q, local_q_out, norm_eps, self.freqs_cis, positions)
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


def _dsa_use_prefill_cp(forward_batch) -> bool:
    """0.5.22 no longer re-exports this decision from ``models.deepseek_v4``."""
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp

    return dsa_use_prefill_cp(forward_batch)


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer.forward",
    type=HookType.REPLACE,
)
def mqa_forward_global_head_layout_kunlun(
    self, x, positions, forward_batch, x_quant=None, defer_all_reduce=False
):
    """Version-pinned MQA forward with deterministic TP-global Q head slots.

    This is intentionally a full method hook: the upstream implementation reads
    the attention backend through a module global, so wrapping it would require
    request-hot-path global rebinding. All non-local Q and sink slots are zeroed,
    and only this rank's local output slice is returned.

    0.5.22 added `defer_all_reduce` and on that release it could only arrive as
    False here, so this hook refused it. 0.5.23 made it the ordinary CUDA route:
    `DeepseekV4CudaDecoderLayer.__init__` sets `self_attn.wo_b.reduce_results =
    False` and `_run_boundary_attention` always passes `defer_all_reduce=True`,
    with the reduction moved into `attn_boundary.finish()` ->
    `AttentionTransport.prepare_output`, which wraps the unreduced rows in an
    `UnreducedOutput(partial, group=...)` -- a declared transport sum, not the
    Blackwell-only fused kernel. So the flag is now honoured by delegating the
    wo_b projection to upstream's own `_project_wo_b` (see the tail).

    0.5.23 changed upstream's body in five places. Three are inert here:

    * the `unified_kv_kernels.env_gate` import moved to
      `srt.layers.attention.dsv4.unified_kv_gate` (followed below);
    * the Q padding buffer was extracted into `MQALayer._zero_padded_q`, which
      this rewrite does not use -- it builds its own TP-global `q_global`;
    * `_wo_a_quant_freqs` adds a `wo_a_quant_freqs` rope kwarg, but it returns
      None unless `_wo_a_fp8_mxscale_prequant` is bound, and that import is
      HIP-only (`deepseek_v4.py:462-484`), so the kwarg can never appear here.

    The remaining two are handled: the new `use_mega_attention` route is
    rejected loudly (see below), and the breakable-graph output buffer is
    allocated with `dtype=x.dtype` to match upstream's fix for TRT-LLM returning
    BF16 under FP8 queries.
    """
    import sglang.srt.models.deepseek_v4 as upstream
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
        is_unified_kv_triton,
    )
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if not upstream.get_attn_tp_context().input_scattered and x.shape[0] == 0:
        return x

    # 0.5.23 新增的 mega attention：上游在 forward 里整条改道
    # （`_forward_mega` + `_project_wo_b`，并把 kernel_num_heads 固定成 64/128）。
    # 门是 `is_dsv41 and _is_cuda and kernel.dsv4_attn_backend == "flashmla_mega"`
    # （`deepseek_v4.py:1509`），基线跑的是 `kunlun_compressed`，所以只能是 False。
    # 这条路径在昆仑没有实现，静默走非 mega 分支等于换了一套数值，所以显式拒绝。
    if getattr(self, "use_mega_attention", False):
        raise NotImplementedError(
            "Kunlun MQA forward 没有 mega attention 实现"
            "（上游 0.5.23 的 flashmla_mega 路径），use_mega_attention=True 不支持。"
        )

    attn_backend = get_attn_backend()
    enable_multi_stream = (
        upstream.envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
        and self.alt_streams is not None
        and upstream.get_is_capture_mode()
        and x.shape[0] <= self._multi_stream_bs_limit
        # 0.5.22 deleted the ``dsa_enable_prefill_cp`` snapshot on MQALayer;
        # with its second argument left empty, ``dsa_use_prefill_cp`` calls
        # ``is_dsa_enable_prefill_cp()`` itself, so ask it directly here.
        and not _dsa_use_prefill_cp(forward_batch)
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
        # 0.5.22 added q_rope_out/k_nope_out/k_rope_out to _forward_prepare (the
        # packed Q/K outputs of the unified_kv fp8 dual-pool). This Q head layout
        # only allocates bf16 q_global/q_out and does not use the dual-pool
        # layout, so the three stay at the default None, equivalent to the
        # upstream non-unified-fp8 branch.
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
                # 0.5.23: TRT-LLM 即使拿 FP8 query 也返回 BF16，所以输出缓冲按
                # 输入 x 的 dtype 分配，而不是跟着 attn_q。
                dtype=x.dtype,
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

    # ERNIE 5.5 HCA folds a cross-head ``VhaPostMix`` into ``_project_wo_a``
    if getattr(self, "postmix", None) is not None:
        o = self.postmix(o)

    if upstream._is_npu:
        # The 0.5.22 rope in-place helper is called ``fused_rope_inplace``
        # (that is how ``deepseek_v4.py:1404`` calls it); the NPU branch is
        # unreachable on Kunlun, but the name must point at a symbol that really
        # exists.
        upstream.fused_rope_inplace(
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

    # wo_b + the TP reduction. Call upstream's own ``_project_wo_b`` instead of
    # re-deriving it: 0.5.23 made `defer_all_reduce=True` the ordinary CUDA
    # route (`DeepseekV4CudaDecoderLayer.__init__` sets
    # `self_attn.wo_b.reduce_results = False` and `_run_boundary_attention`
    # always passes it), so the reduction now belongs to
    # `attn_boundary.finish()`. `_project_wo_b` is where upstream decides
    # whether the defer actually applies (attn_tp_size, `_is_cuda`,
    # `should_skip_mlp_all_reduce`), passes `skip_all_reduce` into `wo_b` and
    # wraps the unreduced rows in `mhc.AttnOutput`. Copying that logic here
    # would be a second opinion on a collective.
    o = self._project_wo_b(o, defer_all_reduce)
    _dsv4_dump(
        self,
        "self_attn.wo_b.output",
        o.partial if isinstance(o, upstream.mhc.AttnOutput) else o,
    )
    _dsv4_dump(
        self,
        "self_attn.output",
        o.partial if isinstance(o, upstream.mhc.AttnOutput) else o,
    )
    return o
