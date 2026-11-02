# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hooks for DeepseekV4DecoderLayer methods (hc_post, hc_pre)."""
from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.jit_kernel.dsv4 import fused_rope_inplace
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.deepseek_v4_rope import v4_rope_inplace_npu
from sglang.srt.layers.dp_attention import attn_tp_all_reduce
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.models.deepseek_v2 import _is_hip, _is_npu
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
 
if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    
import kunlun_ops
import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops.*)
 
_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()

@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_pre",
    type=HookType.REPLACE,
)
def hc_pre_kunlun(
    self, x, hc_fn, hc_scale, hc_base,
    norm: Optional[nn.Module] = None,
    forward_batch: Optional[ForwardBatch] = None,
):
    """ hc pre kunlun """
    assert self.hc_mult == 4, "aiak hc_pre only supports n=4"
 
    y, post, comb = torch.ops.xspeedgate_ops.hc_pre(
        x.contiguous(),                 # 可以直接传3D的 [M, 4, 4096]，这个算子2D 3D都支持
        hc_fn.contiguous(),            # [24, 16384], float32
        hc_scale.contiguous(),         # [3], float32
        hc_base.contiguous(),          # [24], float32
        rms_eps=self.rms_norm_eps,
        hc_pre_eps=self.hc_eps,
        hc_sinkhorn_eps=self.hc_eps,
        mhc_post_mult_value=2.0,
        sinkhorn_iters=self.hc_sinkhorn_iters,
    )
    return y, post, comb, False
 
 
@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post",
    type=HookType.REPLACE,
)
def hc_post_kunlun(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """Kunlun-native hc_post via ``kunlun_ops.hc_post_kunlun_impl``.

    Computes ``out = post * x + comb @ residual`` (fused).
    Only N == 4 (hc_mult) is supported on XPU.
    """
    if x.shape[0] == 0:
        return torch.empty(
            (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
        )

    batch = x.shape[0]
    N = self.hc_mult
    D = x.shape[-1]
    out = torch.empty((batch, N, D), dtype=x.dtype, device=x.device)
    kunlun_ops.hc_post_kunlun_impl(
        post.contiguous(),
        x.contiguous(),
        comb.contiguous(),
        residual.contiguous(),
        out,
        batch, N, D,
    )
    return out


# --------------------------- INT4 checkpoint support ---------------------
# The compressed-tensors targets/ignore lists in the INT4 checkpoint use the
# original HF module names (`attn`/`ffn`/`w1..w3`), while the weights
# themselves are remapped by `remap_weight_name_to_dpsk_hf_format`. The mapper
# is picked up by the model loader to remap the quantization config. The keys
# look like regexes because the checkpoint targets are regex strings; 0.5.14
# WeightsMapper still matches them as plain substrings.
DeepseekV4ForCausalLM.packed_modules_mapping = {
    "wqkv_a": ["wq_a", "wkv"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

DeepseekV4ForCausalLM.hf_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={
        r"\.attn\.": r"\.self_attn\.",
        r"\.ffn\.": r"\.mlp\.",
        r"\.attn_norm": r"\.input_layernorm",
        r"\.ffn_norm": r"\.post_attention_layernorm",
    },
    orig_to_new_suffix={
        r"\.w1$": r"\.gate_proj$",
        r"\.w2$": r"\.down_proj$",
        r"\.w3$": r"\.up_proj$",
    },
)


def _fuse_wqkv_a_weight_scale(weights):
    """Concat the ``wq_a``/``wkv`` per-channel scales for the fused ``wqkv_a``.

    Upstream `load_weights` already fuses ``.weight`` and ``.weight_scale_inv``;
    the int4 checkpoint instead carries ``.weight_scale``, which is fused here
    so the generic loader path picks it up as a plain ``wqkv_a.weight_scale``.
    """
    pending = {}
    for name, tensor in weights:
        is_q = name.endswith(".wq_a.weight_scale")
        if not is_q and not name.endswith(".wkv.weight_scale"):
            yield name, tensor
            continue
        fused_name = name.replace(".wq_a." if is_q else ".wkv.", ".wqkv_a.")
        bucket = pending.setdefault(fused_name, {})
        shard: Literal["q", "kv"] = "q" if is_q else "kv"
        assert shard not in bucket, f"duplicate shard {shard} for {fused_name}"
        bucket[shard] = tensor
        if len(bucket) == 2:
            yield fused_name, torch.cat([bucket["q"], bucket["kv"]], dim=0)
            pending.pop(fused_name)
    assert not pending, f"unpaired wqkv_a weight scales: {sorted(pending)}"


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
    type=HookType.AROUND,
)
def load_weights_kunlun(original_fn, self, weights, is_nextn=False):
    """Pre-process the int4/int8 weight stream, then defer to upstream."""
    if envs.SGLANG_OPT_FUSE_WQA_WKV.get():
        weights = _fuse_wqkv_a_weight_scale(weights)
    return original_fn(self, weights, is_nextn)


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.MQALayer.forward",
    type=HookType.REPLACE,
)
def forward_kunlun(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        x_quant=None,
    ) -> torch.Tensor:
        """Run the Kunlun DeepSeek V4 attention forward pass."""
 
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            return x
 
        attn_backend = get_attn_backend()
        if TYPE_CHECKING:
            assert isinstance(
                attn_backend,
                (DeepseekV4AttnBackend, DeepseekV4HipRadixBackend),
            )
 
        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch))
            and not (_is_hip and self.compressor is None)
        )
 
        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            # FlashMLA's fp8 sparse decode kernel only specializes h_q for {64, 128}.
            # Pad the per-rank heads to 64 (not the full n_heads) when they fit, to
            # dispatch the cheaper decode::head64 variant; attn_sink is sliced to
            # this rank and padded to match.
            padded_num_heads = 64 if self.n_local_heads <= 64 else self.n_heads
            q_padded = x.new_empty(x.shape[0], padded_num_heads, self.head_dim)
            tp_slice = slice(0, self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]
            if self._attn_sink_local is None:
                # Build once on the first forward (post weight load); a per-call
                # rebuild would replay a fill+copy per layer in the decode graph.
                rank = self.tp_rank
                sink = self.attn_sink.new_zeros(padded_num_heads)
                sink[: self.n_local_heads] = self.attn_sink[
                    rank * self.n_local_heads : (rank + 1) * self.n_local_heads
                ]
                self._attn_sink_local = sink
 
        if enable_multi_stream:
            # Multi-stream path always fuses cache write into the K kernel,
            # so the bf16 KV intermediate is gone.
            if _is_hip:
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
 
        # The cache write is always fused / already done by _forward_prepare* --
        # tell the backend to skip its own store_cache. When `kv is None`
        # (no DSA-CP), pass `q` as a sentinel for the `k is v` assert; the
        # attention path doesn't read it once `save_kv_cache=False`.
        attn_k = kv if kv is not None else q
        from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )
 
        if is_unified_kv_triton():
            o = attn_backend.forward(
                q=q_out if q_out is not None else q,
                k=attn_k,
                v=attn_k,
                layer=self.attn_mqa,
                forward_batch=forward_batch,
                compress_ratio=self.compress_ratio,
                attn_sink=self.attn_sink,
                save_kv_cache=kv is not None,
            )
        else:
            attn_q = q_padded if q_padded is not None else q
            save_kv_cache = False
            if forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph():
                # Imported here (not at module scope) because this symbol lives in
                # the hook target module; a top-level import would pull it in
                # during hook registration.
                from sglang.srt.models.deepseek_v4 import (
                    bcg_deepseek_v4_attention_with_output,
                )
 
                o = attn_q.new_empty(
                    (*attn_q.shape[:-1], self.attn_mqa.v_head_dim),
                )
                bcg_deepseek_v4_attention_with_output(
                    attn_q,
                    attn_k,
                    o,
                    self.attn_mqa.layer_id,
                    self.compress_ratio,
                    self._attn_sink_local,
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
                    attn_sink=self._attn_sink_local,
                    save_kv_cache=save_kv_cache,
                )
            o = o[:, tp_slice, :]
        if _is_npu:
            v4_rope_inplace_npu(
                o[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions,
                inverse=True,
            )
        else:
            fused_rope_inplace(
                o[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
                inverse=True,
            )
 
        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm
 
            T, G, D = o.shape
            R = self.o_lora_rank
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D).contiguous(),
                group_size=128,
                scale_ue8m0=True,
            )
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            # print("xspeed_gate enisum!!!!!!!!!!")
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.ops.xspeedgate_ops.einsum_tgd_grd_tgr(o.contiguous(), wo_a)
 
        o, _ = self.wo_b(o.flatten(1))
        if self.tp_size > 1 and self.tp_size < get_tensor_model_parallel_world_size():
            o = attn_tp_all_reduce(o)
 
        return o
