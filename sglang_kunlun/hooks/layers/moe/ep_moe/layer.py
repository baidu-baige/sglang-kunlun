"""Kunlun plugin hooks for ``sglang.srt.layers.moe.ep_moe.layer``."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.server_args import get_global_server_args
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph as is_in_piecewise_cuda_graph,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.utils import get_bool_env_var, dispose_tensor
from sglang.srt.layers.activation import SiluAndMul

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )


from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config, W8A8Int8MoEMethod
from kunlun_ops import (
    dequant2d_per_token,
    m_grouped_gemm_bf16_bf16_bf16_nt_contiguous_v3,
    m_grouped_gemm_bf16_bf16_bf16_nt_masked_v3,
    m_grouped_gemm_fp16_fp16_bf16_nt_contiguous_castte_v3,
    m_grouped_gemm_fp16_fp16_bf16_nt_contiguous_v3,
    m_grouped_gemm_fp16_fp16_bf16_nt_masked_castte_v3,
    m_grouped_gemm_fp16_fp16_bf16_nt_masked_v3,
    m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3,
    m_grouped_gemm_I8_I8_bf16_nt_masked,
    m_grouped_gemm_I8_I8_bf16_nt_masked_v3,
    per_token_dequant2d_with_mask,
    silu_and_mul_mask_fwd,
)
logger = logging.getLogger(__name__)

_FP16_DTYPE_NAMES = frozenset(("fp16", "float16", "half"))


class ReusedBuffer:
    """
    ReusedBuffer
    """
    def __init__(self, buffer: torch.Tensor):
        """
        init
        """
        self.buffer = buffer
        self.buffer_size_in_bytes = self.buffer.numel() * self.buffer.element_size()

    def empty(self, shape, dtype):
        """
        empty
        """
        # Calculate the number of elements needed based on the requested dtype
        num_elements = torch.Size(shape).numel()
        element_size = torch.tensor([], dtype=dtype).element_size()
        requested_size_in_bytes = num_elements * element_size
        # Check if buffer has enough space
        assert (
            requested_size_in_bytes <= self.buffer_size_in_bytes
        ), "Buffer does not have enough space."

        # Create a view of the buffer with the requested shape and dtype
        # Reinterpret the buffer as the desired dtype without actual data conversion
        allocated_tensor = self.buffer.view(dtype=dtype)[:num_elements].view(shape)
        return allocated_tensor


class Bfp16orFp16NormalMoeBuffer:
    """
    condition:
    (1) N <= M <= local_num_experts * N
    (2) F < H. e.g.: F = 2048, H = 7168
    solution:
        recv_x   -> recv_x_all_i8 -> recv_x_all_bfp16 -> gateup_out_bfp16 -> 
        down_inp_bfp16 -> down_out_bfp16 -> x_to_combine
        [N, H]i8 -> [M, H]i8      -> [M, H]bfp16      -> [M, 2*F]bfp16    -> 
        [M, F]bfp16    -> [M, H]bfp16    -> [N, H]bfp16
        buffer0: (recv_x_all_i8, gateup_out_bfp16, down_out_bfp16, x_to_combine) : 
        max(M * H * 2, M * 2*F * 2, N * H * 2)
        buffer1: (recv_x_all_bfp16, down_inp_bfp16): max(M * H * 2, M * F * 2)
    """

    def __init__(
        self,
        num_tokens,
        num_selected,
        hidden_size,
        ffn_hidden_size,
        device,
        is_bfp16=True,
    ):
        """
        init
        """

        self.N = num_tokens
        self.M = num_selected
        self.H = hidden_size
        self.F = ffn_hidden_size
        self.device = device
        self.inter_dtype = torch.bfloat16 if is_bfp16 else torch.float16
        self._init_new()

    def _init_new(self):
        """
        init_new
        """

        buffer0_size = max(
            self.M * self.H * 2, self.M * 2 * self.F * 2, self.N * self.H * 2
        )
        buffer1_size = max(self.M * self.H * 2, self.M * self.F * 2)
        try:
            self.buffer0 = ReusedBuffer(
                torch.empty(buffer0_size, dtype=torch.uint8, device=self.device)
            )
            self.buffer1 = ReusedBuffer(
                torch.empty(buffer1_size, dtype=torch.uint8, device=self.device)
            )
        except Exception as e:
            logger.info(
                f"Failed to allocate memory for Moe infer buffers: N={self.N}, M={self.M}. \
                F={self.F} H={self.H}. buffer0 size: {buffer0_size}, buffer1 size: {buffer1_size}"
            )
            raise e

    @property
    def recv_x_all_int8(self):
        """
        recv_x_all_int8
        """

        return self.buffer0.empty((self.M, self.H), torch.int8)

    @property
    def recv_x_all(self):
        """
        recv_x_all
        """

        return self.buffer1.empty((self.M, self.H), self.inter_dtype)

    @property
    def gateup_output(self):
        """
        gateup_output is bfp16
        """
        return self.buffer0.empty((self.M, self.F * 2), torch.bfloat16)

    @property
    def down_inp_bfp16(self):
        """
        down_inp_bfp16
        """

        return self.buffer1.empty((self.M, self.F), self.inter_dtype)

    @property
    def down_output(self):
        """
        down_output
        """

        return self.buffer0.empty((self.M, self.H), torch.bfloat16)

    @property
    def x_to_combine(self):
        """
        x_to_combine
        """

        return self.buffer0.empty((self.N, self.H), torch.bfloat16)


class W8A8NormalMoeBuffer:
    """
    condition:
    (1) N <= M <= local_num_experts * N
    (2) F < H. e.g.: F = 2048, H = 7168
    solution:
    recv_x   -> recv_x_all -> gateup_out    -> down_inp_bfp16 -> down_inp_i8 -> down_out -> x_to_combine
    [N, H]i8 -> [M, H]i8   -> [M, 2*F]bfp16 -> [M, F]bfp16    -> [M, F]i8    -> [M, H]bfp16 - - comibine- -> [N, H]bfp16
    buffer0: (recv_x_all, down_inp_bfp16, x_to_combine) : max(M * H * 2, M * F * 2, N * H * 2)
    buffer1: (gateup_out, down_out): max(M * 2 * F * 2, M * H * 2)
    """

    def __init__(
        self,
        num_tokens,
        num_selected,
        hidden_size,
        ffn_hidden_size,
        device,
    ):
        """
        init
        """

        self.N = num_tokens
        self.M = num_selected
        self.H = hidden_size
        self.F = ffn_hidden_size
        self.device = device
        self._init_new()

    def _init_new(self):
        """
        init_new
        """

        buffer0_size = max(
            self.M * self.H * 1, self.M * self.F * 2, self.N * self.H * 2
        )
        buffer1_size = max(self.M * 2 * self.F * 2, self.M * self.H * 2)
        self.buffer0 = ReusedBuffer(
            torch.empty(buffer0_size, dtype=torch.uint8, device=self.device)
        )
        self.buffer1 = ReusedBuffer(
            torch.empty(buffer1_size, dtype=torch.uint8, device=self.device)
        )

    @property
    def recv_x_all_int8(self):
        """
        recv_x_all_int8
        """

        # return self.buffer0[: self.M * self.H * 1].view(-1, self.H).view(torch.int8)
        return self.buffer0.empty((self.M, self.H), torch.int8)

    @property
    def gateup_output(self):
        """
        gateup_output
        """

        # return self.buffer1[: self.M * 2 * self.F * 2].view(-1, self.F * 2).view(torch.bfloat16)
        return self.buffer1.empty((self.M, self.F * 2), torch.bfloat16)

    @property
    def down_inp_bfp16(self):
        """
        down_inp_bfp16
        """

        return self.buffer0.empty((self.M, self.F), torch.bfloat16)

    @property
    def down_input_int8(self):
        """
        down_input_int8
        """

        return torch.empty(
            (self.M, self.F),
            dtype=torch.int8,
            device=self.device,
        )

    @property
    def down_output(self):
        """
        down_output
        """

        # return self.buffer1[: self.M * self.H * 2].view(-1, self.H).view(torch.bfloat16)
        return self.buffer1.empty((self.M, self.H), torch.bfloat16)

    @property
    def x_to_combine(self):
        """
        x_to_combine
        """

        return self.buffer1.empty((self.N, self.H), torch.bfloat16)


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        self.should_fuse_routed_scaling_factor_in_topk = True
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        # if self.deprecate_flag:
        #     return self.dispatcher

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        elif isinstance(quant_config, W8A8Int8Config):
            self.use_fp8_w8a8 = True
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.activation_scheme = None
            self.fp8_dtype = torch.int8
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4afp8 = False

        self.deepep_mode = get_deepep_mode()
        self.act_fn = SiluAndMul()
 
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """forward"""
        if is_in_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return torch.ops.sglang.moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """forward_impl"""

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        # TODO: can we call super().forward here?
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        hidden_states = self.dispatcher.combine(
            combine_input=combine_input,
        )
        if get_global_server_args().dtype in _FP16_DTYPE_NAMES:
            # DeepEP combine 返回 bf16，收窄到 fp16 前先 clamp，避免累加后的 MoE
            # 输出超出 fp16 范围。
            limit = int(os.environ.get("SGLANG_FP16_LIMIT_IN_MOE", "10"))
            hidden_states = hidden_states.clamp(min=-limit, max=limit).to(
                torch.float16
            )

        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch"""
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def forward_deepgemm_contiguous(
        self,
        dispatch_output: "DeepEPNormalOutput",
    ):
        """
        forward_deepgemm_contiguous
        """
        import kunlun_ops
        hidden_states, hidden_states_scale, topk_idx, topk_weights, num_recv_tokens_per_expert = (
            dispatch_output
        )
        hidden_states_fp8 = (hidden_states, hidden_states_scale)
        forward_int8: bool = self.w2_weight.dtype == torch.int8

        def wrapper_m_grouped_gemm_nt_contiguous_v3(
            lhs: torch.Tensor,
            rhs: torch.Tensor,
            out: torch.Tensor,
            m_indices: torch.Tensor,
            lod_force_sdnn: bool = False,
        ):
            assert lhs.dtype == rhs.dtype
            assert lhs.dtype in [torch.bfloat16, torch.float16]
            if lhs.dtype == torch.bfloat16:
                return m_grouped_gemm_bf16_bf16_bf16_nt_contiguous_v3(
                    lhs, rhs, out, m_indices, lod_force_sdnn
                )
            else:
                return m_grouped_gemm_fp16_fp16_bf16_nt_contiguous_v3(
                    lhs, rhs, out, m_indices, lod_force_sdnn
                )

        recv_x_int8, recv_x_scale = hidden_states_fp8
        device = recv_x_int8.device
        num_tokens, hidden_size = recv_x_int8.shape
        ffn_hidden_size = self.w13_weight.shape[1] // 2
        if num_tokens == 0:
            return torch.empty_like(recv_x_int8, dtype=torch.bfloat16)
        if num_recv_tokens_per_expert is None:
            return hidden_states_fp8.bfloat16()
        assert isinstance(num_recv_tokens_per_expert, list)
        num_all_tokens = sum(num_recv_tokens_per_expert)
        assert num_all_tokens >= num_tokens, f"{num_all_tokens} >= {num_tokens}"
        topk = topk_idx.shape[1]

        weight_dtype = self.w13_weight.dtype
        num_recv_tokens_per_expert_cpu_tensor = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            device="cpu",
        )
        num_recv_tokens_per_expert_list_start_cpu = torch.zeros(
            len(num_recv_tokens_per_expert) + 1, dtype=torch.int32
        )
        num_recv_tokens_per_expert_list_start_cpu[1:] = torch.cumsum(
            num_recv_tokens_per_expert_cpu_tensor,
            dim=0,
            dtype=torch.int32,
        )

        if forward_int8:
            moe_buffer = W8A8NormalMoeBuffer(
                num_tokens,
                num_all_tokens,
                hidden_size,
                ffn_hidden_size,
                recv_x_int8.device,
            )
        else:
            moe_buffer = Bfp16orFp16NormalMoeBuffer(
                num_tokens,
                num_all_tokens,
                hidden_size,
                ffn_hidden_size,
                recv_x_int8.device,
                is_bfp16=self.w13_weight.dtype == torch.bfloat16
                or self.moe_castte_mode,
            )

        if get_bool_env_var("XSGL_NATIVE_DEEPEP_NORMAL"):
            recv_x_all_int8 = torch.zeros(
                num_all_tokens, hidden_size, device=device, dtype=weight_dtype
            )
            input_scale = torch.zeros(
                num_all_tokens, 1, device=device, dtype=torch.float32
            )
            m_indices = num_recv_tokens_per_expert.to(torch.int32)
            token_counts = torch.zeros(num_tokens, dtype=torch.int32, device=device)
            token_to_m = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
            token_weights = torch.zeros(
                num_all_tokens, dtype=torch.bfloat16, device=device
            )
            num_recv_tokens_per_expert_list_start = (
                num_recv_tokens_per_expert_list_start_cpu.to(device)
            )
            for token_idx in range(num_tokens):
                for j in range(topk):
                    if topk_idx[token_idx, j] != -1:
                        expert = topk_idx[token_idx, j].item()
                        m_idx = num_recv_tokens_per_expert_list_start[expert]
                        recv_x_all_int8[m_idx, :] = recv_x_int8[token_idx, :]
                        input_scale[m_idx, :] = recv_x_scale[token_idx, :]
                        token_to_m[token_idx, token_counts[token_idx]] = m_idx
                        token_counts[token_idx] += 1
                        token_weights[m_idx] = topk_weights[token_idx][j]
                        num_recv_tokens_per_expert_list_start[expert] += 1
        else:
            recv_x_all_int8 = moe_buffer.recv_x_all_int8
            input_scale = torch.empty(
                num_all_tokens, 1, device=device, dtype=torch.float32
            )
            m_indices = torch.empty(num_all_tokens, dtype=torch.int32, device=device)
            token_counts = torch.empty(num_tokens, dtype=torch.int32, device=device)
            token_to_m = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
            token_weights = torch.empty(
                num_all_tokens, dtype=torch.bfloat16, device=device
            )
            torch.ops.custom_ops.dispatch_convert(
                recv_x_int8,
                recv_x_scale,
                topk_idx,
                topk_weights,
                num_recv_tokens_per_expert_list_start_cpu.to(
                    recv_x_int8.device, non_blocking=True
                ),
                num_tokens,
                num_all_tokens,
                hidden_size,
                self.num_local_experts,
                recv_x_all=recv_x_all_int8,
                recv_x_all_scale=input_scale,
                m_indices=m_indices,
                token_counts=token_counts,
                token_to_m=token_to_m,
                token_weights=token_weights,
            )
            m_indices = num_recv_tokens_per_expert_cpu_tensor.to(
                recv_x_int8.device, non_blocking=True
            )

        # GroupGemm-0

        gateup_output = moe_buffer.gateup_output
        if forward_int8:
            m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
                (recv_x_all_int8, input_scale),
                (self.w13_weight, self.w13_weight_scale),
                gateup_output.view(num_all_tokens, self.w13_weight.shape[1]),
                m_indices,
            )
        else:
            recv_x_all = moe_buffer.recv_x_all
            dequant2d_per_token(
                recv_x_all_int8, input_scale, recv_x_all, is_absmax=True
            )

            if recv_x_all.dtype != self.w13_weight.dtype:
                recv_x_all = recv_x_all.to(self.w13_weight.dtype)
            # dequant2d(recv_x_all_int8, input_scale, recv_x_all)
            wrapper_m_grouped_gemm_nt_contiguous_v3(
                recv_x_all,
                self.w13_weight,
                gateup_output.view(num_all_tokens, self.w13_weight.shape[1]),
                m_indices,
            )

        # Act
        down_input = self.act_fn(gateup_output)
        d = down_input.shape[1]

        down_output = moe_buffer.down_output
        if forward_int8:
            # GroupGemm-1
            down_input_int8 = moe_buffer.down_input_int8
            kunlun_ops.quant2d(
                down_input,
                down_input_int8,
                input_scale,
            )
            m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
                (down_input_int8, input_scale),
                (self.w2_weight, self.w2_weight_scale),
                down_output.view(num_all_tokens, self.w2_weight.shape[1]),
                m_indices,
            )
        else:
            if down_input.dtype != self.w2_weight.dtype:
                down_input = down_input.to(self.w2_weight.dtype)
            wrapper_m_grouped_gemm_nt_contiguous_v3(
                down_input,
                self.w2_weight,
                down_output.view(num_all_tokens, self.w2_weight.shape[1]),
                m_indices,
            )

        x_to_combine = torch.zeros(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )

        if get_bool_env_var("XSGL_NATIVE_DEEPEP_NORMAL"):
            for token_idx in range(num_tokens):
                for j in range(token_counts[token_idx]):
                    m_idx = token_to_m[token_idx, j]
                    x_to_combine[token_idx] += down_output[m_idx] * token_weights[m_idx]
        else:
            token_to_m = token_to_m.to(torch.int64)
            torch.ops.custom_ops.combine_convert(
                down_output,
                token_counts,
                token_to_m,
                token_weights,
                num_tokens,
                num_all_tokens,
                hidden_size,
                algo=0,
                y=x_to_combine,
            )

        return x_to_combine

    def forward_deepgemm_masked_bfp16(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        masked_m: torch.Tensor,
        expected_m: int,
    ):
        """
        forward_deepgemm_masked_bfp16
        """
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        def wrapper_m_grouped_gemm_nt_masked_v3(
            lhs: torch.Tensor,
            rhs: torch.Tensor,
            out: torch.Tensor,
            mask_m: torch.Tensor,
            expected_m: int,
            lod_force_sdnn: bool = False,
        ):
            assert lhs.dtype == rhs.dtype
            if lhs.dtype == torch.bfloat16:
                return m_grouped_gemm_bf16_bf16_bf16_nt_masked_v3(
                    lhs, rhs, out, mask_m, expected_m, lod_force_sdnn
                )
            else:
                return m_grouped_gemm_fp16_fp16_bf16_nt_masked_v3(
                    lhs, rhs, out, mask_m, expected_m, lod_force_sdnn
                )

        # GroupGemm-0
        if isinstance(hidden_states_fp8, tuple):
            hidden_states_fp16 = torch.empty_like(
                hidden_states_fp8[0], dtype=self.w13_weight.dtype
            )
            if (hidden_states_fp8[0].dtype == torch.bfloat16
                or hidden_states_fp8[0].dtype == torch.float16):
                hidden_states_fp16 = hidden_states_fp8[0]
            else:
                hidden_states_fp16 = torch.empty_like(
                    hidden_states_fp8[0], dtype=self.w13_weight.dtype
                )
                per_token_dequant2d_with_mask(
                    hidden_states_fp8[0],
                    hidden_states_fp8[1],
                    masked_m,
                    hidden_states_fp16,
                    is_absmax=True,
                )
            # dequant2d(hidden_states_fp8[0], hidden_states_fp8[1], hidden_states_fp16)
        else:
            assert (
                hidden_states_fp8.dtype == torch.bfloat16
                or hidden_states_fp8.dtype == torch.float16
            )
            hidden_states_fp16 = hidden_states_fp8

        num_groups, m, k = hidden_states_fp16.size()
        n = self.w13_weight.size(1)
        expected_m = min(expected_m, m)

        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
        )
        # GroupGemm-0
        if hidden_states_fp16.dtype != self.w13_weight.dtype:
            hidden_states_fp16 = hidden_states_fp16.to(self.w13_weight.dtype)

        wrapper_m_grouped_gemm_nt_masked_v3(
            hidden_states_fp16, self.w13_weight, gateup_output, masked_m, expected_m
        )
        dispose_tensor(hidden_states_fp8[0])
        dispose_tensor(hidden_states_fp16)

        # Act
        down_input = torch.empty(
            (
                num_groups,
                m,
                n // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul_mask_fwd(
            gateup_output.view(-1, gateup_output.shape[2]),
            down_input.view(-1, gateup_output.shape[2] // 2),
            masked_m,
        )
        del gateup_output
        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )
        if down_input.dtype != self.w2_weight.dtype:
            down_input = down_input.to(self.w2_weight.dtype)
        wrapper_m_grouped_gemm_nt_masked_v3(
            down_input, self.w2_weight, down_output, masked_m, expected_m
        )

        return down_output

    def forward_deepgemm_masked(
        self,
        dispatch_output: "DeepEPLLOutput",
    ):
        """
        forward_deepgemm_masked
        """
        import kunlun_ops
        hidden_states, hidden_states_scale,  _, _, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        hidden_states_fp8 = (hidden_states, hidden_states_scale)
        if self.w13_weight.dtype != torch.int8:
            return self.forward_deepgemm_masked_bfp16(
                hidden_states_fp8, masked_m, expected_m
            )
        # GroupGemm-0
        num_groups, m, k = hidden_states_fp8[0].size()
        n = self.w13_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
        )
        w13_scale = (
            self.w13_weight_scale_inv
            if self.use_block_quant
            else self.w13_weight_scale
        )
        if hidden_states_fp8[0].dtype == torch.float16:
            hidden_states_fp8 = (
                hidden_states_fp8[0].to(torch.int8),
                hidden_states_fp8[1],
            )
        kunlun_ops.m_grouped_gemm_I8_I8_bf16_nt_masked(
            hidden_states_fp8,
            (self.w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states_fp8[0])

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
        scale_block_size = gateup_output.shape[2] // 2  # per_col quant

        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        kunlun_ops.silu_and_per_token_group_quant_I8(gateup_output,
                                          down_input,
                                          down_input_scale,
                                          masked_m)
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input, down_input_scale
        )
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )
        w2_scale = (
            self.w2_weight_scale_inv
            if self.use_block_quant
            else self.w2_weight_scale
        )
        m_grouped_gemm_I8_I8_bf16_nt_masked(
            down_input_fp8,
            (self.w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):
        """run_moe_core"""
        if self.deprecate_flag:
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.use_fp8_w8a8:
                output = self.forward_deepgemm_contiguous(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                output = self.forward_deepgemm_contiguous(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if self.use_fp8_w8a8:
                output = self.forward_deepgemm_masked(dispatch_output)
            elif (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            else:
                output = self.forward_deepgemm_masked(dispatch_output)

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )
        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        """combine"""
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        """forward_flashinfer_cutedsl"""
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
        )
        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        """forward_cutlass_w4afp8"""
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        """forward_cutlass_w4afp8_masked"""
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        assert (
            envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ), "W4AFP8 does not support FP8 dispatch; please set SGLANG_DEEPEP_BF16_DISPATCH=1."
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

@plugin_hook(
    "sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class",
    type=HookType.REPLACE,
)
def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    """Return the Kunlun MoE implementation class for the given quantization."""
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
        return DeepEPMoE

    return FusedMoE
