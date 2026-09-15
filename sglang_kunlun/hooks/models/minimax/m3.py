"""Kunlun hooks for the upstream MiniMax-M3 MoE.

Ported from the 0.5.13 line. Two hooks from that line are intentionally NOT
ported because 0.5.19 upstream implements the behavior natively:

* ``MiniMaxM3MoE.__init__`` AFTER + ``forward_normal`` AROUND (dual stream for
  shared experts): upstream now creates ``alt_stream = get_stream("alt")``
  (``_is_cuda`` is true on Kunlun through xpytorch SYMBOL_REWRITE) and overlaps
  shared/routed experts inside ``forward_normal`` (routing moved to
  ``_forward_router_experts``).
* ``MiniMaxM3MoE._compute_router_logits`` REPLACE stays: the Kunlun torch build
  still lacks ``aten::mm.dtype`` for the upstream
  ``torch.mm(..., out_dtype=torch.float32)`` router GEMM.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _router_logits_matmul(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """用 Kunlun matmul 替代上游 MiniMax-M3 的 BF16 router GEMM。

    上游 BF16 router GEMM 做的是：
    ``hidden_states[*, hidden_size] @ gate.weight.T[hidden_size, num_experts]``，
    其中 ``hidden_states`` 和 ``gate.weight`` 都是 BF16，输出
    ``router_logits[*, num_experts]`` 是 FP32。

    上游用 ``torch.mm(..., out_dtype=torch.float32)`` 表达这个计算，但该
    调用会落到 ``aten::mm.dtype``；当前 Kunlun torch29 即使输入 tensor
    已经在 ``cuda:0`` 也不能执行这个 op。

    ``kunlun_ops.matmul`` 的参数约定是 ``x[M, K]``、``weight[N, K]``、
    ``out[M, N]``，内部计算 ``x @ weight.T``。因此这里保留 BF16 的
    ``hidden_states`` 和 ``gate.weight``，只显式创建 FP32 ``out``，得到与
    上游相同 shape 和 dtype 的 router logits。
    """
    x = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
    w = weight.contiguous()
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device=x.device)
    import kunlun_ops

    x = x.to(w.dtype).contiguous()
    # matmul不支持fp16输入，fp32输出，只能用bf16的输入规避；out必须是fp32，否则会严重掉精度
    kunlun_ops.matmul(x, w, out)

    return out.view(*hidden_states.shape[:-1], w.shape[0])


@plugin_hook(
    "sglang.srt.models.minimax_m3.MiniMaxM3MoE._compute_router_logits",
    type=HookType.REPLACE,
)
def minimax_m3_moe_compute_router_logits_kunlun(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """替换 Kunlun 不支持的上游 BF16 router GEMM 分支。

    当 ``self.bf16_router_gemm`` 为 True 时，上游会调用
    ``torch.mm(..., out_dtype=torch.float32)`` 计算 FP32 router logits；
    当前 Kunlun torch29 不支持对应的 ``aten::mm.dtype``，因此改用
    ``_router_logits_matmul`` 保持 BF16 输入、FP32 输出的计算语义。

    当 ``self.bf16_router_gemm`` 为 False 时，不改变上游行为，仍调用
    ``self.gate(hidden_states.to(torch.float32))``。
    """
    if self.bf16_router_gemm:
        return _router_logits_matmul(hidden_states, self.gate.weight)
    router_logits, _ = self.gate(hidden_states.to(torch.float32))
    return router_logits
