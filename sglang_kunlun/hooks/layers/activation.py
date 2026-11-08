"""Kunlun activation and normalization hooks for Kimi-K3."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.activation.SituAndMul.forward_cuda",
    type=HookType.REPLACE,
)
def situ_and_mul_forward_kunlun(self, x: torch.Tensor) -> torch.Tensor:
    """Use the XSpeedGate SiTU custom op on Kunlun."""
    import xspeedgate_ops  # noqa: F401

    linear_beta = self.linear_beta if self.linear_beta is not None else 0.0
    return torch.ops.xspeedgate_ops.situ_and_mul.default(
        x.contiguous(),
        self.beta,
        linear_beta,
    )


@plugin_hook(
    "sglang.kernels.ops.attention.fla.fused_norm_gate.rms_norm_gated",
    type=HookType.REPLACE,
)
def fla_rms_norm_gated_kunlun(
    x,
    g,
    weight,
    bias,
    activation="swish",
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6,
):
    """Run the Kimi-K3 output gated RMSNorm with XSpeedGate.

    Kimi-K3 uses a sigmoid gate without bias, residual, or prenorm output.  The
    XSpeedGate op accepts two-dimensional inputs, so each final-dimension row
    is normalized independently before the original layout is restored.
    """
    if activation != "sigmoid":
        raise NotImplementedError(
            "Kunlun Kimi-K3 gated RMSNorm requires activation='sigmoid'"
        )
    if bias is not None:
        raise NotImplementedError("Kunlun Kimi-K3 gated RMSNorm does not use bias")
    if residual is not None or prenorm or residual_in_fp32:
        raise NotImplementedError(
            "Kunlun Kimi-K3 gated RMSNorm does not support residual/prenorm"
        )
    if x.dim() < 1 or g.dim() < 1:
        raise ValueError(
            "expected gated RMSNorm inputs with at least one dimension, "
            f"got x={x.shape}, gate={g.shape}"
        )
    if x.numel() != g.numel() or x.shape[-1] != g.shape[-1]:
        raise ValueError(
            "x and gate must have matching flattened rows, "
            f"got x={x.shape}, gate={g.shape}"
        )
    if weight is None or weight.numel() != x.shape[-1]:
        raise ValueError(
            f"weight must contain {x.shape[-1]} elements, "
            f"got {getattr(weight, 'shape', None)}"
        )
    if g.dtype != x.dtype or weight.dtype != x.dtype:
        raise ValueError(
            "x, gate, and weight must have the same dtype, got "
            f"{x.dtype}, {g.dtype}, {weight.dtype}"
        )

    import xspeedgate_ops  # noqa: F401

    original_shape = x.shape
    hidden_size = x.shape[-1]
    output, _, _, _ = torch.ops.xspeedgate_ops.layer_norm_gated_fwd.default(
        x.reshape(-1, hidden_size).contiguous(),
        g.reshape(-1, hidden_size).contiguous(),
        weight.reshape(-1).contiguous(),
        None,  # bias
        "sigmoid",
        eps,
        None,  # residual
        None,  # out_dtype; preserve x.dtype
        None,  # residual_dtype
        True,  # is_rms_norm
        1,  # normalize every flattened row independently
        None,  # g_stride_n
    )
    return output.reshape(original_shape)


@plugin_hook(
    "sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd",
    type=HookType.REPLACE,
)
def fla_l2norm_fwd_kunlun(
    x: torch.Tensor, eps: float = 1e-6, output_dtype=None
) -> torch.Tensor:
    """Use the Torch-native L2 normalization fallback on Kunlun."""
    from sglang_kunlun.kernels.k3_norms import l2_norm

    return l2_norm(x, eps=eps, output_dtype=output_dtype)


@plugin_hook(
    "sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul",
    type=HookType.REPLACE,
)
def fused_sigmoid_mul_kunlun(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
    quant: bool = False,
    round_sigmoid: bool = False,
) -> torch.Tensor:
    """Torch-native replacement for the Triton ``fused_sigmoid_mul``.

    Base computes ``attn_output * sigmoid(gate)`` through a Triton hidden-block
    kernel (``_fused_sigmoid_mul_kernel[grid]``), which on Kunlun is a plain
    function stub ('function' object is not subscriptable). ERNIE 5.5's gated
    attention output (``Ernie5_5HcaAttention._project_wo_b``) calls it with a 2D
    ``o.flatten(1)`` and a 2D (possibly strided) gate slice and
    ``round_sigmoid=True``.

    ``round_sigmoid`` reproduces the two-op torch spelling bit for bit: round the
    sigmoid to the gate dtype before the multiply. The strided 3D-gate path
    (Qwen3.5: gate ``[num_tokens, num_heads, head_dim]`` against a 2D output) is
    handled by flattening the gate's trailing dims. ``quant`` (per-token FP8
    output) is unused by the Kunlun model paths, so it is rejected rather than
    silently approximated.
    """
    if quant:
        raise NotImplementedError(
            "Kunlun fused_sigmoid_mul does not implement quant=True"
        )
    if gate.ndim == 3 and attn_output.ndim == 2:
        gate = gate.reshape(gate.shape[0], -1)
    if round_sigmoid:
        # sigmoid rounded to gate dtype first, then multiply (two-op spelling).
        sig = torch.sigmoid(gate)
        result = attn_output * sig.to(attn_output.dtype)
    else:
        # Match the kernel's fp32 sigmoid accumulation before the cast back.
        sig = torch.sigmoid(gate.float())
        result = (attn_output.float() * sig).to(attn_output.dtype)
    if inplace:
        attn_output.copy_(result)
        return attn_output
    return result


@plugin_hook(
    "sglang.kernels.ops.activation.multimax.multimax",
    type=HookType.REPLACE,
)
def multimax_kunlun(x: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor):
    """Torch-native replacement for the Triton ERNIE 5.5 ``multimax`` logit head.

    ERNIE 5.5's ``_compute_lm_head`` calls ``multimax`` (``activation/multimax.py``),
    which launches ``_multimax_kernel[grid]`` for bf16 inputs (a bare stub on
    Kunlun). The module already ships ``multimax_torch`` as the eager reference
    (``x + t0*relu(r0-x) + t1*relu(x-r1) + t2*relu(r2-x)^2 + t3*relu(x-r3)^2``);
    reuse it for every dtype. The Triton path rounds each primitive to bf16 for
    bit-identical output; the eager accumulation differs slightly, to be checked
    at the phase-3 precision step.
    """
    from sglang.kernels.ops.activation.multimax import multimax_torch

    return multimax_torch(x, ranges, ts)
