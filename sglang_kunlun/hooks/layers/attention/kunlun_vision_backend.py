"""Kunlun-side vision attention backend (using kunlun_ops flash attention).

Why this backend exists: upstream `VisionAttention.__init__` selects `qkv_backend`
by platform (`_determine_attention_backend` in `sglang/srt/layers/attention/vision.py`).
Kunlun pretends to be CUDA and `torch.cuda.get_device_capability()` returns (8, 6),
which lands in the "CUDA (other)" branch in that function's docstring -> `triton_attn` ->
`VisionTritonAttention` -> `context_attention_fwd`. That triton kernel always raises
`AssertionError: libcuda.so cannot found!` on the P800, so the first ViT block crashes.

`QKV_BACKEND_IMPL` (`vision.py:1049`) is a module-level dict; we add a `kunlun`
entry to it (the same naming scheme as `attention_registry.py` registering `kunlun`
as the backbone attention backend), selected via `--mm-attention-backend kunlun`.
The upstream switch's choices (`arg_groups/fields/mm.py`) are hardcoded, so we also
use an AROUND hook to append `kunlun` to the choices after the parser is built.

Implementation origin: b1b0584 of kunlun-0.5.19 (AICapX-841 [feat] support mm
backend), i.e. `kunlun_ops.flash_attn_varlen_func`. This version **replaces** this
file's original pure-torch SDPA implementation: for the 1080P real input tier (ViT
seq 8418) it is a net win; see `engram/task2_vit_attention_b1b0584.md` for numbers.

Differences from b1b0584 are all tightenings rather than algorithm changes:

- Sliding window / attention sink / `output_ws` now raise explicit errors. b1b0584
  swallowed them into `**kwargs` and dropped them, while `VisionAttention.forward`
  passes these two arguments every time (`vision.py:1530-1545`), and the kernel call
  below is unconditionally full-context with `causal=False` -- the day a model sets a
  sliding window it would silently compute wrong results.
- `forward_metadata.cu_seqlens` is used as the fallback: DeepSeek ViT passes metadata
  into every layer.

The `softmax_scale` premultiplication is kept as in b1b0584: `kunlun_ops` divides by
`sqrt(head_dim)` internally again, so the caller must multiply it back first.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


class KunlunVisionAttention(nn.Module):
    """Kunlun implementation of vision attention: kunlun_ops varlen flash attention."""

    def __init__(self, **kwargs):
        super().__init__()
        from sglang.srt.environ import envs

        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            raise NotImplementedError(
                "KunlunVisionAttention 还不支持 SGLANG_VIT_ENABLE_CUDA_GRAPH；"
                "该模式要求 caller 传 output_ws 并把 cu_seqlens 拆成三张张量"
            )
        if kwargs.get("flatten_batch"):
            raise NotImplementedError(
                "KunlunVisionAttention 只实现非 flatten 布局；"
                "DeepSeek ViT 走的是 [b, s] 矩形布局，flatten_batch 为 False"
            )
        if kwargs.get("softmax_in_single_precision"):
            raise NotImplementedError(
                "KunlunVisionAttention 没有实现 softmax_in_single_precision "
                "那条显式 fp32 路径"
            )
        self.head_size = kwargs.get("head_dim")
        self.num_heads = kwargs.get("num_heads")
        self.num_kv_heads = kwargs.get("num_kv_heads") or self.num_heads
        self.softmax_scale = kwargs.get("softmax_scale")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        bsz: int = 1,
        seq_len: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        forward_metadata=None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        """forward"""
        if kwargs.get("output_ws") is not None:
            raise NotImplementedError(
                "KunlunVisionAttention 不支持 output_ws（ViT CUDA graph 模式）"
            )
        window = kwargs.get("window_size", (-1, -1))
        if window != (-1, -1) or kwargs.get("s_aux") is not None:
            # The kernel call below is full-context and non-causal; silently dropping either one
            # would compute wrong results.
            raise NotImplementedError(
                "KunlunVisionAttention 不支持滑窗/sink：window_size=%s s_aux=%s"
                % (window, kwargs.get("s_aux") is not None)
            )
        if isinstance(cu_seqlens, list):
            raise RuntimeError(
                "KunlunVisionAttention 不支持 list 形式的 cu_seqlens："
                "kunlun_ops.flash_attn_varlen_func 需要 CPU 侧的 cu_seqlens 元数据"
            )

        # Prefer metadata: DeepSeek ViT passes forward_metadata into every layer.
        if cu_seqlens is None and forward_metadata is not None:
            cu_seqlens = getattr(forward_metadata, "cu_seqlens", None)

        from sglang.srt.layers.attention.vision import resolve_seqlens

        cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
        if cu_seqlens.device != q.device:
            raise RuntimeError(
                "KunlunVisionAttention 要求 cu_seqlens 在 %s，实际在 %s"
                % (q.device, cu_seqlens.device)
            )
        # The kernel needs two copies of the offsets: device-side int32 for addressing, host-side for the
        # segmented loop bounds.
        cu_seqlens_xpu = cu_seqlens.to(dtype=torch.int32).contiguous()
        cu_seqlens_cpu = cu_seqlens_xpu.cpu().contiguous()
        seq_lens = cu_seqlens_xpu[1:] - cu_seqlens_xpu[:-1]
        max_seqlen = int(seq_lens.max().item())

        import kunlun_ops

        compute_dtype = (
            q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        )
        scale = softmax_scale if softmax_scale is not None else self.softmax_scale
        # kunlun_ops divides by sqrt(head_dim) internally again, so multiply it back first here.
        kunlun_scale = None if scale is None else scale * (q.shape[-1] ** 0.5)
        out, _ = kunlun_ops.flash_attn_varlen_func(
            q.to(dtype=compute_dtype).contiguous(),
            k.to(dtype=compute_dtype).contiguous(),
            v.to(dtype=compute_dtype).contiguous(),
            cu_seqlens_qo=cu_seqlens_xpu,
            cu_seqlens_kv=cu_seqlens_xpu,
            max_seqlen_qo=max_seqlen,
            max_seqlen_kv=max_seqlen,
            dropout_p=0.0,
            softmax_scale=kunlun_scale,
            causal=False,
            deterministic=False,
            is_varlen=True,
            is_prefill=True,
            cu_seqlens_qo_cpu=cu_seqlens_cpu,
        )
        return out.to(dtype=q.dtype)


# `--mm-attention-backend` choices are hardcoded in the Arg in arg_groups/fields/mm.py,
# and Arg is a frozen msgspec struct that cannot be modified. So after the parser is built,
# directly extend argparse's choices.
@plugin_hook(
    "sglang.srt.server_args.ServerArgs.add_cli_args", type=HookType.AROUND
)
def allow_kunlun_mm_backend(original_fn, parser, *args, **kwargs):
    """kunlun_mm_backend"""
    result = original_fn(parser, *args, **kwargs)
    for action in getattr(parser, "_actions", ()):
        if "--mm-attention-backend" in getattr(action, "option_strings", ()):
            choices = getattr(action, "choices", None)
            if choices is not None and "kunlun" not in choices:
                action.choices = list(choices) + ["kunlun"]
                logger.info("已把 kunlun 加入 --mm-attention-backend 的 choices")
    return result


def install_kunlun_vision_backend() -> None:
    """Point `QKV_BACKEND_IMPL["kunlun"]` at the Kunlun implementation."""
    try:
        from sglang.srt.layers.attention import vision as vision_mod
    except Exception as e:  # noqa: BLE001 - must not stall startup when vision deps are missing
        logger.warning("Kunlun vision backend: 无法 import vision 模块: %s", e)
        return
    impl = getattr(vision_mod, "QKV_BACKEND_IMPL", None)
    if not isinstance(impl, dict):
        logger.warning(
            "Kunlun vision backend: vision.QKV_BACKEND_IMPL 不存在（上游布局漂移？）"
        )
        return
    impl["kunlun"] = KunlunVisionAttention
    logger.debug(
        "Kunlun vision backend 已装到 QKV_BACKEND_IMPL[kunlun]（kunlun_ops "
        "flash attention）；多模态加 --mm-attention-backend kunlun",
    )


install_kunlun_vision_backend()
