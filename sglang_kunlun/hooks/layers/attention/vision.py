"""Kunlun multimodal vision attention backend."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class VisionKunlunAttention(nn.Module):
    """Kunlun multimodal attention backend."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bsz: int,
        seq_len: int,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run multimodal packed attention through Kunlun kernels."""

        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get() and isinstance(cu_seqlens, list):
            raise RuntimeError(
                "VisionKunlunAttention does not support ViT CUDA Graph list "
                "cu_seqlens yet because kunlun_ops.flash_attn_varlen_func "
                "requires CPU cu_seqlens metadata and MiniMax M3 does not enter this path."
            )

        from sglang.srt.layers.attention.vision import resolve_seqlens

        cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
        if cu_seqlens.device != q.device:
            raise RuntimeError(
                "VisionKunlunAttention expects cu_seqlens on device {}\n".format(q.device)
            )
        cu_seqlens_xpu = cu_seqlens.to(dtype=torch.int32).contiguous()
        cu_seqlens_cpu = cu_seqlens_xpu.cpu().contiguous()
        seq_lens_xpu = cu_seqlens_xpu[1:] - cu_seqlens_xpu[:-1]
        max_seqlen = int(seq_lens_xpu.max().item())

        import kunlun_ops

        compute_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        kunlun_softmax_scale = (
            None if softmax_scale is None else softmax_scale * (q.shape[-1] ** 0.5)
        )
        out, _ = kunlun_ops.flash_attn_varlen_func(
            q.to(dtype=compute_dtype).contiguous(),
            k.to(dtype=compute_dtype).contiguous(),
            v.to(dtype=compute_dtype).contiguous(),
            cu_seqlens_qo=cu_seqlens_xpu,
            cu_seqlens_kv=cu_seqlens_xpu,
            max_seqlen_qo=max_seqlen,
            max_seqlen_kv=max_seqlen,
            dropout_p=0.0,
            softmax_scale=kunlun_softmax_scale,
            causal=False,
            deterministic=False,
            is_varlen=True,
            is_prefill=True,
            cu_seqlens_qo_cpu=cu_seqlens_cpu,
        )
        return out.to(dtype=q.dtype)


def _extend_mm_attention_backend_choices() -> None:
    """Declare ``kunlun`` as a valid ``--mm-attention-backend`` value."""
    import dataclasses
    import sys
    import typing

    from sglang.srt.arg_groups.arg_utils import Arg
    from sglang.srt.server_args import ServerArgs

    raw = ServerArgs.__annotations__.get("mm_attention_backend")
    if raw is None:
        return
    # server_args.py uses ``from __future__ import annotations``, so the entry is
    # a string; evaluate just this one field in its own module namespace instead
    # of resolving every annotation on the class.
    hint = (
        raw
        if not isinstance(raw, str)
        else eval(raw, vars(sys.modules[ServerArgs.__module__]))  # noqa: S307
    )
    metadata = getattr(hint, "__metadata__", ())
    arg_meta = next((meta for meta in metadata if isinstance(meta, Arg)), None)
    if arg_meta is None or not arg_meta.choices or "kunlun" in arg_meta.choices:
        return

    base_type = typing.get_args(hint)[0]
    rest = tuple(meta for meta in metadata if meta is not arg_meta)
    # Tuple subscript: starred unpacking inside ``Annotated[...]`` needs 3.11.
    ServerArgs.__annotations__["mm_attention_backend"] = typing.Annotated[
        (
            base_type,
            dataclasses.replace(arg_meta, choices=[*arg_meta.choices, "kunlun"]),
            *rest,
        )
    ]
    logger.info("Declared 'kunlun' as a --mm-attention-backend choice")


def _register_kunlun_mm_backend() -> None:
    """Register the Kunlun multimodal attention backend."""
    from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL

    QKV_BACKEND_IMPL["kunlun"] = VisionKunlunAttention
    logger.info("Registered 'kunlun' multimodal attention backend")


_extend_mm_attention_backend_choices()
_register_kunlun_mm_backend()