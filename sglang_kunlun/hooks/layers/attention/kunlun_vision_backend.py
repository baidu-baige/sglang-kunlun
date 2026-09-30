"""昆仑侧的 vision attention backend（纯 torch，不依赖 triton）。

上游 `VisionAttention.__init__` 按平台选 `qkv_backend`
（`sglang/srt/layers/attention/vision.py:1268 _determine_attention_backend`）。
昆仑伪装成 CUDA，且 `torch.cuda.get_device_capability()` 返回 (8, 6)，落进该函数
docstring 里那句 "CUDA (other)" → `triton_attn` → `VisionTritonAttention` →
`context_attention_fwd`（`sglang/kernels/ops/attention/prefill_attention.py:195`
的 `_fwd_kernel`）。这个 triton kernel 在 P800 上必然抛
`AssertionError: libcuda.so cannot found!`，ViT 的第一个 block 就挂，所以带
`--skip-server-warmup` 才能启动（warmup 会发一张 dummy 图片进 ViT）。

`QKV_BACKEND_IMPL`（`vision.py:1049`）是模块级 dict，这里**新增一个 `kunlun` 条目**指向
昆仑实现（与 `hooks/layers/attention/attention_registry.py` 把 `kunlun` 注册成主干
attention backend 是同一套命名）。用 `--mm-attention-backend kunlun` 选中。

上游那个开关的 choices（`arg_groups/fields/mm.py:38`）是硬编码的 9 个名字，所以这里同时用
AROUND hook 在 parser 建好后把 `kunlun` 追加进 choices——不改 sglang 的字段声明。

语义对上游两条 N 卡实现（`VisionTritonAttention` / `VisionSdpaAttention`）：`q/k/v`
是 `[b*s, head, head_size]`，按 `[b, s]` 矩形布局理解，掩码是
`(row < seq_len_b) & (col < seq_len_b)`（`vision.py:341-350`）。DeepSeek 的 ViT 是
**一张图一条完整双向序列**，`get_image_feature` 又是逐图调用
（`models/deepseek_v4.py:4089`），所以实际永远是 `bsz == 1`、`cu_seqlens == [0, s]`：
这一档直接不加掩码，省掉上游那份 `[b,1,s,s]` 的 bool 掩码和
`cu_seqlens.cpu().tolist()` 的每层 host 同步。
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


class KunlunVisionAttention(nn.Module):
    """Vision attention 的昆仑实现：纯 torch SDPA。"""

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
                "KunlunVisionAttention 用 SDPA 自带的高精度累加，"
                "没有实现 softmax_in_single_precision 那条显式 fp32 路径"
            )
        self.head_size = kwargs.get("head_dim")
        self.num_heads = kwargs.get("num_heads")
        self.num_kv_heads = kwargs.get("num_kv_heads") or self.num_heads
        self.softmax_scale = kwargs.get("softmax_scale")
        self.dropout = kwargs.get("dropout") or 0.0

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
            raise NotImplementedError(
                f"KunlunVisionAttention 不支持滑窗/sink：window_size={window}"
            )

        # 元数据优先：DeepSeek ViT 每层都带 forward_metadata 进来。
        if forward_metadata is not None and cu_seqlens is None:
            cu_seqlens = forward_metadata.cu_seqlens

        scale = softmax_scale if softmax_scale is not None else self.softmax_scale
        if scale is None:
            # 与 context_attention_fwd 的 sm_scale 默认值一致（prefill_attention.py:187）。
            scale = 1.0 / math.sqrt(q.shape[-1])

        total, heads, head_dim = q.shape
        s = total // bsz
        kv_heads = k.shape[1]

        q_ = q.view(bsz, s, heads, head_dim).transpose(1, 2)
        k_ = k.view(bsz, s, kv_heads, head_dim).transpose(1, 2)
        v_ = v.view(bsz, s, kv_heads, head_dim).transpose(1, 2)

        mask = self._block_mask(cu_seqlens, bsz, s, q.device)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=scale,
        )
        return out.transpose(1, 2).reshape(total, heads, head_dim)

    @staticmethod
    def _block_mask(
        cu_seqlens: Optional[torch.Tensor], bsz: int, s: int, device
    ) -> Optional[torch.Tensor]:
        """`[b, 1, s, s]` 的块掩码；全双向时返回 None。

        判据只用张量形状，**不做 host 同步**：`bsz == 1` 且只有两个边界时，
        这批 token 必然是一条长度为 `s` 的完整序列（`patch_embed` 的输出长度就是
        `cu_seqlens[1]`），不需要掩码。
        """
        if cu_seqlens is None:
            return None
        if bsz == 1 and cu_seqlens.shape[0] == 2:
            return None
        ar = torch.arange(s, device=device)
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        valid = ar[None, :] < lens[:, None]  # [b, s]
        return valid[:, None, :, None] & valid[:, None, None, :]


# `--mm-attention-backend` 的 choices 写死在 arg_groups/fields/mm.py 的 Arg 里，而 Arg 是
# msgspec 的 frozen struct，改不动。所以在 parser 建好之后直接扩 argparse 的 choices。
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
    """把 `QKV_BACKEND_IMPL["kunlun"]` 指向昆仑实现。"""
    try:
        from sglang.srt.layers.attention import vision as vision_mod
    except Exception as e:  # noqa: BLE001 - 没有 vision 依赖时不该拖垮启动
        logger.warning("Kunlun vision backend: 无法 import vision 模块: %s", e)
        return
    impl = getattr(vision_mod, "QKV_BACKEND_IMPL", None)
    if not isinstance(impl, dict):
        logger.warning(
            "Kunlun vision backend: vision.QKV_BACKEND_IMPL 不存在（上游布局漂移？）"
        )
        return
    impl["kunlun"] = KunlunVisionAttention
    logger.info(
        "Kunlun vision backend 已装到 QKV_BACKEND_IMPL['kunlun']；"
        "多模态加 --mm-attention-backend kunlun"
    )


install_kunlun_vision_backend()
