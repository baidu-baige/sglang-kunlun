"""MiniMax 稀疏注意力 prefill / decode 共用的 Kunlun 工具函数。"""

from functools import lru_cache
from typing import Optional, Tuple

import torch


@lru_cache(maxsize=1)
def _backend_classes():
    """一次性解析并缓存需要做 isinstance 判断的后端类。

    这里用延迟 import 是为了避开 hook 加载期的循环依赖，但下面几个函数每个稀疏层都会
    调用，每次重走 import machinery 约 7 us（60 层 × 2 个 leaf 量级下累计到毫秒），
    所以把结果缓存起来。
    """
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        MiniMaxHybridAttnBackend,
    )
    from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

    from sglang_kunlun.hooks.layers.attention.kunlun_backend import (
        KunlunAttentionBackend,
    )

    return MiniMaxHybridAttnBackend, TboAttnBackend, KunlunAttentionBackend


def _get_dense_backend():
    """返回当前 MiniMax hybrid 后端背后的 Kunlun dense 后端。"""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    hybrid_cls, tbo_cls, kunlun_cls = _backend_classes()
    backend = get_attn_backend()
    if isinstance(backend, tbo_cls):
        backend = backend.primary
    if not isinstance(backend, hybrid_cls):
        raise RuntimeError("Kunlun MiniMax requires MiniMaxHybridAttnBackend")
    if not isinstance(backend.dense, kunlun_cls):
        raise RuntimeError("Kunlun MiniMax requires KunlunAttentionBackend")
    return backend.dense


def _get_metadata():
    """返回当前 Kunlun 后端持有的 forward metadata。"""
    metadata = _get_dense_backend().forward_metadata
    if metadata is None:
        raise RuntimeError("Kunlun attention metadata is not initialized")
    return metadata


def _get_metadata_and_page_size() -> Tuple[object, int]:
    """一次后端查找同时返回 ``(forward_metadata, kv_page_size)``。

    上游的 prefill / decode-main leaf 签名里没有 page size（它们按 token 粒度经
    ``req_to_token`` 寻址），而 Kunlun 需要页粒度：``page_table`` 的列是页，
    ``_paged_cache`` 也要按页切分。稀疏 block 大小（``block_size_k``）与页大小是两个
    独立的量，不能互相顶替。
    """
    backend = _get_dense_backend()
    metadata = backend.forward_metadata
    if metadata is None:
        raise RuntimeError("Kunlun attention metadata is not initialized")
    page_size = backend.page_size
    if page_size <= 0:
        raise RuntimeError(f"invalid Kunlun KV page size: {page_size}")
    return metadata, page_size


def _lod_pair(
    device_lod: torch.Tensor,
    host_lod: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 Kunlun 算子需要的 ``(host, device)`` LOD 二元组。

    Kunlun 后端已经在 ``metadata.extra_attn_metadata`` 里备好了累积序列长度的 host
    副本，直接复用可以省掉每层一次 D2H 拷贝（以及随之而来的 device 同步）。没有填充
    host 副本的 forward mode 退回显式 ``.cpu()``。1-D 切片必然连续，无需 contiguous。
    """
    device_lod = device_lod[: batch_size + 1]
    if host_lod is not None and host_lod.shape[0] >= batch_size + 1:
        return host_lod[: batch_size + 1], device_lod
    return device_lod.cpu(), device_lod


def _lods(metadata, batch_size: int):
    """返回当前 batch 的 ``(q_lod_cpu, q_lod, kv_lod_cpu, kv_lod)``。"""
    extra = getattr(metadata, "extra_attn_metadata", None)
    q_lod_cpu, q_lod = _lod_pair(
        metadata.cu_seqlens_q, getattr(extra, "cu_seqlens_q_cpu", None), batch_size
    )
    kv_lod_cpu, kv_lod = _lod_pair(
        metadata.cu_seqlens_k, getattr(extra, "kv_lod_cpu", None), batch_size
    )
    return q_lod_cpu, q_lod, kv_lod_cpu, kv_lod


def _full_page_table(page_table: torch.Tensor, batch_size: int) -> torch.Tensor:
    """返回全宽 page table，只按 batch 维切片。

    不按 ``max_seq_len_k`` 裁剪列，原因有两点：

    * CUDA graph replay 时 Python 代码只在 capture 期执行，``metadata.max_seq_len_k``
      这类 Python int 会被冻结成 capture 时的长度，按它裁剪列会把每次 replay 的注意力
      窗口钉死，产生重复/脏输出。预分配宽度是 capture-invariant 的。
    * 列切片会产生非连续视图，而算子只取 ``data_ptr()``，必须再拷贝一份；行切片则是
      零拷贝。每个稀疏层都要走一次，代价可观。

    每序列的有效范围由 live 的 ``seq_lens`` / KV LOD 在算子内部界定。
    """
    if page_table.dim() != 2 or page_table.shape[0] < batch_size:
        raise RuntimeError("Kunlun MiniMax page table must be a 2D batch table")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("Kunlun MiniMax page table must use an integer dtype")
    return page_table[:batch_size]


def _paged_cache(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    """把 NHD slot-major 或 HND 分页 cache 归一成 HND 分页布局。

    kunlun_ops 的 msa 算子只取 ``data_ptr()``，不接收 stride（见 kunlun_ops
    wrapper/kernel/attention/msa_block_score.cpp 与 msa_sparse_attention.cpp），
    所以返回值必须是 HND 顺序下真正连续的内存，否则算子会按连续布局误读。

    NHD 池 ``[slots, heads, dim]`` 经 view+permute 后，仅在 ``heads == 1`` 时恰好仍
    连续；``heads > 1`` 时必须让 KV pool 直接产出 HND（``SGLANG_USE_HND_KVCACHE=true``），
    在这里 ``.contiguous()`` 会每层拷贝整块 KV cache，不可接受。
    """
    if cache.dim() == 4:
        if cache.shape[2] != page_size:
            raise RuntimeError("Kunlun MiniMax cache page size mismatch")
        return cache
    if cache.dim() != 3 or cache.shape[0] % page_size != 0:
        raise RuntimeError("Kunlun MiniMax cache must be page-aligned 3D or 4D")
    slots, heads, dim = cache.shape
    paged = cache.view(slots // page_size, page_size, heads, dim).permute(0, 2, 1, 3)
    if not paged.is_contiguous():
        raise RuntimeError(
            f"Kunlun MiniMax requires a contiguous HND paged cache, but the NHD "
            f"slot-major pool with num_kv_heads={heads} yields a strided view. "
            f"Set SGLANG_USE_HND_KVCACHE=true so the KV pool allocates HND directly."
        )
    return paged
