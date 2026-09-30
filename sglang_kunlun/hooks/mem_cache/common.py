"""Kunlun memory-cache hooks."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.utils import get_num_new_pages, next_power_of_2


logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype",
    type=HookType.AROUND,
)
def configure_kv_cache_dtype_kunlun(original_fn, **kwargs):
    """把 `--kv-cache-dtype fp16` 解析成 `torch.float16`。

    昆仑的 DSV4 池子是 fp16/bf16 的"半 cache"（见下面 `dsv4_create_buffer_kunlun`），
    所以本平台跑 fp16 KV cache。上游的解析表没有 fp16 分支，会落到
    `raise ValueError(f"Unsupported kv_cache_dtype: ...")`，因此在这里拦掉。字符串槽返回
    None —— 上游用 None 表示未量化池（auto 分支就不设这个标记，backend 靠它判断是否 descale）。

    配套改动：`sitecustomize.py` 把 fp16 加回 argparse choices；`platform/srt.py` 在上游
    DeepSeek-V4 hook 断言这个值时先换成 bfloat16、跑完再声明回来。
    """
    if kwargs.get("server_args_kv_cache_dtype") == "fp16":
        return None, torch.float16
    return original_fn(**kwargs)


@plugin_hook(
    "sglang.srt.mem_cache.allocator.swa.SWATokenToKVPoolAllocator._free_swa_pages_cuda",
    type=HookType.REPLACE,
)
def free_swa_pages_kunlun(self, full_page_representatives: torch.Tensor):
    """Free SWA pages through upstream's torch path instead of its Triton one.

    main fused the resolve-and-clear of `full_to_swa_index_mapping` into
    `get_and_clear_swa_pages_kernel`, picked by `free_index.is_cuda`. XPU
    tensors answer True there, and the Kunlun triton shim hands back the plain
    function behind `@triton.jit`, so the launch dies with "'function' object is
    not subscriptable" the first time a request finishes.

    `_free_swa_pages_none_cuda` is upstream's own non-CUDA branch with the same
    contract (same mapping clear, same `_SWA_PEER_MAPPED` expectation, same
    debug assertions) and is what this platform ran before the fusion landed.
    """
    return self._free_swa_pages_none_cuda(full_page_representatives)


@plugin_hook(
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc_decode",
    type=HookType.REPLACE,
)
def paged_alloc_decode_kunlun(self, seq_lens, seq_lens_cpu, last_loc):
    """Decode allocation with a CPU-side page-count gate and zero-page fast path."""
    if self.debug_mode:
        assert torch.all(
            (last_loc + 2) % self.page_size == seq_lens % self.page_size
        )
    bs = len(seq_lens)
    if self.need_sort and bs > len(self.free_pages):
        self.merge_and_sort_free()

    num_new_pages = get_num_new_pages(
        seq_lens=seq_lens_cpu,
        page_size=self.page_size,
        decode=True,
    )
    if num_new_pages > len(self.free_pages):
        self.merge_and_sort_free()
    if num_new_pages > len(self.free_pages):
        return None

    out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
    if num_new_pages == 0:
        out_indices.copy_(last_loc.to(out_indices.dtype).contiguous() + 1)
        return out_indices

    torch.ops.xspeedgate_ops.alloc_decode_kernel(
        seq_lens,
        last_loc.to(torch.int32).contiguous(),
        self.free_pages,
        out_indices,
        next_power_of_2(bs),
        self.page_size,
        bs,
    )

    if self.debug_mode:
        assert len(torch.unique(out_indices)) == len(out_indices)

    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool.get_bytes_per_token",
    type=HookType.AROUND,
)
def dsv4_get_bytes_per_token_kunlun(original_fn, self):
    """Return the Kunlun half-cache byte footprint for one token."""
    if self.store_dtype in (torch.bfloat16, torch.float16):
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.store_dtype.itemsize
    return original_fn(self)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool.create_buffer",
    type=HookType.AROUND,
)
def dsv4_create_buffer_kunlun(original_fn, self, *, num_pages: int):
    """Allocate the Kunlun half-precision KV cache buffer."""
    if self.store_dtype not in (torch.bfloat16, torch.float16):
        return original_fn(self, num_pages=num_pages)

    dim_per_token = self.qk_nope_head_dim + self.qk_rope_head_dim
    self.kv_cache_total_dim = dim_per_token
    self.bytes_per_page_padded = (
        self.page_size * dim_per_token * self.store_dtype.itemsize
    )
    if (
        os.environ.get("DSV4_MTP_PROBE") == "1"
        and os.environ.get("RANK", "0") == "0"
        and not getattr(dsv4_create_buffer_kunlun, "_probe_logged", False)
    ):
        logger.warning(
            "[DSV4_CALLSTACK] half-cache pool buffer store_dtype=%s "
            "shape=(%d, %d) page_size=%d dim_per_token=%d",
            self.store_dtype,
            num_pages,
            self.page_size * dim_per_token,
            self.page_size,
            dim_per_token,
        )
        dsv4_create_buffer_kunlun._probe_logged = True
    dim_per_page = self.page_size * dim_per_token
    total_bytes = num_pages * dim_per_page * self.store_dtype.itemsize
    return _alloc_2m_aligned(total_bytes, self.store_dtype, self.device).reshape(
        num_pages, dim_per_page
    )


_ALIGNMENT_2M = 2 * 1024 * 1024


def _alloc_2m_aligned(total_bytes: int, dtype: torch.dtype, device: str):
    """Allocate a tensor whose ``data_ptr()`` is 2MB-aligned with 2MB-aligned
    backing pages.

    The Kunlun peermem/XDR path maps device memory at 2MB granularity, so
    ``ibv_reg_mr`` on a host VA that is not 2MB-aligned fails with EINVAL(22).
    Allocating ``ALIGN_2M(total_bytes) + 2MB`` guarantees both that the returned
    view starts on a 2MB boundary and that physical pages cover the full
    ``ALIGN_2M(total_bytes)`` MR registration range from that boundary. The view
    itself exposes exactly ``total_bytes`` so callers can reshape as expected.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    assert total_bytes % element_size == 0
    aligned_bytes = (total_bytes + _ALIGNMENT_2M - 1) // _ALIGNMENT_2M * _ALIGNMENT_2M
    flat = torch.zeros(
        aligned_bytes + _ALIGNMENT_2M,
        dtype=torch.uint8,
        device=device,
    )
    offset = (_ALIGNMENT_2M - flat.data_ptr() % _ALIGNMENT_2M) % _ALIGNMENT_2M
    return flat[offset : offset + total_bytes].view(dtype)


@plugin_hook(
    "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator.__init__",
    type=HookType.AFTER,
)
def dsv4_pool_configurator_kv_bytes_kunlun(result, self, kvc):
    """按昆仑半 cache 的槽位大小定 DSV4 各池子的容量。

    `DSV4PoolConfigurator` 自己算 `kv_bytes = qk_nope + qk_rope*2 + 8`（448+128+8=584 B，
    N 卡 FlashMLA 的 fp8 nope + bf16 rope + ue8m0 scale 打包），**从不问 pool 的
    `get_bytes_per_token()`**，所以上面那个 `dsv4_get_bytes_per_token_kunlun` 影响不到容量。
    昆仑一个槽位是 `(qk_nope + qk_rope)` 个 fp16 元素 = 1024 B，比它大 1.75 倍，于是
    configurator 发的 token 额度是实际能装下的 1.72 倍（`bytes_per_full_token` 口径
    3966 → 应为 6826，full_token 5804544 → 3372544）。表现是 `--mem-fraction-static 0.75`
    在建池阶段 OOM（0.6 只是踩着卡的物理上限侥幸过关），以及 `full token usage` 恒为 0.00。

    上游自己的 bf16 latent 池就是 `attn_head_dim * 2`（`self._unified` 分支），这里取同一个值。

    只改 `kv_bytes` 不够：`bytes_per_full_token` 在 `__init__` 内就算完了，所以要连带重算
    依赖它的四个缓存值。不依赖 `kv_bytes` 的 `swa_cap_tokens` / `swa_prefix_tails` 不用动，
    而 `_get_swa_fixed_bytes()` 是调用时才读 `kv_bytes` 的方法，会自动跟上。
    """
    if self.kv_cache_dtype_str not in ("fp16", "float16", "bf16", "bfloat16"):
        return result
    if getattr(self, "_unified", False):
        return result  # unified KV 本来就按 bf16 latent 定容量

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        get_dsv4_indexer_bytes_per_token,
    )

    old_kv_bytes = self.kv_bytes
    self.kv_bytes = self.attn_head_dim * 2  # fp16/bf16 各 2 字节

    if self.encoder_replay:
        # `request_window_bytes` 也按 kv_bytes 算，但只在 encoder bounded replay 下非零。
        raise NotImplementedError(
            "encoder SWA bounded replay 的 request_window_bytes 未按昆仑半 cache 重算"
        )

    cfg = kvc.model_config
    low_ratio_index_bytes = get_dsv4_indexer_bytes_per_token(
        self.indexer_head_dim, use_fp4_indexer=True
    )
    self.low_ratio_bytes_per_full_token = sum(
        (self.kv_bytes + low_ratio_index_bytes) / cfg.compress_ratios[l]
        for l in cfg.hf_config.kv_source_layer_ids
        if kvc.layer_info.start_layer <= l < kvc.layer_info.end_layer
        and cfg.compress_ratios[l] in (1, 2)
    )
    self.bytes_per_swa_token = self._get_bytes_per_swa_token()
    self.bytes_per_full_token = self._get_bytes_per_full_token()
    if self.is_speculative:
        self.bytes_per_full_token *= self._spec_infl
        self.bytes_per_swa_token *= self._spec_infl

    logger.info(
        "Kunlun half-cache DSV4 pool sizing: kv_bytes %d -> %d, "
        "bytes_per_full_token %.2f, bytes_per_swa_token %.2f",
        old_kv_bytes,
        self.kv_bytes,
        self.bytes_per_full_token,
        self.bytes_per_swa_token,
    )
    return result


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4IndexerPool._create_buffer",
    type=HookType.AROUND,
)
def dsv4_indexer_pool_create_buffer_kunlun(original_fn, self):
    """Allocate the Kunlun indexer pool with RDMA-safe 2MB backing pages."""
    from sglang_kunlun.hooks.utils.common import _is_kunlun

    if not _is_kunlun():
        return original_fn(self)

    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

    page_bytes = self.page_size * self.get_bytes_per_token()
    num_pages = (self.size + self.page_size + 1) // self.page_size
    dtype = self.index_k_with_scale_buffer_dtype
    total_bytes = num_pages * page_bytes * torch.tensor([], dtype=dtype).element_size()
    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                _alloc_2m_aligned(total_bytes, dtype, self.device).reshape(
                    num_pages, page_bytes
                )
                for _ in range(self.layer_num)
            ]


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_compress_state.CompressStatePool."
    "_alloc_kv_score_buffer",
    type=HookType.AROUND,
)
def compress_state_alloc_kv_score_buffer_kunlun(
    original_fn, self, *, dtype, device, enable_memory_saver
):
    """Allocate the compress-state kv+score buffer with RDMA-safe 2MB pages.

    ``kv_score_buffer.kv_score`` is published to the PD peer through
    ``DeepSeekV4TokenToKVPool.get_state_buf_infos()``, so it has to satisfy the
    same 2MB alignment constraint as the KV and indexer pools.
    """
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
    from sglang.srt.mem_cache.memory_pool import (
        TorchMemorySaverAdapter,
        maybe_init_custom_mem_pool,
    )
    from sglang_kunlun.hooks.utils.common import _is_kunlun

    if not _is_kunlun():
        return original_fn(
            self,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

    self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=enable_memory_saver
    )
    self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
        maybe_init_custom_mem_pool(device=device)
    )
    total_bytes = (
        self._size * self.last_dim * torch.tensor([], dtype=dtype).element_size()
    )
    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.kv_score_buffer = KVAndScore(
                _alloc_2m_aligned(total_bytes, dtype, device).reshape(
                    self._size, self.last_dim
                )
            )
    self.kv_score_buffer.clear()


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool.MambaPool._should_fuse_slot_ops",
    type=HookType.AROUND,
)
def disable_fused_mamba_slot_ops(original_fn, self) -> bool:
    """Disable fused slot operations only on the Kunlun platform.

    Returns:
        ``False`` on Kunlun so the upstream Torch clear/copy fallback runs;
        ``True`` elsewhere so native fused operations remain unchanged.
    """
    from sglang.srt.platforms import current_platform

    if getattr(current_platform, "device_name", None) == "kunlun":
        return False
    return original_fn(self)
