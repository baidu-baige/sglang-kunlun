"""Kunlun memory-cache hooks."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.utils import get_num_new_pages, next_power_of_2


logger = logging.getLogger(__name__)


def low_ratio_index_k_int8_enabled() -> bool:
    """Select the int8 low-ratio index cache consistently for sizing and IO."""
    return os.environ.get("DSV4_LOW_RATIO_INDEX_KCACHE", "fp4").lower() == "int8"


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype",
    type=HookType.AROUND,
)
def configure_kv_cache_dtype_kunlun(original_fn, **kwargs):
    """把 `--kv-cache-dtype fp16` / `int8` 解析成昆仑自己的 KV dtype。

    昆仑的 DSV4 池子是 fp16/bf16 的"半 cache"（见下面 `dsv4_create_buffer_kunlun`），
    所以本平台跑 fp16 KV cache。上游的解析表没有 fp16 分支，会落到
    `raise ValueError(f"Unsupported kv_cache_dtype: ...")`，因此在这里拦掉。字符串槽返回
    None —— 上游用 None 表示未量化池（auto 分支就不设这个标记，backend 靠它判断是否 descale）。

    `int8` 同理：GLM-5.3 w4a8 的生产口径是 `--kv-cache-dtype int8`（NSA/DSA 的混合
    KV layout：per-tile INT8 nope + FP32 scales + BF16 rope），0.5.8 产线分支在
    `model_runner.configure_kv_cache_dtype` 里自带这个分支，0.5.23 的上游解析表没有，
    于是 boot 阶段就 `ValueError: Unsupported kv_cache_dtype: int8.`。`sitecustomize.py`
    已经把 int8 加回 argparse choices、`hooks/server_args.py` 已经让它过了 DSA 的
    dtype 断言，所以这里是最后一道缺口。

    speculative draft 保持平铺的 model-dtype 池，和 0.5.8 产线一致：draft 只有一层 MTP，
    容量代价可忽略，而 draft attention 没有 dequant / mixed-row 处理。字符串槽给
    `"auto"` —— 上游自己的 DFLASH 分支就用这个值标记"未量化池"（`kv_cache_dtype.py`），
    而 0.5.23 的 `kv_cache_dtype_str` 是 per-runner 的，draft 不能继承 target 的 `int8`。

    配套改动：`sitecustomize.py` 把 fp16/int8 加回 argparse choices；`platform/srt.py`
    在上游 DeepSeek-V4 hook 断言这个值时先换成 bfloat16、跑完再声明回来。
    """
    requested = kwargs.get("server_args_kv_cache_dtype")
    draft_override = kwargs.get("speculative_draft_kv_cache_dtype")
    if kwargs.get("is_draft_worker") and draft_override is not None:
        # Mirror upstream's precedence: an explicit draft dtype wins.
        requested = draft_override
    if requested == "fp16":
        return None, torch.float16
    if requested == "int8":
        # The draft worker gets the same int8 mixed row as the target. It used
        # to be given a flat model-dtype pool ("auto", model_dtype) on the
        # theory that the one-layer MTP draft was too small to matter, but that
        # is both unlike the 0.5.8 line (its draft stores through the same
        # `nsa_kv_cache_store_int8` path) and actively harmful here:
        #   * the fp16 draft pool is 1152 B/token against int8's 656 B/token,
        #     i.e. it costs memory rather than saving it;
        #   * an fp16 draft pool cannot use `write_mixed_int8_rows`, so its
        #     writes fall through to upstream's `_scatter_mla_rows` -> the
        #     platform has no Triton -> `set_mla_kv_buffer_naive`, which during
        #     decode-CUDA-graph capture asks for a 7.5 GiB transient and OOMs
        #     (measured 2026-11-11: `Capture cuda graph failed: CUDA out of
        #     memory. Tried to allocate 7.57 GiB`, frame
        #     `mla_buffer.py:154 set_mla_kv_buffer_naive` -> `__floordiv__`).
        # The draft attention is the same KunlunNSA backend class, so it reads
        # the mixed row through `dequant_mixed_kv_table` exactly like the target.
        return None, torch.int8
    return original_fn(**kwargs)


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
    type=HookType.AROUND,
)
def calculate_mla_kv_cache_dim_kunlun(original_fn, *, model_config, kv_cache_dtype):
    """Report the int8 mixed row's per-token width to every sizing site.

    Port of the 0.5.17 hook of the same name. 0.5.23 shrank the signature
    (``server_args`` is gone; every caller passes keywords:
    ``kv_cache_configurator.py:2383``, ``pool_configurator.py:380/684``,
    ``page_interleave.py:193``), and it still special-cases only fp8: int8
    falls through to ``kv_lora_rank + qk_rope_head_dim`` (576), which is 80
    bytes short of the 656-byte row the int8 write path consumes
    (``kernels/mixed_int8_kv.py``). Every consumer multiplies the result by
    ``element_size(kv_cache_dtype)``, and that is 1 for int8, so the byte count
    is the element count here.

    Three places need the number and each one breaks differently without it:

    * the MLA/DSA pool sizers price a token from it
      (``pool_configurator.py:380/684``, ``page_interleave.py:193``), so 576
      under-counts the pool by 12% and the sizer hands out ~1.14x more tokens
      than the buffers hold;
    * the MTP/NextN geometry check (``kv_cache_configurator.py:2024``) compares
      it against ``target_pool.kv_cache_dim``, which the Kunlun DSA pool now
      sets to the mixed width -- without this hook the boot dies on
      "MTP and target must use matching DSA cache geometry";
    * ``_page_shard_row_bytes`` adds the indexer bytes on top of it.

    Keyed off ``kv_lora_rank`` -- the DSA pool stores the compressed latent as
    its noPE part -- and off ``DSATokenToKVPool.quant_block_size``, so the width
    cannot drift away from the tile size the writer quantizes with. Gated on
    ``is_deepseek_dsa``: it is the predicate upstream itself uses to pick the
    layout, and it excludes ``DeepseekV4ForCausalLM``, whose single-KV pool has
    its own 604-byte tiling (``dsv4_get_bytes_per_token_kunlun``) and must not
    be priced with the DSA row.
    """
    from sglang.srt.configs.model_config import is_deepseek_dsa
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    from sglang_kunlun.kernels.mixed_int8_kv import mixed_int8_bytes_per_token

    if kv_cache_dtype == torch.int8 and is_deepseek_dsa(model_config.hf_config):
        return mixed_int8_bytes_per_token(
            model_config.kv_lora_rank,
            model_config.qk_rope_head_dim,
            DSATokenToKVPool.quant_block_size,
        )
    return original_fn(
        model_config=model_config,
        kv_cache_dtype=kv_cache_dtype,
    )


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


def _kunlun_half_cache_applies(cfg) -> bool:
    """Whether this DSV4 pool is the Kunlun plain-fp16/bf16 "half cache"."""
    if cfg.kv_cache_dtype_str not in ("fp16", "float16", "bf16", "bfloat16"):
        return False
    # unified_kv already prices a plain latent row, and trtllm uses its own
    # uniform-FP8 pool; neither is the Kunlun half cache.
    if getattr(cfg, "_unified", False):
        return False
    return True


def _kunlun_paged_kv_bytes_per_token(cfg) -> int:
    """One Kunlun KV slot in bytes: ``(qk_nope + qk_rope)`` fp16 elements."""
    return (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) * 2


@plugin_hook(
    "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator."
    "_get_paged_kv_bytes_per_token",
    type=HookType.AROUND,
)
def dsv4_paged_kv_bytes_per_token_kunlun(original_fn, self, compress_ratio: int = 0):
    """Price a paged DSV4 KV token at the Kunlun slot size.

    0.5.22 moved the per-token KV byte count out of ``self.kv_bytes`` and into
    this method, which derives it from the *reader's* page layout
    (``select_dsv4_kv_layout()`` -> ``KVLayout.V4.page_bytes(page_size)``, i.e.
    448 B fp8 nope + 128 B bf16 rope + 8 B ue8m0 scales = 584 B/token, padded to
    the TMA row stride). That is the NVIDIA FlashMLA packing. The Kunlun pool
    stores a plain fp16/bf16 latent instead -- ``(qk_nope + qk_rope)`` halves,
    1024 B/token -- and ``kernel_ops._assert_v4_kv_layout`` refuses every packed
    layout, so the slot width is the same at every compression ratio.

    Under-counting by 1.75x is not a rounding error: the configurator hands out
    1.72x more tokens than fit, which OOMs while building the pool (and made
    ``full token usage`` read 0.00 when it did squeeze through). Correcting it
    here instead of patching ``bytes_per_full_token`` afterwards means the SWA
    term, the c4/c128 terms and the ratio-1/2 term all pick it up from upstream's
    own formulas, so they keep tracking upstream drift.
    """
    if not _kunlun_half_cache_applies(self):
        return original_fn(self, compress_ratio)
    return _kunlun_paged_kv_bytes_per_token(self)


@plugin_hook(
    "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator.__init__",
    type=HookType.AFTER,
)
def dsv4_pool_configurator_kv_bytes_kunlun(result, self, kvc):
    """Re-price the DSV4 pool coefficients at the Kunlun slot size.

    ``_get_paged_kv_bytes_per_token`` above covers the SWA and c4/c128 terms, but
    ``_compressed_bytes_per_full_token`` prices ratio 1 and 2 straight off
    ``self.kv_bytes`` (the FlashMLA 584 B constant), and both coefficients were
    already computed by the time this AFTER hook runs. Correct ``kv_bytes`` and
    re-derive the two coefficients through upstream's own methods, re-applying
    the speculative inflation it stored in ``_spec_infl`` -- rather than adding a
    hand-rolled delta, which is what the 0.5.19 version of this hook did and
    which stopped matching the formula in 0.5.22.
    """
    del kvc
    if low_ratio_index_k_int8_enabled():
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            get_dsv4_indexer_bytes_per_token,
        )

        self.low_ratio_index_bytes = get_dsv4_indexer_bytes_per_token(
            self.indexer_head_dim, use_fp4_indexer=False
        )
    if not _kunlun_half_cache_applies(self):
        if low_ratio_index_k_int8_enabled():
            self.bytes_per_full_token = self._get_bytes_per_full_token() * self._spec_infl
        return result

    old_kv_bytes = self.kv_bytes
    self.kv_bytes = _kunlun_paged_kv_bytes_per_token(self)
    if getattr(self, "encoder_replay", False):
        # request_window_bytes is a fixed-byte bias computed from kv_bytes before
        # this hook runs; recomputing it would mean duplicating that formula.
        logger.warning(
            "Kunlun half-cache DSV4 sizing: --enable-encoder-swa-bounded-replay "
            "prices its request window from kv_bytes before this hook runs, so "
            "request_window_bytes still carries the %d B FlashMLA slot.",
            old_kv_bytes,
        )
    self.bytes_per_swa_token = self._get_bytes_per_swa_token() * self._spec_infl
    self.bytes_per_full_token = self._get_bytes_per_full_token() * self._spec_infl

    logger.info(
        "Kunlun half-cache DSV4 pool sizing: kv_bytes %d -> %d, "
        "bytes_per_swa_token %.2f, bytes_per_full_token %.2f",
        old_kv_bytes,
        self.kv_bytes,
        self.bytes_per_swa_token,
        self.bytes_per_full_token,
    )
    return result


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4IndexerPool.__init__",
    type=HookType.AROUND,
)
def dsv4_indexer_pool_init_kunlun(original_fn, self, *args, **kwargs):
    """Use 132-byte int8 rows for explicitly forced low-ratio pools.

    C4 leaves use_fp4_indexer unspecified in 0.5.22 and retains its own layout.
    Bind the signature to support both positional and keyword callers.
    """
    import inspect

    bound = inspect.signature(original_fn).bind(self, *args, **kwargs)
    self.index_k_int8 = (
        low_ratio_index_k_int8_enabled()
        and bound.arguments.get("use_fp4_indexer") is True
    )
    if self.index_k_int8:
        bound.arguments["use_fp4_indexer"] = False
    result = original_fn(*bound.args, **bound.kwargs)
    if self.index_k_int8:
        logger.info("Low-ratio index-K cache: int8, %d B/token", self.get_bytes_per_token())
    return result


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4IndexerPool.get_index_k_dequant",
    type=HookType.AROUND,
)
def dsv4_indexer_pool_index_k_dequant_kunlun(original_fn, self, layer_id, slots=None):
    """Read int8 payloads and fp32 scales without FP4 reinterpretation."""
    if not getattr(self, "index_k_int8", False):
        return original_fn(self, layer_id, slots)
    from sglang_kunlun.kernels.dsv41_low_ratio import load_index_k_cache_int8

    buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
    if slots is None:
        slots = torch.arange(self.size, device=buf.device)
    return load_index_k_cache_int8(
        buf, slots, page_size=self.page_size, dim=self.index_head_dim
    )


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


@plugin_hook(
    "sglang.srt.speculative.base_spec_worker.BaseSpecWorker._build_hicache_draft_plan",
    type=HookType.AROUND,
)
def decline_packed_hicache_mtp_kunlun(original_fn, self, *args, **kwargs):
    """Keep the model-dtype draft KV pool out of the target's packed host row.

    `configure_kv_cache_dtype_kunlun` above gives the target an int8 KV pool and
    the EAGLE draft a flat model-dtype (fp16) pool, matching the 0.5.8
    production split: the draft attention has no dequant / mixed-row handling,
    and one MTP layer costs little.

    0.5.23's hicache assembler packs the draft layers into the target's host
    page envelope, and `check_packed_kv_rows`
    (`mem_cache/hybrid_cache/host_pool_config.py:255-270`) requires identical
    device row shape AND dtype, so the boot dies on every rank with
    `ValueError: packed draft KV row (576, torch.float16) differs from target
    KV row (576, torch.int8)`.

    Upstream already has the right fallback: `HiCacheDraftMode.SIDECAR` --
    "draft KV pools that share the target KV slots but are stored apart from
    it" (`cache_init_params.py:72-73`), the legacy non-packed hicache
    behaviour, which skips the packed-row check entirely
    (`base_spec_worker.py:335-340`). So this hook lets upstream decide and only
    rewrites a PACKED verdict into the SIDECAR one when the pools actually
    disagree.

    The decision is driven by `store_dtype` -- the same fact the upstream check
    itself compares (`host_pool_config.py:249-253` `_kv_row_signature`) -- and
    not by `server_args.kv_cache_dtype`. The earlier version of this fix read
    the arg, registered successfully, and silently delegated: by the time the
    scheduler builds the KV cache the arg no longer read `int8`, so the guard
    never fired while the crash kept happening.
    """
    plan = original_fn(self, *args, **kwargs)

    from sglang.srt.speculative.base_spec_worker import (
        HiCacheDraftMode,
        HiCacheDraftPlan,
    )

    if plan.mode != HiCacheDraftMode.PACKED:
        return plan
    try:
        target_pool = self.target_worker.model_runner.token_to_kv_pool
        drafts = tuple(r.token_to_kv_pool for r in self._draft_model_runners())
    except Exception:  # layout not reachable; leave upstream's verdict alone
        return plan
    target_dtype = getattr(target_pool, "store_dtype", None)
    if not drafts or all(
        getattr(d, "store_dtype", None) == target_dtype for d in drafts
    ):
        return plan

    logger.info(
        "Kunlun: declining packed hicache MTP -- target KV row dtype %s, draft "
        "%s; upstream requires them equal. Falling back to the SIDECAR draft "
        "plan (draft pools stored apart from the target, which is the legacy "
        "non-packed hicache behaviour).",
        target_dtype,
        [getattr(d, "store_dtype", None) for d in drafts],
    )
    # Upstream assigns these *before* returning the PACKED plan
    # (`base_spec_worker.py:302-304`), and the KV-cache builder consumes the
    # field directly (`kv_cache_builder.py:113`), independently of the plan this
    # hook returns. Leaving them set keeps the pack alive and the boot dies
    # exactly as before -- measured: the hook logged its decline at 00:08:55 and
    # the packed-row ValueError still followed 57 lines later.
    target_model_runner = self.target_worker.model_runner
    target_model_runner.mtp_draft_device_pools = ()
    target_model_runner.mtp_draft_mamba_pools = ()
    return HiCacheDraftPlan(
        mode=HiCacheDraftMode.SIDECAR,
        device_pools=drafts[:1],
    )
