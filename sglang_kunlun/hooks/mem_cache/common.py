"""Kunlun memory-cache hooks."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.utils import get_num_new_pages, next_power_of_2


logger = logging.getLogger(__name__)


_INDEX_K_INT8: bool | None = None

# Upstream's own DeepSeek-V4.1 discriminator (`arg_groups/deepseek_v4_hook.py`
# `validate_deepseek_v41_features`). `is_deepseek_v4()` cannot stand in for it:
# V4 and V4.1 share `architectures`, `model_type` is the only separator.
# `DeepseekV41TextConfig` keeps its own class-level `model_type`, hence both.
_DSV41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")


def _model_is_dsv41() -> bool | None:
    """Whether the served checkpoint is DeepSeek-V4.1; None while not knowable.

    Read through `model_config_of(get_server_args())`, which memoises a
    `ModelConfig` built from the server args alone. `get_model()` is the wrong
    source here -- it is not published while the pools are being built, which is
    exactly when the layout gets decided (see `dsv41_engram_int8.py` for the
    same trap).
    """
    try:
        from sglang.srt.arg_groups.overrides import model_config_of
        from sglang.srt.runtime_context import get_server_args

        model_type = getattr(
            model_config_of(get_server_args()).hf_config, "model_type", None
        )
    except Exception:  # noqa: BLE001 - server args not published yet
        return None
    return None if model_type is None else model_type in _DSV41_MODEL_TYPES


def _index_k_int8_op_available() -> bool:
    """Probe the scatter op that the int8 132 B layout is written with."""
    try:
        import xspeedgate_ops  # noqa: F401  # registers torch.ops.xspeedgate_ops

        return hasattr(torch.ops.xspeedgate_ops, "set_k_and_s_paged_quant")
    except Exception:  # noqa: BLE001 - no wheel just means fp4
        return False


def _resolve_index_k_int8() -> tuple[bool, bool]:
    """Resolve the layout as `(int8?, settled?)`.

    `settled` is False only while the model config is still unreadable, so the
    caller can retry instead of pinning fp4 on a V4.1 run for good.
    """
    requested = os.environ.get("DSV4_LOW_RATIO_INDEX_KCACHE", "").strip().lower()
    if requested and requested not in ("int8", "fp4"):
        logger.warning(
            "DSV4_LOW_RATIO_INDEX_KCACHE=%r is neither int8 nor fp4; ignoring it",
            requested,
        )
        requested = ""
    if requested == "fp4":
        return False, True

    # An explicit `int8` skips the model gate on purpose: that is how a new
    # model gets its first run on this layout.
    if requested != "int8":
        is_dsv41 = _model_is_dsv41()
        if is_dsv41 is None:
            return False, False
        if not is_dsv41:
            return False, True

    if not _index_k_int8_op_available():
        logger.warning(
            "int8 low-ratio index-K cache needs "
            "torch.ops.xspeedgate_ops.set_k_and_s_paged_quant, which this "
            "xspeedgate_ops build does not export; using fp4"
        )
        return False, True
    return True, True


def low_ratio_index_k_int8_enabled() -> bool:
    """Select the int8 low-ratio index cache consistently for sizing and IO.

    int8 is the default on DeepSeek-V4.1, the only model it has been validated
    on. Every other checkpoint that reaches these hooks -- V4.0 and the other
    `DeepseekV4ForCausalLM*` variants, which share V4.1's `architectures` --
    keeps the 68 B fp4 rows. `DSV4_LOW_RATIO_INDEX_KCACHE` overrides either way:
    `fp4` for an A/B run on V4.1, `int8` to try the layout on a new model.

    Resolved once and then frozen. The row width is baked into the pool when it
    allocates, while the scoring paths re-read this flag on every forward -- an
    answer that changed mid-run would price one layout against a buffer written
    in the other, silently and with no error.
    """
    global _INDEX_K_INT8
    if _INDEX_K_INT8 is not None:
        return _INDEX_K_INT8
    value, settled = _resolve_index_k_int8()
    if settled:
        _INDEX_K_INT8 = value
    return value


def _freeze_index_k_int8() -> bool:
    """Pin the layout before the first pool bakes it into a row width."""
    global _INDEX_K_INT8
    if _INDEX_K_INT8 is None:
        value, settled = _resolve_index_k_int8()
        if not settled:
            logger.warning(
                "low-ratio index-K layout frozen before the model config was "
                "readable; using fp4"
            )
        _INDEX_K_INT8 = value
    return _INDEX_K_INT8


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
        _freeze_index_k_int8()
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
