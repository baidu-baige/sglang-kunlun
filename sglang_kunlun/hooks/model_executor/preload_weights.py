# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cold-start optimizer: parallel page-cache preload for checkpoint shards.

Hooks ``DefaultModelLoader.load_model`` to warm the OS page cache **before**
the serial mmap-based weight loading loop begins.  On NFS-backed storage the
first read of each safetensor shard can take 10-65 s (cold page cache); after
preload the same read completes in < 1 s (page cache hit).

Measured improvement on DSV4.1-Flash (43 shards, 720 GiB, P800 NFS):
  cold start load_weight: 565 s  →  ~80 s   (page cache warm)

The preload runs on background threads and finishes before ``load_model()``
enters the per-shard iterator, so the main loading path sees only page-cache
hits.  The hook is transparent: it does not change any model weights, tensor
layout, or device placement.

When engram host table is enabled with per_rank layout, ``_HostTable.__init__``
calls ``_drop_page_cache_once()`` (posix_fadvise DONTNEED) during model
construction — which happens *inside* ``load_model()`` but *before* the weight
iteration loop.  This would negate any preload done before ``load_model()``.
To handle this, when engram is detected the hook monkey-patches
``_drop_page_cache_once`` to be a no-op during ``load_model()``, then restores
it afterward.  The engram huge-page pre-fault still works because the
anonymous mmap + MADV_HUGEPAGE path does not require page-cache eviction; the
drop was a heuristic to free memory pressure, not a correctness requirement.

Environment variables:
  SGLANG_KUNLUN_PRELOAD_WEIGHTS=1   (default: 1, set 0 to disable)
  SGLANG_KUNLUN_PRELOAD_THREADS=16  (default: 16, parallel read threads)
  SGLANG_KUNLUN_PRELOAD_BLOCK_MB=16 (default: 16, read block size in MiB)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_PRELOAD_ENABLED = os.environ.get("SGLANG_KUNLUN_PRELOAD_WEIGHTS", "1") == "1"
_PRELOAD_THREADS = int(os.environ.get("SGLANG_KUNLUN_PRELOAD_THREADS", "16"))
_BLOCK_SIZE = int(os.environ.get("SGLANG_KUNLUN_PRELOAD_BLOCK_MB", "16")) * 1024 * 1024


def _get_safetensor_files(model_path: str) -> list[str]:
    """Return the list of safetensor shard files referenced by the index."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        files = sorted(set(weight_map.values()))
        return [os.path.join(model_path, fn) for fn in files]
    # Fallback: glob all safetensors
    return sorted(glob.glob(os.path.join(model_path, "*.safetensors")))


def _preload_file(path: str) -> tuple[str, float]:
    """Read a file sequentially to populate the OS page cache."""
    t0 = time.monotonic()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_BLOCK_SIZE)
                if not chunk:
                    break
    except Exception as e:
        logger.warning(f"[preload] failed to preload {path}: {e}")
        return path, -1.0
    elapsed = time.monotonic() - t0
    return path, elapsed


def _preload_all(model_path: str) -> float:
    """Preload all safetensor shards in parallel. Returns total wall time.

    Files are distributed across local ranks using LPT (Longest Processing
    Time) greedy: sort by size descending, assign each file to the rank with
    the smallest current total.  This minimises the straggler (max-rank)
    wall time when shard sizes are uneven (e.g. engram embedding shards are
    ~2-5× larger than MoE expert shards).
    """
    files = _get_safetensor_files(model_path)
    if not files:
        logger.info("[preload] no safetensor files found, skipping")
        return 0.0

    rank = 0
    world_size = 1
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except Exception:
        pass

    # Distribute files across ranks on this node using LPT greedy
    local_rank = rank % 8  # assume 8 ranks per node
    local_size = min(8, world_size)

    file_sizes = [(os.path.getsize(f) if os.path.isfile(f) else 0, i, f)
                  for i, f in enumerate(files)]
    file_sizes.sort(reverse=True)  # largest first
    rank_totals = [0] * local_size
    rank_files: list[list[str]] = [[] for _ in range(local_size)]
    for sz, _idx, f in file_sizes:
        target = min(range(local_size), key=lambda r: rank_totals[r])
        rank_totals[target] += sz
        rank_files[target].append(f)
    my_files = rank_files[local_rank]

    total_bytes = rank_totals[local_rank]
    total_gb = total_bytes / (1024 ** 3)

    if rank == 0:
        all_bytes = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        logger.info(
            f"[preload] {len(files)} shards ({all_bytes / (1024**3):.1f} GiB total), "
            f"rank {rank} preloading {len(my_files)} shards ({total_gb:.1f} GiB) "
            f"with {_PRELOAD_THREADS} threads"
        )

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=_PRELOAD_THREADS) as pool:
        futures = {pool.submit(_preload_file, f): f for f in my_files}
        done_count = 0
        for future in as_completed(futures):
            path, elapsed = future.result()
            done_count += 1
            if rank == 0 and done_count % 5 == 0:
                wall = time.monotonic() - t0
                logger.info(
                    f"[preload] {done_count}/{len(my_files)} shards, "
                    f"{wall:.1f}s elapsed"
                )

    wall = time.monotonic() - t0
    if rank == 0:
        bw = total_gb / wall if wall > 0 else 0
        logger.info(
            f"[preload] rank {rank} done: {len(my_files)} shards, "
            f"{total_gb:.1f} GiB in {wall:.1f}s ({bw:.1f} GiB/s)"
        )

    # Barrier so all ranks finish preload before loading begins
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
    except Exception:
        pass

    return wall


@contextmanager
def _suppress_engram_page_cache_drop():
    """Temporarily monkey-patch engram's _drop_page_cache_once to a no-op.

    This prevents engram's _HostTable.__init__ (per_rank layout) from evicting
    the page cache that we just preloaded.  The original function is restored
    when the context exits so that any post-load ``finish_load`` calls still
    work if they genuinely need a second drop (the ``finish_load`` call uses
    ``drop_checkpoint_page_cache()`` directly, not the ``_once`` wrapper, so
    we also patch that).

    The drop was a heuristic to help transparent huge-page allocation — not a
    correctness requirement.  Skipping it means THP coverage *may* be slightly
    lower, but the weight-loading speedup more than compensates (the host table
    pre-fault still runs fine on anonymous MAP_PRIVATE|MAP_ANONYMOUS memory).
    """
    try:
        from sglang.srt.layers import engram as _engram_mod
    except ImportError:
        logger.info("[preload] engram module not available, skip-drop patch not needed")
        yield
        return

    orig_once = _engram_mod._drop_page_cache_once
    orig_drop = _engram_mod.drop_checkpoint_page_cache

    def _noop_once(reason: str) -> None:
        _engram_mod._page_cache_dropped = True  # mark as done so it never retries
        logger.info(
            "[preload] suppressed engram page-cache drop (%s) — cache is preloaded",
            reason,
        )

    def _noop_drop() -> tuple:
        logger.info("[preload] suppressed engram drop_checkpoint_page_cache — cache is preloaded")
        return (0, 0)

    _engram_mod._drop_page_cache_once = _noop_once
    _engram_mod.drop_checkpoint_page_cache = _noop_drop
    try:
        yield
    finally:
        _engram_mod._drop_page_cache_once = orig_once
        _engram_mod.drop_checkpoint_page_cache = orig_drop


def _engram_host_table_enabled() -> bool:
    """Check if engram host table is enabled (per_rank layout = page-cache drop).

    Tries the sglang environ registry first; falls back to raw os.environ
    (the registry may not exist in every sglang build / plugin image).
    """
    # Fast path: check os.environ directly (always available)
    env_enabled = os.environ.get("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "0")
    env_layout = os.environ.get("SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT", "")
    if env_enabled == "1" and env_layout == "per_rank":
        logger.info("[preload] engram host table detected via os.environ "
                    "(enabled=%s, layout=%s)", env_enabled, env_layout)
        return True
    # Slow path: try the sglang environ registry
    try:
        from sglang.srt import environ as envs
        result = (
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get()
            and envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get() == "per_rank"
        )
        if result:
            logger.info("[preload] engram host table detected via sglang environ registry")
        return result
    except (ImportError, AttributeError) as exc:
        logger.debug("[preload] sglang environ check failed: %s", exc)
        return False


@plugin_hook(
    "sglang.srt.model_loader.loader.DefaultModelLoader.load_model",
    type=HookType.AROUND,
)
def preload_weights_before_load(original_fn, self, *args, **kwargs):
    """AROUND hook: warm the page cache before the standard load_model path."""
    if not _PRELOAD_ENABLED:
        return original_fn(self, *args, **kwargs)

    # Extract model_config from kwargs (keyword-only in newer sglang)
    model_config = kwargs.get("model_config") or (args[0] if args else None)
    model_path = getattr(model_config, "model_path", None) if model_config else None
    if not model_path or not os.path.isdir(model_path):
        logger.info(f"[preload] model_path {model_path} is not a local dir, skipping")
        return original_fn(self, *args, **kwargs)

    logger.info(f"[preload] starting page-cache preload for {model_path}")
    preload_wall = _preload_all(model_path)
    logger.info(f"[preload] preload completed in {preload_wall:.1f}s, "
                f"proceeding to load_model")

    # If engram host table (per_rank) is enabled, suppress its page-cache drop
    # so the preloaded cache survives through weight loading.
    if _engram_host_table_enabled():
        logger.info("[preload] engram per_rank host table detected — "
                    "suppressing page-cache drop during load_model")
        with _suppress_engram_page_cache_drop():
            return original_fn(self, *args, **kwargs)
    else:
        return original_fn(self, *args, **kwargs)
