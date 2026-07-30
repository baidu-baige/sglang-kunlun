"""Memory-pool probes moved out of ``sglang_kunlun/hooks/mem_cache/common.py``."""

from __future__ import annotations

import logging
import os

from ._dispatch import SiteRegistry

logger = logging.getLogger(__name__)

_LOGGED = set()


def log_half_cache_buffer(pool, num_pages: int, dim_per_token: int) -> None:
    """Log the Kunlun half-precision KV buffer geometry once per process."""
    if (
        os.environ.get("DSV4_MTP_PROBE") != "1"
        or os.environ.get("RANK", "0") != "0"
        or "half_cache_buffer" in _LOGGED
    ):
        return
    _LOGGED.add("half_cache_buffer")
    logger.warning(
        "[DSV4_CALLSTACK] half-cache pool buffer store_dtype=%s "
        "shape=(%d, %d) page_size=%d dim_per_token=%d",
        pool.store_dtype,
        num_pages,
        pool.page_size * dim_per_token,
        pool.page_size,
        dim_per_token,
    )


_SITES = SiteRegistry()
capture = _SITES.capture


@_SITES.site("half_cache_buffer")
def _site_half_cache_buffer(scope) -> None:
    log_half_cache_buffer(scope["self"], scope["num_pages"], scope["dim_per_token"])
