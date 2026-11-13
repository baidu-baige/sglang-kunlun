#!/usr/bin/env python3
"""Guard the DSA indexer's K layout against a well-meant "skip the copy" edit.

The mixed int8 KV row is ``[int8 nope 512 B | fp32 scales 16 B | bf16 rope 128 B]``
(656 B at page_size 64, head_dim 128).  The indexer slices out the nope part and
needs it as int8:

    k_fp8 = kv_cache_fp8[:, :page_size * head_dim].view(pages, block, 1, 128)

Two ways to finish that, and they are NOT equivalent:

    .to(torch.int8)     contiguous, stride(0) = 8192   <- pages packed
    .view(torch.int8)   stride(0) = 8448               <- 256 B gap per page

Same bytes, different memory layout, and the paged MQA-logits kernel is
sensitive to it: switching to ``.view`` here is what took the SLO cell from
TPOT 38.60 ms to 139.82 ms (2026-11-12).  This test pins the invariant so the
next person who notices the redundant-looking copy gets a failing test and the
reason, instead of a 15-minute SLO run.

Run:  python debug/test_index_k_layout.py
"""

from __future__ import annotations

import sys

import torch

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS_KV = 1
PAGES = 16


def mixed_row_width() -> int:
    return PAGE_SIZE * HEAD_DIM + PAGE_SIZE * 4 + PAGE_SIZE * 2


def build_pool(device: str = "cpu") -> torch.Tensor:
    return torch.zeros((PAGES, mixed_row_width()), dtype=torch.uint8, device=device)


def view_like_production(pool: torch.Tensor) -> torch.Tensor:
    """Exactly what ``kunlun_dsa_indexer`` does at the paged call site."""
    return pool[:, : PAGE_SIZE * HEAD_DIM].view(
        pool.shape[0], PAGE_SIZE, NUM_HEADS_KV, HEAD_DIM
    )


def main() -> int:
    pool = build_pool()
    sliced = view_like_production(pool)

    converted = sliced.to(torch.int8)
    reinterpreted = sliced.view(torch.int8)

    failures = []

    if not converted.is_contiguous():
        failures.append(".to(torch.int8) should be contiguous")
    if converted.stride(0) != PAGE_SIZE * HEAD_DIM:
        failures.append(
            f".to(torch.int8) stride(0) should be {PAGE_SIZE * HEAD_DIM}, "
            f"got {converted.stride(0)}"
        )

    if reinterpreted.is_contiguous():
        failures.append(
            ".view(torch.int8) unexpectedly contiguous -- if the pool layout "
            "changed, re-measure the kernel before dropping the copy"
        )
    if reinterpreted.stride(0) == converted.stride(0):
        failures.append(
            ".view and .to produced the same stride(0); the pool no longer has a "
            "gap between pages, so the copy may be removable after all"
        )

    if not torch.equal(reinterpreted.view(torch.uint8), sliced):
        failures.append(".view(torch.int8) must be byte-identical to the source")

    print(f"pool row width      : {mixed_row_width()} B")
    print(f"sliced contiguity   : {sliced.is_contiguous()}  stride(0)={sliced.stride(0)}")
    print(f".to(int8)           : contiguous={converted.is_contiguous()}  stride(0)={converted.stride(0)}")
    print(f".view(int8)         : contiguous={reinterpreted.is_contiguous()}  stride(0)={reinterpreted.stride(0)}")
    print(f"same bytes          : {torch.equal(reinterpreted.view(torch.uint8), sliced)}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: the production expression yields a contiguous, page-packed K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
