#!/usr/bin/env python3
"""Latency of the new XSpeedGate ops against the torch path they replace.

A full SLO cell takes ~15 minutes and moves one number; when an op regresses it
shows up as "TPOT tripled" with no indication of which kernel did it. These
benches run in seconds on an idle card and answer the same question directly,
so a vendor kernel is never promoted past the torch fallback without a paired
measurement first.

Run:  python debug/bench_new_ops.py [--iters N] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

# Importing the package is what registers the ops on ``torch.ops``; without it
# the namespace lookup below raises AttributeError.
import xspeedgate_ops  # noqa: F401

# Decode operating point of the SLO cell this work is measured against:
# batch <= 8 is the cuda-graph ceiling, next_n = speculative_num_draft_tokens,
# and 77k context / page_size 64 gives the page-table width.
BS = 8
NEXT_N = 4
CONTEXT = 77000
PAGE = 64
PAGE_COLS = CONTEXT // PAGE
EXPANDED = BS * NEXT_N


def timeit(fn, iters: int, warmup: int = 20) -> float:
    """Median device time per call, in microseconds.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls per timing round.
        warmup: Untimed calls issued before the first round.

    Returns:
        Median of ``5`` rounds of ``iters`` calls, in microseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        s = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - s) / iters * 1e6)
    return statistics.median(samples)


def bench_metadata_copy(dev: str, iters: int) -> list[tuple[str, float, float]]:
    """fused_dsa_metadata_copy vs the torch copy chain it replaces."""
    rows, cols = EXPANDED, PAGE_COLS

    def make(mode):
        src = dict(
            cache=torch.randint(0, 1000, (BS,), dtype=torch.int32, device=dev),
            cu=torch.randint(0, 1000, (BS + 1,), dtype=torch.int32, device=dev),
            page=torch.randint(0, 1000, (rows, cols), dtype=torch.int32, device=dev),
            nsa=torch.randint(0, 1000, (EXPANDED,), dtype=torch.int32, device=dev),
            dcu=torch.randint(0, 1000, (EXPANDED + 1,), dtype=torch.int32, device=dev),
            rpt=torch.randint(0, 1000, (rows, cols // 4), dtype=torch.int32, device=dev),
        )
        dst = {k: torch.zeros_like(v) for k, v in src.items()}
        return src, dst, mode

    src, dst, mode = make(1)

    def torch_path():
        dst["cache"].copy_(src["cache"])
        dst["cu"][1:].copy_(src["cu"][1:])
        dst["page"][:, :cols].copy_(src["page"])
        dst["nsa"].copy_(src["nsa"])
        dst["dcu"][1 : 1 + EXPANDED].copy_(src["dcu"][1 : 1 + EXPANDED])
        dst["rpt"][: rows, : cols // 4].copy_(src["rpt"])

    def kernel_path():
        torch.ops.xspeedgate_ops.fused_dsa_metadata_copy(
            src["cache"], src["cu"], src["page"], src["nsa"], None, src["dcu"], src["rpt"],
            None, None,
            dst["cache"], dst["cu"], dst["page"], dst["nsa"], None, dst["dcu"], dst["rpt"],
            None, None,
            src["cache"].numel(), src["cu"].numel() - 1, src["nsa"].numel(), 0,
            EXPANDED, 0, 0,
        )

    kernel_path()  # must not raise before timing
    t_torch = timeit(torch_path, iters)
    t_kernel = timeit(kernel_path, iters)
    return [("fused_dsa_metadata_copy", t_torch, t_kernel)]


def bench_gather_spec_extras(dev: str, iters: int) -> list[tuple[str, float, float]]:
    """gather_spec_extras vs the four index_selects plus casts it replaces."""
    topk = 8
    hidden = 6144
    idx = torch.randint(0, BS * NEXT_N, (BS * NEXT_N,), dtype=torch.int64, device=dev)
    topk_p = torch.rand((BS * NEXT_N, topk), dtype=torch.float32, device=dev)
    topk_i = torch.randint(0, 256, (BS * NEXT_N, topk), dtype=torch.int32, device=dev)
    out_tok = torch.randint(0, 256, (BS * NEXT_N,), dtype=torch.int32, device=dev)
    hid = torch.rand((BS * NEXT_N, hidden), dtype=torch.float16, device=dev)

    def torch_path():
        i = idx.to(torch.long).contiguous()
        return (
            topk_p[i].contiguous(),
            topk_i[i].contiguous(),
            out_tok[i].contiguous(),
            hid[i].contiguous(),
        )

    def kernel_path():
        return torch.ops.xspeedgate_ops.gather_spec_extras(idx, topk_p, topk_i, out_tok, hid)

    try:
        kernel_path()
    except Exception as exc:  # noqa: BLE001
        print(f"  gather_spec_extras unavailable ({type(exc).__name__}: {exc}); skipped")
        return []
    t_torch = timeit(torch_path, iters)
    t_kernel = timeit(kernel_path, iters)
    return [("gather_spec_extras", t_torch, t_kernel)]


def main() -> None:
    """Run every available bench and print a kernel/torch ratio table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no XPU device available")

    rows: list[tuple[str, float, float]] = []
    rows += bench_metadata_copy(args.device, args.iters)
    rows += bench_gather_spec_extras(args.device, args.iters)

    print(f"\n{'op':32} {'torch us':>10} {'kernel us':>10} {'ratio':>8}  verdict")
    bad = 0
    for name, t_torch, t_kernel in rows:
        ratio = t_kernel / t_torch if t_torch else float("inf")
        verdict = "FASTER" if ratio < 0.95 else ("same" if ratio <= 1.05 else "REGRESSION")
        bad += verdict == "REGRESSION"
        print(f"{name:32} {t_torch:10.2f} {t_kernel:10.2f} {ratio:8.2f}  {verdict}")
    print(f"\n{len(rows)} op(s) measured, {bad} regression(s)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
