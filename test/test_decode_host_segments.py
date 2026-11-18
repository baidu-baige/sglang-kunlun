#!/usr/bin/env python3
"""Isolated cost of the decode step's host segments, at the two measured shapes.

Why this exists. The per-stage probe (`[SEG]`, `hooks/stage_timing.py`) says
where a decode step's host time goes, and against the 0.5.8 fork 0.5.23 is
behind on exactly three of its columns (ms per step on rank 0, 77k context):

    bs/rank                    2.1-2.4        2.9-3.1
    draft          0.5.23 4.63-5.04   vs 0.5.8 2.64      -> +2.2
    draft_extend   0.5.23 3.36-3.39   vs 0.5.8 2.54      -> +0.8
    copy_result    0.5.23 0.93-1.01   vs 0.5.8 0.18      -> +1.0

An end-to-end cell costs 15 minutes, moves one number, and cannot say which
call inside a column did it -- the `copy_result_to_cpu` hypothesis (upstream's
dedicated copy_stream and its cross-stream wait) survived a whole boot + sweep
before the numbers came back unchanged (1.01 / 1.31 against a 0.93 / 1.20
baseline), i.e. it was falsified by an experiment that should never have been
run end to end. So: every candidate gets measured here first, at the two shapes
the segment table was taken at, and only a candidate that wins here earns a
boot.

Shapes come from the `[SEG]` rows of `fixedbs_*` at 77k / 600-out:
`k=3 -> bs 2.1-2.4` and `k=5 -> bs 2.9-3.1` per rank, `num_draft_tokens=4`.

NOTE on timing: `torch.cuda.Event.elapsed_time` returns 0.0 on this XPU stack
(verified 2026-11-18), so every measurement here is host wall around an
explicit `torch.cuda.synchronize()`. That measures submit + execute, which is
the right quantity for a host segment: it is what the scheduler thread pays.

    python test/test_decode_host_segments.py                 # both shapes
    python test/test_decode_host_segments.py --iters 200
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

# `import sglang` runs an HF patch that this image's transformers rejects; the
# shim is the documented workaround for standalone scripts (HANDOVER_TPOT.md
# section 6).
try:
    import transformers

    if not hasattr(transformers.PretrainedConfig, "__class_validators__"):
        transformers.PretrainedConfig.__class_validators__ = []
except Exception:  # noqa: BLE001 - only needed on this image
    pass

# Shape A and B: the two decode batches the segment table was measured at.
SHAPES = ((2, "k=3 / bs~2.1-2.4"), (3, "k=5 / bs~2.9-3.1"))
NEXT_N = 4
CONTEXT = 77000
PAGE = 64

# Calls per decode step, so a per-call number can be read against the `[SEG]`
# column it belongs to. Three forwards per step (draft, verify, draft-extend)
# are what the decode log's `cudaGraphLaunch` count shows.
CALLS_PER_STEP = {
    "copy_to_cpu": 1,
    "d2h_upstream_pinned_alloc": 1,
    "d2h_058_to_cpu": 1,
    "d2h_cached_pinned": 1,
    "copy_to_cpu_via_copy_stream": 1,
    "page_table_mask_torch": 3,
    "page_table_mask_op": 3,
    "metadata_copy_torch": 3,
    "metadata_copy_op": 3,
}


def _timeit(fn, iters, warmup=10):
    """Host wall per call, with the device drained before and after.

    Returns (median_us, p90_us). Events are unusable here (see the module
    docstring), so this is deliberately submit+execute, not kernel time.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1e6)
    samples.sort()
    return statistics.median(samples), samples[int(0.9 * (len(samples) - 1))]


def case_copy_to_cpu(bs, device):
    """The four D2H readbacks `GenerationBatchResult.copy_to_cpu` issues.

    0.5.8 does exactly this on the forward stream (`managers/utils.py:57-85`);
    0.5.23 wraps it in `copy_stream.wait_stream(forward_stream)` plus a stream
    context (`scheduler.py:5096-5120`). Both variants are measured so the
    difference is a number rather than an argument -- the end-to-end run said
    they are the same, and this is where that should have been settled.
    """
    rows = bs * NEXT_N
    next_token_ids = torch.randint(0, 150000, (rows,), device=device, dtype=torch.int64)
    accept_lens = torch.randint(1, NEXT_N + 1, (bs,), device=device, dtype=torch.int32)
    new_seq_lens = torch.randint(1, CONTEXT, (bs,), device=device, dtype=torch.int32)
    logprobs = torch.randn(rows, device=device, dtype=torch.float32)
    copy_stream = torch.cuda.Stream(device=device)

    def direct():
        next_token_ids.to("cpu", non_blocking=True)
        accept_lens.to("cpu", non_blocking=True)
        new_seq_lens.to("cpu", non_blocking=True)
        logprobs.to("cpu", non_blocking=True)

    def via_copy_stream():
        copy_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(copy_stream):
            direct()

    return {"copy_to_cpu": direct, "copy_to_cpu_via_copy_stream": via_copy_stream}


def case_page_table_mask(bs, device):
    """`kernel_ops._dsa_write_page_tables`: torch path vs the fused op.

    The torch path's `cols.unsqueeze(0) < kv_lens.unsqueeze(1)` over
    [rows, max_cols] is the `tensor_calc_n1_1k<13,...>` the decode trace shows
    at 3/step, and 0.5.8 has no equivalent.
    `xspeedgate_ops::dsa_write_page_tables` (shipped in
    `sglang_kunlun/extra_ops`) replaces all four full-width passes. This case
    exists to keep that claim measured at the two shapes AND byte-exact, since
    the op is already the live path.
    """
    rows = bs * NEXT_N
    n_req = max(bs, 1)
    max_cols = ((CONTEXT + PAGE - 1) // PAGE) * PAGE
    req_to_token = torch.randint(
        0, 1 << 20, (n_req, max_cols), device=device, dtype=torch.int32
    )
    req_pool_indices = torch.arange(n_req, device=device, dtype=torch.int32)
    req_row = torch.arange(rows, device=device, dtype=torch.int32) % n_req
    kv_lens = torch.randint(1, CONTEXT, (rows,), device=device, dtype=torch.int32)
    pt_op = torch.full((rows, max_cols), -1, device=device, dtype=torch.int32)
    pt_torch = pt_op.clone()
    real_op = torch.full(
        (rows, max_cols // PAGE), -1, device=device, dtype=torch.int32
    )
    real_torch = real_op.clone()

    op = getattr(torch.ops.xspeedgate_ops, "dsa_write_page_tables", None)

    def fused():
        op(
            req_to_token,
            req_pool_indices,
            req_row,
            kv_lens,
            pt_op,
            real_op,
            int(max_cols),
            PAGE,
        )

    def torch_path():
        # The same body as the fallback in `_dsa_write_page_tables`, so the two
        # measurements are of the same work and the equality check is meaningful.
        req_idx = req_pool_indices.index_select(0, req_row).long()
        vals = req_to_token.index_select(0, req_idx)[:, :max_cols].to(torch.int32)
        cols = torch.arange(max_cols, device=device, dtype=torch.int32)
        live = cols.unsqueeze(0) < kv_lens.unsqueeze(1)
        view = pt_torch[:rows, :max_cols]
        view.copy_(torch.where(live, vals, view))
        sub_vals = vals[:, ::PAGE] // PAGE
        sub_live = live[:, ::PAGE]
        rview = real_torch[:rows, : sub_vals.shape[1]]
        rview.copy_(torch.where(sub_live, sub_vals, rview))

    def check():
        """Byte-exactness, the first of the four gates an op has to pass."""
        if op is None:
            return "op absent"
        fused()
        torch_path()
        torch.cuda.synchronize()
        same_pt = torch.equal(pt_op, pt_torch)
        same_real = torch.equal(real_op, real_torch)
        return f"page_table_1 equal={same_pt} real_page_table equal={same_real}"

    cases = {"page_table_mask_torch": torch_path}
    if op is not None:
        cases["page_table_mask_op"] = fused
    return cases, check


def case_async_d2h(bs, device):
    """`utils.common.async_d2h` vs 0.5.8's `.to("cpu")` vs a cached pinned buffer.

    This is the prime suspect for the `copy_result_to_cpu` column. Upstream's
    primitive (`srt/utils/common.py:608-617`) does, per tensor and per step:

        host = torch.empty(shape, dtype, pin_memory=True)   # fresh cudaHostAlloc
        host.copy_(tensor, non_blocking=True)
        tensor.record_stream(current_stream())

    0.5.8 has no such primitive -- `managers/utils.py:57-85` calls
    `tensor.to("cpu", non_blocking=True)` on each tensor. A pinned allocation is
    a driver call, it is not cached, and the decode path runs it on every
    readback: the shape of a cost that shows up as 0.93-1.20 ms/step against
    0.5.8's flat 0.18.

    The third variant is what this codebase already does elsewhere for the same
    reason (`KunlunDSAMetadata.refresh_host_mirrors`,
    `MLPSyncBatchInfo.all_gather`): allocate the pinned buffer once per
    (shape, dtype) and copy into it.
    """
    rows = bs * NEXT_N
    # What a GLM-5.3 decode step actually reads back: no logprobs and no hidden
    # states in the SLO cell, so the live set is the token ids plus the three
    # spec length vectors (`managers/utils.py:168-232`).
    tensors = [
        torch.randint(0, 150000, (rows,), device=device, dtype=torch.int64),
        torch.randint(1, NEXT_N + 1, (bs,), device=device, dtype=torch.int32),
        torch.randint(1, NEXT_N + 1, (bs,), device=device, dtype=torch.int32),
        torch.randint(1, NEXT_N + 1, (bs,), device=device, dtype=torch.int32),
    ]

    def upstream():
        for t in tensors:
            host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            host.copy_(t, non_blocking=True)
            t.record_stream(torch.cuda.current_stream())

    def fork_058():
        for t in tensors:
            t.to("cpu", non_blocking=True)

    cache = {}

    def cached_pinned():
        for i, t in enumerate(tensors):
            key = (i, tuple(t.shape), t.dtype)
            host = cache.get(key)
            if host is None:
                host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                cache[key] = host
            host.copy_(t, non_blocking=True)
            t.record_stream(torch.cuda.current_stream())

    return {
        "d2h_upstream_pinned_alloc": upstream,
        "d2h_058_to_cpu": fork_058,
        "d2h_cached_pinned": cached_pinned,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import xspeedgate_ops  # noqa: F401  registers torch.ops.xspeedgate_ops
    try:
        from sglang_kunlun import extra_ops  # noqa: F401  adds the C ops
    except Exception as exc:  # noqa: BLE001
        print(f"extra_ops unavailable ({exc}); the fused variants will be skipped")

    device = torch.device(a.device)
    torch.cuda.set_device(device)
    print(
        f"device={a.device} iters={a.iters} next_n={NEXT_N} context={CONTEXT} "
        f"page={PAGE}\n"
        "us/call is host wall around an explicit synchronize (cuda events read "
        "0.0 on this stack)\n"
    )
    hdr = f"{'segment':<34}{'shape':<22}{'us/call':>10}{'p90':>9}{'ms/step':>10}"
    print(hdr)
    print("-" * len(hdr))
    checks = []
    for bs, label in SHAPES:
        rows = bs * NEXT_N
        shape = f"bs={bs} rows={rows}"
        cases = dict(case_copy_to_cpu(bs, device))
        cases.update(case_async_d2h(bs, device))
        pt_cases, pt_check = case_page_table_mask(bs, device)
        cases.update(pt_cases)
        for name, fn in cases.items():
            med, p90 = _timeit(fn, a.iters)
            per_step = med * CALLS_PER_STEP.get(name, 1) / 1000.0
            print(f"{name:<34}{shape:<22}{med:>10.1f}{p90:>9.1f}{per_step:>10.3f}")
        checks.append(f"{label}: {pt_check()}")
        print()
    for line in checks:
        print(line)
    print(
        "\nms/step = us/call x calls/step, so a row can be read straight against\n"
        "the matching `[SEG]` column (`draft`, `draft_extend`, `copy_result_to_cpu`).\n"
        "A candidate that does not win here does not get an end-to-end run."
    )


if __name__ == "__main__":
    main()
