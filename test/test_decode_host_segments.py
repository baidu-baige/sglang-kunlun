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
    # `init_forward_metadata` runs once per forward and there are three
    # forwards per decode step (draft / verify / draft-extend).
    "store_lens_and_cu_torch": 3,
    "host_mirror_staged_d2h": 3,
    "host_mirror_7_blocking": 3,
    "host_mirror_7_direct": 3,
    "host_lengths_from_cpu": 3,
    # The three step_* rows already include their own repetition, so their
    # ms/step column is the number to read directly.
    "step_host_lengths_current": 1,
    "step_host_lengths_dedup": 1,
    "step_host_lengths_lean_dedup": 1,
    "store_lens_and_cu_lean": 3,
    "op_launch_floor": 3,
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


def case_metadata(bs, device):
    """What `fused_dsa_decode_metadata` costs besides the page-table write.

    `kernel_ops.py:7561-7604` is: two `_dsa_store_lens_and_cu` (a cumsum plus
    three small copies each) and one `_dsa_write_page_tables` (measured
    separately in `case_page_table_mask`). The question this answers is whether
    the lens/cu half is worth an operator of its own -- it is three tiny
    cumsums over `bs` elements, so the expectation is that it is launch-bound,
    not work-bound, and an op would only remove launches.
    """
    seq = torch.randint(1, CONTEXT, (bs,), device=device, dtype=torch.int32)
    cache_seqlens = torch.zeros(bs, device=device, dtype=torch.int32)
    cu_k = torch.zeros(bs + 1, device=device, dtype=torch.int32)
    dsa_cache = torch.zeros(bs, device=device, dtype=torch.int32)
    dsa_cu_k = torch.zeros(bs + 1, device=device, dtype=torch.int32)
    topk = 2048

    def store(lengths, out_lengths, out_cu):
        n = lengths.numel()
        out_lengths[:n].copy_(lengths)
        out_cu[:1].zero_()
        out_cu[1 : n + 1].copy_(torch.cumsum(lengths, dim=0, dtype=torch.int32))

    def torch_path():
        store(seq, cache_seqlens, cu_k)
        store(torch.clamp(seq, max=topk), dsa_cache, dsa_cu_k)

    # Same values, fewer launches: write the leading zero directly into the
    # destination and accumulate with `out=`, which drops the `torch.cat`
    # allocation and its copy (2 of the 8 launches).
    def lean():
        capped = torch.clamp(seq, max=topk)
        cache_seqlens.copy_(seq)
        dsa_cache.copy_(capped)
        cu_k[0].zero_()
        dsa_cu_k[0].zero_()
        torch.cumsum(seq, dim=0, dtype=torch.int32, out=cu_k[1 : bs + 1])
        torch.cumsum(capped, dim=0, dtype=torch.int32, out=dsa_cu_k[1 : bs + 1])

    def check():
        """Byte-exactness of the lean form against the current one."""
        cache_seqlens.zero_()
        cu_k.zero_()
        dsa_cache.zero_()
        dsa_cu_k.zero_()
        torch_path()
        ref = [t.clone() for t in (cache_seqlens, cu_k, dsa_cache, dsa_cu_k)]
        for t in (cache_seqlens, cu_k, dsa_cache, dsa_cu_k):
            t.zero_()
        lean()
        same = all(
            torch.equal(a, b)
            for a, b in zip(ref, (cache_seqlens, cu_k, dsa_cache, dsa_cu_k))
        )
        return f"lens/cu lean form byte-exact: {same}"

    return {"store_lens_and_cu_torch": torch_path, "store_lens_and_cu_lean": lean}, check


def case_op_launch_floor(bs, device):
    """What one fused XSpeedGate launch costs, to bound a future lens/cu op.

    A fused lens/cu operator would do the arithmetic the torch path spreads
    over 8 launches, so its cost is dominated by the launch, not the work. The
    smallest shipped fused op on a table this size is the floor: measured
    against `store_lens_and_cu_torch` it says how much a new operator could
    possibly return.
    """
    op = getattr(torch.ops.xspeedgate_ops, "dsa_write_page_tables", None)
    if op is None:
        return {}
    req_to_token = torch.zeros((bs, 64), device=device, dtype=torch.int32)
    req_pool_indices = torch.arange(bs, device=device, dtype=torch.int32)
    req_row = torch.arange(bs, device=device, dtype=torch.int32)
    kv_lens = torch.full((bs,), 32, device=device, dtype=torch.int32)
    pt = torch.zeros((bs, 64), device=device, dtype=torch.int32)

    def floor():
        op(req_to_token, req_pool_indices, req_row, kv_lens, pt, None, 64, 1)

    return {"op_launch_floor": floor}


def case_host_mirrors(bs, device):
    """The three ways this backend can land the per-step host lengths.

    K lengths and their prefix sums are NOT shape constants: every decode step
    they grow by that row's accept length, so something has to refresh them
    once per forward. The three variants are the ones the code actually has:

    * `host_mirror_7_blocking` -- seven separate `.cpu()` reads, the form before
      the 2026-11-13 batching (`HANDOVER_TPOT.md` section 4, 1.16 ms/step).
    * `host_mirror_staged_d2h` -- `refresh_host_mirrors`: stage per dtype into
      one buffer, one pinned D2H per dtype, then ONE
      `current_stream().synchronize()` (`kunlun_nsa_backend.py:333-364`). The
      synchronize is the part worth pricing: it drains the device queue in the
      middle of host graph prep.
    * `host_lengths_from_cpu` -- `refresh_host_lengths_from_cpu`
      (`:366-418`): derive the same four vectors from the padded
      `seq_lens_cpu` with no device read at all. This is the path the backend
      prefers when the graph layout is fixed and `dsa_index_kpool <= 1`.
    """
    rows = bs * NEXT_N
    dev32 = lambda n: torch.randint(1, CONTEXT, (n,), device=device, dtype=torch.int32)
    mirrored = [dev32(bs), dev32(bs + 1), dev32(bs + 1), dev32(rows), dev32(bs + 1),
                dev32(rows + 1), dev32(bs + 1)]
    hosts = [torch.empty_like(t, device="cpu") for t in mirrored]
    total = sum(t.numel() for t in mirrored)
    dev_buf = torch.empty(total, device=device, dtype=torch.int32)
    host_buf = torch.empty(total, dtype=torch.int32, pin_memory=True)
    stream_mod = torch.get_device_module()

    def seven_blocking():
        for i, t in enumerate(mirrored):
            hosts[i].copy_(t.cpu())

    def seven_direct():
        """The same seven reads, straight into the stable host storage.

        `t.cpu()` allocates a fresh host tensor per mirror and then copies it
        into the buffer the graph was recorded against; `host.copy_(t)` skips
        that intermediate. Both keep the destination storage, which is the only
        invariant `refresh_host_mirrors` has to hold.
        """
        for i, t in enumerate(mirrored):
            hosts[i].copy_(t)

    def staged():
        off = 0
        spans = []
        for t in mirrored:
            n = t.numel()
            dev_buf[off : off + n].copy_(t.reshape(-1))
            spans.append((off, n))
            off += n
        host_buf.copy_(dev_buf, non_blocking=True)
        stream_mod.current_stream().synchronize()
        for host, (off, n) in zip(hosts, spans):
            host.reshape(-1).copy_(host_buf[off : off + n])

    seq_cpu = torch.randint(1, CONTEXT, (bs,), dtype=torch.int32)
    offsets = torch.arange(1, NEXT_N + 1, dtype=torch.int32)
    dst = [torch.empty(bs, dtype=torch.int32), torch.empty(bs + 1, dtype=torch.int32),
           torch.empty(rows, dtype=torch.int32), torch.empty(rows + 1, dtype=torch.int32)]

    def from_cpu():
        # The target-verify branch, i.e. the widest of the three the method has.
        cache = seq_cpu + NEXT_N
        expanded = (seq_cpu[:, None] + offsets).reshape(-1)
        capped = expanded.clamp(max=2048)
        cu_k = torch.cat(
            (torch.zeros(1, dtype=torch.int32), cache.cumsum(0, dtype=torch.int32))
        )
        dsa_cu_k = torch.cat(
            (torch.zeros(1, dtype=torch.int32), capped.cumsum(0, dtype=torch.int32))
        )
        for d, v in zip(dst, (cache, cu_k, expanded, dsa_cu_k)):
            d.copy_(v)

    def from_cpu_lean():
        """Same four vectors, writing straight into the destinations.

        Drops the `torch.cat` allocations (two allocs plus two copies become a
        zeroed first element and a `cumsum(out=...)`), so it is 6 launches
        instead of 8 with identical values.
        """
        cache = seq_cpu + NEXT_N
        expanded = (seq_cpu[:, None] + offsets).reshape(-1)
        capped = expanded.clamp(max=2048)
        dst[0].copy_(cache)
        dst[2].copy_(expanded)
        dst[1][0].zero_()
        dst[3][0].zero_()
        torch.cumsum(cache, dim=0, dtype=torch.int32, out=dst[1][1:])
        torch.cumsum(capped, dim=0, dtype=torch.int32, out=dst[3][1:])

    # The model's own observation is 3.89 calls per decode step
    # (`[SEG] ... refresh_host_lengths_from_cpu=0.43(n=3.89)`), i.e. the same
    # vectors are derived more than once per step. These two step-level
    # variants price that: 4 recomputes versus 1 recompute behind a signature
    # check. The check is `(forward_mode, data_ptr, last element)` on a CPU
    # tensor, so it cannot cost anything on its own.
    sig = [0, 0, 0]

    def signature_ok():
        key = (7, seq_cpu.data_ptr(), int(seq_cpu[-1]))
        if key == tuple(sig):
            return True
        sig[:] = key
        return False

    def step_current():
        for _ in range(4):
            from_cpu()

    def step_dedup():
        for _ in range(4):
            if not signature_ok():
                from_cpu()

    def step_lean_dedup():
        for _ in range(4):
            if not signature_ok():
                from_cpu_lean()

    def check_host_lengths():
        """Byte-exactness of the lean form against the current body."""
        want = []
        from_cpu()
        want = [d.clone() for d in dst]
        for d in dst:
            d.zero_()
        from_cpu_lean()
        same = all(torch.equal(a, b) for a, b in zip(want, dst))
        return f"host_lengths lean form byte-exact: {same}"

    return {
        "host_mirror_7_blocking": seven_blocking,
        "host_mirror_7_direct": seven_direct,
        "host_mirror_staged_d2h": staged,
        "host_lengths_from_cpu": from_cpu,
        "step_host_lengths_current": step_current,
        "step_host_lengths_dedup": step_dedup,
        "step_host_lengths_lean_dedup": step_lean_dedup,
    }, check_host_lengths


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
        md_cases, md_check = case_metadata(bs, device)
        cases.update(md_cases)
        hm_cases, hm_check = case_host_mirrors(bs, device)
        cases.update(hm_cases)
        cases.update(case_op_launch_floor(bs, device))
        pt_cases, pt_check = case_page_table_mask(bs, device)
        cases.update(pt_cases)
        for name, fn in cases.items():
            med, p90 = _timeit(fn, a.iters)
            per_step = med * CALLS_PER_STEP.get(name, 1) / 1000.0
            print(f"{name:<34}{shape:<22}{med:>10.1f}{p90:>9.1f}{per_step:>10.3f}")
        checks.append(f"{label}: {pt_check()}")
        checks.append(f"{label}: {hm_check()}")
        checks.append(f"{label}: {md_check()}")
        print()
    for line in checks:
        print(line)
    print()
    for line in check_refresh_host_mirrors(device):
        print(line)
    from check_shipped_helpers import check_real_helpers

    for line in check_real_helpers(device):
        print(line)
    print(
        "\nms/step = us/call x calls/step, so a row can be read straight against\n"
        "the matching `[SEG]` column (`draft`, `draft_extend`, `copy_result_to_cpu`).\n"
        "A candidate that does not win here does not get an end-to-end run."
    )




def check_refresh_host_mirrors(device):
    """Correctness gate for `KunlunDSAMetadata.refresh_host_mirrors`.

    The invariants that matter, in the order they broke historically:

    1. after a refresh every host mirror equals its device tensor;
    2. a producer that writes IN PLACE is picked up -- this is the one a cached
       plan can silently miss, and a stale K length under-plans the kernel's kv
       window (0.5.8 shipped exactly that bug on its MTP precompute replay
       path, `nsa_backend.py:1443-1512`);
    3. the host storage is stable across refreshes, because the CUDA graph was
       recorded against those buffers;
    4. a producer that SWAPS a device buffer invalidates the cache.

    Built on a stub carrying the real methods: the metadata is a frozen
    dataclass with a long field list, and none of it is reachable from here.
    """
    from sglang_kunlun.hooks.layers.attention.kunlun_nsa_backend import (
        KunlunDSAMetadata,
    )

    class _Stub:
        _MIRRORED = KunlunDSAMetadata._MIRRORED
        _mirror_pairs = KunlunDSAMetadata._mirror_pairs
        refresh_host_mirrors = KunlunDSAMetadata.refresh_host_mirrors

    stub = _Stub()
    sizes = {name: 3 + i for i, (_, name) in enumerate(_Stub._MIRRORED)}
    for host_name, dev_name in _Stub._MIRRORED:
        dev = torch.randint(
            1, 9999, (sizes[dev_name],), device=device, dtype=torch.int32
        )
        setattr(stub, dev_name, dev)
        setattr(stub, host_name, None)

    results = []
    stub.refresh_host_mirrors()          # first call rebuilds and refreshes
    stub.refresh_host_mirrors()          # second call takes the cached path
    equal = all(
        torch.equal(getattr(stub, h), getattr(stub, d).cpu())
        for h, d in _Stub._MIRRORED
    )
    results.append(f"(1) mirrors equal after refresh: {equal}")

    ptrs = {h: getattr(stub, h).data_ptr() for h, _ in _Stub._MIRRORED}
    for _, dev_name in _Stub._MIRRORED:
        getattr(stub, dev_name).add_(7)  # in-place producer update
    stub.refresh_host_mirrors()
    equal = all(
        torch.equal(getattr(stub, h), getattr(stub, d).cpu())
        for h, d in _Stub._MIRRORED
    )
    results.append(f"(2) in-place device update picked up: {equal}")
    stable = all(
        getattr(stub, h).data_ptr() == ptrs[h] for h, _ in _Stub._MIRRORED
    )
    results.append(f"(3) host storage stable: {stable}")

    for _, dev_name in _Stub._MIRRORED:
        old = getattr(stub, dev_name)
        setattr(stub, dev_name, old.clone() + 1)  # swapped buffer, same shape
    stub.refresh_host_mirrors()
    equal = all(
        torch.equal(getattr(stub, h), getattr(stub, d).cpu())
        for h, d in _Stub._MIRRORED
    )
    results.append(f"(4) swapped device buffer picked up: {equal}")

    # The fixed-Q-LOD filter must leave those three mirrors alone.
    for _, dev_name in _Stub._MIRRORED:
        getattr(stub, dev_name).add_(5)
    stub.refresh_host_mirrors(fixed_query_lod=True)
    skipped = [
        h
        for h, d in _Stub._MIRRORED
        if d in ("cu_seqlens_q", "dsa_cu_seqlens_q", "cum_q_lod")
    ]
    untouched = all(
        not torch.equal(getattr(stub, h), getattr(stub, d).cpu())
        for h, d in _Stub._MIRRORED
        if h in skipped
    )
    refreshed = all(
        torch.equal(getattr(stub, h), getattr(stub, d).cpu())
        for h, d in _Stub._MIRRORED
        if h not in skipped
    )
    results.append(
        f"(5) fixed_query_lod skips the Q lods: {untouched} and refreshes the "
        f"rest: {refreshed}"
    )
    return results


if __name__ == "__main__":
    main()
