"""Microbenchmark: host-side readback of the DP-attention sync row block.

Reproduces ``MLPSyncBatchInfo.all_gather``'s tail exactly
(dp_attn.py:286-301): a ``[num_dp_ranks, attn_tp*attn_cp, 11]`` int64 CPU
tensor sliced to ``[:, 0, :]``, then 11 separate ATen ops over it.

dtype is int64: ``_get_local_tensor`` / ``_get_fallback_tensor`` default to
``dtype=torch.int64`` and ``all_gather`` never overrides it (dp_attn.py:176,
195, 233-234).

No GPU is touched; at this point in the method the tensor is already on host.
"""

import statistics
import time

import torch

N_COLS = 11


def make_tensor(dp_size: int, attn_tp: int, seed: int = 0) -> torch.Tensor:
    """Build the sync row block ``all_gather`` reads back.

    Args:
        dp_size: Number of DP ranks.
        attn_tp: attention TP x CP product.
        seed: Generator seed.

    Returns:
        ``(dp_size, N_COLS)`` int64 tensor on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    full = torch.randint(
        0, 4096, (dp_size, attn_tp, N_COLS), dtype=torch.int64, generator=g
    )
    for col in (2, 3, 6, 8, 9):
        full[:, :, col] = torch.randint(
            0, 2, (dp_size, attn_tp), dtype=torch.int64, generator=g
        )
    return full[:, 0, :]  # the exact object dp_attn.py works on


def variant_aten(t):
    """(a) current 11-op sequence, dp_attn.py:288-300."""
    return (
        t[:, 0].tolist(),
        t[:, 1].tolist(),
        bool(t[:, 2].min()),
        bool(t[:, 3].max()),
        bool(t[:, 6].min()),
        int(t[:, 7].max()),
        bool(t[:, 8].min()),
        bool(t[:, 9].max()),
        t[:, 10].tolist(),
        t[:, 5].tolist(),
    )


def variant_tolist(t):
    """(b) one .tolist(), then pure python."""
    rows = t.tolist()
    return (
        [r[0] for r in rows],
        [r[1] for r in rows],
        all(r[2] for r in rows),
        any(r[3] for r in rows),
        all(r[6] for r in rows),
        max(r[7] for r in rows),
        all(r[8] for r in rows),
        any(r[9] for r in rows),
        [r[10] for r in rows],
        [r[5] for r in rows],
    )


def variant_numpy(t):
    """(c) one .numpy() view, then numpy column ops."""
    a = t.numpy()
    return (
        a[:, 0].tolist(),
        a[:, 1].tolist(),
        bool(a[:, 2].min()),
        bool(a[:, 3].max()),
        bool(a[:, 6].min()),
        int(a[:, 7].max()),
        bool(a[:, 8].min()),
        bool(a[:, 9].max()),
        a[:, 10].tolist(),
        a[:, 5].tolist(),
    )


def bench(fn, t, iters=20000, reps=7):
    """Time ``fn(t)`` and return the best round.

    Args:
        fn: Callable taking the tensor.
        t: Tensor to pass.
        iters: Calls per round.
        reps: Number of rounds.

    Returns:
        Best (minimum) round, in nanoseconds per call.
    """
    best = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn(t)
        best.append((time.perf_counter_ns() - t0) / iters)
    return min(best), statistics.median(best)


def breakdown(t):
    """Per-op cost of the current sequence, to show where the time sits."""
    ops = (
        ("t[:, 0]            ", lambda: t[:, 0]),
        ("t[:, 0].tolist()   ", lambda: t[:, 0].tolist()),
        ("t[:, 2].min()      ", lambda: t[:, 2].min()),
        ("bool(t[:, 2].min())", lambda: bool(t[:, 2].min())),
        ("int(t[:, 7].max()) ", lambda: int(t[:, 7].max())),
        ("t.tolist()         ", lambda: t.tolist()),
        ("t.numpy()          ", lambda: t.numpy()),
    )
    for name, fn in ops:
        lo, _ = bench(lambda _t: fn(), t)
        print(f"      {name} {lo:9.0f} ns")


def main():
    """Run every readback variant across the shapes the scheduler sees."""
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}")
    for dp_size, attn_tp in ((8, 1), (8, 4), (8, 8), (1, 1), (32, 1)):
        t = make_tensor(dp_size, attn_tp)
        assert t.shape == (dp_size, N_COLS) and t.dtype is torch.int64
        ref = variant_aten(t)
        assert variant_tolist(t) == ref, "tolist variant differs"
        assert variant_numpy(t) == ref, "numpy variant differs"
        print(
            f"dp_size={dp_size} attn_tp*attn_cp={attn_tp} "
            f"contiguous={t.is_contiguous()} shape={tuple(t.shape)}"
        )
        res = {}
        for name, fn in (
            ("(a) 11 aten ops  ", variant_aten),
            ("(b) tolist+python", variant_tolist),
            ("(c) numpy columns", variant_numpy),
        ):
            lo, med = bench(fn, t)
            res[name.strip()[:3]] = lo
            print(f"    {name} min={lo:9.0f} ns/call  median={med:9.0f} ns/call")
        print(
            f"    saving (a)->(b) = {res['(a)'] - res['(b)']:.0f} ns/call "
            f"= {(res['(a)'] - res['(b)']) / 1000:.3f} us/step"
        )
        print()


if __name__ == "__main__":
    main()
