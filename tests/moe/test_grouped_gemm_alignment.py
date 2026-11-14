"""Which grouped-GEMM variant does the vendor pick, and what does expert padding cost?

DeepEP is given `expert_alignment=128`, so each local expert's row count is
rounded up before the grouped GEMM: with a steady decode batch the real rows per
expert are ~1-2 while `m_max` becomes 128. 0.5.8 and 0.5.23 differ 1.6x on this
GEMM at identical call counts, so this measures the padding cost directly.

Masked signature (from `kunlun_ops` docs):
    lhs   fp16 [num_groups, m_max, k]
    rhs   (int8 [num_groups, n, k//2] int4-packed, fp32 [g, ceil(n/128), ceil(k/128)])
    out   fp16 [m_sum, n]
    mask_m int32 [num_groups]   valid rows per expert
"""

import torch
import torch_xmlir  # noqa: F401
import kunlun_ops

HIDDEN, INTER, E = 6144, 2048, 32
ALIGN = 128
BS, TOP_K = 6, 8


def bench(fn, iters=50):
    # cuda.Event.elapsed_time is not implemented on this XPU build (returns 0),
    # so time with the host clock around one synchronise per iteration.
    import time as _t

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = _t.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()          # one sync for the whole batch
    return (_t.perf_counter() - start) * 1000.0 / iters


def run(m_max, rows_per_expert, n=INTER * 2):
    # Mirrors ep_moe/layer.py:1318-1330 (the production call shape).
    # 0.5.8's call passes 3-D activations `[num_groups, m, k]`
    # (ep_moe/layer.py:1186 there), and `expected_m = min(expected_m, m)`.
    lhs = torch.randint(-127, 127, (E, m_max, HIDDEN), dtype=torch.int8, device="cuda")
    lhs_scale = torch.rand(E, m_max, (HIDDEN + ALIGN - 1) // ALIGN,
                           dtype=torch.float32, device="cuda")
    w = torch.randint(-128, 127, (E, n, HIDDEN // 2), dtype=torch.int8, device="cuda")
    ws = torch.rand(E, (n + ALIGN - 1) // ALIGN, (HIDDEN + ALIGN - 1) // ALIGN,
                    dtype=torch.float32, device="cuda")
    mask_m = torch.full((E,), rows_per_expert, dtype=torch.int32, device="cuda")
    out = torch.empty(E, m_max, n, dtype=torch.bfloat16, device="cuda")

    def call():
        kunlun_ops.m_grouped_gemm_I8_I4_bf16_nt_masked_v3(
            (lhs, lhs_scale), (w, ws), out, mask_m, min(rows_per_expert, m_max)
        )

    try:
        ms = bench(call)
        return f"m_max={m_max:5d} rows/expert={rows_per_expert:3d}  {ms:7.3f} ms"
    except Exception as exc:
        return f"m_max={m_max:5d} rows/expert={rows_per_expert:3d}  FAIL {str(exc)[:70]}"


if __name__ == "__main__":
    real = BS * TOP_K // E  # ~1.5 rows per expert at bs 6
    for m_max in (2, 16, ALIGN, 2 * ALIGN):
        print(run(m_max, max(1, real)))
    # same real work, different padding: isolates the alignment overhead
    print(run(real, real))
