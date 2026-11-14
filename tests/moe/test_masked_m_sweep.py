"""How does the masked MoE GEMM scale with the *real* row count per expert?

The traces disagree on this kernel (0.5.8 84.7 us/call vs 0.5.23 138.5 us/call at
identical call counts), but both sides pass exactly the same tensors:
`x int8 (32,256,6144)`, `w int8 (32,4096,3072)`, `ws fp32 (32,4096,1)`,
`masked_m int32 (32,)`, `expected_m=9`, `m=256`.

So the only free variable is what `masked_m` holds. This sweeps it with m_max
fixed at the production 256, which says which real row count reproduces 84.7 us
and which reproduces 138.5 us.

Run: python3 tests/moe/test_masked_m_sweep.py
"""

import time

import torch
import torch_xmlir  # noqa: F401
import kunlun_ops

E, K, N = 32, 6144, 4096      # 32 local experts, gate/up
M_MAX = 256                   # production m (expert_alignment padded)
ALIGN = 128


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000.0 / iters


def run(rows_per_expert):
    x = torch.randint(-127, 127, (E, M_MAX, K), dtype=torch.int8, device="cuda")
    xs = torch.rand(E, M_MAX, 1, dtype=torch.float32, device="cuda")
    w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device="cuda")
    ws = torch.rand(E, N, 1, dtype=torch.float32, device="cuda")
    out = torch.empty(E, M_MAX, N, dtype=torch.bfloat16, device="cuda")
    masked_m = torch.full((E,), rows_per_expert, dtype=torch.int32, device="cuda")

    def call():
        kunlun_ops.m_grouped_gemm_I8_I4_bf16_nt_masked_v3(
            (x, xs), (w, ws), out, masked_m, min(rows_per_expert, M_MAX)
        )

    ms = bench(call)
    return ms * 1000.0, ms     # us/call, ms/call


if __name__ == "__main__":
    print(f"{'rows/expert':>12s} {'us/call':>9s} {'ms/call':>9s}")
    for r in (1, 2, 4, 9, 16, 32, 64, 128, 256):
        try:
            us, ms = run(r)
            print(f"{r:12d} {us:9.1f} {ms:9.3f}")
        except Exception as exc:
            print(f"{r:12d}  FAIL {str(exc)[:60]}")
