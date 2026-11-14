"""Compare the two int4 grouped-GEMM entry points at the decode shape.

`compressed_tensors_moe.py` (aima_sglang) calls `kunlun_ops.moe_fc_v3(...)` with
sorted tokens, while our `ep_moe/layer.py` calls
`m_grouped_gemm_I8_I4_bf16_nt_masked_v3(...)` with a masked layout. On the decode
trace the first shows up as `moe_bf16_rn_km_nmk` (13.0 ms/step on 0.5.8) and the
second as `..._kn_nmk` (17.8 ms/step on 0.5.23), so this measures both at the same
shape instead of inferring from names.

Shapes: decode batch 6 x topk 8 = 48 rows, 32 local experts, hidden 6144,
gate/up 4096, down 2048 -- the values the SLO cell actually runs at.
"""

import time

import torch
import torch_xmlir  # noqa: F401
import kunlun_ops

ROWS, E, K = 48, 32, 6144
N_UP, N_DOWN = 4096, 2048
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


def masked_case(n):
    x = torch.randint(-127, 127, (E, ROWS // E + 1, K), dtype=torch.int8, device="cuda")
    xs = torch.rand(E, ROWS // E + 1, (K + ALIGN - 1) // ALIGN, dtype=torch.float32, device="cuda")
    w = torch.randint(-128, 127, (E, n, K // 2), dtype=torch.int8, device="cuda")
    ws = torch.rand(E, (n + ALIGN - 1) // ALIGN, (K + ALIGN - 1) // ALIGN,
                    dtype=torch.float32, device="cuda")
    out = torch.empty(E, ROWS // E + 1, n, dtype=torch.bfloat16, device="cuda")
    m = torch.full((E,), ROWS // E + 1, dtype=torch.int32, device="cuda")
    return lambda: kunlun_ops.m_grouped_gemm_I8_I4_bf16_nt_masked_v3(
        (x, xs), (w, ws), out, m, ROWS // E + 1)


def fc_v3_case(n, per=2):
    rows = E * per          # lod must end exactly at the row count
    x = torch.randint(-127, 127, (rows, K), dtype=torch.int8, device="cuda")
    xs = torch.rand(rows, 1, dtype=torch.float32, device="cuda")
    w = torch.randint(-128, 127, (E, n, K // 2), dtype=torch.int8, device="cuda")
    ws = torch.rand(E, (n + ALIGN - 1) // ALIGN, (K + ALIGN - 1) // ALIGN,
                    dtype=torch.float32, device="cuda")
    lod = torch.arange(0, (E + 1) * per, per, dtype=torch.int32, device="cuda")
    idx = torch.arange(rows, dtype=torch.int32, device="cuda")
    y = torch.zeros(rows, n, dtype=torch.bfloat16, device="cuda")
    return lambda: kunlun_ops.moe_fc_v3(
        x=x, x_perchannel_max=xs, weight=w, w_perchannel_max=ws,
        sorted_tokens_num_lod=lod, sorted_tokens_idx=idx, moe_topk=8,
        y=y, use_pack_int4=True, sort_mode=True)


def fp16_out_case(n):
    """Same W4A8 math, but `fp16_I4_fp16` output (the variant _MOE_FP16_OUT targets)."""
    m = ROWS // E + 1
    x = torch.randn(E, m, K, dtype=torch.float16, device="cuda")
    w = torch.randint(-128, 127, (E, n, K // 2), dtype=torch.int8, device="cuda")
    ws = torch.rand(E, (n + ALIGN - 1) // ALIGN, (K + ALIGN - 1) // ALIGN,
                    dtype=torch.float32, device="cuda")
    out = torch.empty(E, m, n, dtype=torch.float16, device="cuda")
    mm = torch.full((E,), m, dtype=torch.int32, device="cuda")
    return lambda: kunlun_ops.m_grouped_gemm_fp16_I4_fp16_nt_masked_v3(
        x, (w, ws), out, mm, m)


if __name__ == "__main__":
    for n, label in ((N_UP, "gate_up"), (N_DOWN, "down")):
        try:
            print(f"fp16_out   {label:8s} n={n:5d}  {bench(fp16_out_case(n)):7.3f} ms")
        except Exception as exc:
            print(f"fp16_out   {label:8s} n={n:5d}  FAIL {str(exc)[:80]}")
    for name, case in (("masked_v3", masked_case), ("moe_fc_v3", fc_v3_case)):
        for n, label in ((N_UP, "gate_up"), (N_DOWN, "down")):
            try:
                ms = bench(case(n))
                print(f"{name:10s} {label:8s} n={n:5d}  {ms:7.3f} ms")
            except Exception as exc:
                print(f"{name:10s} {label:8s} n={n:5d}  FAIL {str(exc)[:80]}")
