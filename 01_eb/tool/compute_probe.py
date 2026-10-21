"""单卡"真算"探针：小 GEMM + 逐行平方和 + `isfinite().all()` 同步。

**为什么光有 XCCL 探针不够**（2026-10-17 实测）：设备处于"半坏"状态时，
8 张卡显示 `0MiB / 0%`、8 进程 XCCL 探针 **8/8 通过**，但**真实的计算 kernel 照样挂死** ——
当时 `mhc_pre_big_fuse_with_norm_tilelang` 卡在 `torch.isfinite(post_mix).all()` 上，
单卡 util 100%、进程 CPU 101% 空转十几分钟，且**不产生新的 Xid**。
也就是说"通信能通"不等于"算力能用"。

由 `preflight.sh` 调用，也可单跑：
    PROBE_DEVICE=0 python -u compute_probe.py
"""
import os
import time

import torch
import torch_xmlir  # noqa: F401  初始化 XPU 运行时

dev = int(os.environ.get("PROBE_DEVICE", "0"))
torch.cuda.set_device(dev)
t0 = time.time()

# 1) GEMM（bf16）—— 走真实的矩阵乘核
a = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
c = (a @ b).float()
assert torch.isfinite(c).all().item(), "GEMM 输出出现非有限值"

# 2) fp32 逐行平方和 —— 与挂死那条路径同型（mhc_pre 的 sqrsum）
x = torch.randn(2048, 16384, dtype=torch.bfloat16, device="cuda")
xf = x.float()
y = (xf * xf).sum(-1)
assert torch.isfinite(y).all().item(), "sqrsum 输出出现非有限值"

torch.cuda.synchronize()
print(f"COMPUTE_OK dev={dev} in {time.time() - t0:.2f}s", flush=True)
