"""8-rank XCCL 健康探针（doc/3 第五节的那一条）。

干净机器上期望 8/8 打印 `[rankN] all_reduce ok, sum=36.0`。
坏状态下的特征：rank 0/1/2 卡死（rank1/2 精确停在 `.cpu()` 这个 D2H 拷贝上），
此时进程会 CPU 空转、不退出 —— **只等不杀**。

由 `preflight.sh` 调用，也可单跑：
    RANK=$r WORLD_SIZE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
      python -u xccl_probe.py
"""
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
ws = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(rank)
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    world_size=ws,
    rank=rank,
)

# 1) all_reduce：每 rank 贡献 rank+1，SUM 后全 rank 都应得 36.0
t = torch.tensor([float(rank + 1)], device="cuda")
dist.all_reduce(t)

# 2) 复现路径：int64 op + D2H 拷贝（坏状态下 rank1/2 卡在这）
p8l = torch.arange(8, dtype=torch.int64, device="cuda") * (rank + 1)
_ = p8l.cpu()

torch.cuda.synchronize()
print(f"[rank{rank}] all_reduce ok, sum={t.item()}", flush=True)
dist.destroy_process_group()
