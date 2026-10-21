# EB5 整网快速复现（INT8，8 卡）

本目录是 `code/xy/sglang-kunlun` 自带的一套独立复现材料，不依赖源工作区的
`01_eb_adp`。适配层位于 `sglang_kunlun/eb5_adapt/`，由插件
`sglang_kunlun.hooks.eb5_kunlun_patch` 在服务启动时加载。

```
01_eb/
  README.md
  tool/
    env.sh                 # venv / XPU / KLX 环境
    run_eb5_int8.sh        # 起服务的入口（INT8 包装）
    run_eb5_kunlun.sh      # 实际 launcher
    preflight.sh           # 起服务前的设备/XCCL/算力闸门
    stop_kunlun.sh         # 安全停机（不发 kill -9）
    verify_eb5.sh          # 等 ready + health + 4 并发请求
    xccl_probe.py / compute_probe.py
```

## 0. 约定

```bash
W=/root/paddlejob/gpfsspace/bingoo/p800_eb
```

脚本会根据自身位置向上查找含 `env/bin/python` 的目录作为 `$W`，一般不需要手工设 `BINGOO`。

## 1. 前置条件

### 1.1 两个 editable 安装

- **fork**：`$W/code/sglang`
- **插件**：必须是本仓库 `$W/code/xy/sglang-kunlun`（不是源工作区的 `code/sglang-kunlun`）

确认当前装的是哪个：

```bash
source "$W/code/xy/sglang-kunlun/01_eb/tool/env.sh"
python -c "import sglang_kunlun; print(sglang_kunlun.__file__)"
# 期望：.../code/xy/sglang-kunlun/sglang_kunlun/__init__.py
```

如果不是，重装为本仓库（会替换掉原来的 editable 安装）：

```bash
env/bin/pip install -e "$W/code/xy/sglang-kunlun" --no-deps --no-build-isolation
```

> 需要切回源工作区时，反向执行：
> `env/bin/pip install -e "$W/code/sglang-kunlun" --no-deps --no-build-isolation`

### 1.2 fork 需要已打 INT8 补丁

```bash
grep -c EB5_INT8_ONLINE "$W/code/sglang/python/sglang/srt/internal/environ.py"   # 应为 1
```

### 1.3 设备干净

```bash
source "$W/code/xy/sglang-kunlun/01_eb/tool/env.sh"
xpu-smi | grep -c '0MiB / 98304MiB'      # 必须是 8
```

## 2. 起服务（后台）

```bash
cd "$W"
LAYERS=43 EB5_GPUS=0,1,2,3,4,5,6,7 CONTEXT_LENGTH=32768 MEM_FRACTION=0.88 \
EXTRA_ARGS="--dsa-topk-backend torch" \
bash "$W/code/xy/sglang-kunlun/01_eb/tool/run_eb5_int8.sh" \
     "$W/models/hf_checkpoint-48800" 30082
```

脚本内部会先 `stop_kunlun.sh` + `preflight.sh`，任一不过不会启动；launcher 用
`nohup` 后台运行，日志在 `$W/logs/eb5_kunlun_*.log`。

四个参数都不能省：

| 参数 | 值 | 原因 |
|---|---|---|
| `LAYERS` | `43` | 默认 8 是减层模型 |
| `EB5_GPUS` | `0,1,2,3,4,5,6,7` | 默认只有 4–7 |
| `MEM_FRACTION` | `0.88` | 0.85 会 `ValueError`，KV pool 分不出来 |
| `EXTRA_ARGS` | `--dsa-topk-backend torch` | EB5 必需 |

## 3. 验证

```bash
bash "$W/code/xy/sglang-kunlun/01_eb/tool/verify_eb5.sh" 30082
```

脚本会：

1. 在最新日志里等待 `fired up and ready to roll`（约 3 分钟）
2. 重试 `/health_generate` 直到 200（刚 ready 的瞬时 503 属正常）
3. 并发发 4 个**不同** prompt（相同 prompt 会被路由到同一 DP rank）
4. 检查日志无 `OutOfMemory|CUDA error|died`

通过判据：`通过 4/4` 且致命错误计数为 0。

手工验证等价命令：

```bash
L=$(ls -t "$W"/logs/eb5_kunlun_*.log | head -1)
grep -a "fired up and ready to roll" "$L"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:30082/health_generate

for p in "What is the capital of France?" "1+1=2, 2+2=4, 3+3=" \
         "写一句话描述春天。" "列出三种常见的水果，用逗号分隔。"; do
  curl -s --max-time 240 http://127.0.0.1:30082/generate \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"$p\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":32}}" &
done; wait
```

## 4. 停机

```bash
bash "$W/code/xy/sglang-kunlun/01_eb/tool/stop_kunlun.sh"
```

**不要 `kill -9`**：会留下杀不掉的 XPU kernel，拖坏整个 XPU runtime（详见源仓库
`01_eb_adp/doc/3` 的「最重要的一条」）。

## 5. 硬约束

- **必须 INT8**：BF16 整网 84.37 GB/卡，KV pool 直接拒绝（`minimum viable = 0.8793`），
  0.88 也会首前向 OOM。
- 起服务的入口必须是 `sglang-kunlun-launch`，不要用 `python -m sglang.launch_server`：
  后者会在 `_kunlun_pre_shim()` 之前 import fork 的 eb5 模型，触发
  `sgl_kernel` / compat alias 连锁失败。

## 6. 实测基线

2026-10-21 INT8 8 卡、43 层：

- ready 约 177–186 s
- 4/4 请求 HTTP 200，单请求约 25–44 s
- 日志：`applied 142 EB5 Kunlun patches from sglang_kunlun.eb5_adapt`
- 致命错误计数 0
