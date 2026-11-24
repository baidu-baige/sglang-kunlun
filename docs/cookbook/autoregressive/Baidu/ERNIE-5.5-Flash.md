# ERNIE-5.5-Flash (Kunlun)

## Overview

This page deploys **ERNIE-5.5-Flash-Preview-INT4** on **Kunlun (XPU)** with SGLang, served through the `sglang-kunlun` plugin stack (`SGLANG_PLATFORM=kunlun`). The checkpoint ships W4A8 weights; the recipes below cover prefill/decode (PD) disaggregation with DSpark speculative decoding and the MegaMoE all-to-all backend.

## Key Features

- PD Disaggregation
- DSpark
- MegaMoE
- Parallelism: TP / CP / DP-Attention, MoE backend + EP

## Deployment

### 1. Model

Download from ModelScope: [baidu-aicapx/ERNIE-5.5-Flash-Preview-INT4](https://modelscope.cn/models/baidu-aicapx/ERNIE-5.5-Flash-Preview-INT4).

### 2. Docker image

```bash
# Download
wget https://aihc-private-hcd.cdn.bcebos.com/opensource/sglang-kunlun/ernie/images/sglang-kunlun-0.5.23-ernie5.5.tar
# Load
docker load -i sglang-kunlun-0.5.23-ernie5.5.tar
```

> **Note:** 
> - Use the `sglang` code bundled in the image at `/workspace/sglang`; it differs slightly from the upstream code at https://github.com/sgl-project/sglang.
>
> - For the versions of the other dependencies, refer to `dockerfile/Dockerfile`.

### 3. Start Server

When using prefix cache, remove `--disable-radix-cache`.

Prefill (`p.sh`):

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

LOG_PATH=./logs_eb5_pd_prefill/eb5-prefill-512k-$(date +%Y%m%d%H%M).log
HOST_IP=10.129.128.219
model_path=/home/models/ERNIE-5.5-Flash-Preview-INT4

# ===== 代码路径（插件必须在 sglang 前面）
export PYTHONPATH=/workspace/sglang-kunlun:/workspace/sglang/python:${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_PLUGINS=kunlun,kunlun_hooks
export SGLANG_EXTERNAL_MODEL_PACKAGE=""
export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1

# ===== 设备与平台
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_FAST_SWIGLU=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_CUDNN_ENABLED=1
export XMLIR_ENABLE_FAST_FC=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export FORCE_NN_LINEAR=1
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export XPU_ENABLE_PROFILER_TRACING=1             # 抓 trace 的硬前提，启动时就得带上
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
unset XPU_DUMMY_EVENT
# ===== mHC / DSA 由昆仑 hook 接管，关掉上游不适用的融合路径
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
export KUNLUN_MHC_FUSE_FLEET_SINKHORN=1          # 设 0 回 torch 小算子链
export SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.05   # 框架默认 0.2

# ===== 精度 / 量化（权重自带 W4A8，所以不传 --quantization）
export EB5_INT8_DENSE=t2                         # dense 在线 INT8 的 tier2（路由专家仍走 checkpoint 的 W4A8）
export EB5_INT8_ONLINE=0
export SGLANG_FP16_LIMIT_IN_MOE=65504

# ===== 集合通信 / a2a（megamoe）
export BKCL_RDMA_VERBS=1
export BKCL_ENABLE_XDR=1
export BKCL_ENABLE_IBGDA=1                       # XDR 的 QP 约定靠它生效，缺了 combine 会写错远端地址
export BKCL_USE_PEERMEM_XDR=0
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export NCCL_IB_GID_INDEX=3
export XSHMEM_MODE=1
# megamoe 必须覆盖的几条：hydra linear 和 megamoe kernel 不能同占设备
export XMLIR_USE_HYDRA_LINEAR=0
export BKCL_USE_RDMA=1
export BKCL_RING_OPT=1
export BKCL_USE_AR=1
export XSHMEM_QP_NUM_PER_RANK=32                 # megamoe 的口径
export XSHMEM_SYMMETRIC_SIZE=8589934592          # 8GiB
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8192

# ===== PD 传输
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=360000
export MC_TCP_ENABLE_CONNECTION_POOL=true
export MOONCAKE_USE_RDMA=TRUE

# 换集群先确认：有的环境必须用 mlx5_0
disaggregation_ib_device=$(ibdev2netdev | awk '$1 != "mlx5_0" && $NF == "(Up)" {print $1}' | paste -sd, -)
[ -n "$disaggregation_ib_device" ] || disaggregation_ib_device=mlx5_2,mlx5_3,mlx5_4,mlx5_5

mkdir -p "$(dirname -- "$LOG_PATH")"
echo "[p] host=$HOST_IP ib=$disaggregation_ib_device port=31100 boot=31500 log=$LOG_PATH"
nohup python3 -u -m sglang.launch_server \
  --model-path $model_path \
  --served-model-name ernie55-w4a8 \
  --host $HOST_IP --port 31100 \
  --trust-remote-code \
  --dtype float16 --kv-cache-dtype fp16 \
  --context-length 524288 \
  --mem-fraction-static 0.6 \
  --chunked-prefill-size 32768 \
  --max-running-requests 32 \
  --disable-cuda-graph \
  --attention-backend ernie55 \
  --tp 8 --ep 8 \
  --attn-dp-size 8 \
  --enable-dp-lm-head \
  --moe-a2a-backend megamoe \
  --speculative-algorithm DSPARK \
  --speculative-num-draft-tokens 5 \
  --speculative-dspark-block-size 4 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 31500 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$disaggregation_ib_device" \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --disable-custom-all-reduce \
  --load-balance-method round_robin \
  --reasoning-parser ernie_lite --tool-call-parser ernie_lite \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
  --enable-metrics --enable-cache-report \
  --watchdog-timeout 30000 \
  --page-size 256 \
  --swa-full-tokens-ratio 0.1 \
  > "$LOG_PATH" 2>&1 &
```

Decode (`d.sh`):

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

LOG_PATH=./logs_eb5_pd_decode/eb5-decode-512k-$(date +%Y%m%d%H%M).log
HOST_IP=10.129.128.159
model_path=/home/models/ERNIE-5.5-Flash-Preview-INT4

# ===== 代码路径（插件必须在 sglang 前面）
export PYTHONPATH=/workspace/sglang-kunlun:/workspace/sglang/python:${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_PLUGINS=kunlun,kunlun_hooks
export SGLANG_EXTERNAL_MODEL_PACKAGE=""
export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1

# ===== 设备与平台
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_FAST_SWIGLU=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_CUDNN_ENABLED=1
export XMLIR_ENABLE_FAST_FC=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export FORCE_NN_LINEAR=1
export XPU_ENABLE_PROFILER_TRACING=1
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
# ===== mHC / DSA 由昆仑 hook 接管
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export KUNLUN_MHC_FUSE_FLEET_SINKHORN=1          # 设 0 回 torch 小算子链

# ===== DSPARK（只有 D 侧有 draft worker，P 侧不要设这些）
export SGLANG_RAGGED_VERIFY_MODE=static          # megamoe / deepep a2a 下必须是 static
export SGLANG_DSPARK_USE_GREEDY_DRAFT=1          # draft 取 argmax，省掉 torch 随机数 + step logits buffer

# ===== 精度 / 量化
export EB5_INT8_DENSE=1
export EB5_INT8_ONLINE=0
export SGLANG_FP16_LIMIT_IN_MOE=65504

# ===== 集合通信 / a2a（megamoe）
export BKCL_RDMA_VERBS=1
export BKCL_ENABLE_XDR=1
export BKCL_ENABLE_IBGDA=1
export BKCL_USE_PEERMEM_XDR=0
export BKCL_PROXY_THREAD_NUM=4
export BKCL_QPS_PER_CONNECTION=4
export BKCL_RING_BUFFER_SIZE=33554432
export BKCL_NORMAL_COMBINE_DEBUG=0
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export NCCL_IB_GID_INDEX=3
export CUDA_ENABLE_P2P_NO_UVA=1
export XSHMEM_MODE=1
export XMLIR_USE_HYDRA_LINEAR=0
export BKCL_USE_RDMA=1
export BKCL_RING_OPT=1
export BKCL_USE_AR=1
export XSHMEM_QP_NUM_PER_RANK=32
export XSHMEM_SYMMETRIC_SIZE=2147483648
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=240
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=240
# ===== PD 传输
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=360000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export MC_TCP_ENABLE_CONNECTION_POOL=true
export MOONCAKE_USE_RDMA=TRUE
disaggregation_ib_device=$(ibdev2netdev | awk '$1 != "mlx5_0" && $NF == "(Up)" {print $1}' | paste -sd, -)
[ -n "$disaggregation_ib_device" ] || disaggregation_ib_device=mlx5_2,mlx5_3,mlx5_4,mlx5_5

mkdir -p "$(dirname -- "$LOG_PATH")"
echo "[d] host=$HOST_IP ib=$disaggregation_ib_device port=31200 boot=31500 log=$LOG_PATH"

nohup python3 -u -m sglang.launch_server \
  --model-path $model_path \
  --served-model-name ernie55-w4a8 \
  --host $HOST_IP --port 31200 \
  --trust-remote-code \
  --dtype float16 --kv-cache-dtype fp16 \
  --context-length 524288 \
  --mem-fraction-static 0.76 \
  --max-running-requests 512 \
  --cuda-graph-max-bs-decode 48 \
  --attention-backend ernie55 \
  --tp 8 --ep 8 \
  --attn-dp-size 8 \
  --speculative-algorithm DSPARK \
  --speculative-num-draft-tokens 5 \
  --speculative-dspark-block-size 4 \
  --enable-dp-lm-head \
  --moe-a2a-backend megamoe \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 31500 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$disaggregation_ib_device" \
  --no-enable-tp-lm-head-all-to-all \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --disable-custom-all-reduce \
  --load-balance-method round_robin \
  --sleep-on-idle \
  --reasoning-parser ernie_lite --tool-call-parser ernie_lite \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
  --enable-metrics --enable-cache-report \
  --watchdog-timeout 30000 \
  --page-size 256 \
  --swa-full-tokens-ratio 0.1 \
  --tokenizer-worker-num 4 \
  --dsa-topk-backend torch \
  > "$LOG_PATH" 2>&1 &
```

Router (`router.sh`):

```bash
DATE=$(date +%Y%m%d%H%M)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

log_dir=./pd_router
mkdir -p $log_dir
log_path=$log_dir/ernie-pd_router-${DATE}.log

NO_PROXY=0.0.0.0,localhost,127.0.0.1
no_proxy=0.0.0.0,localhost,127.0.0.1

prefil_ip=10.129.128.219
decode_ip=10.129.128.159

nohup python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://${prefil_ip}:31100 31500 \
    --decode http://${decode_ip}:31200 \
    --health-check-timeout-secs 120 \
    --health-failure-threshold 10000 \
    --cb-failure-threshold 30000 \
    --cb-timeout-duration-secs 30000 \
    --request-timeout-secs 720000 \
    --host 0.0.0.0 --port 8003 > $log_path 2>&1 &

echo "router started, log at $log_path"
```
