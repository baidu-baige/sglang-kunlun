#!/bin/bash
# EB5 / EB-Lite (Ernie Lite) 在昆仑 P800 上起服务。
#
# 参考：
#   * 昆仑侧：`01_eb/tool/env.sh`（本仓库自建 venv 的环境激活脚本）
#   * EB5 侧：fork 的 `ce/scripts/eb/gpu-b/eb5_mtp_tp4dp4ep4_128k_opt.sh`
#              + `scripts/internal/launch_demo_scripts/eb5/common_eb5_env.sh`
#
# 与原始 EB5 脚本的差异（昆仑没有的东西 / 本权重是 bf16）：
#   - 本权重 `hf_checkpoint-48800` 是**纯 BF16**（无 blockwise_fp8）→
#     去掉 `--quantization blockwise_fp8`
#   - 去掉 `--fp8-gemm-backend deep_gemm` / `--moe-runner-backend deep_gemm`
#     （deep_gemm 整包缺失）
#   - 去掉 `--enable-hierarchical-cache` / `--speculative-*`（先跑通主链路，后加）
#   - `SGLANG_PLUGINS` 白名单里加 `eb5_kunlun_patch`（把 EB5 融合算子换成
#     torch 拆解 / KLX 原生算子），见 `sglang_kunlun/hooks/eb5_kunlun_patch.py`
#
# 用法：
#   bash run_eb5_kunlun.sh [MODEL_PATH] [PORT]
set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_BINGOO_DIR="$HERE"
while [ "$_BINGOO_DIR" != / ] && [ ! -x "$_BINGOO_DIR/env/bin/python" ]; do
  _BINGOO_DIR=$(dirname "$_BINGOO_DIR")
done
BINGOO=${BINGOO:-$_BINGOO_DIR}
unset _BINGOO_DIR

# 环境：venv + 动态库搜索路径 + 设备可见性 + 昆仑运行时开关 + KLX 集合通信变量。
# **不要**再单独设 `SGLANG_PLATFORM` / `SGLANG_USE_XPU`：标准 launcher 通过 entry
# points 发现 kunlun 平台，提前设会让 sitecustomize 过早 bootstrap XPU
# （见 `01_eb/tool/env.sh` 第 3 段的注释）。
source "$HERE/env.sh"

MODEL_PATH=${1:-$BINGOO/models/hf_checkpoint-48800}
PORT=${2:-30082}

# 用哪几张卡（默认全 8 张）。TP/DP/EP 都按卡数自动取，与 run_dsv4_newenv.sh 一致。
EB5_GPUS=${EB5_GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<<"$EB5_GPUS")
export XPU_VISIBLE_DEVICES=$EB5_GPUS
export CUDA_VISIBLE_DEVICES=$EB5_GPUS

# 可选：给模型 config 打补丁（JSON）。4 卡装不下完整 43 层（573GB / 4 = 143GB > 96GB），
# 用减层子模型时必须同时改 num_hidden_layers 和 compress_ratios：
#   MODEL_OVERRIDE_ARGS='{"num_hidden_layers":8,"compress_ratios":[128,128,128,128,128,128,128,-2,-2]}'
MODEL_OVERRIDE_ARGS=${MODEL_OVERRIDE_ARGS:-}
# 用数组传，避免 JSON 里带空格时被 word-split。
OVERRIDE_ARG=()
[ -n "$MODEL_OVERRIDE_ARGS" ] && OVERRIDE_ARG=(--json-model-override-args "$MODEL_OVERRIDE_ARGS")
# BF16 权重（每卡 ~67 GB / 可用 96 GB）下 KV cache 只剩 ~25 GB，先跑通：
# 小 context、低并发、关 CUDA graph。
MEM_FRACTION=${MEM_FRACTION:-0.85}  # 0.90 -> 整网(43 层/8 卡)时 KV pool 把卡吃到只剩 28 MB，
# 首个 forward 里 `_store_bf16_rows` 申请 38 MB 临时缓冲就直接 OOM
# （2026-10-20 23:15 实测，`fused_k_norm_rope_flashmla_bf16`）。
# 0.85 留出 ~14 GB 给临时张量；8L 在 0.90/0.85 下都够用。
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}
# KV cache 必须显式给 bf16。默认 `auto` 落到 FP8(uint8) 布局，而 SWA 的 K 写入
# （`set_swa_key_buffer_radix_fused_norm_rope`）在 fp8 布局下走
# `dsv4_torch_shim._e4m3_to_uint8`，里面有一次 `.cpu()`（昆仑没有设备侧 fp32→fp8
# 转换，只能在 host 上转）。2026-10-20 实测：4 rank 首次前向永久停在那个 `.cpu()`，
# watchdog 300 s 后退出，并在 4 张卡上各留一条 KL_XID_KERNEL_EXCEPTION。
# bf16 布局改走 `fused_k_norm_rope_flashmla_bf16` → 插件原生
# `kunlun_ops.rmsnorm` + `xspeedgate_ops.flashinfer_rotary_embedding` +
# `xspeedgate_ops.set_k_and_s_v4`，与 DSV4 已跑通的那条同类。
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}
EXTRA_ARGS=${EXTRA_ARGS:-}
LOG=$BINGOO/logs/eb5_kunlun_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$BINGOO/logs"

# `code/sglang`（fork）与 `code/sglang-kunlun`（插件）都是 editable 安装，
# 不再需要手工拼 PYTHONPATH。
unset PYTHONPATH

# ---- 平台 + 插件（SGLANG_PLUGINS 是白名单，设了就只加载列出的） ----
# **`kunlun` 必须列进来**：白名单对 `sglang.srt.platforms` 与 `sglang.srt.plugins`
# 两个 group 一视同仁（`load_plugins_by_group` 里统一过滤），漏了它平台插件会被跳过，
# 自动回落到 `CudaSRTPlatform`（昆仑伪装成 CUDA），`_kunlun_pre_shim` 不跑、
# `sgl_kernel` 不被 stub，随后 `sglang.srt.layers.quantization` 一挂就是
# `<kunlun-alias-stub>` 连锁失败。
export SGLANG_PLUGINS=kunlun,kunlun_hooks,eb5_internal_runtime_patch,eb5_kunlun_patch
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang.srt.internal.models.eb5
# EB5 的融合算子拆解（torch / KLX 原生）随 sglang_kunlun 包一起安装，由
# eb5_kunlun_patch 从 `sglang_kunlun.eb5_adapt` 导入，不再需要指定目录。
# 仅当要用工作区里的平铺副本临时覆盖时才设置：
#   export EB5_KUNLUN_ADAPT_DIR=<含三个 eb5_*_adapt.py 的目录>

# ---- 昆仑 torch-shim 基线（沿用 DSV4 已验证的那套） ----
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True SGLANG_OPT_USE_JIT_NORM=True
export SGLANG_OPT_USE_TOPK_V2=False SGLANG_TOPK_TRANSFORM_512_TORCH=False
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=False SGLANG_OPT_USE_TILELANG_INDEXER=False SGLANG_OPT_USE_AITER_INDEXER=False
export SGLANG_OPT_USE_FUSED_STORE_CACHE=0 SGLANG_OPT_USE_COMPRESSOR_V2=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
# `mhc_pre` / `mhc_post` 由适配层整体换成散装 torch 实现（与 DSV4 的
# `hc_pre` / `hc_post_torch_impl` 逐式一致：GEMM → sinkhorn → combine → layernorm），
# 不再经过任何融合 kernel，也没有回退开关。
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_HACK_FLASHMLA_BACKEND=kunlun SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_IS_FLASHINFER_AVAILABLE=False SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_FP8_WO_A_GEMM=False SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_PREP_IN_CUDA_GRAPH=False SGLANG_OPT_CP_REARRANGE_TRITON=False
export SGLANG_KUNLUN_INDEXER_FP32=1

# ---- EB5 侧（来自 ce/scripts/eb/gpu-b/eb5_mtp_tp4dp4ep4_128k_opt.sh） ----
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=4
export SGLANG_EB5_MLA_HCA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="$no_proxy"

echo "[run_eb5_kunlun] model=$MODEL_PATH port=$PORT gpus=$EB5_GPUS (n=$NGPU) log=$LOG"

# ---- 起服务前先安全清掉上一次的残留 ----
# 这里**不再**用 `pkill -9 -f "sglang.launch_server|sglang::"`：它有两个真实的坑
#   1) 会打到调用者自己（`pkill -f` 匹配整条命令行，调用命令本身就含这些字符串）；
#   2) 对 launcher 用 -9 会留下 stuck kernel（doc/3 的「最重要的一条」）。
# 换成 stop_kunlun.sh：先 TERM/INT，必要时只对空闲的纯 CPU worker 用 -9 解开
# waitpid 死锁，launcher 绝不用 -9，最后校验 8 卡回到 0 MiB / 0%。
# `SKIP_STOP=1`：别的卡上还有**别人**的任务时用（stop_kunlun 会按进程名清所有
# sglang 进程，不管在哪几张卡）。此时调用方必须先自己确认目标卡干净。
if [ "${SKIP_STOP:-0}" = "1" ]; then
  echo "[run_eb5_kunlun] SKIP_STOP=1，跳过 stop_kunlun（请自行确认 $EB5_GPUS 上的残留）"
elif ! STOP_GPUS=$EB5_GPUS bash "$HERE/stop_kunlun.sh"; then
  echo "[run_eb5_kunlun] ❌ 残留没清干净，放弃启动（不要硬上）"
  exit 1
fi

# ---- 起服务前的健康闸门 ----
# 为什么不能只看 `xpu-smi`：设备级 kernel 异常之后，8 卡照样显示 `0MiB / 0%`、
# 驱动登记 0 个进程 —— **看着完全正常**，但下一次起服务会挂死（2026-10-17 实测：
# EB5 首次前向报 KL_XID_KERNEL_EXCEPTION 之后，下一次起服务在加载权重阶段挂死）。
# 所以闸门必须包含 **8 进程 XCCL 探针**（单卡探针抓不到 rank 0/1/2 挂死）。
# 急用时可用 `SKIP_PREFLIGHT=1` 跳过（不推荐）。
if ! PREFLIGHT_GPUS=$EB5_GPUS bash "$HERE/preflight.sh"; then
  echo "[run_eb5_kunlun] ❌ preflight 没过，放弃启动（别硬上）"
  exit 1
fi

# 入口用插件自带的 `sglang-kunlun-launch`，**不要**用 `python -m sglang.launch_server`：
# 标准入口在 `load_plugins()` 里先加载 general 插件，此时平台还没解析、
# `_kunlun_pre_shim` 没跑、`sgl_kernel` 没被 stub，fork 的 `eb5_internal_runtime_patch`
# 一 import `sglang.srt.internal.models.eb5`（→ quantization → `from sgl_kernel import ...`）
# 就炸，随后 compat_aliases 把 `sglang.srt.layers.quantization` 变成
# `<kunlun-alias-stub>`，连锁失败。`sglang-kunlun-launch` 会先 `import torch`、
# 再显式调 `_kunlun_pre_shim()`，然后才 runpy 进 launch_server。
nohup sglang-kunlun-launch \
  --model-path "$MODEL_PATH" \
  --served-model-name Ernie_Lite \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --dtype bfloat16 \
  --context-length "$CONTEXT_LENGTH" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --disable-cuda-graph \
  --swa-full-tokens-ratio 0.1 \
  --tp-size "$NGPU" --dp-size "$NGPU" --ep-size "$NGPU" \
  "${OVERRIDE_ARG[@]}" \
  --enable-dp-attention \
  --disable-custom-all-reduce \
  --page-size 256 \
  --nccl-port 9999 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --tool-call-parser ernie_lite \
  --reasoning-parser ernie_lite \
  --tokenizer-worker-num 4 \
  --enable-metrics \
  $EXTRA_ARGS \
  > "$LOG" 2>&1 &

echo "[run_eb5_kunlun] pid=$! log=$LOG"
echo "[run_eb5_kunlun] 探活: curl -s http://127.0.0.1:$PORT/health_generate"
