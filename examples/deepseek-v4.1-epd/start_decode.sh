#!/usr/bin/env bash
# Decode role: 8 XPUs (tp8/ep8), disaggregated decode, low-latency DeepEP with
# CUDA graphs.
#
# Required:
#   MODEL_PATH   DeepSeek-V4.1-Flash W4A8 checkpoint
# Optional:
#   MODEL_ID             served model name (default: DeepSeek-V4.1-Flash)
#   PLUGIN_DIR           plugin tree in the image (default: /workspace/sglang_kunlun)
#   POD_IP               decode host IP (default: first address of `hostname -i`)
#   HTTP_PORT            decode engine port (default: 8000)
#   LIBCUDART_PRELOAD    libcudart.so.11.0 path when it is not on the loader path
#   IB_DEVICE_CANDIDATES RDMA devices to probe for the mooncake transfer
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must point to the DeepSeek-V4.1-Flash W4A8 checkpoint}"
export MODEL_ID="${MODEL_ID:-DeepSeek-V4.1-Flash}"
export PLUGIN_DIR="${PLUGIN_DIR:-/workspace/sglang_kunlun}"
export PYTHONPATH="${PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch29_cuda

# --- environment (validated SGLang 0.5.21 decode pose) ----------------------
export LD_LIBRARY_PATH="/root/.cache/kunlun-libshim:/usr/local/xccl/so:/usr/local/lib/:/usr/local/cuda-12.8/:/usr/lib/x86_64-linux-gnu:/usr/lib64"
export PYTHONUNBUFFERED="1"
export TZ="Asia/Shanghai"
export HTTP_PORT="${HTTP_PORT:-8000}"
export PD_SEPARATE_STAGE="decode"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CXPU_VISIBLE_DEVICES="3,4,5,6,7,0,1,2"
export CUDA_DEVICE_ORDER="OAM_ID"
export CUDA_DEVICE_MAX_COLLECTIONS="8"
export CUDA_DISABLE_PRINTF="1"
export CUDA_ENABLE_P2P_NO_UVA="1"
export CUDA_GRAPH_OPTIMIZE_STREAM="1"
export BKCL_INFERENCE="1"
export BKCL_ENABLE_IBGDA="0"
export BKCL_ENABLE_XDR="0"
export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"
export BKCL_RDMA_VERBS="1"
export BKCL_NORMAL_COMBINE_DEBUG="0"
export BKCL_USE_PEERMEM_XDR="1"
export NCCL_IB_GID_INDEX="3"
export XMLIR_FORCE_USE_XPU_GRAPH="1"
export XMLIR_ENABLE_FAST_FC="1"
export XMLIR_CUDNN_ENABLED="1"
export XPU_USE_FAST_SWIGLU="1"
export XPU_USE_DEFAULT_CTX="1"
export XFT_USE_FAST_SWIGLU="1"
export USE_FAST_ALLOC_EXTEND_KUNLUN="1"
export USE_CUDA_ROPE="1"
export SGLANG_PLATFORM="kunlun"
export SGLANG_MAX_NEW_TOKENS_LIMIT="65536"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="1200"
export SGLANG_USE_XPU="1"
export SGLANG_PLUGINS="kunlun_hooks"
export SGLANG_HACK_FLASHMLA_BACKEND="kunlun"
export SGLANG_IS_FLASHINFER_AVAILABLE="False"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="1"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="0"
export SGLANG_DSV4_FP4_EXPERTS="0"
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="True"
export SGLANG_TOPK_TRANSFORM_512_TORCH="False"
export SGLANG_OPT_USE_TOPK_V2="False"
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="0"
export SGLANG_OPT_FP8_WO_A_GEMM="False"
export SGLANG_OPT_USE_TILELANG_MHC_PRE="0"
export SGLANG_OPT_USE_TILELANG_MHC_POST="0"
export SGLANG_OPT_USE_TILELANG_INDEXER="False"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="0"
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL="0"
export SGLANG_OPT_USE_JIT_NORM="True"
export SGLANG_OPT_USE_JIT_INDEXER_METADATA="False"
export SGLANG_OPT_USE_AITER_INDEXER="False"
export SGLANG_OPT_USE_COMPRESSOR_V2="False"
export SGLANG_OPT_USE_FUSED_STORE_CACHE="0"
export SGLANG_OPT_CP_REARRANGE_TRITON="False"
export SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_KUNLUN_INDEXER_FP32="1"
export SGLANG_KUNLUN_CP_DRAIN="0"
export SGLANG_KUNLUN_CP_DRAIN_SWA="0"
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE="1"
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT="per_rank"
export SGLANG_FIX_DSV4_BASE_MODEL_LOAD="1"
export SGLANG_PREP_IN_CUDA_GRAPH="False"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="3600"
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER="1"
export SGLANG_MOONCAKE_SEND_AUX_TCP="TRUE"
export SGLANG_TOOL_STRICT_LEVEL="2"
export ENABLE_DEEP_DP="1"
export DISABLE_TP_WITH_SP="1"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="32"
export ENABLE_DEEP_EP_FP8_DISPATCH="1"
export ENABLE_DEEP_EP_DISPATCH_CONVERT_CUDA="1"
export ENABLE_DEEP_EP_COMBINE_CONVERT_CUDA="1"
export ENABLE_LEGACY_PREFILL_MOE="0"
export ENABLE_SHARED_EXPERTS_OVERLAP="0"
export ENABLE_SELECT_EXPERTS="0"
export ENABLE_SELECT_EXPERTS_KLX="1"
export ENABLE_SILU_AND_MUL_ONE_EXPERT="0"
export ENABLE_COPY_AND_SET_NATIVE="0"
export ENABLE_POST_REORDER_FUSION="0"
export ENABLE_POST_PROC_CUDA="0"
export ENABLE_ONLY_TP0_SEND_KVCACH="0"
export SET_DEEP_DP_MAX_TOKENS="16"
export SET_DEEP_DP_NUM_EXPERTS="256"
export REBALANCE_WEIGHT_UPDATE_STEPS="16"
export XSHMEM_MODE="1"
export XSHMEM_QP_NUM_PER_RANK="48"
export XSHMEM_SYMMETRIC_SIZE="536870912"
export MC_MAX_EP_PER_CTX="3000"
export ENABLE_PREFILL_OVERLAP="1"
export ENABLE_MICRO_BATCH_OVERLAP="1"
export SYNC_COMM="True"
export ENABLE_FLASHINFER_WARMUP="0"
export ENABLE_TRITON_WARMUP="0"
export ENABLE_FP8_TRANS="0"
export USE_CUTLASS_FP8_BLOCK_GEMM="0"
export USE_FUSED_INPUT_TO_FP8="0"
export MOONCAKE_USE_RDMA="TRUE"
export MOONCAKE_ASYNC_SEND="FALSE"
export MOONCAKE_SEND_RECV_TIMEOUT="60"
export MOONCAKE_META_SERVER_URL="${MOONCAKE_META_SERVER_URL:-redis://127.0.0.1:6379}"
export MOONCAKE_META_SERVER_PASSWORD="${MOONCAKE_META_SERVER_PASSWORD:-}"
export AIAK_DEBUG_DISABLE_KV_TRANSFER="0"
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP="1"
export TVM_FFI_DISABLE_TORCH_C_DLPACK="1"
export LOG_COLLECTION="true"

export POD_IP="${POD_IP:-$(hostname -i | awk '{print $1}')}"

if [ -n "${LIBCUDART_PRELOAD:-}" ]; then
    export LD_PRELOAD="${LIBCUDART_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# RDMA devices used for the mooncake KV transfer; adjust to the host.
IB_DEVICES=""
for _d in ${IB_DEVICE_CANDIDATES:-mlx5_3 mlx5_5 mlx5_7 mlx5_8 mlx5_2 mlx5_4 mlx5_6 mlx5_9}; do
    if [ -e "/sys/class/infiniband/${_d}" ]; then
        IB_DEVICES="${IB_DEVICES:+${IB_DEVICES},}${_d}"
    fi
done
[ -z "${IB_DEVICES}" ] && IB_DEVICES=mlx5_3
export IB_DEVICES

# The grammar backend is patched by the plugin's constrained hook; refuse to
# start with --grammar-backend xgrammar when the hook is missing.
SP=/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages
REG="${SP}/sglang_kunlun/hooks/registry.py"
if [ -f "${REG}" ] && ! grep -qE 'sglang_kunlun\.hooks\.constrained|patch_grammar_backend_default' "${REG}"; then
    echo "[boot] FATAL: grammar hook is not wired in ${REG}" >&2
    exit 1
fi

echo "[decode] host=${POD_IP} ib=${IB_DEVICES}"
cd /workspace
exec python3 -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp 8 --ep 8 \
  --port "${HTTP_PORT}" --host "${POD_IP}" \
  --trust-remote-code \
  --kv-cache-dtype float16 --dtype float16 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 8192 \
  --max-running-requests 8 \
  --max-prefill-tokens 250000 \
  --context-length 114688 \
  --max-total-tokens 1048576 \
  --swa-full-tokens-ratio 0.1 \
  --disable-radix-cache \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs 16 \
  --cuda-graph-bs-decode 8 \
  --attention-backend compressed \
  --grammar-backend xgrammar \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  --disable-overlap-schedule \
  --served-model-name "${MODEL_ID}" \
  --skip-server-warmup \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 8200 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-metrics
