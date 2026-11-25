#!/usr/bin/env bash
# Prefill role: 8 XPUs (tp8/ep8/cp8), context-parallel interleave, language-only
# (no ViT / aligner), sends KV to the decode role over mooncake.
#
# Required:
#   MODEL_PATH     DeepSeek-V4.1-Flash W4A8 checkpoint
#   ENCODER_URLS   space-separated encoder worker URLs, e.g.
#                  "http://<enc-ip>:8000 ... http://<enc-ip>:8007"
# Optional:
#   MODEL_ID             served model name (default: DeepSeek-V4.1-Flash)
#   PLUGIN_DIR           plugin tree in the image (default: /workspace/sglang_kunlun)
#   HOST_IP              prefill host IP (default: first address of `hostname -i`)
#   LIBCUDART_PRELOAD    libcudart.so.11.0 path when it is not on the loader path
#   IB_DEVICE_CANDIDATES RDMA devices to probe for the mooncake transfer
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must point to the DeepSeek-V4.1-Flash W4A8 checkpoint}"
: "${ENCODER_URLS:?set ENCODER_URLS to the encoder worker URLs}"
export MODEL_ID="${MODEL_ID:-DeepSeek-V4.1-Flash}"
export PLUGIN_DIR="${PLUGIN_DIR:-/workspace/sglang_kunlun}"
export PYTHONPATH="${PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch29_cuda

# --- environment (validated SGLang 0.5.21 prefill pose) ---------------------
export HTTP_PORT="8000"
export PD_SEPARATE_STAGE="prefill"
export LD_LIBRARY_PATH="/root/.cache/kunlun-libshim:/usr/local/xccl/so:/usr/local/lib/:/usr/local/cuda-12.8/:/usr/lib/x86_64-linux-gnu:/usr/lib64"
export PYTHONUNBUFFERED="1"
export TZ="Asia/Shanghai"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CXPU_VISIBLE_DEVICES="1,2,3,4,5,6,7,0"
export CUDA_ENABLE_P2P_NO_UVA="1"
export CUDA_DEVICE_MAX_COLLECTIONS="8"
export CUDA_DISABLE_PRINTF="1"
export CUDA_GRAPH_OPTIMIZE_STREAM="1"
export BKCL_INFERENCE="1"
export BKCL_ENABLE_IBGDA="0"
export BKCL_ENABLE_XDR="0"
export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"
export BKCL_RDMA_VERBS="1"
export BKCL_NORMAL_COMBINE_DEBUG="0"
export BKCL_PROXY_THREAD_NUM="4"
export BKCL_QPS_PER_CONNECTION="4"
export BKCL_RING_BUFFER_SIZE="33554432"
export BKCL_USE_PEERMEM_XDR="0"
export BKCL_DEEPEP_NORMAL_CLUSTER_NUM="8"
export NCCL_IB_GID_INDEX="3"
export XMLIR_FORCE_USE_XPU_GRAPH="1"
export XMLIR_ENABLE_FAST_FC="1"
export XMLIR_CUDNN_ENABLED="1"
export XPU_USE_FAST_SWIGLU="1"
export XPU_USE_DEFAULT_CTX="1"
export USE_FAST_ALLOC_EXTEND_KUNLUN="False"
export SGLANG_PLATFORM="kunlun"
export SGLANG_USE_XPU="1"
export SGLANG_PLUGINS="kunlun_hooks"
export SGLANG_HACK_FLASHMLA_BACKEND="kunlun"
export SGLANG_IS_FLASHINFER_AVAILABLE="False"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="1"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="0"
export SGLANG_DSV4_FP4_EXPERTS="0"
export SGLANG_DSV41_TORCH_PREFILL_INDEXER="1"
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="True"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="0"
export SGLANG_OPT_USE_TOPK_V2="0"
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="0"
export SGLANG_OPT_FP8_WO_A_GEMM="False"
export SGLANG_OPT_USE_TILELANG_MHC_PRE="0"
export SGLANG_OPT_USE_TILELANG_MHC_POST="0"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="0"
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL="0"
export SGLANG_SET_CPU_AFFINITY="1"
export SGLANG_ENABLE_PREFILL_FWD_THREAD="0"
export SGLANG_TOOL_STRICT_LEVEL="2"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="3600"
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER="1"
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="0"
export ENABLE_DEEP_DP="1"
export DISABLE_TP_WITH_SP="1"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="1024"
export XSHMEM_MODE="0"
export XSHMEM_QP_NUM_PER_RANK="32"
export XSHMEM_SYMMETRIC_SIZE="4294967296"
export MOONCAKE_USE_RDMA="TRUE"
export MOONCAKE_ASYNC_SEND="FALSE"
export MOONCAKE_SEND_RECV_TIMEOUT="60"
export MOONCAKE_META_SERVER_URL="${MOONCAKE_META_SERVER_URL:-redis://127.0.0.1:6379}"
export MOONCAKE_META_SERVER_PASSWORD="${MOONCAKE_META_SERVER_PASSWORD:-}"
export KL_ROPE_CHUNK="0"
export KL_IDX_SCORE_BUDGET_MB="512"
export KL_CP_GATHER_MIN_MB="1024"
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP="1"

export HOST_IP="${HOST_IP:-$(hostname -i | awk '{print $1}')}"
export MASTER_IP="${HOST_IP}"
export KL_IDX_PLANE_COLS="${KL_IDX_PLANE_COLS:-114688}"

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

# tvm-ffi torch-c-dlpack addon: build it once if the image does not ship one.
TVM_DIR="${HOME}/.cache/tvm-ffi"
TVM_LIB="libtorch_c_dlpack_addon_torch29-cuda-6f4089e8.so"
TVM_UTILS="/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/tvm_ffi/utils/_build_optional_torch_c_dlpack.py"
mkdir -p "${TVM_DIR}"
if [ ! -f "${TVM_DIR}/${TVM_LIB}" ] && [ -f "${TVM_UTILS}" ]; then
    echo "[boot] building tvm-ffi torch-c-dlpack addon"
    env -u SGLANG_PLATFORM -u SGLANG_USE_XPU -u SGLANG_PLUGINS \
        TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
        python3 "${TVM_UTILS}" \
        --output-dir "${TVM_DIR}" --libname "${TVM_LIB}" --build-with-cuda \
        || echo "[boot] WARN: tvm-ffi addon build failed"
fi

# The grammar backend is patched by the plugin's constrained hook; refuse to
# start with --grammar-backend xgrammar when the hook is missing.
REG="${PLUGIN_DIR}/sglang_kunlun/hooks/registry.py"
if ! grep -qE 'sglang_kunlun\.hooks\.constrained|patch_grammar_backend_default' "${REG}"; then
    echo "[boot] FATAL: grammar hook is not wired in ${REG}" >&2
    exit 1
fi

echo "[prefill] host=${HOST_IP} ib=${IB_DEVICES}"
cd /workspace
exec python3 -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp 8 --ep 8 \
  --dist-init-addr "${MASTER_IP}:7788" \
  --port "${HTTP_PORT}" --host "${HOST_IP}" \
  --trust-remote-code \
  --dtype float16 --kv-cache-dtype float16 \
  --chunked-prefill-size 32768 \
  --context-length 114688 \
  --max-total-tokens 250000 \
  --max-running-requests 4 --max-prefill-tokens 250000 \
  --swa-full-tokens-ratio 0.5 \
  --disable-radix-cache --disable-cuda-graph \
  --attention-backend kunlun_dsv4 \
  --grammar-backend xgrammar \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  --mem-fraction-static 0.85 \
  --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":64}' \
  --served-model-name "${MODEL_ID}" --skip-server-warmup \
  --moe-a2a-backend deepep --deepep-mode normal \
  --enable-prefill-cp --cp-strategy interleave \
  --enable-decoder-swa-bounded-replay \
  --language-only --image-processor-backend pil \
  --encoder-urls ${ENCODER_URLS} \
  --encoder-bootstrap-port 8997 \
  --encoder-transfer-backend zmq_to_scheduler \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 8200 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --load-balance-method round_robin \
  --enable-metrics
