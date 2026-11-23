#!/usr/bin/env bash
# Encoder role: one single-XPU vision worker per device. Each worker runs the
# vision tower + aligner and pushes embeddings to the prefill
# EncoderBootstrapServer (:8997); it never serves text and never goes through
# the router.
#
# Required:
#   MODEL_PATH   DeepSeek-V4.1-Flash W4A8 checkpoint
#   PREFILL_IP   prefill host IP
# Optional:
#   MODEL_ID            served model name (default: DeepSeek-V4.1-Flash)
#   PLUGIN_DIR          plugin tree in the image (default: /workspace/sglang_kunlun)
#   ENCODE_WORKERS      number of workers (default: 8)
#   ENCODE_BASE_PORT    first worker port (default: 8000)
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must point to the DeepSeek-V4.1-Flash W4A8 checkpoint}"
: "${PREFILL_IP:?set PREFILL_IP to the prefill host IP}"
export MODEL_ID="${MODEL_ID:-DeepSeek-V4.1-Flash}"
export PLUGIN_DIR="${PLUGIN_DIR:-/workspace/sglang_kunlun}"
export PYTHONPATH="${PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
unset XPU_DUMMY_EVENT

source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch29_cuda
export LD_LIBRARY_PATH="/root/miniconda/envs/python310_torch29_cuda/xcudart/lib:${LD_LIBRARY_PATH:-}"

# --- Kunlun platform and DSV4 kernel / torch-shim selection -----------------
export SGLANG_PLATFORM="kunlun"
export SGLANG_USE_XPU="1"
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP="1"
export TVM_FFI_DISABLE_TORCH_C_DLPACK="1"
export SGLANG_HACK_FLASHMLA_BACKEND="kunlun"
export SGLANG_IS_FLASHINFER_AVAILABLE="False"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="1"
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="True"
export SGLANG_OPT_USE_JIT_NORM="True"
export SGLANG_OPT_USE_TOPK_V2="False"
export SGLANG_TOPK_TRANSFORM_512_TORCH="True"
export SGLANG_OPT_USE_JIT_INDEXER_METADATA="False"
export SGLANG_OPT_USE_TILELANG_INDEXER="False"
export SGLANG_OPT_USE_AITER_INDEXER="False"
export SGLANG_OPT_USE_FUSED_STORE_CACHE="1"
export SGLANG_KUNLUN_INDEXER_FP32="1"
export SGLANG_OPT_USE_TILELANG_MHC_PRE="0"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="0"
export SGLANG_OPT_USE_TILELANG_MHC_POST="0"
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="0"
export SGLANG_FIX_DSV4_BASE_MODEL_LOAD="1"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="0"
export SGLANG_DSV4_FP4_EXPERTS="0"
export SGLANG_OPT_FP8_WO_A_GEMM="False"
export SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_DSV41_TORCH_PREFILL_INDEXER="1"
export SGLANG_OPT_DSV41_DEQUANT_ROWS_PER_CTA="0"
export SGLANG_KUNLUN_FUSED_VOCAB_EMBEDDING="0"
export SGLANG_ENABLE_FAST_INPUT_LOGPROBS="0"
export SGLANG_KUNLUN_SHIM_DEBUG="0"

# Encoder-specific: CPU image preprocessing (the GPU path is not validated on
# XPU), the VLM cache, and no page-cache weight preload -- the encoder-only
# loader materialises only the vision / aligner / image modules.
export SGLANG_KUNLUN_PRELOAD_WEIGHTS="0"
export SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU="0"
export SGLANG_VLM_CACHE_SIZE_MB="8192"

ENCODE_WORKERS="${ENCODE_WORKERS:-8}"
ENCODE_BASE_PORT="${ENCODE_BASE_PORT:-8000}"

pids=()
for i in $(seq 0 $((ENCODE_WORKERS - 1))); do
    port=$((ENCODE_BASE_PORT + i))
    echo "[encoder] worker ${i}: port=${port} -> ${PREFILL_IP}:8997"
    CUDA_VISIBLE_DEVICES="${i}" XPU_VISIBLE_DEVICES="${i}" \
        python3 -u -m sglang.launch_server \
        --model-path "${MODEL_PATH}" --served-model-name "${MODEL_ID}" \
        --host 0.0.0.0 --port "${port}" --trust-remote-code \
        --tp 1 --dp 1 --dtype float16 \
        --encoder-only --mm-attention-backend sdpa --image-processor-backend pil \
        --encoder-transfer-backend zmq_to_scheduler \
        --encoder-register-urls "http://${PREFILL_IP}:8997" \
        --max-running-requests 32 &
    pids+=($!)
done

trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
wait
