#!/usr/bin/env bash
# Prefill role: 8 XPUs (tp8/ep8/cp8), context-parallel interleave,
# language-only (no ViT / aligner), sends KV to the decode role over mooncake.
#
# Required environment:
#   MODEL_PATH     DeepSeek-V4.1-Flash W4A8 checkpoint
#   ENCODER_URLS   space-separated encoder worker URLs,
#                  e.g. "http://<enc-ip>:8000 ... http://<enc-ip>:8007"
# Optional:
#   HOST_IP        prefill host IP (default: first address of `hostname -i`)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.common.sh"
require_model_path
source "${HERE}/env.prefill.sh"

export HOST_IP="${HOST_IP:-$(hostname -i | awk '{print $1}')}"
export MASTER_IP="${HOST_IP}"
export KL_IDX_PLANE_COLS="${KL_IDX_PLANE_COLS:-114688}"
: "${ENCODER_URLS:?set ENCODER_URLS to the encoder worker URLs}"

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
  --port "${HTTP_PORT:-8000}" --host "${HOST_IP}" \
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
