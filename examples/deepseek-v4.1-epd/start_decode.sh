#!/usr/bin/env bash
# Decode role: 8 XPUs (tp8/ep8), disaggregated decode, low-latency DeepEP
# with CUDA graphs.
#
# Required environment:
#   MODEL_PATH  DeepSeek-V4.1-Flash W4A8 checkpoint
# Optional:
#   POD_IP      decode host IP (default: first address of `hostname -i`)
#   HTTP_PORT   decode engine port (default: 8000)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.common.sh"
require_model_path
source "${HERE}/env.decode.sh"

export POD_IP="${POD_IP:-$(hostname -i | awk '{print $1}')}"

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
  --port "${HTTP_PORT:-8000}" --host "${POD_IP}" \
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
