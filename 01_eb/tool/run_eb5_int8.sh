#!/bin/bash
# EB5 在线 INT8 量化跑法（昆仑 P800）。包一层 `run_eb5_kunlun.sh`，只负责
# 打开 INT8 的开关并给出 8L 的减层子模型；其余（env / preflight / 端口）沿用原脚本。
#
# 用法：
#   bash run_eb5_int8.sh [MODEL_PATH] [PORT]
#   LAYERS=8 EB5_GPUS=4,5,6,7 bash run_eb5_int8.sh
#
# 量化方案（见 sglang/srt/internal/layers/quantization/blockwise_int8_patch.py）：
#   * 权重：加载后 per-output-channel 对称 INT8（absmax 标度），MoE 专家 CPU 暂存逐 expert 量化
#   * 激活：运行期 per-token 对称 INT8（kunlun_ops.quant2d）
#   * 昆仑这套 kunlun_ops 没有 128×128 block scale 的 INT8 GEMM，故粒度到 per-channel
#
# 显存：EB5 权重 79.5 GB/卡 → ~40 GB/卡（8 卡整网）；8L/4 卡 26.5 GB → ~13 GB。
set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_BINGOO_DIR="$HERE"
while [ "$_BINGOO_DIR" != / ] && [ ! -x "$_BINGOO_DIR/env/bin/python" ]; do
  _BINGOO_DIR=$(dirname "$_BINGOO_DIR")
done
BINGOO=${BINGOO:-$_BINGOO_DIR}
unset _BINGOO_DIR
MODEL_PATH=${1:-$BINGOO/models/hf_checkpoint-48800}
PORT=${2:-30199}

LAYERS=${LAYERS:-8}
EB5_GPUS=${EB5_GPUS:-4,5,6,7}

# ---- INT8 开关 ----
export EB5_INT8_ONLINE=1          # 总开关：把 --quantization blockwise_int8 映射到本实现
export EB5_INT8_MOE_ONLY=${EB5_INT8_MOE_ONLY:-0}      # 1 = 只量化 routed experts
export EB5_INT8_MOE_CPU_STAGING=${EB5_INT8_MOE_CPU_STAGING:-1}

# ---- 减层子模型：num_hidden_layers 与 compress_ratios 必须同时改 ----
if [ -n "$LAYERS" ] && [ "$LAYERS" != "43" ]; then
  RATIOS='[128,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,-2]'
  # 取前 $LAYERS 个，其余丢弃
  RATIOS=$(python3 - "$LAYERS" "$RATIOS" <<'PY'
import json, sys
n = int(sys.argv[1]); r = json.loads(sys.argv[2])
print(json.dumps(r[:n]))
PY
)
  export MODEL_OVERRIDE_ARGS="{\"num_hidden_layers\":$LAYERS,\"compress_ratios\":$RATIOS}"
fi

export EXTRA_ARGS="--deepep-mode normal --quantization blockwise_int8 ${EXTRA_ARGS:-}"

echo "[run_eb5_int8] layers=$LAYERS gpus=$EB5_GPUS port=$PORT override=$MODEL_OVERRIDE_ARGS"
exec bash "$HERE/run_eb5_kunlun.sh" "$MODEL_PATH" "$PORT"
