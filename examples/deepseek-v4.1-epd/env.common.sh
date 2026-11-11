#!/usr/bin/env bash
# Shared environment for the DeepSeek-V4.1-Flash EPD example.
#
# Source this file from every role script.
#
# Required (set before running a role):
#   MODEL_PATH  path to the DeepSeek-V4.1-Flash W4A8 checkpoint
#
# Optional:
#   MODEL_ID             served model name (default: DeepSeek-V4.1-Flash)
#   PLUGIN_DIR           plugin tree in the image (default: /workspace/sglang_kunlun)
#   LIBCUDART_PRELOAD    absolute path to libcudart.so.11.0 when the image does
#                        not put it on the loader path
#   IB_DEVICE_CANDIDATES space-separated RDMA devices to probe for the mooncake
#                        KV transfer use "none" to skip probing

export MODEL_ID="${MODEL_ID:-DeepSeek-V4.1-Flash}"
export PLUGIN_DIR="${PLUGIN_DIR:-/workspace/sglang_kunlun}"
export PYTHONPATH="${PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TZ="${TZ:-Asia/Shanghai}"

# Kunlun runtime: torch_xmlir, kunlun_ops and the operator wheels live in this
# conda environment in the image built from dockerfile/.
source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch29_cuda

# RDMA devices used for the mooncake KV transfer. Adjust to the host: the
# candidates below are the default set on a Kunlun3 P800 node.
IB_DEVICES=""
for _d in ${IB_DEVICE_CANDIDATES:-mlx5_3 mlx5_5 mlx5_7 mlx5_8 mlx5_2 mlx5_4 mlx5_6 mlx5_9}; do
    if [ -e "/sys/class/infiniband/${_d}" ]; then
        IB_DEVICES="${IB_DEVICES:+${IB_DEVICES},}${_d}"
    fi
done
[ -z "${IB_DEVICES}" ] && IB_DEVICES=mlx5_3
export IB_DEVICES

if [ -n "${LIBCUDART_PRELOAD:-}" ]; then
    export LD_PRELOAD="${LIBCUDART_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

require_model_path() {
    : "${MODEL_PATH:?MODEL_PATH must point to the DeepSeek-V4.1-Flash W4A8 checkpoint}"
}
