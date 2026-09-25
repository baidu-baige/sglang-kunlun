#!/usr/bin/env bash
# Baseline runtime environment for a Kunlun XPU deployment node.
#
# Source this file (e.g. `source setup_env.sh`) before starting the SGLang
# server. It mirrors the environment documented in the README quick start and
# is a starting point — adjust device visibility and model-specific kernel
# options for your deployment.
#
# NOTE: Do not set SGLANG_PLATFORM / SGLANG_USE_XPU here. The standard SGLang
# launcher discovers the Kunlun platform through installed entry points, and
# sitecustomize bootstraps XPU too early if the platform is forced.

# Use real XPU events instead of dummy events for synchronization.
unset XPU_DUMMY_EVENT

# Expose the XPU devices available on this host. Replace with the IDs of the
# devices you want to use (e.g. 0,1,2,3,4,5,6,7 on a full P800 node).
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Advertise a consistent CUDA-compatible device view for the XPUs above.
export CUDA_VISIBLE_DEVICES=$XPU_VISIBLE_DEVICES

# The Kunlun runtime intentionally reports no flashinfer; let SGLang bail out
# of flashinfer-only paths early.
export SGLANG_IS_FLASHINFER_AVAILABLE=False

# Skip the sgl-kernel version check against the pinned SGLang version.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# Bind server processes to CPU cores for stable latency.
export SGLANG_SET_CPU_AFFINITY=1

# Route CUDA Graph-compatible APIs to XPU Graph capture and replay.
export XMLIR_FORCE_USE_XPU_GRAPH=1

# Enable the fast SwiGLU implementations in XFT and Kunlun MoE operators.
export XPU_USE_FAST_SWIGLU=1

# Use the XPU runtime's default device context.
export XPU_USE_DEFAULT_CTX=1

# Enable the XMLIR fast fully-connected implementation.
export XMLIR_ENABLE_FAST_FC=1

# Enable the XMLIR runtime's cuDNN-compatible path.
export XMLIR_CUDNN_ENABLED=1

# Enable the CUDA-Graph-optimized stream used by the runtime.
export CUDA_GRAPH_OPTIMIZE_STREAM=1