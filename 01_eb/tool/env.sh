#!/usr/bin/env bash
# Kunlun P800 推理环境激活脚本  ——  source 这个文件，不要直接执行。
#
#   source code/xy/sglang-kunlun/01_eb/tool/env.sh
#
# 内容分四段：
#   1) venv 与动态库搜索路径
#   2) 设备可见性
#   3) 昆仑运行时开关（来自 sglang-kunlun/setup_env.sh）
#   4) 集合通信 / DeepEP 所需 KLX 变量（BKCL_* / XSHMEM_*）
#
# 环境构成见 01_eb/2. 构建环境.md。
#
# 🔴 起服务 / 跑压测前必读：**不要 `kill -9` 卡在 XPU 调用里的进程**（CPU 100% 空转、
# GPU 0% util、显存不涨）。强杀会在设备上留下杀不掉的 kernel，拖坏整个 XPU runtime，
# 并让驱动泄漏 "in use" 引用计数 —— 之后 `xpu-smi -r` 对全部 8 张卡都报
# "currently in use by another process"（即使没有进程持有 /dev/xpu*），只能节点级重启。
# 详见 01_eb/3. eb5模型运行.md 的「最重要的一条」。

# 工作区根目录：从脚本位置向上找含 env/bin/python 的目录（脚本在
# code/xy/sglang-kunlun/01_eb/tool/ 下，离根有好几层），不依赖当前目录。
_BINGOO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ "$_BINGOO_DIR" != / ] && [ ! -x "$_BINGOO_DIR/env/bin/python" ]; do
  _BINGOO_DIR=$(dirname "$_BINGOO_DIR")
done
BINGOO=${BINGOO:-$_BINGOO_DIR}
unset _BINGOO_DIR

# ---------------------------------------------------------------- 1) venv
export VIRTUAL_ENV="$BINGOO/env"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME

# torch_plugin 的导入钩子会 ctypes.CDLL("libpython3.so") 探活；本机基础 CPython 3.10
# 是静态编译（无 libpython3.10.so），klx_stub/ 里放的是空实现占位库，详见 tool/libpython3.c
# 的注释（用 tool/build_libpython3_stub.sh 生成）。同时 torch_xmlir 的扩展 .so 直接
# DT_NEEDED libpython3.10.so.1.0，也由这里提供。
export LD_LIBRARY_PATH="$VIRTUAL_ENV/klx_stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ------------------------------------------------------------ 2) 设备可见性
# 用真实的 XPU event 而不是 dummy event 做同步。
unset XPU_DUMMY_EVENT
export XPU_VISIBLE_DEVICES=${XPU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$XPU_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=OAM_ID

# ------------------------------------------------- 3) 昆仑运行时开关
# 不要设 SGLANG_PLATFORM / SGLANG_USE_XPU：标准 launcher 通过 entry points
# 发现 kunlun 平台，提前设会让 sitecustomize 过早 bootstrap XPU。
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_FAST_SWIGLU=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_ENABLE_FAST_FC=1
export XMLIR_CUDNN_ENABLED=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

# ------------------------------------------- 4) 集合通信 / DeepEP（KLX）
# DeepEP 的 bkcl_xshmem_init 依赖这些；缺 XSHMEM_MODE 时 low-latency 的对称缓冲区
# 会落到主机内存，报 "pointer resides on host memory"。
export BKCL_RDMA_VERBS=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4}
export BKCL_ENABLE_XDR=1
export NCCL_IB_GID_INDEX=3
export CUDA_ENABLE_P2P_NO_UVA=1
export XSHMEM_MODE=1
export XSHMEM_SYMMETRIC_SIZE=4294967296

export PYTHONUNBUFFERED=1
