pip install -e /${YOUR_PATH}/sglang-kunlun

unset XPU_DUMMY_EVENT
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_FAST_SWIGLU=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_ENABLE_FAST_FC=1
export XMLIR_CUDNN_ENABLED=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1

export SGLANG_PLATFORM=kunlun

SGLANG_ENABLE_SPEC_V2=1 
sglang serve \
    --model-path /home/models/MiMo-V2-Flash-W8A8-INT8-Dynamic-official \
    --speculative-algorithm EAGLE \
    --quantization w8a8_int8 \
    --max-total-tokens 131072 \
    --disable-radix-cache \
    --decode-log-interval 1 \
    --host 0.0.0.0 \
    --port 8806 \
    --trust-remote-code \
    --tp-size 8 \
    --max-running-requests 64 \
    --disable-overlap-schedule \
    --attention-backend kunlun \
    --disable-cuda-graph \
    --mem-fraction-static 0.85
