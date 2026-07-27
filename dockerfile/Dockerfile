ARG BASE_IMAGE=iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.42
FROM ${BASE_IMAGE}

ARG XPYTORCH_DOWNLOAD_ADDR="https://klx-sdk-release-public.su.bcebos.com/xpytorch/release/3.6.2.1/xpytorch-cp310-torch290-ubuntu2004-x64.run"
ARG KUNLUN_OPS_DOWNLOAD_ADDR="https://klx-sdk-release-public.su.bcebos.com/kunlunops/20260701/kunlun_ops-0.0.1-cp310-cp310-linux_x86_64.whl"
ARG XSPEEDGATE_OPS_DOWNLOAD_ADDR="http://aihc-private-hcd.bj.bcebos.com/xspeedgate_release/torch29/20260721_165007/xspeedgate_ops-1.3.0+ccfaaa4.torch29-cp310-cp310-linux_x86_64.whl"

RUN rm -rf /workspace/*
RUN mkdir -p /workspace/

COPY sglang-kunlun /workspace/sglang_kunlun

RUN bash /workspace/sglang_kunlun/dockerfile/install.sh \
    --XPYTORCH_DOWNLOAD_ADDR="${XPYTORCH_DOWNLOAD_ADDR}" \
    --KUNLUN_OPS_DOWNLOAD_ADDR="${KUNLUN_OPS_DOWNLOAD_ADDR}" \
    --XSPEEDGATE_OPS_DOWNLOAD_ADDR="${XSPEEDGATE_OPS_DOWNLOAD_ADDR}"

WORKDIR /workspace

ENV LD_LIBRARY_PATH=/usr/local/xre/so:/usr/local/xccl/so:/usr/local/lib/:${LD_LIBRARY_PATH}
