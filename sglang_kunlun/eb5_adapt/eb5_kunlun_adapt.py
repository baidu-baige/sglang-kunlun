"""EB5-lite 算子在昆仑上的适配层 —— 参考 dpsk-v4 的「融合算子拆解」做法。

EB5 单测**直接调用**的这批 kernel 在昆仑上跑不了（tilelang / deep_gemm / fork 自带
nvcc-JIT / triton）。dpsk-v4 的做法是：把融合 kernel 换成**纯 torch 拆解实现**（或
KLX 原生算子）。本模块照此把 EB5 的融合 kernel 换成 torch 实现，语义与 kernel 源码
逐行对应，供 EB5 的 kernel 级单测和服务组网使用。

被替换的 kernel 与拆解：

* ``mhc_pre`` / ``mhc_post``
      整体换成与 DSV4 同构的散算子实现（`mhc_pre_torch_ref` / `mhc_post_torch_ref`）：
      GEMM → split-sinkhorn → combine → layernorm。这是**模型实际调用**的通路。
* ``mhc_post_tilelang`` / ``mhc_pre_big_fuse_with_norm_tilelang`` /
  ``mhc_pre_gemm_sqrsum_tilelang`` / ``mhc_pre_gemm_sqrsum_splitk_kernel``（stage_0 +
  stage_1）/ ``_mhc_pre_gemm_sqrsum_dispatch`` / ``tf32_hc_prenorm_gemm``
      **kernel 级**替代（不是只在 `mhc_pre` / `mhc_post` 调用层绕开）。昆仑 P800
      没有 Triton / tilelang / deep_gemm，所以这些 kernel 逐个用 torch 散算子重写，
      签名 / shape / dtype / 填充值与 kernel 源码逐行对应，由 `apply()` patch 进
      ``sglang.kernels.ops.layernorm.mhc`` 与 ``deep_gemm_wrapper.entrypoint``，
      因此任何**直接调用**这些 kernel 的代码（含 kernel 级单测）都走 torch 实现。
      为什么必须留在 kernel 级：2026-10-20 曾以"模型已不走融合路径"为由整体删除过
      这批替代实现，结果 `test_eb5_hc_pre/post.py` 的 12 个 kernel 级用例全灭 ——
      删掉的是**可测的 kernel 契约**，不是死代码。
"""

from __future__ import annotations

import os

import torch

_LOG2E = 1.4426950408889634


def _import_sibling(name: str):
    """导入同目录模块，兼容两种加载方式。

    * 作为包内模块加载（``sglang_kunlun.eb5_adapt.eb5_kunlun_adapt``）→ 相对导入；
    * ``EB5_KUNLUN_ADAPT_DIR`` 把本目录直接加进 ``sys.path`` → 平铺导入。
    """
    import importlib

    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ModuleNotFoundError as exc:
            # 只吞"相对路径找不到"，不要掩盖被导入模块自身的依赖缺失。
            if exc.name not in (f"{__package__}.{name}", name):
                raise
    return importlib.import_module(name)


def _mhc_trace(tag: str) -> None:
    """``EB5_MHC_TRACE=1`` 时在每个子步骤后同步并打点，用来把挂死定位到具体 op。

    只用于定位：同步会打断异步流水，正常跑不要开。
    """
    if os.environ.get("EB5_MHC_TRACE", "0") == "1":
        torch.cuda.synchronize()
        rank = -1
        try:
            import torch.distributed as _dist

            if _dist.is_initialized():
                rank = _dist.get_rank()
        except Exception:
            pass
        print(f"[eb5-mhc-trace][r{rank}] {tag}", flush=True)


def _dsv4_torch_shim_path() -> str:
    """定位同包 ``kernels/dsv4_torch_shim.py``（不依赖工作区落在哪个绝对路径）。

    优先按本文件位置解析（包内/平铺都成立）；解析不到再退回已安装的
    ``sglang_kunlun`` 包位置。
    """
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(os.path.dirname(here), "kernels", "dsv4_torch_shim.py")
    if os.path.exists(local):
        return local
    try:
        spec = importlib.util.find_spec("sglang_kunlun")
    except Exception:  # noqa: BLE001 - 解析失败就返回本地路径，由调用方报错
        spec = None
    if spec is not None and spec.origin:
        candidate = os.path.join(
            os.path.dirname(spec.origin), "kernels", "dsv4_torch_shim.py"
        )
        if os.path.exists(candidate):
            return candidate
    return local


# ═══════════════════════════════════════════════════════════════════════════ #
# 融合 kernel 的 **kernel 级** torch 替代
#
# 昆仑 P800 不支持 Triton / tilelang / deep_gemm，所以这里对
# `sglang/kernels/ops/layernorm/mhc.py` 与 `deep_gemm_wrapper.entrypoint` 里的
# 融合 kernel **逐个**给出 torch 实现，并由 `apply()` patch 到定义模块上。
# 不是在 `mhc_pre` / `mhc_post` 调用层绕开 —— 凡是直接调这些 kernel 的地方
# （kernel 级单测、prewarm、未来的组网代码）都会拿到 torch 版本。
#
# 三条硬约束（本段所有实现都遵守）：
#  1) 纯 torch，**没有 Python 逐元素循环**；只允许按 split / token 块 / hc 维循环
#     （循环次数是 10 量级，不是 token 量级）；
#  2) 大张量不整块 `.float()`：GEMM 按 token 分块、平方和沿 K 分块、combine 按 hc 维
#     逐项累加，临时块大小有上界（见 `_FP32_GEMM_TOKEN_CHUNK` / `_SQRSUM_K_CHUNK`）；
#  3) 签名 / 输出 shape / dtype / **填充值**与 kernel 一致：输出张量都是调用方传进来
#     的（in-place 写）；split-K 的尾部列（32 - hc_mult3）必须清零，因为 kernel 用
#     `T.clear(out_frag)` 覆盖写、`big_fuse` 又按 gemm_last_dim 列求和。
# ═══════════════════════════════════════════════════════════════════════════ #

#: 一次 `x[lo:hi].float() @ f.T` 的 token 块。512 行 x 16384 列 fp32 ≈ 33 MB。
_FP32_GEMM_TOKEN_CHUNK = 512
#: 平方和沿 K 维的分块。4096 列 x 2048 行 fp32 ≈ 33 MB。
_SQRSUM_K_CHUNK = 4096


def _matmul_bf16_by_fp32(x, f, token_chunk: int = _FP32_GEMM_TOKEN_CHUNK):
    """``x[n,k] (bf16) @ f[j,k].T (fp32) -> out[n,j] (fp32)``，按 token 分块。

    kernel 里是 ``T.gemm(x_frag /*fp32*/, fn_smem /*fp32*/, out_frag)``：x 先 bf16→fp32
    再乘，所以这里只把**当前块**转 fp32，绝不整块 `.float()`。
    """
    n = x.shape[0]
    j = f.shape[0]
    out = torch.empty(n, j, dtype=torch.float32, device=x.device)
    if n == 0:
        return out
    # 转置只做一次（j x k -> k x j，几 MB），避免每块重复转置。
    ft = f.t().contiguous()
    for lo in range(0, n, token_chunk):
        hi = min(lo + token_chunk, n)
        # `.contiguous()` 是刻意的：split 切片是非连续的（行 stride 是整行宽），
        # XPU 上非连续 matmul 会走另一条路径（见 01_eb_adp/problem 的 noc idle timeout）。
        xb = x[lo:hi].float().contiguous()
        out[lo:hi] = xb @ ft
    return out


def _sqrsum_bf16_rows(x, k_chunk: int = _SQRSUM_K_CHUNK):
    """``sqrsum[t] = Σ_k x[t,k]²``（fp32 累加），沿 K 分块，避免整块 `.float()`。

    对应 kernel 的 ``sqrsum_part[i,j] += x_f[..]*x_f[..]`` + `reduce_sum`。
    """
    n, k = x.shape
    acc = torch.zeros(n, dtype=torch.float32, device=x.device)
    for lo in range(0, k, k_chunk):
        hi = min(lo + k_chunk, k)
        xb = x[:, lo:hi].float()
        acc += (xb * xb).sum(-1)
    return acc


# --------------------------------------------------------------------------- #
# hc_post：mhc_post_tilelang
# --------------------------------------------------------------------------- #
def mhc_post_tilelang(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
):
    """``mhc.mhc_post_tilelang`` 的 torch 版（原地写 ``x``）。

    kernel（``mhc.py:1282``）逐行：

        x[n,hco,h] = c[n,hco]*d[n,h] + Σ_hci a[n,hci,hco]*b[n,hci,h]

    参数顺序 / 形状与 kernel 的 ``T.Tensor`` 声明一致：``a (n,hc,hc) fp32``、
    ``b (n,hc,h) bf16``、``c (n,hc) fp32``、``d (n,h) bf16``、``x (n,hc,h) bf16``。
    ``n_thr`` / ``h_blk`` 是 kernel 的 CTA 参数，torch 版无实际作用，只为签名兼容。
    """
    n = b.shape[0]
    if n == 0:
        return x
    for lo in range(0, n, _FP32_GEMM_TOKEN_CHUNK):
        hi = min(lo + _FP32_GEMM_TOKEN_CHUNK, n)
        # a 的 hci 维（dim=1）与 b 的 hci 维收缩，hco 落到输出的 dim=1。
        acc = torch.einsum("nhc,nhd->ncd", a[lo:hi], b[lo:hi].float())
        acc += c[lo:hi].unsqueeze(-1) * d[lo:hi].unsqueeze(1)
        x[lo:hi].copy_(acc.to(x.dtype))
    return x


# --------------------------------------------------------------------------- #
# hc_pre ①：deep_gemm.tf32_hc_prenorm_gemm
# --------------------------------------------------------------------------- #
def tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=None):
    """``deep_gemm.tf32_hc_prenorm_gemm`` 的 torch 版（split-K，原地写 out / sqrsum）。

    kernel 语义（TF32 prenorm GEMM + 同行平方和）：

        out[s, t, :] = x[t, 段_s] @ fn[:, 段_s].T
        sqrsum[s, t] = Σ_{k∈段_s} x[t, k]²

    ``num_splits`` 为 ``None`` 时按 ``out.shape[0]`` 取（调用方两种写法都有）。
    与 ``entrypoint.tf32_hc_prenorm_gemm`` 一致：**空输入直接返回**（不写任何输出）。
    """
    n, k = x.shape
    if n == 0:
        return None
    ns = int(out.shape[0]) if num_splits is None else int(num_splits)
    assert ns == out.shape[0], f"num_splits={ns} != out.shape[0]={out.shape[0]}"
    f = fn.float()
    j = min(fn.shape[0], out.shape[-1])
    # kernel 按 `T.ceildiv(k, ns)` 分段（最后一段可能不满）。
    split = -(-k // ns) if ns > 0 else k
    for s in range(ns):
        lo, hi = s * split, min((s + 1) * split, k)
        # 覆盖写：整行先清零，尾部列（> j）与空 split 都不能留 empty() 的垃圾值。
        out[s].zero_()
        sqrsum[s].zero_()
        if hi <= lo:
            continue
        xs = x[:, lo:hi]
        out[s][:, :j].copy_(_matmul_bf16_by_fp32(xs, f[:, lo:hi]))
        sqrsum[s].copy_(_sqrsum_bf16_rows(xs))
    return None


# --------------------------------------------------------------------------- #
# hc_pre ②：mhc_pre_gemm_sqrsum_tilelang / splitk / dispatch
# --------------------------------------------------------------------------- #
def mhc_pre_gemm_sqrsum_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
):
    """``mhc.mhc_pre_gemm_sqrsum_tilelang`` 的 torch 版（单链、无 split）。

        out[t, j] = Σ_k x[t,k]*fn[j,k]   (j < hc_mult3)
        sqrsum[t] = Σ_k x[t,k]²

    形状契约与 kernel 的 ``T.Tensor`` 声明一致：``out (n, hc_mult3) fp32``、
    ``sqrsum (n,) fp32``（2-D 调用约定，区别于 splitk 的 3-D partial）。
    """
    if not 0 < hc_mult3 <= 32:
        # kernel: assert hc_mult3 <= 32（0 会让下面 j=0，静默什么都不写）
        raise ValueError("hc_mult3 must be in [1, 32]")
    if hc_hidden_size % hidden_block:
        raise ValueError(
            f"hc_hidden_size={hc_hidden_size} not divisible by hidden_block={hidden_block}"
        )
    n = x.shape[0]
    j = min(int(hc_mult3), fn.shape[0], out.shape[1])
    if n == 0:
        return None
    out[:, :j].copy_(_matmul_bf16_by_fp32(x, fn[:j]))
    if j < out.shape[1]:          # 尾部列 kernel 不写，但也不能留垃圾
        out[:, j:].zero_()
    sqrsum.copy_(_sqrsum_bf16_rows(x))
    return None


def mhc_pre_gemm_sqrsum_splitk_kernel(
    hc_mult3: int,
    hc_hidden_size: int,
    split_k: int,
    token_block: int = 32,
    hidden_block: int = 256,
    threads: int = 128,
):
    """``mhc.mhc_pre_gemm_sqrsum_splitk_kernel`` 的 torch 版工厂。

    返回 ``(stage_0, stage_1)`` —— 与 fork 的返回元组个数/顺序一致（`mhc_pre` 里是
    ``kernel_0, _ = ...`` 只取第一个，stage_1 的 reduce 折进了 big_fuse）：

    * ``stage_0(x, fn, out_partial, sqrsum_partial)``：
      ``out_partial[s,t,:hc_mult3] = x[t, 段_s] @ fn[:hc_mult3, 段_s].T``、
      ``sqrsum_partial[s,t] = Σ_{k∈段_s} x[t,k]²``；``out_partial`` 宽 32，
      **尾部 32-hc_mult3 列必须清零**（kernel 只写前 hc_mult3 列，而调用方按 32 列
      分配并按 32 列求和）。`out_partial[:, :n, :].zero_()` 对应 kernel 的
      `T.clear(out_frag)`。
    * ``stage_1(out_partial, sqrsum_partial, out, sqrsum)``：沿 split 求和，把
      ``out (n, hc_mult3)`` / ``sqrsum (n,)`` 写满（kernel 里是 warp reduce）。

    K 分段与 kernel 一致：等长 ``split_size = hc_hidden_size // split_k``。kernel 用
    assert 保证整除，这里提前抛 ValueError，避免静默丢 K 尾部。
    """
    if not 0 < hc_mult3 <= 32:
        raise ValueError("hc_mult3 must be in [1, 32]")
    if hc_hidden_size <= 0 or split_k <= 0 or hc_hidden_size % split_k:
        raise ValueError(
            f"hc_hidden_size={hc_hidden_size} must be positive and divisible by "
            f"positive split_k={split_k}"
        )
    if hc_hidden_size % hidden_block:
        raise ValueError(
            f"hc_hidden_size={hc_hidden_size} not divisible by hidden_block={hidden_block}"
        )
    split_size = hc_hidden_size // split_k
    if split_size % hidden_block:
        raise ValueError(
            f"split_size={split_size} not divisible by hidden_block={hidden_block}"
        )

    def stage_0(x, fn, out_partial, sqrsum_partial):
        n = x.shape[0]
        # 尾部列清零：big_fuse 按 gemm_last_dim(=32) 列求和，单测也是整块对照。
        out_partial[:, :n, :].zero_()
        if n == 0:
            return None
        f = fn.float()
        j = min(int(hc_mult3), f.shape[0])
        for s in range(split_k):
            lo, hi = s * split_size, (s + 1) * split_size
            xs = x[:, lo:hi]
            out_partial[s, :n, :j].copy_(_matmul_bf16_by_fp32(xs, f[:j, lo:hi]))
            sqrsum_partial[s, :n].copy_(_sqrsum_bf16_rows(xs))
        return None

    def stage_1(out_partial, sqrsum_partial, out, sqrsum):
        n = out.shape[0]
        if n == 0:
            return None
        j = min(int(hc_mult3), out.shape[1])
        out[:, :j].copy_(out_partial[:split_k, :n, :j].sum(0))
        if j < out.shape[1]:
            out[:, j:].zero_()
        sqrsum.copy_(sqrsum_partial[:split_k, :n].sum(0))
        return None

    return stage_0, stage_1


def _mhc_pre_gemm_sqrsum_dispatch():
    """``mhc._mhc_pre_gemm_sqrsum_dispatch()`` 的 torch 版。

    原实现只在 SM120 上换 pass_configs，其余 arch 直接返回
    ``mhc_pre_gemm_sqrsum_tilelang``；昆仑两者都跑不了，统一返回 torch 版单链实现
    （`mhc_pre` 会直接调用它的返回值）。
    """
    return mhc_pre_gemm_sqrsum_tilelang


# --------------------------------------------------------------------------- #
# hc_pre ③：mhc_pre_big_fuse_with_norm_tilelang
# --------------------------------------------------------------------------- #
def _mhc_pre_big_fuse_torch(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    hc_mult: int,
    gemm_last_dim: int,
    mhc_align_fleet: int,
    norm_weight=None,
    norm_eps=None,
):
    """``mhc_pre_big_fuse_tilelang`` / ``..._with_norm_tilelang`` 的公共 torch 实现。

    两个 kernel 的差别只在最后一步：带 norm 版把 weighted-sum 先 round 到 bf16、用它
    算 sumsq，再乘 ``rsqrt(sumsq/hidden+norm_eps)*norm_weight``；不带 norm 版直接把
    fp32 weighted-sum round 到 bf16 写 ``layer_input``。这里用 ``norm_weight is None``
    区分，其余（rms / mixes / post_mix / sinkhorn）逐行相同。
    """
    n = residual.shape[0]
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    if n == 0:
        return None
    assert residual.dim() == 3, f"residual must be (n, hc, hidden); got {residual.shape}"

    # 1) rms：沿 split 求和（kernel 是 T.serial(n_splits) 累加，n_splits 只有几十）。
    sq = gemm_out_sqrsum[:n_splits, :n].float().sum(0)                 # (n,)
    denom = hc_mult * hidden_size
    if mhc_align_fleet:
        # PaddleFleet：eps 在 sqrt **外**（`fused_mhc_kernels.py` 的 `norm/sqrt(K)+eps`）。
        rms = 1.0 / (torch.sqrt(sq / denom) + rms_eps)
    else:
        rms = torch.rsqrt(sq / denom + rms_eps)

    # 2) mixes：先按 split 求和，再乘 rms。gemm_out_mul 是 (n_splits, n, gemm_last_dim)。
    mixes = gemm_out_mul[:n_splits, :n, :gemm_last_dim].float().sum(0)
    mixes = mixes * rms.unsqueeze(-1)                                  # (n, hc_mult3)

    # 3) post_mix
    post_mix.copy_(
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )

    # 4) comb_mix：sinkhorn。dim=2 是 k（kernel 的 row），dim=1 是 j（kernel 的 col）。
    cm = (
        mixes[:, 2 * hc_mult : 2 * hc_mult + hc_mult * hc_mult] * hc_scale[2]
        + hc_base[2 * hc_mult : 2 * hc_mult + hc_mult * hc_mult]
    ).reshape(n, hc_mult, hc_mult)
    cm = cm - cm.amax(dim=2, keepdim=True)
    cm = torch.exp2(cm * _LOG2E) if mhc_align_fleet else torch.exp(cm)
    cm = cm / cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(max(int(sinkhorn_repeat) - 1, 0)):
        cm = cm / (cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    comb_mix.copy_(cm.reshape(n, hc_mult * hc_mult))

    # 5) layer_input = Σ_hc pre_mix*residual（[可选] RMSNorm）
    pre_mix = (
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
        + hc_pre_eps
    )                                                                  # (n, hc)
    ol = torch.zeros(n, hidden_size, dtype=torch.float32, device=residual.device)
    for i_hc in range(hc_mult):        # hc 维（4 次），不是 token / 元素维
        ol += pre_mix[:, i_hc : i_hc + 1] * residual[:, i_hc, :]
    # 与 kernel 一致：先 round 到 bf16（lizhenyun 那条注释要求与 torch 实现对齐）。
    ol_bf = ol.to(residual.dtype)
    if norm_weight is None:
        # 不带 norm 的 kernel：`T.copy(ol, layer_input[...])`，ol 就是 fp32 加权和。
        layer_input.copy_(ol_bf)
        return None
    # 带 norm：用 **round 之后** 的值算 sumsq，再乘 rsqrt 与 bf16 权重。
    sumsq = _sqrsum_bf16_rows(ol_bf)                                   # 沿 hidden 分块
    rsqrt_norm = torch.rsqrt(sumsq / hidden_size + norm_eps)
    ol_scaled = ol_bf.float() * rsqrt_norm.unsqueeze(-1)
    ol_scaled *= norm_weight.float()
    layer_input.copy_(ol_scaled.to(layer_input.dtype))
    return None


def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    mhc_align_fleet: int = 0,
):
    """``mhc.mhc_pre_big_fuse_tilelang`` 的 torch 版（**不带** norm 的 big fuse）。

    `mhc_pre` 在 ``norm_weight is None`` 时走这条；昆仑没有 tilelang，同样要能跑。
    """
    return _mhc_pre_big_fuse_torch(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        gemm_last_dim,
        mhc_align_fleet,
    )


def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    mhc_align_fleet: int = 0,
):
    """``mhc.mhc_pre_big_fuse_with_norm_tilelang`` 的 torch 版（原地写三个输出）。

    逐段对应 kernel（``mhc.py:858``）：

    1. ``rms = rsqrt|1/(sqrt+eps) ( Σ_splits sqrsum[i_split,t] / (hc_mult*hidden) )``
    2. ``mixes[j] = (Σ_splits gemm_out_mul[i_split,t,j]) * rms``（只取前 gemm_last_dim 列）
    3. ``post_mix = sigmoid(mixes[hc:2hc]*scale[1]+base[hc:2hc]) * post_mult``
    4. ``comb``：row max → ``exp``/``exp2`` → 行归一(+eps) → 列归一(+eps) →
       ``sinkhorn_repeat-1`` 轮（行/列交替，每步 +eps）
    5. ``layer_input = rmsnorm(Σ_hc pre_mix*residual)``：fp32 逐 hc 累加 →
       **先 round 到 bf16** → 用 round 后的值算 sumsq → ``* rsqrt(sumsq/hidden+norm_eps)
       * norm_weight``。
    """
    return _mhc_pre_big_fuse_torch(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        gemm_last_dim,
        mhc_align_fleet,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def mhc_post_torch_ref(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
):
    """``sglang.kernels.ops.layernorm.mhc.mhc_post`` 的**散装 torch 实现**。

    与 DSV4 在昆仑上跑的 ``hc_post_torch_impl``（`deepseek_v4.py:2739`）逐行同式：

        out[n,hco,h] = post[n,hco] * x[n,h] + Σ_hci comb[n,hci,hco] * residual[n,hci,h]

    ``post_layer_mix`` 允许 ``(n, hc, 1)``（EB5 的调用形状）或 ``(n, hc)``。
    由 `apply()` 无条件 patch，与 `mhc_pre_torch_ref` 配套，完全绕开融合 kernel。
    """
    if post_layer_mix.dim() == residual.dim():   # (n, hc, 1)
        post = post_layer_mix
    else:                                        # (n, hc)
        post = post_layer_mix.unsqueeze(-1)
    # 与 DSV4 的 hc_post_torch_impl 完全一致。
    out = post * x.unsqueeze(1) + (
        comb_res_mix.unsqueeze(-1) * residual.unsqueeze(2)
    ).sum(dim=1)
    _mhc_trace("post:out")
    return out.type_as(x)


def _rmsnorm_torch(x, weight, eps):
    """``input_layernorm`` 的显式写法：fp32 累加 + fp32 权重 + 输出回原 dtype。

    与 ``EB5LayerRMSNorm.forward``（底层 ``F.rms_norm``）和 mHC 融合 kernel 的
    norm 段一致。
    """
    xf = x.float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * rsqrt * weight.float()).to(x.dtype)


def _hc_split_sinkhorn_torch(
    mixes,
    hc_scale,
    hc_base,
    hc_mult,
    sinkhorn_iters,
    eps,
    mhc_align_fleet=False,
    post_mult_value=2.0,
):
    """``hc_split_sinkhorn`` 的 torch 版（DSV4 在昆仑上由插件 hook 到同语义实现）。

    与 `mhc.py::hc_split_sinkhorn_kernel` / `_hc_split_sinkhorn_torch` 的 sinkhorn
    段逐行对应：第一轮 ``comb/row_sum + eps``，之后每轮 row/col 都是 ``/(sum + eps)``。
    """
    flat = mixes.float()
    hc = int(hc_mult)

    pre = torch.sigmoid(flat[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = post_mult_value * torch.sigmoid(
        flat[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = (flat[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).reshape(-1, hc, hc)
    _mhc_trace("sinkhorn:mixes")

    row_max = comb.amax(dim=2, keepdim=True)
    if mhc_align_fleet:
        comb = torch.exp2((comb - row_max) * _LOG2E)
    else:
        comb = torch.exp(comb - row_max)
    _mhc_trace("sinkhorn:softmax")
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    _mhc_trace("sinkhorn:norm0-dim2")
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    _mhc_trace("sinkhorn:norm0-dim1")
    for _it in range(max(int(sinkhorn_iters) - 1, 0)):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        _mhc_trace(f"sinkhorn:iter{_it}-dim2")
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
        _mhc_trace(f"sinkhorn:iter{_it}-dim1")

    return pre, post, comb


def mhc_pre_torch_ref(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight=None,
    norm_eps=None,
    mhc_align_fleet: bool = False,
    mhc_align_accum_order: bool = False,
):
    """``sglang.kernels.ops.layernorm.mhc.mhc_pre`` 的**散装 torch 实现**。

    刻意照抄 DSV4 在昆仑上已经跑通的那条路（`deepseek_v4.py::hc_pre` 的 torch 分支），
    四步一一对应，不调用任何融合 kernel：

        1. hc_pre_torch_impl : mixes = F.linear(x_flat, fn) * rsqrt(mean(x²)+eps)
        2. hc_split_sinkhorn: pre / post / comb
        3. hc_combine       : y = Σ_hc pre * residual
        4. input_layernorm  : y = rmsnorm(y, norm_weight, norm_eps)

    DSV4 第 4 步在模型层单独调，EB5 的 ``mhc_pre`` 契约是把 norm 折进来，所以这里
    只是把同一公式放在同一函数里（与融合 kernel 的 norm 段一致：先 round 到 bf16，
    再 fp32 平方 / 乘 fp32 权重）。

    与融合实现的差别只有 ``mhc_align_fleet`` 的 eps 位置（PaddleFleet 权重要求
    ``1/(sqrt(mean)+eps)``，DSV4 是 ``rsqrt(mean+eps)``），其余完全同 DSV4。

    返回形状与 ``mhc.mhc_pre`` 一致：``(*outer, hc, 1)`` / ``(*outer, hc, hc)`` /
    ``(*outer, hidden)``。
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=residual.dtype, device=residual.device
    )
    if num_tokens == 0:
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    assert fn.shape[0] == hc_mult3, (
        f"fn.shape[0]={fn.shape[0]} != hc_mult3={hc_mult3}"
    )
    assert fn.shape[1] == hc_mult * hidden_size, (
        f"fn.shape[1]={fn.shape[1]} != hc_mult*hidden_size={hc_mult * hidden_size}"
    )

    # 1) 一次连续 GEMM：与 DSV4 `hc_pre_torch_impl` 同式。
    x_flat = residual_flat.reshape(num_tokens, hc_mult * hidden_size).float()
    _mhc_trace("pre:flatten")
    if mhc_align_fleet:
        rms = 1.0 / (
            torch.sqrt(x_flat.square().mean(-1, keepdim=True)) + rms_eps
        )
    else:
        rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(x_flat, fn.float()) * rms
    _mhc_trace("pre:linear")

    # 2) split-sinkhorn：与 DSV4 `hc_split_sinkhorn` 同式。
    pre, post, comb = _hc_split_sinkhorn_torch(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_repeat,
        hc_sinkhorn_eps,
        bool(mhc_align_fleet),
        hc_post_mult_value,
    )
    _mhc_trace("pre:sinkhorn-done")
    post_mix.copy_(post)
    comb_mix.copy_(comb.reshape(num_tokens, hc_mult * hc_mult))

    # 3) combine：与 DSV4 `hc_combine` 同式（fp32 累加后回 bf16）。
    y = (pre.unsqueeze(-1) * x_flat.view(num_tokens, hc_mult, hidden_size)).sum(dim=1)
    y = y.to(residual.dtype)
    _mhc_trace("pre:combine")

    # 4) 模型层 RMSNorm：DSV4 在 `hc_pre` 之后单独调 input_layernorm。
    if norm_weight is None:
        layer_input.copy_(y)
    else:
        assert norm_eps is not None, "norm_eps required when norm_weight is provided"
        layer_input.copy_(_rmsnorm_torch(y, norm_weight, norm_eps))
    _mhc_trace("pre:norm")

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


# --------------------------------------------------------------------------- #
# 融合 rope（按 fork kernel 源码 + 单测参考实现，而非复用插件版本）
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 融合 rope（按 fork kernel 源码 + 单测参考实现，而非复用插件版本）
# --------------------------------------------------------------------------- #
def _rope_interleaved(x_rope: torch.Tensor, freqs_real: torch.Tensor, positions: torch.Tensor):
    """对 x 的最后 rope_dim 维做交错 RoPE。x_rope: (tokens, heads, rope_dim)。"""
    rope_dim = x_rope.shape[-1]
    f = freqs_real[positions.long()].float()                  # (tokens, rope_dim)
    f = f.reshape(*f.shape[:-1], rope_dim // 2, 2).unsqueeze(-3)   # (tokens,1,rope_dim//2,2)
    rp = x_rope.reshape(*x_rope.shape[:-1], rope_dim // 2, 2)
    re = rp[..., 0] * f[..., 0] - rp[..., 1] * f[..., 1]
    im = rp[..., 0] * f[..., 1] + rp[..., 1] * f[..., 0]
    return torch.stack([re, im], -1).reshape(*x_rope.shape)


def fused_q_norm_rope(q_input, q_output, eps, freqs_cis, positions):
    """RMSNorm(no weight) + 末尾 rope_dim 的交错 RoPE。"""
    x = q_input.float()
    normed = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    nope = q_input.shape[-1] - rope_dim
    out = normed.clone()
    if rope_dim > 0:
        out[..., nope:] = _rope_interleaved(normed[..., nope:], freqs_real, positions)
    q_output.copy_(out.to(q_output.dtype))
    return None


# --------------------------------------------------------------------------- #
# 注入
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# wo_a 低秩投影：替代 torch.einsum 的通用收缩
# --------------------------------------------------------------------------- #
def _apply_wo_a_bf16_matmul_torch(o, wo_a, is_decode=False):
    """``[T,G,D] x [G,R,D] -> [T,G,R]``：group 作 batch 的 bmm + 按 token 分块。

    fork 的兜底是 ``torch.einsum("tgd,grd->tgr")``。通用收缩在昆仑上会把
    4-D 中间量物化（T≈1200、G=8、R=1536、D=512 时约 15 GiB），整网 43 层
    首 forward 因此 OOM（2026-10-20 23:23 实测）。这里改成显式 bmm + 分块，
    峰值只有 ``chunk*G*(D+R)*2`` 字节（chunk=1024 时约 30 MB），数值等价。

    ``is_decode`` 仅为保持签名兼容（昆仑上 aiter batched_gemm 不可用）。
    """
    assert o.dim() == 3 and wo_a.dim() == 3, (o.shape, wo_a.shape)
    T, G, D = o.shape
    assert wo_a.shape[0] == G and wo_a.shape[-1] == D, (o.shape, wo_a.shape)
    R = wo_a.shape[1]
    if T == 0:
        return o.new_empty((0, G, R))
    w = wo_a.transpose(1, 2)
    if not w.is_contiguous():
        w = w.contiguous()                                   # [G,D,R]
    out = torch.empty((T, G, R), dtype=o.dtype, device=o.device)
    chunk = 1024
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        x = o[lo:hi].permute(1, 0, 2).contiguous()           # [G,c,D]
        y = torch.bmm(x, w)                                  # [G,c,R]
        out[lo:hi] = y.permute(1, 0, 2)
    return out


def apply() -> list[str]:
    """把上面的 torch 拆解实现 patch 到 fork 的调用点上。返回已 patch 的符号列表。"""
    import importlib

    patched = []

    def _patch(module_name, attr, impl):
        # 用 MLA 模块里的通用注入器：它会把**改名/多处 import 的旧绑定**一并修掉
        # （fork 的 `sglang.srt.models.deepseek_v4` 等模块在 import 期就绑死了这些
        # 融合算子，只改定义模块改不到它们）。
        try:
            eb5_mla_adapt = _import_sibling("eb5_mla_adapt")

            patched.extend(eb5_mla_adapt.patch_everywhere(module_name, attr, impl))
        except Exception:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                return
            if hasattr(mod, attr):
                setattr(mod, attr, impl)
                patched.append(f"{module_name}.{attr}")

    ew0 = "sglang.kernels.ops.attention.dsv4.elementwise"
    _patch(ew0, "fused_q_norm_rope", fused_q_norm_rope)

    mhc_mod = "sglang.kernels.ops.layernorm.mhc"
    # ① 调用层：EB5 的模型前向统一走与 DSV4 同构的散算子实现（融合分支在真实多 rank
    #    前向会挂死，不再保留回退）。`ernie5.py` 在 import 期就
    #    `from ...mhc import mhc_pre as _mhc_pre` / `mhc_post as _mhc_post`，
    #    `patch_everywhere` 会按对象身份把这类旧绑定一并改掉。
    _patch(mhc_mod, "mhc_pre", mhc_pre_torch_ref)
    _patch(mhc_mod, "mhc_post", mhc_post_torch_ref)
    # ② kernel 级：昆仑没有 tilelang / deep_gemm，这些融合 kernel 由 torch 散算子
    #    逐个替代（**不能**只在 ① 的调用层绕开 —— 直调 kernel 的代码同样要能跑）。
    _patch(mhc_mod, "mhc_post_tilelang", mhc_post_tilelang)
    _patch(mhc_mod, "mhc_pre_big_fuse_tilelang", mhc_pre_big_fuse_tilelang)
    _patch(mhc_mod, "mhc_pre_big_fuse_with_norm_tilelang", mhc_pre_big_fuse_with_norm_tilelang)
    _patch(mhc_mod, "mhc_pre_gemm_sqrsum_tilelang", mhc_pre_gemm_sqrsum_tilelang)
    _patch(mhc_mod, "mhc_pre_gemm_sqrsum_splitk_kernel", mhc_pre_gemm_sqrsum_splitk_kernel)
    _patch(mhc_mod, "_mhc_pre_gemm_sqrsum_dispatch", _mhc_pre_gemm_sqrsum_dispatch)
    # deep_gemm 的 tf32 prenorm GEMM（`mhc_pre` 的 `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`
    # 分支、以及 kernel 级单测都直接 import 它）。
    # ③ wo_a 低秩投影：fork 兜底的 torch.einsum 在昆仑上会物化 4-D 中间量
    #    （整网 43 层首 forward 峰值 ~15 GiB -> OOM）。改成 bmm + 分块。
    _patch(
        "sglang.srt.models.deepseek_v4",
        "_apply_wo_a_bf16_matmul",
        _apply_wo_a_bf16_matmul_torch,
    )

    dg_mod = "sglang.srt.layers.deep_gemm_wrapper.entrypoint"
    _patch(dg_mod, "tf32_hc_prenorm_gemm", tf32_hc_prenorm_gemm)

    # --- 融合 rope 家族：复用插件 dsv4_torch_shim 的 torch 拆解（签名一致） ---
    # 注意：插件包 `sglang_kunlun.kernels` 的 __init__ 会拉起只对 sglang 0.5.14 成立的
    # import 链（`sglang.srt.layers.quantization.fp8_kernel` 在 fork 里已改名），
    # 所以这里**按文件直接加载** shim，并给它唯一缺的那个符号打桩。
    try:
        import importlib.util
        import sys
        import types

        _stub_name = "sglang_kunlun.hooks.layers.attention.nsa.triton_kernel"
        if _stub_name not in sys.modules:
            # DSV4 的 indexer 量化路径（fused_q_indexer_rope_hadamard_quant ->
            # dsv4_torch_shim.act_quant_kunlun）会**真的**调它，所以要给能用的实现
            # （与插件 nsa/triton_kernel.act_quant_kunlun 同实现，都是 kunlun_ops.quant2d）。
            # 不要改成 import 插件那个模块：会拉起 sglang_kunlun.hooks.layers.attention
            # 的 nsa redirect，提前 import fork 的 attention 模块，使本模块 _TARGETS 的补丁
            # 被已绑定的旧符号绕过（EB5 单测 19 passed -> 4 failed）。
            def _act_quant_kunlun(x, block_size=128, scale_fmt=None):
                import kunlun_ops

                assert x.is_contiguous(), "Input tensor must be contiguous"
                assert x.size(-1) % block_size == 0
                assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                x_shape = x.shape
                x_blocks = x.view(-1, block_size).contiguous()
                y_int8 = torch.empty(x_blocks.shape, dtype=torch.int8, device=x.device)
                y_max = torch.empty(x_blocks.shape[0], dtype=torch.float32, device=x.device)
                kunlun_ops.quant2d(x_blocks, y_int8, y_max, force_sdnn=True)
                return y_int8.view(x_shape), y_max.view(x_shape[0], -1)

            _m = types.ModuleType(_stub_name)
            _m.act_quant_kunlun = _act_quant_kunlun
            sys.modules[_stub_name] = _m

        _shim_path = _dsv4_torch_shim_path()
        _spec = importlib.util.spec_from_file_location("_eb5_dsv4_torch_shim", _shim_path)
        _shim = importlib.util.module_from_spec(_spec)
        sys.modules["_eb5_dsv4_torch_shim"] = _shim
        _spec.loader.exec_module(_shim)

        ew = "sglang.kernels.ops.attention.dsv4.elementwise"
        for attr, impl_name in (
            ("fused_k_norm_rope_flashmla", "_fused_k_norm_rope_flashmla_torch"),
            ("fused_rope_inplace", "_fused_rope_inplace_torch"),
            ("fused_q_indexer_rope_hadamard_quant", "_fused_q_indexer_rope_hadamard_quant_torch"),
        ):
            impl = getattr(_shim, impl_name, None)
            if impl is not None:
                _patch(ew, attr, impl)
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] rope shim load failed:", _e, file=_sys.stderr)

    # --- attn ② MLA 专用拆解（另见本包的 eb5_mla_adapt.py） ---
    try:
        eb5_mla_adapt = _import_sibling("eb5_mla_adapt")

        patched.extend(eb5_mla_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] mla adapt failed:", _e, file=_sys.stderr)

    # --- HCA / MoE / 前后处理拆解（另见本包的 eb5_extra_adapt.py） ---
    try:
        eb5_extra_adapt = _import_sibling("eb5_extra_adapt")

        patched.extend(eb5_extra_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] extra adapt failed:", _e, file=_sys.stderr)

    # --- 在线 INT8 量化方法注册（另见本包的 eb5_int8_adapt.py） ---
    # 必须发生在 `prepare_server_args()` 之前：`--quantization` 的 argparse
    # choices 在参数解析时才构建。插件入口点早于参数解析，满足该时序。
    try:
        eb5_int8_adapt = _import_sibling("eb5_int8_adapt")

        patched.extend(eb5_int8_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] int8 adapt failed:", _e, file=_sys.stderr)

    # --- DSA transform_index 的 torch 拆解（另见本包的 eb5_transform_index_adapt.py） ---
    try:
        eb5_transform_index_adapt = _import_sibling("eb5_transform_index_adapt")

        patched.extend(eb5_transform_index_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] transform_index adapt failed:", _e, file=_sys.stderr)

    # --- xy fork 专属适配（ERNIE 5.5 算子 + DP 空转守卫，另见 eb5_xy_adapt.py） ---
    # 上游仓库没有这个文件，缺失时静默跳过，保证本模块可以随上游同步。
    try:
        eb5_xy_adapt = _import_sibling("eb5_xy_adapt")

        patched.extend(eb5_xy_adapt.apply())
    except ImportError:
        pass
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] xy adapt failed:", _e, file=_sys.stderr)

    return patched
