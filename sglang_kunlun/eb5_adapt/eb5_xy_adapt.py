"""xy fork 专属的 EB 适配（从 c11d9c8 / d86dfb7 迁入）。

与另外几个 ``eb5_*_adapt.py`` 同级、扁平存放：

* ERNIE 5.5 Flash 的模型级算子（原先注册在 ``kernels/kernel_ops.py``）
* DP 空转 / 短 prompt 的 zero-length kernel 守卫（原先是两个 ``@plugin_hook``）

``eb5_kunlun_adapt.apply()`` 不直接引用本模块；由
``sglang_kunlun.hooks.eb5_kunlun_patch`` 在通用适配之后调用 ``apply()``，
这样本文件可以独立增删，不影响上游同步。
"""

from __future__ import annotations

import torch

try:  # 包内导入
    from .eb5_extra_adapt import _patch_everywhere
except ImportError:  # EB5_KUNLUN_ADAPT_DIR 平铺目录
    from eb5_extra_adapt import _patch_everywhere

# --------------------------------------------------------------------------- #
# xy fork：ERNIE 5.5 Flash 模型级算子（原先散在 kernels/kernel_ops.py）
# --------------------------------------------------------------------------- #
def sigmoid_gate_mul_torch(x, gate, *, round_sigmoid: bool = False):
    """``x * sigmoid(gate)``，可选先对 sigmoid 结果做 dtype 舍入。"""
    s = torch.sigmoid(gate.to(torch.float32))
    if round_sigmoid:
        s = s.to(gate.dtype).to(torch.float32)
    return (x.to(torch.float32) * s).to(x.dtype)


def sigmoid_gate_mul_broadcast_torch(x, gate, *, round_sigmoid: bool = False):
    """与 :func:`sigmoid_gate_mul_torch` 同语义的 broadcast 变体。"""
    s = torch.sigmoid(gate.to(torch.float32))
    if round_sigmoid:
        s = s.to(gate.dtype).to(torch.float32)
    return (x.to(torch.float32) * s).to(x.dtype)


def multimax_torch(x, ranges, ts):
    """ERNIE 5.5 Flash multimax：中间量按 bf16 舍入（与 kernel 逐位一致）。"""

    def rb(v):
        return v.to(torch.bfloat16).float()

    x32 = x.to(torch.float32)
    r0, r1, r2, r3 = [ranges[i].float() for i in range(4)]
    t0, t1, t2, t3 = [ts[i].float() for i in range(4)]
    lower = rb(torch.clamp(r0 - x32, min=0))
    upper = rb(torch.clamp(x32 - r1, min=0))
    lower_sq = rb(torch.clamp(r2 - x32, min=0))
    upper_sq = rb(torch.clamp(x32 - r3, min=0))
    out = rb(x32 + rb(t0 * lower))
    out = rb(out + rb(t1 * upper))
    out = rb(out + rb(t2 * rb(lower_sq * lower_sq)))
    out = rb(out + rb(t3 * rb(upper_sq * upper_sq)))
    return out.to(torch.bfloat16)


def _install_ernie5_ops() -> list:
    """把 ERNIE 5.5 的模型级算子显式 patch 到 fork 的 JIT 辅助函数上。

    这些实现原先由 ``kernels/kernel_ops.py`` 的 ``register_jit_op`` 注册、
    再由 ``kernel_ops.install()`` 统一落地。迁到适配层后不再依赖插件 hook 的
    装载顺序，EB5 插件一加载就生效。
    """
    patched = []
    for module_name, attr, impl in (
        (
            "sglang.kernels.ops.moe.triton_sigmoid_gate_mul",
            "sigmoid_gate_mul",
            sigmoid_gate_mul_torch,
        ),
        (
            "sglang.kernels.ops.moe.triton_sigmoid_gate_mul",
            "sigmoid_gate_mul_broadcast",
            sigmoid_gate_mul_broadcast_torch,
        ),
        ("sglang.kernels.ops.activation.multimax", "multimax", multimax_torch),
    ):
        try:
            patched.extend(_patch_everywhere(module_name, attr, impl))
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-extra-adapt] {module_name}.{attr} failed: {exc}")
    return patched


# --------------------------------------------------------------------------- #
# xy fork：DP 空转 / 短 prompt 的 zero-length kernel 守卫
# （原先是 hooks/layers 下的两个 @plugin_hook）
# --------------------------------------------------------------------------- #
def _make_embed_zero_guard(original):
    def _guard(self, input_):
        if input_.numel() == 0:
            return self.weight.new_empty(
                tuple(input_.shape) + (self.weight.shape[1],)
            )
        return original(self, input_)

    return _guard


def _make_compress_zero_guard(original):
    def _guard(self, compress_ratio, page_table, c_page_size):
        if compress_ratio in self.compressed:
            return original(self, compress_ratio, page_table, c_page_size)

        seq_lens_cpu = self.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = self.seq_lens.tolist()
        if any(int(length) // compress_ratio for length in seq_lens_cpu):
            return original(self, compress_ratio, page_table, c_page_size)

        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            CompressedGather,
        )

        device = self.seq_lens.device
        zeros = torch.zeros(self.num_reqs, dtype=torch.int32, device=device)
        gather = CompressedGather(
            flat_token_ids=torch.empty(0, dtype=torch.int32, device=device),
            compressed_base=zeros,
            swa_base=0 + self.swa_offsets[:-1],
        )
        self.compressed[compress_ratio] = gather
        return gather

    return _guard


def _install_dp_idle_guards() -> list:
    """XPU 内核拒绝 zero-element launch，给两处空 batch 路径加短路。"""
    import importlib
    import sys

    patched = []
    targets = (
        (
            "sglang.srt.layers.vocab_parallel_embedding",
            "VocabParallelEmbedding",
            "_eb5_embed_zero_guard",
            _make_embed_zero_guard,
        ),
        (
            "sglang.srt.layers.attention.dsv4.sparse_prefill_utils",
            "SparsePrefillChunkCache",
            "_eb5_compress_zero_guard",
            _make_compress_zero_guard,
        ),
    )
    for module_path, cls_name, flag, factory in targets:
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-extra-adapt] import {module_path} failed: {exc}")
            continue
        # compat_aliases 可能留下同一文件的多个模块副本，逐个按类对象打。
        for mod_name, module in list(sys.modules.items()):
            if module is None or not mod_name.startswith("sglang."):
                continue
            cls = getattr(module, cls_name, None)
            if not isinstance(cls, type) or getattr(cls, flag, False):
                continue
            original = getattr(cls, "_embed_local_shard", None) if cls_name == (
                "VocabParallelEmbedding"
            ) else getattr(cls, "ensure_compressed", None)
            if original is None:
                continue
            target_attr = (
                "_embed_local_shard"
                if cls_name == "VocabParallelEmbedding"
                else "ensure_compressed"
            )
            setattr(cls, target_attr, factory(original))
            setattr(cls, flag, True)
            patched.append(f"{mod_name}.{cls_name}.{target_attr}")

    return patched


def apply() -> list:
    """应用 xy fork 专属适配，返回已 patch 的符号列表。"""
    patched = []
    for hook in (_install_ernie5_ops, _install_dp_idle_guards):
        try:
            patched.extend(hook())
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-xy-adapt] {hook.__name__} failed: {exc}")
    return patched
