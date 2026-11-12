"""Kunlun bootstrap hook loaded automatically from PYTHONPATH."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


def _enable_kunlun_attention_choice(module) -> None:
    choices = getattr(module, "ATTENTION_BACKEND_CHOICES", None)
    if choices is not None:
        for backend in ("kunlun", "kunlun_compressed"):
            if backend not in choices:
                choices.append(backend)

    try:
        annotations = module.ServerArgs.__annotations__
        raw_annotation = annotations["kv_cache_dtype"]
        # dsv4.1 把这个字段声明成 `Annotated[str, Arg(choices=[...])]`，老版本是字符串化
        # 的 `Literal[...]`。argparse 读的是前者的列表，所以往那里追加；`Arg` 可能是
        # frozen dataclass，就地改列表可以避免重新赋值属性。
        appended = False
        for meta in getattr(raw_annotation, "__metadata__", ()) or ():
            choices = getattr(meta, "choices", None)
            if isinstance(choices, list):
                # ``fp16``: Kunlun runs the model dtype as fp16. ``int8``: the
                # GLM-5.3 w4a8 deployment's KV cache (0.5.8 accepts it; the
                # 0.5.23 arg group does not list it).
                for extra in ("fp16", "int8"):
                    if extra not in choices:
                        choices.append(extra)
                appended = True
        if not appended and isinstance(raw_annotation, str):
            choices_start = raw_annotation.find("choices=[")
            choices_end = raw_annotation.find("]", choices_start)
            if (
                choices_start >= 0
                and choices_end > choices_start
                and "'int8'" not in raw_annotation[choices_start:choices_end]
            ):
                annotations["kv_cache_dtype"] = (
                    raw_annotation[:choices_end]
                    + ", 'fp16', 'int8'"
                    + raw_annotation[choices_end:]
                )
    except Exception:
        pass


def _enable_transformers_compatibility() -> None:
    import transformers
    from huggingface_hub import dataclasses as hub_dataclasses
    from huggingface_hub.errors import StrictDataclassDefinitionError
    from transformers import configuration_utils

    config_type = getattr(configuration_utils, "PretrainedConfig", None)
    if config_type is not None:
        if not hasattr(configuration_utils, "PreTrainedConfig"):
            configuration_utils.PreTrainedConfig = config_type
        if not hasattr(transformers, "PreTrainedConfig"):
            transformers.PreTrainedConfig = config_type
        if not hasattr(config_type, "__class_validators__"):
            # ``__class_validators__`` is installed by huggingface_hub's
            # ``@strict`` decorator, which transformers only applies to
            # ``PretrainedConfig`` from 5.4 on. 0.5.22's
            # ``hf_transformers_patches._patch_layer_types_validation`` reads it
            # unguarded, so on this transformers build ``import sglang`` dies
            # with "type object 'PreTrainedConfig' has no attribute
            # '__class_validators__'" before any Kunlun code runs.
            #
            # An empty list is the honest value here: without the strict
            # decoration there is no ``validate_layer_type`` validator to wrap,
            # so upstream's patch loop should find nothing and do nothing. It is
            # also inert for huggingface_hub, which only iterates this list from
            # the ``__post_init__`` that the skipped decoration would have
            # installed.
            config_type.__class_validators__ = []

    original_strict = hub_dataclasses.strict
    if getattr(original_strict, "_sglang_kunlun_compatible", False):
        return

    def compatible_strict(cls=None, **kwargs):
        def decorate(target):
            try:
                return original_strict(target, **kwargs)
            except StrictDataclassDefinitionError:
                # The decoration is skipped, so ``@strict`` never installs
                # ``__class_validators__``. 0.5.22's
                # ``hf_transformers_patches._patch_layer_types_validation``
                # reads that attribute unguarded (it is a transformers 5.4+
                # mechanism) and ``import sglang`` dies with
                # "type object 'PreTrainedConfig' has no attribute
                # '__class_validators__'". Install the empty list the skipped
                # decoration would have produced: with no validators there is
                # no ``validate_layer_type`` to wrap, which is the right answer
                # for this transformers build, and the patch loop becomes the
                # no-op it should be.
                if not hasattr(target, "__class_validators__"):
                    target.__class_validators__ = []
                return target

        return decorate(cls) if cls is not None else decorate

    compatible_strict._sglang_kunlun_compatible = True
    hub_dataclasses.strict = compatible_strict

    # Newer transformers ship configs whose names this sglang tree registers
    # itself (e.g. ``qwen3_asr``), and AutoConfig.register refuses duplicates.
    # Keep upstream's registration and drop the duplicate instead of failing.
    from transformers.models.auto.configuration_auto import AutoConfig

    original_register = AutoConfig.register
    if getattr(original_register, "_sglang_kunlun_compatible", False):
        return

    def compatible_register(model_type, config, exist_ok=False):
        try:
            return original_register(model_type, config, exist_ok=exist_ok)
        except ValueError:
            return config

    compatible_register._sglang_kunlun_compatible = True
    AutoConfig.register = compatible_register


class _ServerArgsLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        _enable_kunlun_attention_choice(module)


class _ServerArgsFinder(importlib.abc.MetaPathFinder):
    _target = "sglang.srt.server_args"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            if not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _ServerArgsLoader(spec.loader)
                return spec
        return None


def _quiet_plugin_loader() -> None:
    """Demote the upstream plugin loader's per-process banner.

    ``sglang.srt.plugins`` logs "Available plugins for group ...", "Loaded
    plugin ...", "OOT platform plugin activated ..." at INFO, for two groups, in
    each of the ~10 processes a PD role spawns -- 33 identical lines per boot.
    This has to run here rather than from a hook module: the banner is printed
    *while* loading our plugin, so by the time any hook executes it has already
    been emitted. Warnings and errors from that logger still come through.
    """
    import logging

    logging.getLogger("sglang.srt.plugins").setLevel(logging.WARNING)


def _load_sglang_plugins_early() -> None:
    """Run upstream's own plugin loader before the arg-resolution pipeline.

    0.5.23 applies OOT hooks when ``sglang.srt.plugins.load_plugins()`` runs,
    and upstream's own docstring says it "should be called early in every
    process (main, engine core, workers)" -- but nothing in the serving path
    calls it; the hooks only land as a side effect of whoever first resolves
    ``get_platform()``.

    That made hook application depend on flag order, and the PD pair diverged:
    the prefill node resolves the platform early (``--disable-cuda-graph`` ->
    ``parse_cuda_graph_config``), so its 224 hooks were applied before
    ``resolve_once()``; the decode node's first ``get_platform()`` call happens
    *inside* ``_dsa_kv_cache_dtype_default`` itself (``overrides.py:500``
    reads ``get_platform().is_npu``), so the hooks landed while that very frame
    was already executing the unhooked original -- 0 ``Applied ... hook`` lines
    in the decode log and ``AssertionError: DeepSeek DSA only supports
    bf16/bfloat16 or fp8_e4m3 kv_cache_dtype``, with the fix for that assert
    sitting right there in the tree, installed on nothing.

    Calling it here, from the shim that already runs in every process before
    argparse, makes hook application deterministic and order-independent.
    ``load_plugins()`` is idempotent (``_plugins_loaded`` guard), so the later
    implicit call is a no-op.
    """
    try:
        from sglang.srt.plugins import load_plugins
    except Exception:  # upstream too old, or sglang not importable yet
        return
    load_plugins()


_startup_args = [*getattr(sys, "orig_argv", ()), *sys.argv]
_is_package_tool = any(
    arg == "pip"
    or arg.endswith("/pip")
    or arg.endswith("/pip3")
    or arg.endswith("/pip3.10")
    for arg in _startup_args
)

if (
    not _is_package_tool
    and (
        os.environ.get("SGLANG_PLATFORM") == "kunlun"
        or os.environ.get("SGLANG_USE_XPU") == "1"
    )
):
    try:
        _enable_transformers_compatibility()
        os.environ.setdefault("SGLANG_OPT_USE_TOPK_V2", "0")

        from sglang_kunlun import _kunlun_pre_shim

        _kunlun_pre_shim()
        if "sglang.srt.server_args" in sys.modules:
            _enable_kunlun_attention_choice(sys.modules["sglang.srt.server_args"])
        elif not any(isinstance(finder, _ServerArgsFinder) for finder in sys.meta_path):
            sys.meta_path.insert(0, _ServerArgsFinder())
        _quiet_plugin_loader()
        _load_sglang_plugins_early()
    except Exception as exc:
        if os.environ.get("SGLANG_KUNLUN_SHIM_DEBUG"):
            print(f"[kunlun-shim] sitecustomize failed: {exc}", file=sys.stderr)
