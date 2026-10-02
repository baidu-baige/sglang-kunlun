"""Kunlun-side protection for Kimi-K3 packed INT4 weight names.

Upstream ``KimiK3LinearForCausalLM.load_weights`` rewrites every checkpoint
name containing ``weight_packed`` before resolving the parameter
(``srt/models/kimi_k3.py:3085-3086``). That was intended for the serialized
MXFP4 path, but the Kunlun checkpoint uses compressed-tensors W4A16 parameters
whose actual model parameters deliberately keep the ``_packed`` suffix. This
hook preserves those names while delegating all mapping, TP/EP sharding,
truncation filtering, and post-load work to upstream SGLang.

The shard-file trimming in this module is a **startup optimisation only**.
Upstream already discards out-of-range layers correctly:

* PP boundary filter          ``srt/models/kimi_k3.py:3070-3074``
* truncated-config filter     ``srt/models/kimi_k3.py:3075-3081``
* expert ``params_dict`` gate ``srt/models/kimi_k3.py:3151-3154``
* ``post_load_weights`` guard ``srt/models/kimi_k3.py:3201-3202``

So whenever the local layer count cannot be determined this module degrades to
"no trimming" rather than raising.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


logger = logging.getLogger(__name__)


# Only text-model transformer layers are reduced. Vision tower namespaces may
# contain their own layer-like names and must remain fully loadable.
_LAYER_PATTERN = re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)")
# Upstream ``EntryClass`` lists both architectures (``srt/models/kimi_k3.py:3707``);
# the text-only one can be loaded directly since 0.5.19.
_KIMI_K3_ARCHITECTURES = {
    "KimiK3ForConditionalGeneration",
    "KimiK3LinearForCausalLM",
}
# The per-tensor load audit walks every source name, so it is only worth
# enabling for the reduced-layer smoke configurations it was written for
# (``--json-model-override-args '{"text_config":{"num_hidden_layers":N}}'``).
# The full model has 896 experts x 69 layers of source tensors.
_AUDIT_MAX_LAYER_COUNT = 8


@dataclass
class _WeightLoadAudit:
    source_tensors: int = 0
    source_elements: int = 0
    source_bytes: int = 0
    source_names: set[str] = field(default_factory=set)
    duplicate_sources: set[str] = field(default_factory=set)
    callback_targets: set[str] = field(default_factory=set)
    callback_events: int = 0
    callback_failures: int = 0
    direct_copy_name_matches: set[str] = field(default_factory=set)

    def observe_source(self, name: str, tensor: Any) -> None:
        """Record source tensor metadata observed during weight iteration."""
        if name in self.source_names:
            self.duplicate_sources.add(name)
        self.source_names.add(name)
        self.source_tensors += 1
        numel = int(tensor.numel())
        self.source_elements += numel
        self.source_bytes += numel * int(tensor.element_size())
        if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
            self.direct_copy_name_matches.add(name)

    def observe_callback(self, target: str) -> None:
        """Record a parameter weight-loader callback target."""
        self.callback_events += 1
        self.callback_targets.add(target)


class _PackedWeightName(str):
    """A checkpoint name that bypasses K3's MXFP4 rename guard.

    K3 tests ``"weight_packed" in name`` before calling ``replace``
    (``srt/models/kimi_k3.py:3085-3086``, still present in 0.5.19). Returning
    false for that exact membership test keeps the original string value and
    therefore lets the normal expert mapping see ``w13_weight_packed`` and
    ``w2_weight_packed``. All other string operations retain normal semantics.

    Do not "simplify" this away: the rewrite is unconditional upstream and
    every packed compressed-tensors expert weight is silently dropped without
    it.
    """

    def __contains__(self, item: object) -> bool:
        if item == "weight_packed":
            return False
        return super().__contains__(item)


def _config_layer_count(config: Any) -> int | None:
    """Return a usable ``num_hidden_layers`` from a model/text config."""
    count = getattr(config, "num_hidden_layers", None)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        return None
    return count


def _source_layer_count(source: Any) -> int | None:
    """Read the instantiated text-model layer count from a loader source.

    ``--json-model-override-args '{"text_config":{"num_hidden_layers":N}}'``
    lands on ``hf_config.text_config`` (``utils/hf_transformers/config.py``
    updates the sub-config in place), which is also what
    ``KimiK3LinearForCausalLM`` is constructed from
    (``srt/models/kimi_k3.py:3357``). Fall back to the top-level config for the
    text-only architecture, and to ``None`` -- "do not trim" -- when neither
    attribute is present.
    """
    hf_config = getattr(getattr(source, "model_config", None), "hf_config", None)
    if hf_config is None:
        return None
    return _config_layer_count(
        getattr(hf_config, "text_config", None)
    ) or _config_layer_count(hf_config)


def _partial_layer_name(name: str, layer_count: int | None) -> str | None:
    """Drop checkpoint text-model layers at or beyond the instantiated count."""
    if layer_count is None:
        return name
    match = _LAYER_PATTERN.search(name)
    if match is None:
        return name
    if int(match.group(1)) >= layer_count:
        return None
    return name


def _protect_packed_weight_names(
    model: Any,
    weights: Iterable,
    audit: _WeightLoadAudit | None = None,
    layer_count: int | None = None,
) -> Iterator:
    """Yield selected checkpoint weights with partial-layer and packed-name guards."""
    for item in weights:
        if len(item) < 2:
            yield item
            continue
        name = item[0]
        if isinstance(name, str):
            name = _partial_layer_name(name, layer_count)
            if name is None:
                continue
            # Wrap every packed name: upstream kimi_k3.load_weights rewrites
            # any checkpoint name containing "weight_packed" to "weight" (MXFP4
            # convention), which breaks compressed-tensors expert params whose
            # fused parameter names keep the "_packed" suffix. Checkpoint names
            # (e.g. ...experts.42.w1.weight_packed) never equal parameter names
            # directly, so a `name in parameter_names` guard would never fire
            # and all packed expert weights would be silently dropped.
            if "weight_packed" in name:
                name = _PackedWeightName(name)
        if audit is not None and isinstance(name, str):
            audit.observe_source(str(name), item[1])
        if len(item) == 2:
            yield (name, item[1])
        else:
            yield (name, *item[1:])


def _install_weight_loader_audit(model: Any, audit: _WeightLoadAudit):
    originals = []
    for name, parameter in model.named_parameters():
        had_loader = hasattr(parameter, "weight_loader")
        original_loader = getattr(parameter, "weight_loader", None)
        if original_loader is None:
            continue

        def audited_loader(
            param,
            loaded_weight,
            *args,
            _name=name,
            _original=original_loader,
            **kwargs,
        ):
            """Record and delegate one parameter weight-loader callback."""
            audit.observe_callback(_name)
            try:
                return _original(param, loaded_weight, *args, **kwargs)
            except BaseException:
                audit.callback_failures += 1
                raise

        parameter.weight_loader = audited_loader
        originals.append((parameter, had_loader, original_loader))
    return originals


def _restore_weight_loaders(originals) -> None:
    for parameter, had_loader, original_loader in originals:
        if had_loader:
            parameter.weight_loader = original_loader
        else:
            delattr(parameter, "weight_loader")


def _log_weight_load_audit(audit: _WeightLoadAudit) -> None:
    logger.info(
        "Kimi-K3 weight audit: source=%d tensors/%d elements/%d bytes, "
        "loader_callbacks=%d across %d targets, direct_copy_name_matches=%d, "
        "duplicate_sources=%d, callback_failures=%d",
        audit.source_tensors,
        audit.source_elements,
        audit.source_bytes,
        audit.callback_events,
        len(audit.callback_targets),
        len(audit.direct_copy_name_matches),
        len(audit.duplicate_sources),
        audit.callback_failures,
    )


def _partial_shard_layer_count(source: Any) -> int | None:
    """Layer count to trim shard files to, or ``None`` to load every shard."""
    hf_config = getattr(getattr(source, "model_config", None), "hf_config", None)
    architectures = set(getattr(hf_config, "architectures", None) or ())
    if not architectures.intersection(_KIMI_K3_ARCHITECTURES):
        return None
    if not os.path.isdir(getattr(source, "model_or_path", "") or ""):
        return None
    return _source_layer_count(source)


def _select_partial_shards(
    hf_folder: str, weight_files: list[str], layer_count: int
) -> list[str]:
    """Select common tensors and checkpoint layers ``0..layer_count-1``."""
    index_path = os.path.join(hf_folder, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return weight_files
    try:
        with open(index_path, encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        # Purely an optimisation: upstream still filters out-of-range layers.
        logger.debug("Kimi-K3 shard trimming skipped (%s): %s", index_path, exc)
        return weight_files
    selected = set()
    for name, shard in weight_map.items():
        match = _LAYER_PATTERN.search(name)
        if match is None or int(match.group(1)) < layer_count:
            selected.add(os.path.join(hf_folder, shard))
    filtered = [path for path in weight_files if path in selected]
    return filtered or weight_files


@plugin_hook(
    "sglang.srt.model_loader.loader.DefaultModelLoader._get_weights_iterator",
    type=HookType.AROUND,
)
def get_partial_kimi_weights_iterator(original_fn, self, source, **kwargs):
    """Trim the mmap'd shard set to the instantiated Kimi-K3 layers.

    ``**kwargs`` forwards 0.5.19's keyword-only additions
    (``srt/model_loader/loader.py:549-556``:
    ``resolved_source`` / ``startup_prefetch_started`` /
    ``startup_prefetch_active``). Forwarding them is mandatory -- the hook
    would otherwise ``TypeError`` on the first weight load.

    Args:
        original_fn: Upstream weight-iterator method.
        self: The upstream model loader instance.
        source: Model source configuration passed to the loader.
        **kwargs: Upstream keyword-only arguments, forwarded verbatim.

    Returns:
        A generator yielding the upstream weight name/value tuples.
    """
    layer_count = _partial_shard_layer_count(source)
    if layer_count is None:
        yield from original_fn(self, source, **kwargs)
        return

    if kwargs.get("resolved_source") is not None:
        # Upstream skips ``self._prepare_weights`` when the shard list was
        # already resolved (``loader.py:571-574``), so the substitution below
        # would be a no-op. Startup trimming is an optimisation only; the
        # upstream truncation filters still make the load correct.
        logger.debug(
            "Kimi-K3 shard trimming skipped: weight files already resolved by "
            "the startup prefetch path"
        )
        yield from original_fn(self, source, **kwargs)
        return

    original_prepare = self._prepare_weights

    def prepare_partial(model_name_or_path, revision, fall_back_to_pt):
        """Prepare upstream weights and retain only the local layer shards."""
        hf_folder, weight_files, use_safetensors = original_prepare(
            model_name_or_path, revision, fall_back_to_pt
        )
        if use_safetensors:
            weight_files = _select_partial_shards(
                hf_folder, weight_files, layer_count
            )
        return hf_folder, weight_files, use_safetensors

    self._prepare_weights = prepare_partial
    try:
        yield from original_fn(self, source, **kwargs)
    finally:
        self._prepare_weights = original_prepare


@plugin_hook(
    "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.load_weights",
    type=HookType.AROUND,
)
def load_weights_kunlun_w4a16(original_fn, self, weights):
    """Preserve packed names and audit reduced-layer K3 loading metadata.

    Args:
        original_fn: Upstream weight-loading method.
        self: The Kimi-K3 model instance receiving weights.
        weights: Iterable of checkpoint name/value tuples.

    Returns:
        The result returned by the upstream weight-loading method.

    The audit confirms source iterator consumption and custom parameter-loader
    callbacks. MLA fused direct copies are reported separately as predictions;
    no tensor values or parameter references are retained after this call.
    """
    layer_count = _config_layer_count(getattr(self, "config", None))
    audit = None
    originals = []
    if layer_count is not None and layer_count <= _AUDIT_MAX_LAYER_COUNT:
        audit = _WeightLoadAudit()
        originals = _install_weight_loader_audit(self, audit)
    try:
        return original_fn(
            self,
            _protect_packed_weight_names(
                self, weights, audit=audit, layer_count=layer_count
            ),
        )
    finally:
        _restore_weight_loaders(originals)
        if audit is not None:
            _log_weight_load_audit(audit)
            self.kunlun_weight_load_audit = {
                "source_tensors": audit.source_tensors,
                "source_elements": audit.source_elements,
                "source_bytes": audit.source_bytes,
                "loader_callback_events": audit.callback_events,
                "loader_callback_targets": len(audit.callback_targets),
                "direct_copy_name_matches": len(audit.direct_copy_name_matches),
                "duplicate_sources": sorted(audit.duplicate_sources),
                "callback_failures": audit.callback_failures,
            }
