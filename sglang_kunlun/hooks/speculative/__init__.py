"""speculative attention"""
import importlib
import sglang.srt.speculative as upstream_speculative

# HookRegistry resolves dotted targets through parent-module attributes.
upstream_speculative.eagle_worker_v2 = importlib.import_module(
    "sglang.srt.speculative.eagle_worker_v2"
)

from . import draft_utils  # noqa: E402,F401
from . import eagle_worker_v2  # noqa: E402,F401
from . import multi_layer_eagle_worker_v2  # noqa: E402,F401
