"""Wave 2 quantization hooks (w8a8_int8, unquant, int8_kernel).

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/quantization/.
Importing the submodules registers their ``@plugin_hook`` targets in
``HookRegistry``; the actual install happens via
``HookRegistry.apply_hooks()``.
"""

from . import unquant  # noqa: F401
from . import w8a8_int8  # noqa: F401
