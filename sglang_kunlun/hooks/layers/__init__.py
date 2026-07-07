"""Kunlun layer hooks.

Importing this package registers all layer-level ``@plugin_hook`` targets.
"""

from . import linear  # noqa: F401
from . import quantization  # noqa: F401
from . import moe  # noqa: F401
from . import rotary_embedding  # noqa: F401,E402

# Must be imported before upstream resolves attention backend modules.
from . import attention  # noqa: F401,E402
