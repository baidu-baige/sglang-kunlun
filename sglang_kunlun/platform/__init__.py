"""Hardware platform entry point for the Kunlun SGLang plugin."""

from sglang_kunlun import _kunlun_pre_shim

_kunlun_pre_shim()

from .device import KunlunDeviceMixin
from .srt import KunlunSRTPlatform, activate

__all__ = ["KunlunDeviceMixin", "KunlunSRTPlatform", "activate"]
