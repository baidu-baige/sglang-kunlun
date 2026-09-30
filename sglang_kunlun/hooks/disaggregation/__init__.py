"""Disaggregation patches for Kunlun (P800).

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/disaggregation/__init__.py

This module used to round every kv-pool buffer length up to 2MiB before
``KVManager`` was built, on the assumption that Kunlun RDMA registration needs
2MiB-aligned lengths. That is wrong on kunlun_peermem: ``peermem_mmap()``
translates an XPU device address into a host mmap address, and the translation
is only valid while ``[ptr, ptr+len)`` stays inside a single allocation.
Rounding ``len`` up overruns the mmap tail and ``ibv_reg_mr`` fails with
EFAULT / EINVAL / ENOMEM. GLM KDA state tensors are slices of larger buffers,
so they hit this reliably.

The lengths reported by the kv pools are therefore passed through untouched and
no hook is installed here. Keep this module importable: ``hooks/registry.py``
lists it in ``HOOK_MODULES``.
"""

from __future__ import annotations
