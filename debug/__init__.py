"""Precision-alignment debug tooling for sglang-kunlun.

This package holds every probe, tensor dump and diagnostic hook used by the
DSV4 Kunlun precision-alignment investigation. It is deliberately kept
*outside* ``sglang_kunlun`` so that the production package contains no
diagnostics.

Nothing in this package may change numerical behaviour. The invocation contract
is unchanged from before the extraction: the same ``DSV4_*`` / ``TENSOR_DUMP_*``
environment variables, the same dump file names and the same payload keys.
"""
