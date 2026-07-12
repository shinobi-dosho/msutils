"""Deprecated import alias for :mod:`msutils`.

``MSUtils`` was renamed to the lowercase ``msutils``. This shim keeps
``import MSUtils`` / ``from MSUtils import msutils`` working while emitting a
:class:`FutureWarning` (shown by default, unlike ``DeprecationWarning``), then
aliases itself to the real package so that attribute and submodule access
transparently resolve to ``msutils``.
"""
import sys
import warnings

warnings.warn(
    "The 'MSUtils' package has been renamed to 'msutils'. Importing 'MSUtils' "
    "is deprecated and will be removed in a future release; use 'msutils' "
    "(e.g. 'from msutils import msutils') instead.",
    FutureWarning,
    stacklevel=2,
)

import msutils as _msutils

# Replace this shim module with the real package so that any subsequent
# attribute or submodule access (e.g. ``from MSUtils import msutils``) resolves
# against ``msutils`` rather than this empty stub.
sys.modules[__name__] = _msutils
