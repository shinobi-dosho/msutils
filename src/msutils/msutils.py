"""Deprecated module alias.

The core functions used to live in ``msutils.msutils``; they are now available
directly on the top-level ``msutils`` package (e.g. ``msutils.summary``). This
module re-exports them for backwards compatibility and will be removed in a
future release.
"""
import warnings

from ._ms import *  # noqa: F401,F403  (re-export the public API)
from ._ms import STOKES_TYPES, __all__  # noqa: F401

warnings.warn(
    "`msutils.msutils` is deprecated; the functions are now available directly "
    "on the `msutils` package (e.g. `msutils.summary`). This alias will be "
    "removed in a future release.",
    FutureWarning,
    stacklevel=2,
)
