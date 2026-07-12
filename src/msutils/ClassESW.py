"""Deprecated module alias: use :mod:`msutils.weights` instead."""
import warnings

warnings.warn(
    "`msutils.ClassESW` has been renamed to `msutils.weights`; the old name is "
    "deprecated and will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

from .weights import *  # noqa: E402,F401,F403  (re-export the public API)
from .weights import __all__  # noqa: E402,F401
