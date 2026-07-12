"""Deprecated module alias: use :mod:`msutils.flagstats` instead."""
import warnings

warnings.warn(
    "`msutils.flag_stats` has been renamed to `msutils.flagstats`; the old name "
    "is deprecated and will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

from .flagstats import *  # noqa: E402,F401,F403  (re-export the public API)
from .flagstats import __all__  # noqa: E402,F401
