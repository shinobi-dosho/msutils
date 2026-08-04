"""msutils: Measurement Set manipulation utilities.

Everyday MS operations, exposed directly on the package::

    import msutils

    info = msutils.msinfo("obs.ms")            # structured metadata
    print(info.render())

    msutils.addcol("obs.ms", "MODEL_DATA", clone="DATA")
    msutils.copycol("obs.ms", "DATA", "CORRECTED_DATA")

Feature-specific modules require optional extras -- ``msutils[flagstats]``,
``msutils[plots]``, ``msutils[average]``, ``msutils[msv4]``.
"""
from importlib.metadata import PackageNotFoundError, version as _version

from ._ms import (
    STOKES_TYPES,
    addcol,
    addnoise,
    compute_vis_noise,
    copycol,
    delcol,
    renamecol,
    summary,
    sumcols,
    verify_antpos,
)
from .diagnostics import check, du, taql
from .flags import flag_backup, flag_delete, flag_restore, flag_versions
from .flagstats import flagstats
from .info import MSInfo, detect_format, msinfo
from .subset import average, subset

try:
    __version__ = _version("msutils")
except PackageNotFoundError:      # running from a source tree, not installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # metadata
    "msinfo",
    "MSInfo",
    "detect_format",
    "STOKES_TYPES",
    # columns
    "addcol",
    "delcol",
    "renamecol",
    "copycol",
    "sumcols",
    "addnoise",
    "compute_vis_noise",
    "verify_antpos",
    # datasets
    "subset",
    "average",
    # flags
    "flagstats",
    "flag_backup",
    "flag_restore",
    "flag_versions",
    "flag_delete",
    # diagnostics
    "du",
    "check",
    "taql",
    # deprecated
    "summary",
]
