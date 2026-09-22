"""msutils: Measurement Set manipulation utilities.

Everyday MS operations, exposed directly on the package::

    import msutils

    info = msutils.msinfo("obs.ms")  # structured metadata
    print(info.render())

    msutils.addcol("obs.ms", "MODEL_DATA", clone="DATA")
    msutils.copycol("obs.ms", "DATA", "CORRECTED_DATA")

Feature-specific modules require optional extras -- ``msutils[flagstats]``,
``msutils[plots]``, ``msutils[average]``, ``msutils[msv4]``.
"""

from . import gains
from ._ms import (
    STOKES_TYPES,
    addcol,
    addnoise,
    compute_vis_noise,
    copycol,
    delcol,
    renamecol,
    sumcols,
    summary,
    verify_antpos,
)
from ._package import __version__
from .convert import to_msv2, to_msv4
from .diagnostics import check, du, taql
from .flags import flag_backup, flag_delete, flag_restore, flag_versions
from .flagstats import flagstats
from .info import MSInfo, detect_format, msinfo
from .subset import average, subset

__all__ = [
    "STOKES_TYPES",
    "MSInfo",
    "__version__",
    # columns
    "addcol",
    "addnoise",
    "average",
    "check",
    "compute_vis_noise",
    "copycol",
    "delcol",
    "detect_format",
    # diagnostics
    "du",
    "flag_backup",
    "flag_delete",
    "flag_restore",
    "flag_versions",
    # flags
    "flagstats",
    # gain-table operations (also the `gainutils` console script)
    "gains",
    # metadata
    "msinfo",
    "renamecol",
    # datasets
    "subset",
    "sumcols",
    # deprecated
    "summary",
    "taql",
    "to_msv2",
    "to_msv4",
    "verify_antpos",
]
