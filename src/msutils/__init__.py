"""msutils: CASA Measurement Set manipulation utilities.

The core column operations are exposed directly on the package, e.g.::

    import msutils
    msutils.summary(msname)
    msutils.addcol(msname, "MODEL_DATA")

Feature-specific modules (``flagstats``, ``weights``) require optional
extras -- ``pip install msutils[flagstats]`` / ``msutils[plots]``.
"""
from ._ms import (
    STOKES_TYPES,
    summary,
    addcol,
    sumcols,
    copycol,
    compute_vis_noise,
    verify_antpos,
    addnoise,
)
from .info import MSInfo, detect_format, msinfo

__all__ = [
    "STOKES_TYPES",
    "msinfo",
    "MSInfo",
    "detect_format",
    "summary",
    "addcol",
    "sumcols",
    "copycol",
    "compute_vis_noise",
    "verify_antpos",
    "addnoise",
]


def __getattr__(name):
    # Lazily expose the deprecated `msutils.msutils` alias (emits a warning on
    # import) without pulling it in on every package import. Import via
    # importlib rather than `from . import msutils` to avoid recursing back
    # through this __getattr__.
    if name == "msutils":
        import importlib
        return importlib.import_module(__name__ + ".msutils")
    raise AttributeError("module {0!r} has no attribute {1!r}".format(__name__, name))
