"""Smoke tests: the public API imports cleanly and is complete.

These assert importability and API surface only -- behaviour lives in the
MS-backed test modules.
"""
PUBLIC_API = [
    "msinfo", "MSInfo", "detect_format",
    "summary", "addcol", "sumcols", "copycol",
    "compute_vis_noise", "verify_antpos", "addnoise", "STOKES_TYPES",
]


def test_flat_api():
    import msutils

    for name in PUBLIC_API:
        assert hasattr(msutils, name), name
    assert msutils.STOKES_TYPES[1] == "I"


def test_public_api_is_exported():
    import msutils

    for name in msutils.__all__:
        assert hasattr(msutils, name), name


def test_removed_1x_aliases_are_gone():
    """The MSUtils / msutils.msutils / flag_stats / ClassESW shims were
    dropped in 3.0 after one deprecation release."""
    import importlib

    import pytest

    for name in ("MSUtils", "msutils.msutils", "msutils.flag_stats",
                 "msutils.ClassESW"):
        with pytest.raises(ImportError):
            importlib.import_module(name)


def test_importing_msutils_does_not_pull_in_optional_stacks():
    """The base install must not need matplotlib, dask, africanus or xradio.

    Run in a subprocess so the check sees a clean module table -- by the time
    the rest of the suite has run, plenty of optional stacks are imported.
    """
    import os
    import subprocess
    import sys

    code = (
        "import sys, msutils;"
        "heavy = [m for m in ('matplotlib', 'dask', 'daskms', 'africanus',"
        " 'xradio', 'scipy', 'xarray') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    # pytest's `pythonpath` setting applies to this process only, so hand the
    # subprocess the same search path.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, check=True, env=env)
    assert result.stdout.strip() == "", result.stdout


def test_function_and_module_names_that_collide():
    """`flagstats` and `subset` are both a module and a re-exported function.

    The package root binds the function, which shadows the submodule as an
    attribute. Dotted imports still resolve through sys.modules, and both
    forms must keep working -- a CLI command once broke because
    `from . import flagstats` returned the function.
    """
    import msutils
    import msutils.flagstats
    import msutils.subset
    from msutils.flagstats import FlagStats, flagstats
    from msutils.subset import average, subset

    assert callable(msutils.flagstats) and msutils.flagstats is flagstats
    assert callable(msutils.subset) and msutils.subset is subset
    assert FlagStats and average
