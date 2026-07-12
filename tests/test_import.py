"""Smoke tests: the package and its pure-Python entry points import cleanly.

Most of msutils operates on real casacore Measurement Sets, so these tests only
assert importability and public API presence, not MS behaviour.
"""
import sys
import warnings


def test_import_msutils():
    from msutils import msutils

    assert hasattr(msutils, "summary")
    assert hasattr(msutils, "addcol")
    assert msutils.STOKES_TYPES[1] == "I"


def test_msutils_alias_is_deprecated():
    # The warning only fires on first import, so drop any cached shim module.
    sys.modules.pop("MSUtils", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from MSUtils import msutils as aliased

    from msutils import msutils as real

    assert aliased.summary is real.summary
    assert any(issubclass(w.category, FutureWarning) for w in caught)
