"""Smoke tests: the flattened public API imports cleanly and the deprecated
aliases still work (while warning).

Most of msutils operates on real casacore Measurement Sets, so these tests only
assert importability and public API presence, not MS behaviour.
"""
import sys
import warnings

PUBLIC_API = [
    "summary", "addcol", "sumcols", "copycol",
    "compute_vis_noise", "verify_antpos", "addnoise", "STOKES_TYPES",
]


def test_flat_api():
    import msutils

    for name in PUBLIC_API:
        assert hasattr(msutils, name), name
    assert msutils.STOKES_TYPES[1] == "I"


def test_msutils_submodule_alias_is_deprecated():
    # `msutils.msutils` still resolves to the same functions, but warns.
    sys.modules.pop("msutils.msutils", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from msutils import msutils as legacy

    import msutils as pkg
    assert legacy.summary is pkg.summary
    assert any(issubclass(w.category, FutureWarning) for w in caught)


def test_MSUtils_package_alias_is_deprecated():
    # The capitalised `MSUtils` package aliases the real one, but warns.
    sys.modules.pop("MSUtils", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import MSUtils

    import msutils as pkg
    assert MSUtils is pkg
    assert any(issubclass(w.category, FutureWarning) for w in caught)
