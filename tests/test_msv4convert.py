"""MSv4 -> MSv2 validation that does not need the optional xarray stack."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from msutils import _msv4convert
from msutils._tables import open_table


class _DatasetStub:
    def __init__(self, *, groups, dtype=np.complex64):
        self.attrs = {"data_groups": groups}
        self._arrays = {
            "VISIBILITY": SimpleNamespace(dtype=np.dtype(dtype)),
            "UVW": SimpleNamespace(),
        }

    def __contains__(self, name):
        return name in self._arrays

    def __getitem__(self, name):
        return self._arrays[name]


class _NodeStub:
    def __init__(self, dataset):
        self.ds = dataset
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("groups", "dtype", "message"),
    [
        (
            {
                "base": {"correlated_data": "VISIBILITY", "uvw": "UVW"},
                "imaging": {"correlated_data": "VISIBILITY_CORRECTED", "uvw": "UVW"},
            },
            np.complex64,
            "data groups other than 'base'",
        ),
        (
            {"base": {"correlated_data": "VISIBILITY", "uvw": "UVW"}},
            np.complex128,
            "cannot be represented losslessly",
        ),
    ],
)
def test_unrepresentable_content_is_rejected_before_overwrite(
    tmp_path, monkeypatch, groups, dtype, message
):
    """Preflight failures must leave a pre-existing destination untouched."""
    destination = tmp_path / "existing.ms"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("original")

    node = _NodeStub(_DatasetStub(groups=groups, dtype=dtype))
    monkeypatch.setitem(sys.modules, "xarray", ModuleType("xarray"))
    monkeypatch.setattr(_msv4convert._msv4, "_open", lambda *_args: ([node], ["partition"], "zarr"))

    with pytest.raises(ValueError, match=message):
        _msv4convert.to_msv2("input.zarr", str(destination), overwrite=True)

    assert sentinel.read_text() == "original"
    assert node.closed


def test_write_fields_preserves_direction_reference(tmp_path):
    path = str(tmp_path / "direction.ms")
    _msv4convert._create_ms(path, weight_spectrum=False)
    part = SimpleNamespace(
        field_names=("target",),
        field_direction=(1.2, -0.4),
        field_frame="GALACTIC",
    )

    _msv4convert._write_fields(path, [part], {"target": 0})

    with open_table(path + "::FIELD") as tab:
        for column in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
            assert tab.getcolkeywords(column)["MEASINFO"]["Ref"] == "GALACTIC"
        np.testing.assert_allclose(tab.getcell("PHASE_DIR", 0), [[1.2, -0.4]])


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("fk5", "J2000"),
        ("icrs", "ICRS"),
        ("fk4noterms", "B1950_VLA"),
        ("galactic", "GALACTIC"),
        ("altaz", "AZELGEO"),
    ],
)
def test_direction_frame_translation(source, expected):
    assert _msv4convert._direction_frame(source) == expected


def test_mixed_direction_frames_are_rejected():
    parts = [SimpleNamespace(field_frame="J2000"), SimpleNamespace(field_frame="ICRS")]
    with pytest.raises(ValueError, match="different FIELD direction frames"):
        _msv4convert._field_frame(parts)
