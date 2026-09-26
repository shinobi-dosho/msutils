"""Executable Phase 0 probes for an arcae-native MSv2 reconstruction writer.

These are capability tests, not the writer implementation.  Ordinary installs
skip the module; CI's all-extras job installs the pinned reconstruction extra
and executes every probe.  Strict xfails express required invariants that the
first reconstruction profile must refuse until the underlying gap is closed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pytest

arrow_tables = pytest.importorskip(
    "arcae.lib.arrow_tables", reason="arcae not installed (msutils[reconstruction])"
)
Table = arrow_tables.Table
ms_descriptor = arrow_tables.ms_descriptor


PINNED_ARCAE = "0.5.4"
DEFAULT_SUBTABLES = {
    "ANTENNA",
    "DATA_DESCRIPTION",
    "FEED",
    "FIELD",
    "FLAG_CMD",
    "HISTORY",
    "OBSERVATION",
    "POINTING",
    "POLARIZATION",
    "PROCESSOR",
    "SPECTRAL_WINDOW",
    "STATE",
}


def _fixed_column(value_type: str, shape: tuple[int, ...], group: str) -> dict:
    return {
        "_c_order": True,
        "comment": "msutils arcae reconstruction probe",
        "dataManagerGroup": group,
        "dataManagerType": "TiledColumnStMan",
        "keywords": {},
        "maxlen": 0,
        "ndim": len(shape),
        "option": 4,
        "shape": list(shape),
        "valueType": value_type,
    }


def _scalar_column(value_type: str) -> dict:
    return {
        "comment": "msutils arcae reconstruction probe",
        "dataManagerGroup": "StandardStMan",
        "dataManagerType": "StandardStMan",
        "keywords": {},
        "maxlen": 0,
        "option": 0,
        "valueType": value_type,
    }


def _probe_descriptor() -> tuple[dict, dict]:
    descriptor = ms_descriptor("MAIN", complete=False)
    descriptor.update(
        {
            "DATA": _fixed_column("COMPLEX", (2, 2), "VISIBILITY_GROUP"),
            "FLAG": _fixed_column("BOOLEAN", (2, 2), "VISIBILITY_GROUP"),
            "SPECIAL": _fixed_column("DOUBLE", (4,), "SPECIAL_GROUP"),
            "ROW_NOTE": _scalar_column("STRING"),
        }
    )
    dminfo = {
        "*1": {
            "NAME": "VISIBILITY_GROUP",
            "TYPE": "TiledColumnStMan",
            "SPEC": {"DEFAULTTILESHAPE": [2, 2, 2]},
            "COLUMNS": ["DATA", "FLAG"],
        },
        "*2": {
            "NAME": "SPECIAL_GROUP",
            "TYPE": "TiledColumnStMan",
            "SPEC": {"DEFAULTTILESHAPE": [4, 2]},
            "COLUMNS": ["SPECIAL"],
        },
    }
    return descriptor, dminfo


def test_probe_is_pinned_to_reviewed_arcae():
    assert version("arcae") == PINNED_ARCAE


def test_fresh_main_creates_and_links_default_subtables(tmp_path):
    ms = tmp_path / "fresh.ms"
    with Table.ms_from_descriptor(str(ms)) as main:
        keywords = main.tabledesc()["_keywords_"]

    assert (ms / "table.dat").is_file()
    assert DEFAULT_SUBTABLES.issubset(keywords)
    assert all((ms / name / "table.dat").is_file() for name in DEFAULT_SUBTABLES)

    for name in DEFAULT_SUBTABLES:
        with Table.from_filename(f"{ms}::{name}") as subtable:
            assert subtable.nrow() == 0


def test_optional_subtable_is_created_linked_and_reopened(tmp_path):
    ms = tmp_path / "weather.ms"
    with Table.ms_from_descriptor(str(ms)):
        pass

    with Table.ms_from_descriptor(
        str(ms), "WEATHER", ms_descriptor("WEATHER", complete=False)
    ) as weather:
        weather.addrows(1)
        weather.putcol("ANTENNA_ID", np.array([3], dtype=np.int32))
        weather.putcol("INTERVAL", np.array([8.0], dtype=np.float64))
        weather.putcol("TIME", np.array([5_000_000_000.0], dtype=np.float64))

    with Table.from_filename(str(ms)) as main:
        assert "WEATHER" in main.tabledesc()["_keywords_"]
    with Table.from_filename(f"{ms}::WEATHER") as weather:
        np.testing.assert_array_equal(weather.getcol("ANTENNA_ID"), [3])


def test_bounded_addrows_calls_preserve_total(tmp_path):
    ms = tmp_path / "rows.ms"
    with Table.ms_from_descriptor(str(ms)) as main:
        for count in (2, 3, 1):
            main.addrows(count)
        assert main.nrow() == 6


def test_addrows_rejects_count_outside_pinned_c_int(tmp_path):
    ms = tmp_path / "too-many-rows.ms"
    with Table.ms_from_descriptor(str(ms)) as main:
        with pytest.raises(OverflowError):
            main.addrows(2**31)
        assert main.nrow() == 0


def test_indexed_fixed_shape_and_scalar_values_roundtrip(tmp_path):
    ms = tmp_path / "fixed.ms"
    descriptor, dminfo = _probe_descriptor()
    row_ids = np.array([3, 0, 2, 1], dtype=np.int64)
    data = (
        np.arange(16, dtype=np.float32).reshape(4, 2, 2)
        + 1j * np.arange(16, 32, dtype=np.float32).reshape(4, 2, 2)
    ).astype(np.complex64)
    flags = (np.arange(16).reshape(4, 2, 2) % 3 == 0).astype(bool)
    specials = np.array(
        [
            [0.0, -0.0, np.nan, np.inf],
            [-np.inf, 1.5, -2.5, 3.5],
            [4.5, 5.5, 6.5, 7.5],
            [8.5, 9.5, 10.5, 11.5],
        ],
        dtype=np.float64,
    )
    notes = np.array(["row-three", "row-zero", "row-two", "row-one"])

    with Table.ms_from_descriptor(
        str(ms), table_desc=descriptor, dminfo=dminfo, cache_size=1
    ) as main:
        main.addrows(4)
        whole_cells = (row_ids, None, None)
        main.putcol("DATA", data, index=whole_cells)
        main.putcol("FLAG", flags, index=whole_cells)
        main.putcol("SPECIAL", specials, index=(row_ids, None))
        main.putcol("ROW_NOTE", notes, index=(row_ids,))

    inverse = np.argsort(row_ids)
    with Table.from_filename(str(ms), cache_size=1) as main:
        actual_data = main.getcol("DATA")
        actual_flags = main.getcol("FLAG")
        actual_specials = main.getcol("SPECIAL")
        actual_notes = main.getcol("ROW_NOTE")
        flag_descriptor = main.getcoldesc("FLAG")
        groups = {group["NAME"]: group for group in main.getdminfo().values()}

    np.testing.assert_array_equal(actual_data, data[inverse])
    np.testing.assert_array_equal(actual_flags, flags[inverse])
    np.testing.assert_array_equal(actual_notes, notes[inverse])
    np.testing.assert_array_equal(actual_specials, specials[inverse])
    assert actual_data.dtype == np.dtype(np.complex64)
    assert flag_descriptor["valueType"].upper() == "BOOLEAN"
    assert actual_flags.dtype == np.dtype(np.uint8)
    assert actual_specials.dtype == np.dtype(np.float64)
    assert actual_specials.view(np.uint64).tolist() == specials[inverse].view(np.uint64).tolist()
    assert groups["VISIBILITY_GROUP"]["TYPE"] == "TiledColumnStMan"
    assert set(groups["VISIBILITY_GROUP"]["COLUMNS"]) == {"DATA", "FLAG"}


def test_required_measure_metadata_survives_creation(tmp_path):
    ms = tmp_path / "metadata.ms"
    descriptor = ms_descriptor("MAIN", complete=False)
    descriptor["_keywords_"]["MSUTILS_PROFILE"] = "fixed-shape-v1"

    with Table.ms_from_descriptor(str(ms), table_desc=descriptor):
        pass
    with Table.from_filename(str(ms)) as main:
        time = main.getcoldesc("TIME")
        keywords = main.tabledesc()["_keywords_"]

    assert time["keywords"]["MEASINFO"] == {"Ref": "UTC", "type": "epoch"}
    assert time["keywords"]["QuantumUnits"] == ["s"]
    assert keywords["MSUTILS_PROFILE"] == "fixed-shape-v1"


def test_undefined_variable_cell_is_visible_to_capability_check(tmp_path):
    ms = tmp_path / "undefined.ms"
    descriptor = ms_descriptor("MAIN", complete=False)
    descriptor["VARIABLE"] = {
        "comment": "variable-shape probe",
        "dataManagerGroup": "StandardStMan",
        "dataManagerType": "StandardStMan",
        "keywords": {},
        "maxlen": 0,
        "ndim": 2,
        "option": 0,
        "valueType": "FLOAT",
    }

    with Table.ms_from_descriptor(str(ms), table_desc=descriptor) as main:
        main.addrows(2)
        main.putcol(
            "VARIABLE",
            np.arange(6, dtype=np.float32).reshape(1, 2, 3),
            index=(np.array([0], dtype=np.int64), None, None),
        )
        shapes = main.row_shapes("VARIABLE").to_pylist()

    assert shapes == [[2, 3], None]


def test_empty_keyword_array_is_untyped_in_descriptor_transport():
    required = ms_descriptor("MAIN", complete=False)
    complete = ms_descriptor("MAIN", complete=True)

    assert "CATEGORY" not in required["FLAG_CATEGORY"]["keywords"]
    assert complete["FLAG_CATEGORY"]["keywords"]["CATEGORY"] == []


@pytest.mark.xfail(
    strict=True,
    reason="arcae 0.5.4 descriptor JSON cannot preserve an empty array's element type",
)
def test_empty_string_keyword_array_preserves_native_type(tmp_path):
    ms = tmp_path / "typed-empty-keyword.ms"
    writer = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "from arcae.lib.arrow_tables import Table, ms_descriptor;"
                "native = Table.ms_from_descriptor("
                "sys.argv[1], table_desc=ms_descriptor('MAIN', complete=True));"
                "native.close()"
            ),
            str(ms),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert writer.returncode == 0, writer.stderr

    oracle = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, numpy as np;"
                "from casacore.tables import table;"
                "native = table(sys.argv[1], ack=False);"
                "value = native.getcolkeyword('FLAG_CATEGORY', 'CATEGORY');"
                "native.close();"
                "print(np.asarray(value).dtype.kind)"
            ),
            str(ms),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert oracle.stdout.strip() in {"U", "S"}


@pytest.mark.xfail(
    strict=True,
    reason="zero-length native array-cell creation/reopen is not established for arcae 0.5.4",
)
def test_zero_length_variable_cell_roundtrips(tmp_path):
    ms = tmp_path / "zero-length.ms"
    descriptor = ms_descriptor("MAIN", complete=False)
    descriptor["VARIABLE"] = {
        "comment": "zero-length probe",
        "dataManagerGroup": "StandardStMan",
        "dataManagerType": "StandardStMan",
        "keywords": {},
        "maxlen": 0,
        "ndim": 2,
        "option": 0,
        "valueType": "FLOAT",
    }

    with Table.ms_from_descriptor(str(ms), table_desc=descriptor) as main:
        main.addrows(1)
        expected = np.empty((1, 0, 2), dtype=np.float32)
        main.putcol("VARIABLE", expected, index=(np.array([0]), None, None))

    with Table.from_filename(str(ms)) as main:
        assert main.row_shapes("VARIABLE").to_pylist() == [[0, 2]]
        actual = main.getcol("VARIABLE", index=(np.array([0]), None, None))
    assert actual.shape == expected.shape


def test_linked_optional_subtable_survives_parent_relocation(tmp_path):
    staged = tmp_path / "stage.ms"
    published = tmp_path / "published.ms"
    with Table.ms_from_descriptor(str(staged)):
        pass
    with Table.ms_from_descriptor(str(staged), "WEATHER", ms_descriptor("WEATHER", complete=False)):
        pass

    os.rename(staged, published)

    with Table.from_filename(str(published)) as main:
        assert main.tabledesc()["_keywords_"]["WEATHER"] == f"Table: {published}/WEATHER"
    with Table.from_filename(f"{published}::WEATHER"):
        pass
