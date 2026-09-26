"""Exact-native reconstruction using independent casacore read-back.

Tests that need no optional stack (native_logical_id, the capture refusals it
shares, the metadata codecs) run on the bare install; bundle tests need zarr
(``msutils[msv4]``) and restoration tests dask-ms (``msutils[exact-native]``).
"""

import json
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest
from casacore.tables import makearrcoldesc, makescacoldesc, table
from click.testing import CliRunner
from msfactory import make_ms

from msutils import _native_preservation as native
from msutils._native_preservation import (
    NativePreservationIds,
    NativePreservationRefusal,
    _decode,
    _encode,
    _manager_identity,
    _native_tree_id,
    capture_native_preservation,
    materialize_exact_native,
    native_logical_id,
    plan_reconstruction,
    verify_native_preservation,
)
from msutils.cli import cli
from msutils.convert import to_msv2


@pytest.fixture
def zarr_lib():
    return pytest.importorskip("zarr", reason="exact-native bundles need msutils[msv4]")


@pytest.fixture
def daskms():
    return pytest.importorskip("daskms", reason="exact-native restoration needs dask-ms")


def test_manager_identity_excludes_only_standard_stman_index_length():
    source = {
        "*1": {
            "TYPE": "StandardStMan",
            "NAME": "StandardStMan",
            "COLUMNS": ["DATA"],
            "SPEC": {"BUCKETSIZE": 4228, "IndexLength": 394},
        }
    }
    reconstructed = {
        "*1": {
            "TYPE": "StandardStMan",
            "NAME": "StandardStMan",
            "COLUMNS": ["DATA"],
            "SPEC": {"BUCKETSIZE": 4228, "IndexLength": 134},
        }
    }
    assert _manager_identity(source) == _manager_identity(reconstructed)
    reconstructed["*1"]["SPEC"]["BUCKETSIZE"] = 8192
    assert _manager_identity(source) != _manager_identity(reconstructed)
    assert source["*1"]["SPEC"]["IndexLength"] == 394


def _ms(tmp_path, *, ntime=1, set_category=True, name="source.ms"):
    source = tmp_path / name
    make_ms(
        source, nant=2, ntime=ntime, nspw=1, nfield=1, nscan_per_field=1, add_weight_spectrum=True
    )
    if set_category:
        with table(str(source), readonly=False, ack=False) as tab:
            tab.putcolkeyword("FLAG_CATEGORY", "CATEGORY", np.asarray(["test"]))
    return source


def _source(tmp_path, *, ntime=1, set_category=True):
    import zarr

    source = _ms(tmp_path, ntime=ntime, set_category=set_category)
    state = tmp_path / "state.zarr"
    group = zarr.group(str(state))
    group.create_array("VISIBILITY", data=np.zeros((1, 1, 1, 1), dtype=np.complex64))
    return source, state


def _manifest(bundle):
    return json.loads((bundle / "manifest.json").read_text())


def _write_manifest(bundle, manifest):
    (bundle / "manifest.json").write_text(json.dumps(manifest))


def _refusal(code, fn, *args, **kwargs):
    with pytest.raises(NativePreservationRefusal) as exc:
        fn(*args, **kwargs)
    assert exc.value.code == code, exc.value
    return exc.value


# --------------------------------------------------------------------------
# Profile refusals (shared by capture and native_logical_id; base install)


def _add_variable(source, cells):
    with table(str(source), readonly=False, ack=False) as tab:
        tab.addcols(makearrcoldesc("CUSTOM_VARIABLE", 0.0, ndim=1, valuetype="float"))
        for row, value in cells.items():
            tab.putcell("CUSTOM_VARIABLE", row, np.asarray(value, dtype=np.float32))


PROFILE_CASES = {
    "mixed-definedness": (2, {0: [1.0, 2.0]}, 1),
    "ragged-cell": (2, {0: [1.0, 2.0], 1: [3.0]}, 1),
    "zero-length-cell": (1, {0: []}, 0),
    "six-row-mixed": (6, {row: np.arange(row + 1) for row in range(5)}, 5),
}


@pytest.mark.parametrize("case", sorted(PROFILE_CASES))
def test_native_logical_id_shares_the_capture_refusals(tmp_path, case):
    ntime, cells, row = PROFILE_CASES[case]
    source = _ms(tmp_path, ntime=ntime)
    _add_variable(source, cells)
    refusal = _refusal(
        case.replace("six-row-mixed", "mixed-definedness"), native_logical_id, source
    )
    assert (refusal.table, refusal.column, refusal.row) == ("MAIN", "CUSTOM_VARIABLE", row)


def _add_complete_row(tab):
    """Append a row, defining every cell that is defined in row 0.

    A new row leaves variable-shape cells undefined, which the profile would
    refuse as mixed definedness rather than hash.
    """
    tab.addrows(1)
    for column in tab.colnames():
        if tab.iscelldefined(column, 0) and not tab.iscelldefined(column, 1):
            tab.putcell(column, 1, tab.getcell(column, 0))


def test_native_logical_id_is_deterministic_and_content_sensitive(tmp_path):
    base = _ms(tmp_path)
    identity = native_logical_id(base)
    assert identity == native_logical_id(base)
    assert identity.startswith("msutils-logical-hash/v1:")
    copy = tmp_path / "copy.ms"
    shutil.copytree(base, copy)
    assert native_logical_id(copy) == identity
    edits = {
        "cell": lambda tab: tab.putcell("TIME", 0, tab.getcell("TIME", 0) + 1e-6),
        "keyword": lambda tab: tab.putkeyword("SHINOBI_FIXTURE", "x"),
        "column-keyword": lambda tab: tab.putcolkeyword("DATA", "QuantumUnits", np.asarray(["Jy"])),
        "row": _add_complete_row,
    }
    seen = {identity}
    for name, edit in edits.items():
        variant = tmp_path / f"{name}.ms"
        shutil.copytree(base, variant)
        with table(str(variant), readonly=False, ack=False) as tab:
            edit(tab)
        changed = native_logical_id(variant)
        assert changed not in seen, name
        seen.add(changed)


def test_native_logical_id_needs_no_optional_stack(tmp_path):
    source = _ms(tmp_path)
    code = (
        "import sys, msutils;"
        "print(msutils.native_logical_id(sys.argv[1]));"
        "print(','.join(m for m in ('zarr', 'xarray', 'dask', 'daskms') if m in sys.modules))"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-c", code, str(source)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    identity, loaded = result.stdout.splitlines()
    assert identity == native_logical_id(source)
    assert loaded == ""


def test_native_tree_id_ignores_only_index_length(tmp_path):
    source = _ms(tmp_path)
    paths = native._closure(source)
    with native._read_locked(paths) as handles:
        tables = native._read_native(handles, paths, source, None, None)
    base = _native_tree_id(tables)

    def with_spec(key, value):
        doctored = json.loads(json.dumps(tables))
        managers = _decode(doctored[0]["managers"])
        standard = next(m for m in managers.values() if m["TYPE"] == "StandardStMan")
        assert key in standard["SPEC"]
        standard["SPEC"][key] = value
        doctored[0]["managers"] = _encode(managers)
        return _native_tree_id(doctored)

    assert with_spec("IndexLength", 12345) == base
    assert with_spec("BUCKETSIZE", 12345) != base


def test_unsupported_value_type_is_refused_before_reading(tmp_path, monkeypatch):
    # python-casacore cannot create a ushort column at all, so the unsupported
    # type is simulated by narrowing the supported map.
    source = _ms(tmp_path)
    monkeypatch.setattr(
        native, "_VALUE_TYPES", {k: v for k, v in native._VALUE_TYPES.items() if k != "double"}
    )
    refusal = _refusal("column-type", native_logical_id, source)
    assert (refusal.table, refusal.column) == ("MAIN", "UVW")


@pytest.mark.parametrize("name", ["__X", "data"])
def test_reserved_and_case_colliding_names_are_refused(tmp_path, name):
    source = _ms(tmp_path)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.addcols(makescacoldesc(name, 0.0))
    refusal = _refusal("unsupported-name", native_logical_id, source)
    assert (refusal.table, refusal.column) == ("MAIN", name)


def test_metadata_dict_cannot_impersonate_a_codec_tag():
    value = {"kind": "float", "hex": "0x1.0p+0"}
    assert _decode(_encode(value)) == value


@pytest.mark.parametrize(
    "envelope",
    [
        {"$type": "array", "dtype": "<f8", "shape": [2**40], "base64": ""},
        {"$type": "array", "dtype": "<f8", "shape": [0, 2**40], "base64": ""},
        {"$type": "array", "dtype": "O", "shape": [1], "base64": "AAAAAA=="},
        {"$type": "array", "dtype": "<f8", "shape": [2], "base64": "AAAAAA=="},
        {"$type": "array", "dtype": "<f8", "shape": [1], "base64": "AAAAAA=="},
        {"$type": "text-array", "dtype": "<U100000000", "shape": [1], "items": ["x"]},
        {"$type": "text-array", "dtype": "<U4", "shape": [2], "items": ["x"]},
    ],
)
def test_metadata_decoder_refuses_unbounded_or_malformed_arrays(envelope):
    with pytest.raises(NativePreservationRefusal):
        _decode(envelope)


# --------------------------------------------------------------------------
# Capture


def test_capture_refuses_mixed_undefinedness_without_a_bundle(zarr_lib, tmp_path):
    source, state = _source(tmp_path, ntime=2)
    _add_variable(source, {0: [1.0, 2.0]})
    bundle = tmp_path / "bundle"
    refusal = _refusal("mixed-definedness", capture_native_preservation, source, state, bundle)
    assert (refusal.table, refusal.column, refusal.row) == ("MAIN", "CUSTOM_VARIABLE", 1)
    assert not bundle.exists()


def test_six_row_custom_variable_refuses_without_target(zarr_lib, tmp_path):
    source, state = _source(tmp_path, ntime=6)
    _add_variable(source, {row: np.arange(row + 1) for row in range(5)})
    refusal = _refusal(
        "mixed-definedness", capture_native_preservation, source, state, tmp_path / "bundle"
    )
    assert (refusal.column, refusal.row) == ("CUSTOM_VARIABLE", 5)
    assert not (tmp_path / "bundle").exists()


def test_ordinary_capture_completes_with_user_locks(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = tmp_path / "bundle"
    command = [
        sys.executable,
        "-c",
        "from msutils import capture_native_preservation; import sys; capture_native_preservation(*sys.argv[1:])",
        str(source),
        str(state),
        str(bundle),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert plan_reconstruction(state, bundle).tables[0]["id"] == "MAIN"


@pytest.mark.parametrize("kind", ["column", "keyword", "reference"])
def test_capture_refuses_names_outside_planner_profile(zarr_lib, tmp_path, kind):
    source, state = _source(tmp_path)
    with table(str(source), readonly=False, ack=False) as tab:
        if kind == "column":
            tab.addcols(makearrcoldesc("A-B", 0.0, ndim=1, valuetype="float"))
        elif kind == "keyword":
            tab.putkeyword("K-X", "ordinary")
        else:
            tab.putkeyword("A-B", tab.getkeyword("ANTENNA"))
    bundle = tmp_path / "bundle"
    with pytest.raises(NativePreservationRefusal, match="unsupported-name"):
        capture_native_preservation(source, state, bundle)
    assert not bundle.exists()


def test_capture_refuses_chunk_above_ceiling(zarr_lib, tmp_path, monkeypatch):
    source, state = _source(tmp_path)
    monkeypatch.setattr(native, "_MAX_CHUNK_BYTES", 1)
    bundle = tmp_path / "bundle"
    refusal = _refusal("chunk-size", capture_native_preservation, source, state, bundle)
    assert (refusal.table, refusal.column, refusal.row) == ("MAIN", "UVW", 0)
    assert not bundle.exists()


@pytest.mark.parametrize(
    ("ceiling", "value", "column"),
    [("_MAX_CHUNK_BYTES", 1, "UVW"), ("_MAX_STRING_CHUNK_ELEMENTS", 0, "TYPE")],
)
def test_payload_ceilings_do_not_limit_native_identity(
    zarr_lib, tmp_path, monkeypatch, ceiling, value, column
):
    """The chunk ceilings bound what a payload stores, not what an MS may
    contain: capture refuses, native_logical_id still identifies the MS."""
    source, state = _source(tmp_path)
    identity = native_logical_id(source)
    monkeypatch.setattr(native, ceiling, value)
    refusal = _refusal("chunk-size", capture_native_preservation, source, state, tmp_path / "b")
    assert refusal.column == column
    monkeypatch.setattr(native, "_READ_BYTES", 1)  # one row per identity read
    assert native_logical_id(source) == identity


@pytest.mark.parametrize("block_rows", [0, -1, True, 1.5])
def test_capture_validates_block_rows(zarr_lib, tmp_path, block_rows):
    source, state = _source(tmp_path)
    with pytest.raises(ValueError, match="block_rows"):
        capture_native_preservation(source, state, tmp_path / "bundle", block_rows=block_rows)


def test_auto_chunking_targets_a_decoded_size(monkeypatch):
    assert native._chunk_rows("float64", (), 10**9, None, "MAIN", "TIME") == 8 * 1024 * 1024
    assert native._chunk_rows("complex64", (32768, 4), 10**9, None, "MAIN", "DATA") == 64
    assert native._chunk_rows("float64", (), 5, None, "MAIN", "TIME") == 5
    assert native._chunk_rows("string", (), 10**6, None, "T", "NAME") == 4096
    assert native._chunk_rows("float64", (), 100, 7, "MAIN", "TIME") == 7
    monkeypatch.setattr(native, "_MAX_STRING_CHUNK_ELEMENTS", 10)
    with pytest.raises(NativePreservationRefusal, match="chunk-size"):
        native._chunk_rows("string", (4,), 100, 3, "T", "NAME")


def test_capture_refuses_source_change_during_read(zarr_lib, tmp_path, monkeypatch):
    source, state = _source(tmp_path)
    original = native._capture_table
    changed = False

    def changing_capture(*args):
        nonlocal changed
        result = original(*args)
        if not changed:
            (source / "external-mutation").write_text("changed while reading")
            changed = True
        return result

    monkeypatch.setattr(native, "_capture_table", changing_capture)
    with pytest.raises(NativePreservationRefusal, match="source-changed"):
        capture_native_preservation(source, state, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_capture_refuses_ambiguous_empty_keyword(zarr_lib, tmp_path):
    source, state = _source(tmp_path, set_category=False)
    refusal = _refusal(
        "empty-typed-keyword", capture_native_preservation, source, state, tmp_path / "bundle"
    )
    assert refusal.column == "FLAG_CATEGORY"


@pytest.mark.parametrize("location", ["ms", "zarr", "linked-ms"])
def test_capture_refuses_contained_bundle_before_creating_it(zarr_lib, tmp_path, location):
    source, state = _source(tmp_path)
    if location == "ms":
        destination = source / "bundle"
    elif location == "zarr":
        destination = state / "bundle"
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        destination = alias / "bundle"
    with pytest.raises(NativePreservationRefusal, match="bundle-contained"):
        capture_native_preservation(source, state, destination)
    assert not destination.exists()


def test_capture_refuses_marker_only_zarr(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    (state / "VISIBILITY" / "zarr.json").unlink()
    with pytest.raises(NativePreservationRefusal, match="zarr-extra-entry"):
        capture_native_preservation(source, state, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_capture_refuses_a_zarr_v2_msv4_tree(zarr_lib, tmp_path):
    """Both trees must be Zarr v3: a v2 MSv4 tree has to be re-exported."""
    source = _ms(tmp_path)
    state = tmp_path / "v2.zarr"
    group = zarr_lib.open_group(str(state), mode="w", zarr_format=2)
    group.create_array("VISIBILITY", shape=(1,), dtype="complex64")
    refusal = _refusal(
        "zarr-format", capture_native_preservation, source, state, tmp_path / "bundle"
    )
    assert refusal.reason.startswith("msv4: ") and "zarr_format=3" in refusal.reason
    assert not (tmp_path / "bundle").exists()


def test_capture_refuses_msv4_tree_without_arrays(zarr_lib, tmp_path):
    source = _ms(tmp_path)
    state = tmp_path / "empty.zarr"
    zarr_lib.group(str(state))
    _refusal("zarr-empty", capture_native_preservation, source, state, tmp_path / "bundle")


def test_capture_readback_guard_catches_a_silently_altered_payload(zarr_lib, tmp_path, monkeypatch):
    source, state = _source(tmp_path, ntime=2)
    original = native._PayloadSink.column

    class Altering:
        def __init__(self, array):
            self._array = array

        def __setitem__(self, index, values):
            values = values.copy()
            values.reshape(-1)[0] = values.reshape(-1)[0] + 1
            self._array[index] = values

    def column(self, table_id, name, *args):
        array = original(self, table_id, name, *args)
        return Altering(array) if (table_id, name) == ("MAIN", "TIME") else array

    monkeypatch.setattr(native._PayloadSink, "column", column)
    bundle = tmp_path / "bundle"
    refusal = _refusal("payload-readback", capture_native_preservation, source, state, bundle)
    assert (refusal.table, refusal.column) == ("MAIN", "TIME")
    assert not bundle.exists()
    assert not list(tmp_path.glob(".bundle.*"))


def test_bundle_layout_manifest_and_payload_types(zarr_lib, tmp_path):
    source, state = _source(tmp_path, ntime=2)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.addcols(makescacoldesc("CUSTOM_UCHAR", 0, valuetype="uchar"))
        tab.putcol("CUSTOM_UCHAR", np.asarray([0, 255], dtype=np.uint8))
        tab.addcols(makearrcoldesc("CUSTOM_UNWRITTEN", 0.0, ndim=1, valuetype="float"))
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    assert sorted(os.listdir(bundle)) == ["manifest.json", "native.zarr"]
    manifest = _manifest(bundle)
    assert manifest["schema"] == "msutils-native-preservation/v2"
    assert manifest["profile"] == "fixed-shape-defined-or-empty/v1"
    assert manifest["hash"] == "msutils-logical-hash/v1"
    assert manifest["payload"]["path"] == "native.zarr"
    payload = bundle / "native.zarr"
    tables = {item["id"]: item for item in manifest["tables"]}
    assert set(os.listdir(payload)) == {"zarr.json", *tables}
    columns = {col["name"]: col for col in tables["MAIN"]["columns"]}
    with table(str(source), ack=False) as tab:
        assert [col["name"] for col in tables["MAIN"]["columns"]] == tab.colnames()

    def meta(*parts):
        return json.loads(payload.joinpath(*parts, "zarr.json").read_text())

    assert meta("MAIN", "FLAG")["data_type"] == "bool"
    assert meta("MAIN", "CUSTOM_UCHAR")["data_type"] == "uint8"
    assert columns["CUSTOM_UCHAR"]["dtype"] == "uint8"
    assert meta("ANTENNA", "NAME")["data_type"] == "string"
    assert meta("MAIN", "DATA")["data_type"] == "complex64"
    for name in ("CUSTOM_UNWRITTEN", "FLAG_CATEGORY"):
        assert columns[name] == {
            "name": name,
            "defined": False,
            "shape": None,
            "dtype": "float32" if name == "CUSTOM_UNWRITTEN" else "bool",
            "digest": None,
        }
        assert not (payload / "MAIN" / name).exists()
    assert tables["POINTING"]["rows"] == 0
    assert all(not col["defined"] for col in tables["POINTING"]["columns"])
    assert sorted(os.listdir(payload / "POINTING")) == ["zarr.json"]
    array = meta("MAIN", "TIME")
    assert array["attributes"] == {} and "dimension_names" not in array
    assert [codec["name"] for codec in array["codecs"]] == ["bytes", "zstd"]


def test_bundle_binds_the_three_logical_ids(zarr_lib, tmp_path):
    from msutils import logical_id

    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    manifest = _manifest(bundle)
    assert manifest["msv4"]["logical_id"] == logical_id(state)
    assert manifest["payload"]["logical_id"] == logical_id(bundle / "native.zarr")
    assert manifest["native_logical_id"] == native_logical_id(source)
    ids = verify_native_preservation(state, bundle)
    assert ids == NativePreservationIds(
        manifest["msv4"]["logical_id"],
        manifest["payload"]["logical_id"],
        manifest["native_logical_id"],
    )
    # Chunking is a storage choice: it never enters an ID.
    other = capture_native_preservation(source, state, tmp_path / "other", block_rows=1)
    assert verify_native_preservation(state, other) == ids


# --------------------------------------------------------------------------
# Planning refusals


def test_plan_refuses_v1_bundles_with_an_actionable_message(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    manifest = _manifest(bundle)
    manifest["schema"] = "msutils-native-preservation/v1"
    _write_manifest(bundle, manifest)
    refusal = _refusal("bundle-version", plan_reconstruction, state, bundle)
    assert "capture_native_preservation" in refusal.reason and "v2" in refusal.reason
    manifest["schema"] = "msutils-native-preservation/v2"
    _write_manifest(bundle, manifest)
    (bundle / "cells").mkdir()
    _refusal("bundle-version", plan_reconstruction, state, bundle)


@pytest.mark.parametrize("key", ["schema", "profile", "hash", "native_model"])
def test_plan_refuses_unknown_versions(zarr_lib, tmp_path, key):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    manifest = _manifest(bundle)
    manifest[key] = manifest[key] + "-future"
    _write_manifest(bundle, manifest)
    _refusal("bundle-version", plan_reconstruction, state, bundle)


def test_plan_refuses_extra_bundle_entries(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    (bundle / "sidecar.json").write_text("{}")
    _refusal("bundle-extra-entry", plan_reconstruction, state, bundle)


def test_plan_refuses_tampered_zarr_and_payload(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    assert plan_reconstruction(state, bundle).tables[0]["id"] == "MAIN"
    marker = state / "zarr.json"
    original = marker.read_bytes()
    marker.write_text('{"zarr_format":99}')
    _refusal("zarr-format", plan_reconstruction, state, bundle)
    marker.write_bytes(original)
    chunk = next((bundle / "native.zarr" / "MAIN" / "TIME" / "c").rglob("0"))
    saved = chunk.read_bytes()
    chunk.write_bytes(b"tampered")
    _refusal("zarr-unreadable", plan_reconstruction, state, bundle)
    chunk.write_bytes(saved)
    time = zarr_lib.open_array(
        zarr_lib.storage.LocalStore(str(bundle / "native.zarr" / "MAIN" / "TIME")), mode="r+"
    )
    time[0] = time[0] + 1.0
    refusal = _refusal("payload-changed", plan_reconstruction, state, bundle)
    assert (refusal.table, refusal.column) == ("MAIN", "TIME")


def test_plan_refuses_changed_msv4_content(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    document = json.loads((state / "VISIBILITY" / "zarr.json").read_text())
    document["attributes"] = {"units": "Jy"}
    (state / "VISIBILITY" / "zarr.json").write_text(json.dumps(document))
    _refusal("zarr-changed", plan_reconstruction, state, bundle)


def test_plan_refuses_incomplete_coverage(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    manifest = _manifest(bundle)
    manifest["coverage"].pop()
    _write_manifest(bundle, manifest)
    with pytest.raises(NativePreservationRefusal, match="coverage"):
        plan_reconstruction(state, bundle)


def _flip(hex_id):
    prefix, digest = hex_id[:-1], hex_id[-1]
    return prefix + ("0" if digest != "0" else "1")


MANIFEST_DAMAGE = {
    "bad-rows": (lambda m, main, col: main.update(rows="six"), "row-count"),
    "bad-digest": (lambda m, main, col: col.update(digest="x" * 64), "bundle-column"),
    "missing-digest": (lambda m, main, col: col.update(digest=None), "bundle-column"),
    "wrong-dtype": (lambda m, main, col: col.update(dtype="float32"), "bundle-column"),
    "escape-path": (lambda m, main, col: m["payload"].update(path="../x"), "bundle-manifest"),
    "bad-id": (lambda m, main, col: m.update(native_logical_id="sha256:0"), "bundle-manifest"),
    "extra-key": (lambda m, main, col: m.update(block_rows=64), "bundle-manifest"),
    "column-digest": (
        lambda m, main, col: col.update(digest=_flip(col["digest"])),
        "payload-changed",
    ),
    "native-id": (
        lambda m, main, col: m.update(native_logical_id=_flip(m["native_logical_id"])),
        "bundle-integrity",
    ),
    "payload-id": (
        lambda m, main, col: m["payload"].update(logical_id=_flip(m["payload"]["logical_id"])),
        "bundle-integrity",
    ),
    "msv4-id": (
        lambda m, main, col: m["msv4"].update(logical_id=_flip(m["msv4"]["logical_id"])),
        "zarr-changed",
    ),
}


@pytest.mark.parametrize("damage", sorted(MANIFEST_DAMAGE))
def test_plan_refuses_malformed_or_edited_manifest(zarr_lib, tmp_path, damage):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    manifest = _manifest(bundle)
    main = manifest["tables"][0]
    col = next(col for col in main["columns"] if col["name"] == "TIME")
    fn, code = MANIFEST_DAMAGE[damage]
    fn(manifest, main, col)
    _write_manifest(bundle, manifest)
    _refusal(code, plan_reconstruction, state, bundle)


def _swap_first_columns(manifest):
    columns = manifest["tables"][0]["columns"]
    columns[0], columns[1] = columns[1], columns[0]


RECORD_DAMAGE = {
    "column-order": (_swap_first_columns, "column-coverage"),
    "table-extra-key": (lambda m: m["tables"][0].update(block_rows=64), "bundle-table"),
    "table-missing-key": (lambda m: m["tables"][1].pop("components"), "bundle-table"),
    "column-extra-key": (
        lambda m: m["tables"][0]["columns"][0].update(blocks=[]),
        "bundle-column",
    ),
}


@pytest.mark.parametrize("damage", sorted(RECORD_DAMAGE))
def test_plan_refuses_records_verification_would_reject(zarr_lib, tmp_path, damage):
    """Planning (and so verify_native_preservation) must refuse every record
    shape that verification would only reject after writing the target."""
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    manifest = _manifest(bundle)
    fn, code = RECORD_DAMAGE[damage]
    fn(manifest)
    _write_manifest(bundle, manifest)
    _refusal(code, verify_native_preservation, state, bundle)
    target = tmp_path / "target.ms"
    _refusal(code, materialize_exact_native, state, target, bundle)
    assert not target.exists()
    assert not list(tmp_path.glob(".target.ms.*"))


def test_expected_native_id_mismatch_refuses_before_writing(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    recorded = _manifest(bundle)["native_logical_id"]
    target = tmp_path / "target.ms"

    class RecordingWriter:
        called = False

        def write(self, plan, destination):
            RecordingWriter.called = True

    refusal = _refusal(
        "native-id-mismatch",
        materialize_exact_native,
        state,
        target,
        bundle,
        writer=RecordingWriter(),
        expected_native_logical_id=_flip(recorded),
    )
    assert recorded in refusal.reason
    assert not RecordingWriter.called
    assert not target.exists()
    assert not list(tmp_path.glob(".target.ms.*"))
    with pytest.raises(NativePreservationRefusal, match="native-id-mismatch"):
        to_msv2(
            str(state),
            str(target),
            fidelity="exact-native-v1",
            preservation=str(bundle),
            expected_native_logical_id=_flip(recorded),
        )
    assert not target.exists()


def test_expected_native_id_is_exact_mode_only(tmp_path):
    with pytest.raises(ValueError, match="expected_native_logical_id"):
        to_msv2(
            str(tmp_path / "state.zarr"),
            str(tmp_path / "target.ms"),
            expected_native_logical_id="msutils-logical-hash/v1:" + "0" * 64,
        )


def test_plan_refuses_symlink_inside_payload(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    chunk = next(
        p
        for p in (bundle / "native.zarr" / "MAIN" / "TIME").rglob("*")
        if p.is_file() and p.name != "zarr.json"
    )
    external = tmp_path / "external"
    external.write_bytes(chunk.read_bytes())
    chunk.unlink()
    chunk.symlink_to(external)
    _refusal("zarr-symlink", plan_reconstruction, state, bundle)


def _add_payload_array(zarr, payload, name, dtype="int32"):
    zarr.create_array(
        store=zarr.storage.LocalStore(str(payload)), name=name, shape=(1,), dtype=dtype
    )


def _edit_payload_meta(payload, path, **changes):
    document = json.loads((payload / path / "zarr.json").read_text())
    document.update(changes)
    (payload / path / "zarr.json").write_text(json.dumps(document))


PAYLOAD_DAMAGE = {
    "extra-array": lambda z, p: _add_payload_array(z, p, "MAIN/EXTRA"),
    "missing-array": lambda z, p: shutil.rmtree(p / "MAIN" / "TIME"),
    "extra-group": lambda z, p: z.create_group(store=z.storage.LocalStore(str(p)), path="EXTRA"),
    "array-attributes": lambda z, p: _edit_payload_meta(p, "MAIN/TIME", attributes={"a": 1}),
    "dimension-names": lambda z, p: _edit_payload_meta(p, "MAIN/TIME", dimension_names=["row"]),
    "group-attributes": lambda z, p: _edit_payload_meta(p, "ANTENNA", attributes={"a": 1}),
    "shape": lambda z, p: _edit_payload_meta(p, "MAIN/TIME", shape=[2]),
    "dtype": lambda z, p: _edit_payload_meta(p, "MAIN/ANTENNA1", data_type="int64"),
}


@pytest.mark.parametrize("damage", sorted(PAYLOAD_DAMAGE))
def test_plan_refuses_payload_structure_changes(zarr_lib, tmp_path, damage):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    PAYLOAD_DAMAGE[damage](zarr_lib, bundle / "native.zarr")
    _refusal("payload-structure", plan_reconstruction, state, bundle)


def test_existing_target_survives_exact_refusal(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    target = tmp_path / "target.ms"
    target.mkdir()
    (target / "sentinel").write_text("intact")
    with pytest.raises(FileExistsError):
        materialize_exact_native(state, target, bundle)
    assert (target / "sentinel").read_text() == "intact"


def test_exact_mode_rejects_mapped_only_options(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    for options in (
        {"weight_spectrum": False},
        {"weight_spectrum": True},
        {"rowchunk": 8},
        {"rowchunk": 64},
        {"overwrite": True},
    ):
        with pytest.raises(ValueError):
            to_msv2(
                str(state),
                str(tmp_path / "target.ms"),
                fidelity="exact-native-v1",
                preservation=str(bundle),
                **options,
            )
    assert not (tmp_path / "target.ms").exists()


def test_exact_cli_reports_refusal_as_click_error(zarr_lib, tmp_path):
    _, state = _source(tmp_path)
    target = tmp_path / "target.ms"
    result = CliRunner().invoke(
        cli, ["materialize", str(state), str(target), "--fidelity", "exact-native-v1"]
    )
    assert result.exit_code != 0
    assert "Error: exact-native-v1 requires preservation" in result.output
    assert not target.exists()


def test_failed_writer_removes_private_candidate(zarr_lib, tmp_path):
    source, state = _source(tmp_path)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    target = tmp_path / "target.ms"

    class FailedWriter:
        def write(self, plan, destination):
            destination.mkdir()
            (destination / "partial").write_text("partial")
            raise RuntimeError("write failure")

    with pytest.raises(RuntimeError, match="write failure"):
        materialize_exact_native(state, target, bundle, writer=FailedWriter())
    assert not target.exists()
    assert not list(tmp_path.glob(".target.ms.*"))


# --------------------------------------------------------------------------
# Restoration (dask-ms)


def _assert_same_tables(old_path, new_path, suffixes):
    for suffix in suffixes:
        with (
            table(str(old_path) + suffix, ack=False) as old,
            table(str(new_path) + suffix, ack=False) as new,
        ):
            assert old.nrows() == new.nrows()
            assert old.info() == new.info()
            assert old.colnames() == new.colnames()
            for column in old.colnames():
                for row in range(old.nrows()):
                    assert old.iscelldefined(column, row) == new.iscelldefined(column, row)
                    if old.iscelldefined(column, row):
                        original, actual = old.getcell(column, row), new.getcell(column, row)
                        if isinstance(original, dict):  # string array cell
                            assert original == actual
                            continue
                        original, actual = np.asarray(original), np.asarray(actual)
                        assert original.shape == actual.shape, (column, row)
                        if original.dtype.kind in "US":
                            np.testing.assert_array_equal(original, actual)
                        else:
                            assert original.dtype.str == actual.dtype.str, (column, row)
                            assert original.tobytes() == actual.tobytes(), (column, row)


def test_daskms_exact_native_roundtrip(zarr_lib, daskms, tmp_path):
    source, state = _source(tmp_path)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.putkeyword("SHINOBI_FIXTURE_PROFILE", "fixed-shape")
        tab.putcolkeyword("DATA", "QuantumUnits", np.asarray(["Jy"]))
        tab.putcell("WEIGHT", 0, np.asarray([2, 3, 5, 7], dtype=np.float32))
        tab.putcell("SIGMA", 0, np.asarray([1, 2, 3, 4], dtype=np.float32))
        tab.putcell("FLAG_ROW", 0, True)
        original_info = tab.info()
        original_info["subType"] = "exact-test"
        original_info["readme"] = "custom readme"
        tab.putinfo(original_info)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    target = tmp_path / "target.ms"
    info = to_msv2(
        str(state),
        str(target),
        fidelity="exact-native-v1",
        preservation=str(bundle),
        expected_native_logical_id=native_logical_id(source),
    )
    assert info.nrows == 1
    with table(str(target), ack=False) as tab:
        assert str(target / "ANTENNA") in tab.getkeyword("ANTENNA")
        np.testing.assert_array_equal(tab.getcell("WEIGHT", 0), [2, 3, 5, 7])
        np.testing.assert_array_equal(tab.getcell("SIGMA", 0), [1, 2, 3, 4])
        assert tab.getcell("FLAG_ROW", 0)
        assert tab.getkeyword("SHINOBI_FIXTURE_PROFILE") == "fixed-shape"
        assert list(tab.getcolkeyword("DATA", "QuantumUnits")) == ["Jy"]
        assert tab.info()["subType"] == "exact-test"
        assert tab.info()["readme"] == "custom readme\n"
    with table(str(target) + "::ANTENNA", ack=False) as antenna:
        assert antenna.nrows() == 2
    assert native_logical_id(target) == _manifest(bundle)["native_logical_id"]


def test_rowids_preserve_order_across_chunks(zarr_lib, daskms, tmp_path, monkeypatch):
    source, state = _source(tmp_path, ntime=2)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.putcell("FLAG_ROW", 0, True)
        tab.putcell("FLAG_ROW", 1, False)
        original_times = tab.getcol("TIME").copy()
    bundle = capture_native_preservation(source, state, tmp_path / "bundle", block_rows=1)
    chunks = json.loads((bundle / "native.zarr" / "MAIN" / "TIME" / "zarr.json").read_text())
    assert chunks["chunk_grid"]["configuration"]["chunk_shape"] == [1]
    # One row per write batch as well, so ROWID ordering is exercised on write.
    monkeypatch.setattr(native, "_READ_BYTES", 1)
    target = tmp_path / "target.ms"
    materialize_exact_native(state, target, bundle)
    with table(str(target), ack=False) as tab:
        assert tab.getcol("FLAG_ROW").tolist() == [True, False]
        np.testing.assert_array_equal(tab.getcol("TIME"), original_times)


def test_source_can_be_hidden_after_capture(zarr_lib, daskms, tmp_path):
    source, state = _source(tmp_path, ntime=6)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle", block_rows=2)
    source.rename(tmp_path / "hidden-source")
    target = tmp_path / "target.ms"
    materialize_exact_native(state, target, bundle)
    with table(str(target), ack=False) as tab:
        assert tab.nrows() == 6


def _f32(bits):
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def _f64(bits):
    return np.array([bits], dtype=np.uint64).view(np.float64)[0]


def _matrix_values(value_type, nrows):
    """Row values with the special cases a lossy path would disturb."""
    if value_type == "boolean":
        return np.array([True, False, True, False][:nrows])
    if value_type == "uchar":
        return np.array([0, 255, 7, 128][:nrows], dtype=np.uint8)
    if value_type == "short":
        return np.array([-(2**15), 2**15 - 1, 0, -1][:nrows], dtype=np.int16)
    if value_type == "int":
        return np.array([-(2**31), 2**31 - 1, 0, -1][:nrows], dtype=np.int32)
    if value_type == "uint":
        return np.array([0, 2**32 - 1, 1, 2**31][:nrows], dtype=np.uint32)
    if value_type == "int64":
        return np.array([-(2**63), 2**63 - 1, 0, -1][:nrows], dtype=np.int64)
    if value_type == "float":
        return np.array(
            [_f32(0x7FC00001), _f32(0xFF800001), -0.0, np.inf][:nrows], dtype=np.float32
        )
    if value_type == "double":
        return np.array([_f64(0x7FF8000000000001), _f64(0xFFF0000000000001), -0.0, -np.inf][:nrows])
    if value_type == "complex":
        return np.array(
            [complex(-0.0, -0.0), complex(_f32(0x7FC00001), -0.0), complex(np.inf, 1), 2j][:nrows],
            dtype=np.complex64,
        )
    if value_type == "dcomplex":
        return np.array(
            [complex(-0.0, 0.0), complex(_f64(0x7FF8000000000002), 1), -1, 0][:nrows],
            dtype=np.complex128,
        )
    return np.array(["", "é𝄞", "x" * 300, "plain"][:nrows], dtype=object)


MATRIX_TYPES = [
    "boolean",
    "uchar",
    "short",
    "int",
    "uint",
    "int64",
    "float",
    "double",
    "complex",
    "dcomplex",
    "string",
]


def test_dtype_matrix_restores_bit_for_bit(zarr_lib, daskms, tmp_path):
    source, state = _source(tmp_path, ntime=2)
    with table(str(source), readonly=False, ack=False) as tab:
        nrows = tab.nrows()
        for value_type in MATRIX_TYPES:
            values = _matrix_values(value_type, nrows)
            default = "" if value_type == "string" else values[0].item()
            name = f"M_{value_type.upper()}"
            tab.addcols(makescacoldesc(f"{name}_S", default, valuetype=value_type))
            tab.addcols(makearrcoldesc(f"{name}_F", default, shape=[2, 3], valuetype=value_type))
            tab.addcols(makearrcoldesc(f"{name}_V", default, ndim=1, valuetype=value_type))
            for row in range(nrows):
                cell = values[row]
                tab.putcell(f"{name}_S", row, cell.item() if hasattr(cell, "item") else cell)
                fixed = np.resize(np.roll(values, row), 6).reshape(2, 3)
                variable = np.resize(np.roll(values, -row), 3)
                if value_type == "string":
                    fixed, variable = fixed.astype(str), variable.astype(str)
                tab.putcell(f"{name}_F", row, fixed)
                tab.putcell(f"{name}_V", row, variable)
        # The P3 hazard: a complex column that is signed zero everywhere.
        tab.addcols(makescacoldesc("M_ALL_NEGZERO", 0j, valuetype="complex"))
        tab.putcol("M_ALL_NEGZERO", np.full(nrows, complex(-0.0, -0.0), np.complex64))
    bundle = capture_native_preservation(source, state, tmp_path / "bundle", block_rows=1)
    manifest = _manifest(bundle)
    columns = {col["name"]: col for col in manifest["tables"][0]["columns"]}
    assert columns["M_UCHAR_S"]["dtype"] == "uint8"
    assert columns["M_STRING_F"]["shape"] == [2, 3]
    target = tmp_path / "target.ms"
    materialize_exact_native(state, target, bundle)
    _assert_same_tables(source, target, [""])
    assert native_logical_id(source) == manifest["native_logical_id"]
    assert native_logical_id(target) == manifest["native_logical_id"]


def _relayout(zarr, source, destination, *, chunks, compressors, shard):
    """Losslessly rewrite a Zarr v3 tree with a different physical layout."""
    source_group = zarr.open_group(
        zarr.storage.LocalStore(str(source), read_only=True), mode="r", use_consolidated=False
    )
    store = zarr.storage.LocalStore(str(destination))
    zarr.create_group(store=store, zarr_format=3, attributes=dict(source_group.attrs))

    def walk(group, prefix):
        for name, item in group.members():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(item, zarr.Group):
                zarr.create_group(
                    store=store, path=path, zarr_format=3, attributes=dict(item.attrs)
                )
                walk(item, path)
                continue
            values = item[...]
            string = values.dtype.kind in "OUT"
            shape_chunks = tuple(max(1, c) for c in chunks(item.shape))
            kwargs = {}
            if shard and item.ndim and not string:
                kwargs["shards"] = tuple(2 * c for c in shape_chunks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                array = zarr.create_array(
                    store=store,
                    name=path,
                    shape=item.shape,
                    dtype=str if string and values.dtype.kind != "U" else item.dtype,
                    chunks=shape_chunks,
                    compressors=compressors,
                    attributes=dict(item.attrs),
                    dimension_names=item.metadata.dimension_names,
                    fill_value=item.fill_value,
                    config={"write_empty_chunks": True},
                    zarr_format=3,
                    **kwargs,
                )
                array[...] = values

    walk(source_group, "")


def _relayout_in_place(zarr, root, **layout):
    fresh = root.with_name(root.name + ".relayout")
    _relayout(zarr, root, fresh, **layout)
    shutil.rmtree(root)
    fresh.rename(root)


@pytest.mark.parametrize("which", ["payload", "msv4", "both"])
def test_lossless_relayout_of_either_tree_keeps_the_bundle_valid(zarr_lib, daskms, tmp_path, which):
    from zarr.codecs import GzipCodec

    source, state = _source(tmp_path, ntime=3)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    manifest = _manifest(bundle)
    if which in ("payload", "both"):
        payload = bundle / "native.zarr"
        before = sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*"))
        _relayout_in_place(
            zarr_lib,
            payload,
            chunks=lambda s: (1,) * len(s),
            compressors=[GzipCodec(level=9)],
            shard=True,
        )
        assert sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*")) != before
    if which in ("msv4", "both"):
        _relayout_in_place(zarr_lib, state, chunks=lambda s: s, compressors=None, shard=False)
    plan = plan_reconstruction(state, bundle)
    assert plan.msv4_logical_id == manifest["msv4"]["logical_id"]
    assert plan.payload_logical_id == manifest["payload"]["logical_id"]
    assert plan.native_logical_id == manifest["native_logical_id"]
    target = tmp_path / "target.ms"
    materialize_exact_native(state, target, bundle)
    _assert_same_tables(source, target, ["", "::ANTENNA"])


def test_payload_changed_after_planning_is_not_published(zarr_lib, daskms, tmp_path):
    source, state = _source(tmp_path, ntime=2)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    target = tmp_path / "target.ms"

    class SwappingWriter(native.DaskMsWriter):
        def write(self, plan, destination):
            time = zarr_lib.open_array(
                zarr_lib.storage.LocalStore(str(plan.payload / "MAIN" / "TIME")), mode="r+"
            )
            time[1] = time[1] + 1.0
            super().write(plan, destination)

    with pytest.raises(NativePreservationRefusal) as exc:
        materialize_exact_native(state, target, bundle, writer=SwappingWriter())
    assert exc.value.code in ("verify-cell", "verify-payload-changed")
    assert (exc.value.table, exc.value.column) == ("MAIN", "TIME")
    assert not target.exists()
    assert not list(tmp_path.glob(".target.ms.*"))


def test_verification_names_the_first_differing_row(zarr_lib, daskms, tmp_path):
    source, state = _source(tmp_path, ntime=3)
    bundle = capture_native_preservation(source, state, tmp_path / "bundle")
    target = tmp_path / "target.ms"

    class CorruptingWriter(native.DaskMsWriter):
        def write(self, plan, destination):
            super().write(plan, destination)
            with table(str(destination), readonly=False, ack=False) as tab:
                tab.putcell("EXPOSURE", 2, tab.getcell("EXPOSURE", 2) + 1.0)

    refusal = _refusal(
        "verify-cell", materialize_exact_native, state, target, bundle, writer=CorruptingWriter()
    )
    assert (refusal.table, refusal.column, refusal.row) == ("MAIN", "EXPOSURE", 2)
    assert not target.exists()


def test_real_msv2_export_exact_restore_and_reexport(zarr_lib, daskms, tmp_path):
    pytest.importorskip("xarray_ms")
    import xarray as xr
    import xarray.testing as xt

    from msutils import logical_id

    source = _ms(tmp_path, ntime=6)
    with table(str(source), readonly=False, ack=False) as main:
        main.putcell("WEIGHT", 0, np.asarray([2, 3, 5, 7], dtype=np.float32))
        main.putcell("SIGMA", 0, np.asarray([1, 2, 3, 4], dtype=np.float32))
        main.putcell("FLAG_ROW", 0, True)
        main.putkeyword("SHINOBI_FIXTURE_PROFILE", "six-row-fixed")
        main.putcolkeyword("DATA", "QuantumUnits", np.asarray(["Jy"]))
    with table(str(source) + "::PROCESSOR", readonly=False, ack=False) as processor:
        processor.addrows(1)
        processor.putcell("TYPE", 0, "CORRELATOR")
        processor.putcell("SUB_TYPE", 0, "test")
        processor.putcell("MODE_ID", 0, 0)
        processor.putcell("FLAG_ROW", 0, False)
    state = tmp_path / "real-export.zarr"
    settings = {
        "engine": "xarray-ms:msv2",
        "auto_corrs": True,
        "partition_schema": ["DATA_DESC_ID", "FIELD_ID"],
        "driver": "arcae",
        "driver_kwargs": {"cache_size": 64},
        "chunks": {},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_datatree(source, **settings) as tree:
            tree.to_zarr(state, mode="w", compute=True, consolidated=True)
    with xr.open_datatree(state, engine="zarr") as reopened:
        exported = reopened.load()
    bundle = capture_native_preservation(source, state, tmp_path / "bundle", block_rows=2)
    assert logical_id(state) == _manifest(bundle)["msv4"]["logical_id"]
    hidden = tmp_path / "hidden-source"
    source.rename(hidden)
    target = tmp_path / "restored" / "source.ms"
    materialize_exact_native(state, target, bundle)
    second_state = tmp_path / "second-export.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_datatree(target, **settings) as tree:
            tree.to_zarr(second_state, mode="w", compute=True, consolidated=True)
    with xr.open_datatree(second_state, engine="zarr") as reopened:
        restored = reopened.load()
    assert {node.path for node in exported.subtree} == {node.path for node in restored.subtree}
    for node in exported.subtree:
        xt.assert_equal(node.ds, restored[node.path].ds)
    _assert_same_tables(
        hidden,
        target,
        ["", "::PROCESSOR", "::FEED", "::FIELD", "::OBSERVATION", "::SPECTRAL_WINDOW"],
    )
