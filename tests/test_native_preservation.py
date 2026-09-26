"""Exact-native reconstruction using independent casacore read-back."""

import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
from casacore.tables import makearrcoldesc, table
from click.testing import CliRunner
from msfactory import make_ms

from msutils._native_preservation import (
    NativePreservationRefusal,
    _decode,
    _encode,
    _manager_identity,
    capture_native_preservation,
    materialize_exact_native,
    plan_reconstruction,
)
from msutils.cli import cli
from msutils.convert import to_msv2

zarr_lib = pytest.importorskip("zarr", reason="exact-native tests require msutils[exact-native]")


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


def _source(tmp_path, *, ntime=1, set_category=True):
    source = tmp_path / "source.ms"
    make_ms(
        source, nant=2, ntime=ntime, nspw=1, nfield=1, nscan_per_field=1, add_weight_spectrum=True
    )
    if set_category:
        with table(str(source), readonly=False, ack=False) as tab:
            tab.putcolkeyword("FLAG_CATEGORY", "CATEGORY", np.asarray(["test"]))
    zarr = tmp_path / "state.zarr"
    group = zarr_lib.group(str(zarr))
    data = np.zeros((1, 1, 1, 1), dtype=np.complex64)
    if hasattr(group, "create_array"):
        group.create_array("VISIBILITY", data=data)
    else:
        group.create_dataset("VISIBILITY", data=data)
    return source, zarr


def test_capture_refuses_mixed_undefinedness_without_a_bundle(tmp_path):
    source, zarr = _source(tmp_path, ntime=2)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.addcols(makearrcoldesc("CUSTOM_VARIABLE", 0.0, ndim=1, valuetype="float"))
        tab.putcell("CUSTOM_VARIABLE", 0, np.asarray([1.0, 2.0], dtype=np.float32))
    bundle = tmp_path / "bundle"
    with pytest.raises(NativePreservationRefusal) as exc:
        capture_native_preservation(source, zarr, bundle)
    assert exc.value.code == "mixed-definedness"
    assert (exc.value.table, exc.value.column, exc.value.row) == ("MAIN", "CUSTOM_VARIABLE", 1)
    assert not bundle.exists()


def test_ordinary_capture_completes_with_user_locks(tmp_path):
    source, zarr = _source(tmp_path)
    bundle = tmp_path / "bundle"
    command = [
        sys.executable,
        "-c",
        "from msutils import capture_native_preservation; import sys; capture_native_preservation(*sys.argv[1:])",
        str(source),
        str(zarr),
        str(bundle),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    assert plan_reconstruction(zarr, bundle).tables[0]["id"] == "MAIN"


@pytest.mark.parametrize("kind", ["column", "keyword", "reference"])
def test_capture_refuses_names_outside_planner_profile(tmp_path, kind):
    source, zarr = _source(tmp_path)
    with table(str(source), readonly=False, ack=False) as tab:
        if kind == "column":
            tab.addcols(makearrcoldesc("A-B", 0.0, ndim=1, valuetype="float"))
        elif kind == "keyword":
            tab.putkeyword("K-X", "ordinary")
        else:
            tab.putkeyword("A-B", tab.getkeyword("ANTENNA"))
    bundle = tmp_path / "bundle"
    with pytest.raises(NativePreservationRefusal, match="unsupported-name"):
        capture_native_preservation(source, zarr, bundle)
    assert not bundle.exists()


def test_capture_refuses_ragged_cells(tmp_path):
    source, zarr = _source(tmp_path, ntime=2)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.addcols(makearrcoldesc("CUSTOM_VARIABLE", 0.0, ndim=1, valuetype="float"))
        tab.putcell("CUSTOM_VARIABLE", 0, np.asarray([1.0, 2.0], dtype=np.float32))
        tab.putcell("CUSTOM_VARIABLE", 1, np.asarray([3.0], dtype=np.float32))
    with pytest.raises(NativePreservationRefusal) as exc:
        capture_native_preservation(source, zarr, tmp_path / "bundle")
    assert (exc.value.code, exc.value.column, exc.value.row) == (
        "ragged-cell",
        "CUSTOM_VARIABLE",
        1,
    )


def test_capture_refuses_zero_length_cell(tmp_path):
    source, zarr = _source(tmp_path)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.addcols(makearrcoldesc("CUSTOM_VARIABLE", 0.0, ndim=1, valuetype="float"))
        tab.putcell("CUSTOM_VARIABLE", 0, np.asarray([], dtype=np.float32))
    with pytest.raises(NativePreservationRefusal) as exc:
        capture_native_preservation(source, zarr, tmp_path / "bundle")
    assert (exc.value.code, exc.value.column, exc.value.row) == (
        "zero-length-cell",
        "CUSTOM_VARIABLE",
        0,
    )


def test_capture_refuses_block_larger_than_planner_limit(tmp_path, monkeypatch):
    from msutils import _native_preservation as native

    source, zarr = _source(tmp_path)
    monkeypatch.setattr(native, "_MAX_BLOCK_BYTES", 1)
    bundle = tmp_path / "bundle"
    with pytest.raises(NativePreservationRefusal) as exc:
        capture_native_preservation(source, zarr, bundle)
    assert exc.value.code == "block-size"
    assert not bundle.exists()


def test_capture_refuses_source_change_during_read(tmp_path, monkeypatch):
    from msutils import _native_preservation as native

    source, zarr = _source(tmp_path)
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
        capture_native_preservation(source, zarr, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_capture_refuses_ambiguous_empty_keyword(tmp_path):
    source, zarr = _source(tmp_path, set_category=False)
    with pytest.raises(NativePreservationRefusal) as exc:
        capture_native_preservation(source, zarr, tmp_path / "bundle")
    assert (exc.value.code, exc.value.column) == ("empty-typed-keyword", "FLAG_CATEGORY")


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


@pytest.mark.parametrize("location", ["ms", "zarr", "linked-ms"])
def test_capture_refuses_contained_bundle_before_creating_it(tmp_path, location):
    source, zarr = _source(tmp_path)
    if location == "ms":
        destination = source / "bundle"
    elif location == "zarr":
        destination = zarr / "bundle"
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        destination = alias / "bundle"
    with pytest.raises(NativePreservationRefusal, match="bundle-contained"):
        capture_native_preservation(source, zarr, destination)
    assert not destination.exists()


def test_capture_refuses_marker_only_zarr(tmp_path):
    source, zarr = _source(tmp_path)
    marker = zarr / "VISIBILITY" / ".zarray"
    if not marker.exists():
        marker = zarr / "VISIBILITY" / "zarr.json"
    marker.unlink()
    with pytest.raises(NativePreservationRefusal, match="zarr-empty"):
        capture_native_preservation(source, zarr, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_plan_refuses_tampered_zarr_and_payload(tmp_path):
    source, zarr = _source(tmp_path)
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle")
    plan = plan_reconstruction(zarr, bundle)
    assert plan.tables[0]["id"] == "MAIN"
    marker = zarr / ".zgroup"
    if not marker.exists():
        marker = zarr / "zarr.json"
    original = marker.read_bytes()
    marker.write_text('{"zarr_format":99}')
    with pytest.raises(NativePreservationRefusal, match="zarr-unreadable"):
        plan_reconstruction(zarr, bundle)
    marker.write_bytes(original)
    manifest = json.loads((bundle / "manifest.json").read_text())
    block = next(
        col["blocks"][0] for tab in manifest["tables"] for col in tab["columns"] if col["blocks"]
    )
    (bundle / block["path"]).write_bytes(b"tampered")
    with pytest.raises(NativePreservationRefusal, match="block-integrity"):
        plan_reconstruction(zarr, bundle)


def test_plan_refuses_incomplete_coverage(tmp_path):
    source, zarr = _source(tmp_path)
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coverage"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(NativePreservationRefusal, match="coverage"):
        plan_reconstruction(zarr, bundle)


@pytest.mark.parametrize("damage", ["bad-rows", "bad-size", "escape-path", "symlink"])
def test_plan_refuses_malformed_manifest_and_payload(tmp_path, damage):
    source, zarr = _source(tmp_path)
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    table_record = next(item for item in manifest["tables"] if item["id"] == "MAIN")
    block = next(col["blocks"][0] for col in table_record["columns"] if col["blocks"])
    if damage == "bad-rows":
        table_record["rows"] = "six"
    elif damage == "bad-size":
        block["size"] = 1
    elif damage == "escape-path":
        block["path"] = "cells/../manifest.json"
    else:
        block_path = bundle / block["path"]
        external = tmp_path / "external.npy"
        external.write_bytes(block_path.read_bytes())
        block_path.unlink()
        block_path.symlink_to(external)
    if damage != "symlink":
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(NativePreservationRefusal):
        plan_reconstruction(zarr, bundle)


def test_daskms_exact_native_roundtrip(tmp_path):
    pytest.importorskip("daskms")
    source, zarr = _source(tmp_path)
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
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle")
    target = tmp_path / "target.ms"
    info = to_msv2(str(zarr), str(target), fidelity="exact-native-v1", preservation=str(bundle))
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


def test_rowids_preserve_order_across_blocks(tmp_path):
    pytest.importorskip("daskms")
    source, zarr = _source(tmp_path, ntime=2)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.putcell("FLAG_ROW", 0, True)
        tab.putcell("FLAG_ROW", 1, False)
        original_times = tab.getcol("TIME").copy()
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle", block_rows=1)
    target = tmp_path / "target.ms"
    materialize_exact_native(zarr, target, bundle)
    with table(str(target), ack=False) as tab:
        assert tab.getcol("FLAG_ROW").tolist() == [True, False]
        np.testing.assert_array_equal(tab.getcol("TIME"), original_times)


def test_source_can_be_hidden_after_capture(tmp_path):
    pytest.importorskip("daskms")
    source, zarr = _source(tmp_path, ntime=6)
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle", block_rows=2)
    source.rename(tmp_path / "hidden-source")
    target = tmp_path / "target.ms"
    materialize_exact_native(zarr, target, bundle)
    with table(str(target), ack=False) as tab:
        assert tab.nrows() == 6


def test_six_row_custom_variable_refuses_without_target(tmp_path):
    source, zarr = _source(tmp_path, ntime=6)
    with table(str(source), readonly=False, ack=False) as tab:
        tab.addcols(makearrcoldesc("CUSTOM_VARIABLE", 0.0, ndim=1, valuetype="float"))
        for row in range(5):
            tab.putcell("CUSTOM_VARIABLE", row, np.arange(row + 1, dtype=np.float32))
    target = tmp_path / "target.ms"
    with pytest.raises(NativePreservationRefusal) as exc:
        capture_native_preservation(source, zarr, tmp_path / "bundle")
    assert (exc.value.code, exc.value.column, exc.value.row) == (
        "mixed-definedness",
        "CUSTOM_VARIABLE",
        5,
    )
    assert not target.exists()


def test_real_msv2_export_exact_restore_and_reexport(tmp_path):
    pytest.importorskip("daskms")
    pytest.importorskip("xarray_ms")
    import xarray as xr
    import xarray.testing as xt

    source, _ = _source(tmp_path, ntime=6)
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
    zarr = tmp_path / "real-export.zarr"
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
            tree.to_zarr(zarr, mode="w", compute=True, consolidated=True)
    with xr.open_datatree(zarr, engine="zarr") as reopened:
        exported = reopened.load()
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle", block_rows=2)
    hidden = tmp_path / "hidden-source"
    source.rename(hidden)
    target = tmp_path / "restored" / "source.ms"
    materialize_exact_native(zarr, target, bundle)
    second_zarr = tmp_path / "second-export.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_datatree(target, **settings) as tree:
            tree.to_zarr(second_zarr, mode="w", compute=True, consolidated=True)
    with xr.open_datatree(second_zarr, engine="zarr") as reopened:
        restored = reopened.load()
    assert {node.path for node in exported.subtree} == {node.path for node in restored.subtree}
    for node in exported.subtree:
        xt.assert_equal(node.ds, restored[node.path].ds)
    for suffix in ("", "::PROCESSOR", "::FEED", "::FIELD", "::OBSERVATION", "::SPECTRAL_WINDOW"):
        with (
            table(str(hidden) + suffix, ack=False) as old,
            table(str(target) + suffix, ack=False) as new,
        ):
            assert old.nrows() == new.nrows()
            assert old.info() == new.info()
            assert old.colnames() == new.colnames()
            for column in old.colnames():
                for row in range(old.nrows()):
                    assert old.iscelldefined(column, row) == new.iscelldefined(column, row)
                    if old.iscelldefined(column, row):
                        original = np.asarray(old.getcell(column, row))
                        actual = np.asarray(new.getcell(column, row))
                        assert original.shape == actual.shape
                        if original.dtype.kind in "US":
                            np.testing.assert_array_equal(original, actual)
                        else:
                            assert original.dtype.str == actual.dtype.str
                            assert original.tobytes() == actual.tobytes()


def test_existing_target_survives_exact_refusal(tmp_path):
    source, zarr = _source(tmp_path)
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle")
    target = tmp_path / "target.ms"
    target.mkdir()
    (target / "sentinel").write_text("intact")
    with pytest.raises(FileExistsError):
        materialize_exact_native(zarr, target, bundle)
    assert (target / "sentinel").read_text() == "intact"


def test_exact_mode_rejects_mapped_only_options(tmp_path):
    source, zarr = _source(tmp_path)
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle")
    for options in (
        {"weight_spectrum": False},
        {"weight_spectrum": True},
        {"rowchunk": 8},
        {"rowchunk": 64},
        {"overwrite": True},
    ):
        with pytest.raises(ValueError):
            to_msv2(
                str(zarr),
                str(tmp_path / "target.ms"),
                fidelity="exact-native-v1",
                preservation=str(bundle),
                **options,
            )
    assert not (tmp_path / "target.ms").exists()


def test_exact_cli_reports_refusal_as_click_error(tmp_path):
    _, zarr = _source(tmp_path)
    target = tmp_path / "target.ms"
    result = CliRunner().invoke(
        cli, ["materialize", str(zarr), str(target), "--fidelity", "exact-native-v1"]
    )
    assert result.exit_code != 0
    assert "Error: exact-native-v1 requires preservation" in result.output
    assert not target.exists()


def test_failed_writer_removes_private_candidate(tmp_path):
    source, zarr = _source(tmp_path)
    bundle = capture_native_preservation(source, zarr, tmp_path / "bundle")
    target = tmp_path / "target.ms"

    class FailedWriter:
        def write(self, plan, destination):
            destination.mkdir()
            (destination / "partial").write_text("partial")
            raise RuntimeError("write failure")

    with pytest.raises(RuntimeError, match="write failure"):
        materialize_exact_native(zarr, target, bundle, writer=FailedWriter())
    assert not target.exists()
    assert not list(tmp_path.glob(".target.ms.*"))
