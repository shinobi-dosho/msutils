"""Correlation selections change cells and their polarization metadata together."""

from functools import partial
from importlib import import_module

import numpy as np
import pytest
from casacore.tables import makearrcoldesc, maketabdesc, table
from click.testing import CliRunner
from msfactory import make_ms

import msutils
from msutils.cli import cli

_COLUMNS = (
    "DATA",
    "MODEL_DATA",
    "CORRECTED_DATA",
    "FLOAT_DATA",
    "LAG_DATA",
    "FLAG",
    "FLAG_CATEGORY",
    "WEIGHT",
    "SIGMA",
    "WEIGHT_SPECTRUM",
    "SIGMA_SPECTRUM",
)


@pytest.fixture
def corr_ms(tmp_path):
    path = make_ms(tmp_path / "corr.ms", add_weight_spectrum=True, flag_every=None)
    with table(path, readonly=False, ack=False) as tab:
        for column in ("MODEL_DATA", "CORRECTED_DATA", "LAG_DATA"):
            tab.addcols(maketabdesc(makearrcoldesc(column, 0j, shape=[4, 4], valuetype="complex")))
            tab.putcol(column, tab.getcol("DATA") * (2 if column == "MODEL_DATA" else 3))
        tab.addcols(maketabdesc(makearrcoldesc("FLOAT_DATA", 0.0, shape=[4, 4])))
        tab.putcol("FLOAT_DATA", tab.getcol("DATA").real)
        tab.addcols(maketabdesc(makearrcoldesc("SIGMA_SPECTRUM", 0.0, shape=[4, 4])))
        nrows = tab.nrows()
        values = np.arange(nrows * 16, dtype=np.float32).reshape(nrows, 4, 4) + 1
        tab.putcol("WEIGHT_SPECTRUM", values)
        tab.putcol("SIGMA_SPECTRUM", 1 / np.sqrt(values))
        tab.putcol("WEIGHT", values[:, 0])
        tab.putcol("SIGMA", 1 / np.sqrt(values[:, 0]))
        tab.putcol("FLAG", values.astype(int) % 5 == 0)
        tab.putcolkeyword("DATA", "QuantumUnits", ["Jy"])
        tab.putcolkeyword("FLAG_CATEGORY", "CATEGORY", ["one", "two", "three"])
        # Three categories distinguish the category axis from correlation.
        for row in range(0, nrows, 2):
            categories = np.arange(48).reshape(3, 4, 4) % (row + 2) == 0
            tab.putcell("FLAG_CATEGORY", row, categories)
    return path


@pytest.mark.parametrize(
    "corrs,indices", [(["XX", "YY"], [0, 3]), (["3", 1], [3, 1]), (["yy"], [3])]
)
def test_subset_slices_every_column_and_preserves_rows(
    corr_ms, tmp_path, monkeypatch, corrs, indices
):
    module = import_module("msutils.subset")
    monkeypatch.setattr(
        module, "_copy_correlations", partial(module._copy_correlations, rowchunk=2)
    )
    out = str(tmp_path / "selected.ms")
    msutils.subset(corr_ms, out, corrs=corrs, fields=[2], spws=[1], reindex=True)
    with table(corr_ms, ack=False) as source, table(out, ack=False) as target:
        kept = np.flatnonzero(
            (source.getcol("FIELD_ID") == 2) & (source.getcol("DATA_DESC_ID") == 1)
        )
        np.testing.assert_array_equal(target.getcol("TIME"), source.getcol("TIME")[kept])
        np.testing.assert_array_equal(target.getcol("UVW"), source.getcol("UVW")[kept])
        for column in _COLUMNS:
            for row, original in enumerate(kept):
                assert target.iscelldefined(column, row) == source.iscelldefined(
                    column, int(original)
                )
                if source.iscelldefined(column, int(original)):
                    np.testing.assert_array_equal(
                        target.getcell(column, row),
                        np.take(source.getcell(column, int(original)), indices, axis=-1),
                    )
        assert target.getcolkeywords("DATA") == source.getcolkeywords("DATA")
        assert list(target.getcolkeyword("FLAG_CATEGORY", "CATEGORY")) == ["one", "two", "three"]
    with (
        table(corr_ms + "::POLARIZATION", ack=False) as source,
        table(out + "::POLARIZATION", ack=False) as target,
    ):
        assert target.getcell("NUM_CORR", 0) == len(indices)
        np.testing.assert_array_equal(
            target.getcell("CORR_TYPE", 0), source.getcell("CORR_TYPE", 0)[indices]
        )
        np.testing.assert_array_equal(
            target.getcell("CORR_PRODUCT", 0), source.getcell("CORR_PRODUCT", 0)[indices]
        )
    assert msutils.msinfo(corr_ms, level="meta").polarizations[0].num_corr == 4
    assert msutils.check(out).ok


def _second_setup(path, codes):
    with table(path + "::POLARIZATION", readonly=False, ack=False) as tab:
        tab.addrows(1)
        tab.putcell("NUM_CORR", 1, len(codes))
        tab.putcell("CORR_TYPE", 1, np.asarray(codes))
        products = [divmod((code - 5) % 4, 2) for code in codes]
        tab.putcell("CORR_PRODUCT", 1, np.asarray(products))
    with table(path + "::DATA_DESCRIPTION", readonly=False, ack=False) as tab:
        # DDID 0 references SPW 1 / POL 1, DDID 1 references SPW 0 / POL 0.
        tab.putcol("POLARIZATION_ID", [1, 0])
        tab.putcol("SPECTRAL_WINDOW_ID", [1, 0])


@pytest.mark.parametrize("averaged", [False, True])
def test_resolves_names_per_setup(corr_ms, tmp_path, averaged):
    if averaged:
        pytest.importorskip("africanus.averaging")
    _second_setup(corr_ms, [12, 11, 10, 9])
    out = str(tmp_path / "setups.ms")
    # Unit channel bins let us compare cells directly, while exercising the writer.
    msutils.subset(corr_ms, out, corrs=["XX", "YY"], chan_bin=1 if averaged else None)
    with table(corr_ms, ack=False) as source, table(out, ack=False) as target:
        # Averaging writes groups in sorted order rather than source row order.
        for ddid, indices in ((0, [3, 0]), (1, [0, 3])):
            with (
                source.query(
                    f"DATA_DESC_ID=={ddid}",
                    sortlist="FIELD_ID, SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2",
                ) as src,
                target.query(
                    f"DATA_DESC_ID=={ddid}",
                    sortlist="FIELD_ID, SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2",
                ) as dst,
            ):
                expected = src.getcol("DATA")[:, :, indices]
                flags = src.getcol("FLAG")[:, :, indices]
                np.testing.assert_allclose(dst.getcol("DATA")[~flags], expected[~flags], rtol=2e-7)
                np.testing.assert_array_equal(dst.getcol("FLAG"), flags)
    info = msutils.msinfo(out, level="meta")
    assert [p.corr_labels for p in info.polarizations] == [["XX", "YY"], ["XX", "YY"]]


@pytest.mark.parametrize(
    "corrs,match",
    [
        (["RR"], "unknown correlation"),
        ([-1], "index -1"),
        ([4], "index 4"),
        (["XX", 0], "duplicate"),
    ],
)
def test_invalid_selection_leaves_existing_output(corr_ms, tmp_path, corrs, match):
    out = tmp_path / "existing.ms"
    out.mkdir()
    sentinel = out / "sentinel"
    sentinel.write_text("original")
    with pytest.raises(ValueError, match=match):
        msutils.subset(corr_ms, str(out), corrs=corrs, overwrite=True)
    assert sentinel.read_text() == "original"


def test_missing_name_only_matters_for_used_setups(corr_ms, tmp_path):
    _second_setup(corr_ms, [5, 6, 7, 8])
    out = str(tmp_path / "linear.ms")
    msutils.subset(corr_ms, out, spws=[0], corrs=["XX", "YY"])
    assert msutils.msinfo(out, level="meta").polarizations[1].corr_labels == [
        "RR",
        "RL",
        "LR",
        "LL",
    ]
    with pytest.raises(ValueError, match="polarization 1"):
        msutils.subset(corr_ms, str(tmp_path / "mixed.ms"), corrs=["XX", "YY"])
    assert not (tmp_path / "mixed.ms").exists()


@pytest.mark.parametrize("command", ["subset", "average", "subset-average"])
def test_cli_selection_and_weighted_averaging(corr_ms, tmp_path, command):
    averaging = command != "subset"
    if averaging:
        pytest.importorskip("africanus.averaging")
    out = str(tmp_path / "cli.ms")
    args = [
        "average" if command == "average" else "subset",
        corr_ms,
        out,
        "--corr",
        "YY",
        "--corr",
        "XX",
    ]
    if averaging:
        args += ["--chan-bin", "2", "--time-bin", "24"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert msutils.msinfo(out).polarizations[0].corr_labels == ["YY", "XX"]
    if averaging:
        reference = str(tmp_path / "reference.ms")
        msutils.average(corr_ms, reference, time_bin=24, chan_bin=2)
        with table(reference, ack=False) as src, table(out, ack=False) as dst:
            for column in ("DATA", "FLAG", "WEIGHT", "SIGMA", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM"):
                np.testing.assert_allclose(dst.getcol(column), src.getcol(column)[..., [3, 0]])


def test_cli_reports_bad_correlation(corr_ms, tmp_path):
    result = CliRunner().invoke(cli, ["subset", corr_ms, str(tmp_path / "bad.ms"), "--corr", "ZZ"])
    assert result.exit_code == 1
    assert "unknown correlation 'ZZ'" in result.output


def test_tiled_visibility_column_can_shrink(corr_ms, tmp_path):
    with table(corr_ms, readonly=False, ack=False) as tab:
        values = tab.getcol("DATA")
        desc = tab.getcoldesc("DATA")
        tab.removecols("DATA")
        desc["dataManagerType"] = "TiledColumnStMan"
        desc["dataManagerGroup"] = "TiledData"
        tab.addcols(
            {"DATA": desc},
            {
                "TYPE": "TiledColumnStMan",
                "NAME": "TiledData",
                "SPEC": {"DEFAULTTILESHAPE": [4, 4, 16]},
                "COLUMNS": ["DATA"],
            },
        )
        tab.putcol("DATA", values)
    out = str(tmp_path / "tiled.ms")
    msutils.subset(corr_ms, out, corrs=["XX", "YY"])
    with table(out, ack=False) as tab:
        np.testing.assert_array_equal(tab.getcol("DATA"), values[..., [0, 3]])


@pytest.mark.parametrize("averaged", [False, True])
def test_variable_correlation_counts(corr_ms, tmp_path, averaged):
    if averaged:
        pytest.importorskip("africanus.averaging")
    _second_setup(corr_ms, [12, 9])
    with table(corr_ms, readonly=False, ack=False) as tab:
        ddids = tab.getcol("DATA_DESC_ID")
        for column in (c for c in _COLUMNS if c != "FLAG_CATEGORY"):
            values = tab.getcol(column)
            desc = tab.getcoldesc(column)
            tab.removecols(column)
            desc.pop("shape", None)
            desc["option"] = 0
            tab.addcols({column: desc})
            for row, ddid in enumerate(ddids):
                value = np.take(values[row], [3, 0], axis=-1) if ddid == 0 else values[row]
                tab.putcell(column, row, value)
        for row in np.flatnonzero(ddids == 0):
            if tab.iscelldefined("FLAG_CATEGORY", int(row)):
                tab.putcell(
                    "FLAG_CATEGORY",
                    int(row),
                    np.take(tab.getcell("FLAG_CATEGORY", int(row)), [3, 0], axis=-1),
                )
    out = str(tmp_path / "variable.ms")
    msutils.subset(corr_ms, out, corrs=["YY", "XX"], chan_bin=1 if averaged else None)
    with table(corr_ms, ack=False) as source, table(out, ack=False) as target:
        for ddid, indices in ((0, [0, 1]), (1, [3, 0])):
            with (
                source.query(
                    f"DATA_DESC_ID=={ddid}",
                    sortlist="FIELD_ID, SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2",
                ) as src,
                target.query(
                    f"DATA_DESC_ID=={ddid}",
                    sortlist="FIELD_ID, SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2",
                ) as dst,
            ):
                expected = np.take(src.getcol("DATA"), indices, axis=-1)
                flags = np.take(src.getcol("FLAG"), indices, axis=-1)
                np.testing.assert_allclose(dst.getcol("DATA")[~flags], expected[~flags], rtol=2e-7)
                np.testing.assert_array_equal(dst.getcol("FLAG"), flags)
    assert [p.corr_labels for p in msutils.msinfo(out).polarizations] == [
        ["YY", "XX"],
        ["YY", "XX"],
    ]


def test_averaging_reconciles_flags_after_correlation_selection(ms, tmp_path):
    pytest.importorskip("africanus.averaging")
    with table(ms, readonly=False, ack=False) as tab:
        flags = np.zeros_like(tab.getcol("FLAG"))
        flags[..., [0, 3]] = True
        tab.putcol("FLAG", flags)
        tab.putcol("FLAG_ROW", np.zeros(tab.nrows(), bool))
    out = str(tmp_path / "flagged.ms")
    msutils.average(ms, out, corrs=["XX", "YY"], chan_bin=2)
    with table(out, ack=False) as tab:
        assert tab.nrows() > 0
        assert tab.getcol("FLAG").all()
        assert tab.getcol("FLAG_ROW").all()
