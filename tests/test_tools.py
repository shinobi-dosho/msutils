"""Tests for the new everyday tools: column removal, subsetting, averaging,
flag versioning, disk usage, conformance checking and TaQL passthrough.
"""

import os

import numpy
import pytest

import msutils
from msutils.diagnostics import check, du, taql
from msutils.flags import flag_backup, flag_delete, flag_restore, flag_versions
from msutils.subset import subset

# --------------------------------------------------------------------------
# delcol / renamecol


def test_delcol_removes_columns(ms):
    msutils.addcol(ms, "CORRECTED_DATA", clone="DATA")
    assert "CORRECTED_DATA" in msutils.msinfo(ms, level="meta").column_names

    assert msutils.delcol(ms, "CORRECTED_DATA") == ["CORRECTED_DATA"]
    assert "CORRECTED_DATA" not in msutils.msinfo(ms, level="meta").column_names


def test_delcol_multiple_and_missing(ms):
    msutils.addcol(ms, "MODEL_DATA", clone="DATA")
    removed = msutils.delcol(ms, "MODEL_DATA", "NOT_THERE")
    assert removed == ["MODEL_DATA"]  # missing one skipped, not fatal


def test_delcol_protects_required_columns(ms):
    with pytest.raises(ValueError, match="required by the MSv2 standard"):
        msutils.delcol(ms, "TIME")
    # still there
    assert "TIME" in msutils.msinfo(ms, level="meta").column_names


def test_delcol_force_overrides(ms):
    # FLAG_CATEGORY is required but unused; forcing is the documented escape
    msutils.delcol(ms, "ARRAY_ID", force=True)
    assert "ARRAY_ID" not in msutils.msinfo(ms, level="meta").column_names


def test_delcol_needs_arguments(ms):
    with pytest.raises(ValueError, match="no columns given"):
        msutils.delcol(ms)


def test_renamecol(ms):
    # copycol rather than addcol: an added-but-uninitialised column has no
    # defined contents, so there would be nothing meaningful to compare.
    msutils.copycol(ms, "DATA", "MODEL_DATA")
    original = _getcol(ms, "MODEL_DATA")
    msutils.renamecol(ms, "MODEL_DATA", "OLD_MODEL")

    names = msutils.msinfo(ms, level="meta").column_names
    assert "OLD_MODEL" in names and "MODEL_DATA" not in names
    assert numpy.array_equal(_getcol(ms, "OLD_MODEL"), original)


def test_renamecol_validation(ms):
    msutils.addcol(ms, "MODEL_DATA", clone="DATA")
    with pytest.raises(ValueError, match="has no column"):
        msutils.renamecol(ms, "NOPE", "X")
    with pytest.raises(ValueError, match="already has a column"):
        msutils.renamecol(ms, "MODEL_DATA", "DATA")
    with pytest.raises(ValueError, match="cannot be renamed"):
        msutils.renamecol(ms, "TIME", "T")


# --------------------------------------------------------------------------
# subset


def test_subset_by_field(ms, tmp_path):
    out = str(tmp_path / "one_field.ms")
    subset(ms, out, fields=["DEEP_2"])

    info = msutils.msinfo(out)
    assert info.nrows == 72  # 1 field of 3
    assert {s.field_id for s in info.scans} == {2}


def test_subset_preserves_original_ids(ms, tmp_path):
    """Unlike CASA split, ids are not renumbered."""
    out = str(tmp_path / "kept_ids.ms")
    subset(ms, out, fields=["DEEP_2"])

    info = msutils.msinfo(out)
    # the FIELD subtable still carries all three rows, and the kept field is
    # still id 2 -- not renumbered to 0
    assert info.nfields == 3
    assert info.fields[2].name == "DEEP_2"
    assert {s.field_id for s in info.scans} == {2}


def test_subset_by_spw(ms, tmp_path):
    out = str(tmp_path / "one_spw.ms")
    subset(ms, out, spws=[1])
    info = msutils.msinfo(out)
    assert info.nrows == 108  # half the rows
    assert {sid for s in info.scans for sid in s.spw_ids} == {1}


def test_subset_by_scan_and_antenna(ms, tmp_path):
    out = str(tmp_path / "narrow.ms")
    subset(ms, out, scans=[1], antennas=["m000"])
    info = msutils.msinfo(out)
    assert info.nscans == 1
    pairs = taql("SELECT DISTINCT ANTENNA1, ANTENNA2 FROM $1", out)
    assert all(0 in (a, b) for a, b in zip(pairs["ANTENNA1"], pairs["ANTENNA2"], strict=True))


def test_subset_with_extra_taql(ms, tmp_path):
    out = str(tmp_path / "taql.ms")
    subset(ms, out, taql="ANTENNA1==0 AND ANTENNA2==1")
    assert msutils.msinfo(out).nrows == 36


def test_subset_output_is_valid(ms, tmp_path):
    out = str(tmp_path / "valid.ms")
    subset(ms, out, fields=[0])
    assert check(out).ok


def test_subset_refuses_to_clobber(ms, tmp_path):
    out = str(tmp_path / "twice.ms")
    subset(ms, out, fields=[0])
    with pytest.raises(FileExistsError, match="already exists"):
        subset(ms, out, fields=[0])
    subset(ms, out, fields=[1], overwrite=True)  # explicit overwrite is fine
    assert msutils.msinfo(out).nrows == 72


def test_subset_empty_selection_raises(ms, tmp_path):
    with pytest.raises(ValueError, match="matched no rows"):
        subset(ms, str(tmp_path / "empty.ms"), taql="ANTENNA1==99")


def test_subset_unknown_field(ms, tmp_path):
    with pytest.raises(ValueError, match="unknown field"):
        subset(ms, str(tmp_path / "x.ms"), fields=["NO_SUCH"])


# --------------------------------------------------------------------------
# average (needs codex-africanus)


@pytest.fixture
def africanus():
    return pytest.importorskip(
        "africanus.averaging", reason="codex-africanus not installed (msutils[average])"
    )


def test_average_time_and_channel(ms, tmp_path, africanus):
    out = str(tmp_path / "avg.ms")
    msutils.average(ms, out, time_bin=24.0, chan_bin=2)

    before, after = msutils.msinfo(ms), msutils.msinfo(out)
    # 3 timestamps of 8 s collapse into one bin
    assert after.nrows == before.nrows / 3
    assert after.spws[0].num_chan == 2
    assert after.nscans == before.nscans  # scans are never merged
    assert check(out).ok


def test_average_rebuilds_the_channel_grid(ms, tmp_path, africanus):
    out = str(tmp_path / "avg_freq.ms")
    msutils.average(ms, out, chan_bin=2)

    spw = msutils.msinfo(out).spws[0]
    # channels 0,1 (1400, 1410 MHz) average to 1405 MHz with double the width
    assert spw.chan_freq == [1405e6, 1425e6]
    assert spw.chan_width == [20e6, 20e6]
    assert spw.total_bandwidth == 40e6


def test_average_channel_only_keeps_all_rows(ms, tmp_path, africanus):
    out = str(tmp_path / "chan_only.ms")
    msutils.average(ms, out, chan_bin=4)
    after = msutils.msinfo(out)
    assert after.nrows == msutils.msinfo(ms).nrows
    assert after.spws[0].num_chan == 1


def test_average_with_selection(ms, tmp_path, africanus):
    out = str(tmp_path / "avg_sel.ms")
    msutils.average(ms, out, chan_bin=2, fields=["DEEP_2"])
    assert {s.field_id for s in msutils.msinfo(out).scans} == {2}


def test_average_validates_arguments(ms, tmp_path, africanus):
    with pytest.raises(ValueError, match="chan_bin must be"):
        msutils.average(ms, str(tmp_path / "a.ms"), chan_bin=0)
    with pytest.raises(ValueError, match="time_bin must be"):
        msutils.average(ms, str(tmp_path / "b.ms"), time_bin=0)
    with pytest.raises(ValueError, match="has no column"):
        msutils.average(ms, str(tmp_path / "c.ms"), datacolumn="NOPE")


def test_average_reconciles_inconsistent_flags(ms, tmp_path, africanus):
    """FLAG_ROW disagreeing with FLAG must not abort the averaging.

    africanus raises "flag_row and flag arrays mismatch" for such input, and
    the synthetic MS is exactly that case: whole rows flagged in FLAG with
    FLAG_ROW left False.
    """
    from casacore.tables import table

    with table(ms, readonly=False, ack=False) as tab:
        flag_row = numpy.zeros(tab.nrows(), bool)
        flag_row[1] = True  # row-flagged, but FLAG says otherwise
        tab.putcol("FLAG_ROW", flag_row)

    out = str(tmp_path / "mixed_flags.ms")
    msutils.average(ms, out, chan_bin=2)
    assert msutils.msinfo(out).nrows > 0


# --------------------------------------------------------------------------
# flag versions


def test_flag_backup_and_restore(ms):
    original = _getcol(ms, "FLAG")
    flag_backup(ms, name="pristine", description="before anything")

    from casacore.tables import table

    with table(ms, readonly=False, ack=False) as tab:
        tab.putcol("FLAG", numpy.ones_like(original))
    assert _getcol(ms, "FLAG").all()

    flag_restore(ms, "pristine")
    assert numpy.array_equal(_getcol(ms, "FLAG"), original)


def test_flag_versions_listing(ms):
    flag_backup(ms, name="first", description="one")
    flag_backup(ms, name="second", description="two")

    versions = {v.name: v for v in flag_versions(ms)}
    assert set(versions) == {"first", "second"}
    assert versions["first"].description == "one"
    assert versions["second"].nrows == 216


def test_flag_versions_live_beside_the_ms(ms):
    flag_backup(ms, name="v1")
    assert os.path.isdir(ms + ".flagversions")
    assert os.path.isdir(os.path.join(ms + ".flagversions", "flags.v1"))


def test_flag_backup_default_name_is_a_timestamp(ms):
    version = flag_backup(ms)
    assert version.name.startswith("backup_")
    assert flag_versions(ms)[0].name == version.name


def test_flag_backup_refuses_to_clobber(ms):
    flag_backup(ms, name="v1")
    with pytest.raises(FileExistsError, match="already exists"):
        flag_backup(ms, name="v1")
    flag_backup(ms, name="v1", overwrite=True)  # explicit is fine


def test_flag_delete(ms):
    flag_backup(ms, name="temp")
    flag_delete(ms, "temp")
    assert flag_versions(ms) == []
    with pytest.raises(FileNotFoundError):
        flag_delete(ms, "temp")


def test_flag_restore_unknown_version(ms):
    with pytest.raises(FileNotFoundError, match="no flag version"):
        flag_restore(ms, "never_saved")


def test_flag_restore_rejects_row_count_mismatch(ms, tmp_path):
    flag_backup(ms, name="v1")
    smaller = str(tmp_path / "smaller.ms")
    subset(ms, smaller, fields=[0])
    # move the version next to the smaller MS to simulate a stale backup
    import shutil

    shutil.copytree(ms + ".flagversions", smaller + ".flagversions")
    with pytest.raises(ValueError, match="the MS has changed"):
        flag_restore(smaller, "v1")


def test_flag_version_name_validation(ms):
    with pytest.raises(ValueError, match="invalid flag version name"):
        flag_backup(ms, name="../escape")
    with pytest.raises(ValueError, match="invalid flag version name"):
        flag_backup(ms, name="")


def test_no_versions_for_fresh_ms(ms):
    assert flag_versions(ms) == []


# --------------------------------------------------------------------------
# du


def test_du_accounts_for_the_bytes(ms):
    usage = du(ms)
    assert usage.total > 0
    assert usage.storage
    # main table + subtables should account for essentially the whole tree
    accounted = usage.main_table + sum(usage.subtables.values())
    assert accounted > 0.9 * usage.total
    assert "ANTENNA" in usage.subtables


def test_du_lists_columns_per_manager(ms):
    usage = du(ms)
    columns = {c for group in usage.storage for c in group.columns}
    assert {"DATA", "TIME", "FLAG"} <= columns


def test_du_render_and_dict(ms):
    usage = du(ms)
    assert "Disk usage" in usage.render()
    assert str(usage) == usage.render()
    assert usage.to_dict()["total"] == usage.total


# --------------------------------------------------------------------------
# check


def test_check_passes_on_a_valid_ms(ms):
    report = check(ms)
    assert report.ok
    assert report.errors == []
    assert "OK" in report.render()


def test_check_reports_missing_column(ms):
    msutils.delcol(ms, "ARRAY_ID", force=True)
    report = check(ms)
    assert not report.ok
    assert any("ARRAY_ID" in e for e in report.errors)
    assert "ERROR" in report.render()


def test_check_reports_dangling_reference(ms):
    from casacore.tables import table

    with table(ms, readonly=False, ack=False) as tab:
        field = tab.getcol("FIELD_ID")
        field[0] = 99  # no such FIELD row
        tab.putcol("FIELD_ID", field)

    report = check(ms)
    assert not report.ok
    assert any("FIELD_ID references" in e for e in report.errors)


def test_check_warns_about_missing_data_column(ms):
    msutils.delcol(ms, "DATA")
    report = check(ms)
    assert any("no DATA or FLOAT_DATA" in w for w in report.warnings)


def test_check_to_dict(ms):
    assert check(ms).to_dict()["ok"] is True


# --------------------------------------------------------------------------
# taql passthrough


def test_taql_returns_arrays(ms):
    result = taql("SELECT DISTINCT FIELD_ID FROM $1", ms)
    assert sorted(result["FIELD_ID"].tolist()) == [0, 1, 2]


def test_taql_aggregation(ms):
    result = taql("SELECT GCOUNT() AS N FROM $1", ms)
    assert int(result["N"][0]) == 216


def test_taql_column_selection(ms):
    result = taql("SELECT FROM $1", ms, columns=["TIME"])
    assert list(result) == ["TIME"]
    assert len(result["TIME"]) == 216


def _getcol(msname, col):
    from casacore.tables import table

    with table(msname, ack=False) as tab:
        return tab.getcol(col)
