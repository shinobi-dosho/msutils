"""Tests for msutils.msinfo and the MSInfo model.

The synthetic MS from ``msfactory`` is built to have exactly the properties
the old ``summary()`` got wrong -- several fields with distinct intents,
several scans per field, several SPWs, and a real DATA_DESCRIPTION mapping --
so these assert the corrected behaviour rather than just "it runs".
"""
import json
import math

import pytest

import msutils
from msutils.info import (Registry, detect_format, format_dec, format_ra,
                          geodetic_to_itrf, itrf_to_geodetic)
from msutils.info._model import SCHEMA_VERSION, Field, format_duration
from msutils.info._render import format_bytes, format_frequency


@pytest.fixture(scope="module")
def info(base_ms):
    return msutils.msinfo(base_ms, level="data")


# --------------------------------------------------------------------------
# top-level facts

def test_basic_counts(info):
    # 3 fields x 2 scans x 3 times x 2 spws x 6 baselines
    assert info.nrows == 3 * 2 * 3 * 2 * 6
    assert info.format == "MSv2"
    assert (info.nantennas, info.nfields, info.nspws, info.nscans) == (4, 3, 2, 6)
    assert info.nbaselines == 6          # 4 antennas -> 6 cross-correlations
    assert info.nchan_total == 8         # 2 SPWs x 4 channels
    assert info.size_bytes > 0


def test_observation_metadata(info):
    assert info.observation.telescope == "MEERKAT"
    assert info.observation.observer == "msutils-tests"
    assert info.observation.project == "TEST-2024-01"
    # T0 is MJD 60310 = 2024-01-01
    assert info.start_utc.startswith("2024-01-01T00:00:00")


def test_integration_time_and_baselines(info):
    assert info.integration_times == [8.0]
    # max_baseline is the physical antenna separation; max_uv_distance is the
    # projected uv extent. The old MAXBL conflated the two, reporting a
    # uv distance under a baseline-length name.
    assert info.max_baseline > 0
    assert info.max_uv_distance > 0


def test_flag_counts(info):
    # every third row flagged, over 4 chan x 4 corr
    assert info.nvisibilities == info.nrows * 4 * 4
    assert info.nflagged == pytest.approx(info.nvisibilities / 3, rel=0.01)
    assert info.flagged_fraction == pytest.approx(1 / 3, rel=0.01)


# --------------------------------------------------------------------------
# the things summary() got wrong

def test_scan_duration_includes_final_integration(info):
    # 3 timestamps x 8 s. max(TIME) - min(TIME) is only 16 s; the scan really
    # covers 24 s. This is the bug that made summary() understate every scan.
    for scan in info.scans:
        assert scan.duration == pytest.approx(24.0)


def test_field_period_sums_corrected_scan_durations(info):
    field = info.fields["PKS1934-638"]
    total = sum(info.scans[n].duration for n in field.scan_numbers)
    assert total == pytest.approx(48.0)


def test_intents_are_resolved_per_field_not_dumped_raw(info):
    # Each field gets its own STATE row / intent. summary() returned the raw
    # OBS_MODE column for every field.
    assert info.fields["PKS1934-638"].intents == [
        "CALIBRATE_BANDPASS#ON_SOURCE", "CALIBRATE_FLUX#ON_SOURCE"]
    assert info.fields["J0217+0144"].intents == ["CALIBRATE_PHASE#ON_SOURCE"]
    assert info.fields["DEEP_2"].intents == ["OBSERVE_TARGET#ON_SOURCE"]


def test_scans_carry_their_own_intents(info):
    for scan in info.scans:
        assert scan.intents == info.fields[scan.field_id].intents


def test_chan_width_is_available(info):
    # summary()'s hard-coded SPW column list omitted CHAN_WIDTH entirely.
    assert info.spws[0].chan_width == [10e6] * 4
    assert info.spws[0].total_bandwidth == 40e6


def test_data_description_mapping_is_explicit(info):
    assert len(info.data_descriptions) == 2
    assert [d.spw_id for d in info.data_descriptions] == [0, 1]
    assert all(d.pol_id == 0 for d in info.data_descriptions)


def test_polarization_setup(info):
    pol = info.polarizations[0]
    assert pol.num_corr == 4
    assert pol.corr_labels == ["XX", "XY", "YX", "YY"]
    assert pol.name == "XX,XY,YX,YY"


# --------------------------------------------------------------------------
# addressability

def test_fields_addressable_by_id_and_name(info):
    assert info.fields[0] is info.fields["PKS1934-638"]
    assert info.fields.names == ["PKS1934-638", "J0217+0144", "DEEP_2"]
    assert info.fields.ids == [0, 1, 2]


def test_antennas_addressable_by_name(info):
    assert info.antennas["m002"].id == 2
    assert info.antennas[2].dish_diameter == 13.5


def test_scans_keyed_by_scan_number(info):
    assert info.scans[1].number == 1
    assert info.scans[1].field_name == "PKS1934-638"
    assert [s.number for s in info.scans] == [1, 2, 3, 4, 5, 6]


def test_columns_addressable_by_name(info):
    data = info.columns["DATA"]
    assert data.shape == [4, 4]
    assert data.dtype == "complex"
    assert data.nbytes == 8 * 16 * info.nrows
    assert info.columns["ANTENNA1"].is_scalar


def test_unknown_key_raises_with_context(info):
    with pytest.raises(KeyError, match="no entry named"):
        info.fields["NOT_A_FIELD"]
    with pytest.raises(KeyError, match="no entry with id"):
        info.fields[99]


def test_registry_is_a_sequence():
    reg = Registry([Field(id=5, name="a"), Field(id=7, name="b")])
    assert len(reg) == 2
    assert [f.name for f in reg] == ["a", "b"]
    # integer keys are ids, not positions
    assert reg[7].name == "b"
    assert reg[0:1][0].name == "a"


# --------------------------------------------------------------------------
# serialisation and rendering

def test_to_dict_is_json_serialisable_and_versioned(info):
    payload = json.loads(info.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["observation"]["telescope"] == "MEERKAT"
    assert len(payload["fields"]) == 3
    # derived properties are included, not just stored fields
    assert payload["fields"][0]["ra_hms"]
    assert payload["spws"][0]["centre_freq"] == pytest.approx(1.415e9)


def test_save_writes_json(info, tmp_path):
    out = tmp_path / "info.json"
    info.save(str(out))
    assert json.loads(out.read_text())["nrows"] == info.nrows


def test_msinfo_outfile_argument(base_ms, tmp_path):
    out = tmp_path / "direct.json"
    msutils.msinfo(base_ms, outfile=str(out))
    assert json.loads(out.read_text())["format"] == "MSv2"


def test_render_mentions_the_essentials(info):
    text = info.render()
    for expected in ("MEERKAT", "PKS1934-638", "DEEP_2", "XX,XY,YX,YY",
                     "Fields (3)", "Spectral windows (2)"):
        assert expected in text
    assert str(info) == text


def test_verbose_render_adds_detail(info):
    plain, verbose = info.render(), info.render(verbose=True)
    assert len(verbose) > len(plain)
    for expected in ("Scans (6)", "Antennas (4)", "Columns (22)", "m003"):
        assert expected in verbose


# --------------------------------------------------------------------------
# levels

def test_meta_level_skips_the_main_table(single_spw_ms):
    info = msutils.msinfo(single_spw_ms, level="meta")
    assert info.nrows > 0                 # from table.nrows(), not a scan
    assert info.nscans == 0               # no main-table pass, so no scans
    assert info.nfields == 1              # subtables are still read
    assert info.nflagged is None
    assert info.max_baseline > 0          # antenna geometry, no main table


def test_full_level_does_not_read_bulk_columns(base_ms):
    info = msutils.msinfo(base_ms, level="full")
    assert info.nscans == 6
    # UVW and FLAG are only read at level="data"
    assert info.nflagged is None
    assert info.flagged_fraction is None
    assert info.max_uv_distance is None
    # max_baseline is antenna geometry, so it is available regardless
    assert info.max_baseline > 0


def test_bad_level_rejected(base_ms):
    with pytest.raises(ValueError, match="level must be one of"):
        msutils.msinfo(base_ms, level="everything")


def test_unflagged_ms(single_spw_ms):
    info = msutils.msinfo(single_spw_ms, level="data")
    assert info.nflagged == 0
    assert info.flagged_fraction == 0.0
    assert info.nspws == 1


# --------------------------------------------------------------------------
# format detection

def test_detect_format_msv2(base_ms):
    assert detect_format(base_ms) == "MSv2"


def test_detect_format_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_format(str(tmp_path / "nope.ms"))


def test_detect_format_rejects_plain_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="neither a casacore MSv2"):
        detect_format(str(plain))


def test_detect_format_zarr(tmp_path):
    ps = tmp_path / "obs.ps"
    (ps / "ms_0").mkdir(parents=True)
    (ps / "ms_0" / ".zgroup").write_text("{}")
    assert detect_format(str(ps)) == "MSv4"


# --------------------------------------------------------------------------
# formatting helpers

def test_coordinate_formatting():
    assert format_ra(0.0) == "00h00m00.000"
    assert format_ra(math.pi) == "12h00m00.000"
    assert format_dec(0.0) == "+00d00m00.00"
    assert format_dec(-math.pi / 6).startswith("-30d")


def test_duration_formatting():
    assert format_duration(8) == "8.0s"
    assert format_duration(90) == "1m30s"
    assert format_duration(3725) == "1h02m05s"


def test_size_and_frequency_formatting():
    assert format_bytes(None) == "-"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KiB"
    assert format_frequency(1.4e9) == "1.4 GHz"
    assert format_frequency(10e6) == "10 MHz"
    assert format_frequency(None) == "-"


def test_itrf_geodetic_roundtrip(info):
    for antenna in info.antennas:
        lon, lat, height = itrf_to_geodetic(*antenna.position)
        x, y, z = geodetic_to_itrf(lon, lat, height)
        assert (x, y, z) == pytest.approx(antenna.position, abs=1e-6)
    # the synthetic array sits at roughly MeerKAT's location
    assert info.antennas[0].latitude == pytest.approx(-30.71, abs=0.05)
    assert info.antennas[0].longitude == pytest.approx(21.44, abs=0.05)
