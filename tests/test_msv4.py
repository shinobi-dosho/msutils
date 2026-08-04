"""MSv4-schema reading, engine selection, and MSv2 -> MSv4 conversion.

The point of these tests is not that xradio or xarray-ms work, but that *the
same MSInfo* comes out however the data is read. The MSv4 reader folds an
n-dimensional, pre-partitioned layout back into the shared model -- converting
unix time to MJD and synthesising the DATA_DESCRIPTION indirection MSv4 lacks
-- and it must do that identically whether the tree came from stored zarr,
from xradio, or from xarray-ms viewing a plain MSv2.
"""
import pytest

import msutils

xradio = pytest.importorskip(
    "xradio.measurement_set",
    reason="xradio not installed (msutils[convert])")


@pytest.fixture(scope="module")
def processing_set(base_ms, tmp_path_factory):
    """Convert the synthetic MS to an MSv4 processing set once."""
    from msutils.convert import to_msv4

    out = str(tmp_path_factory.mktemp("msv4") / "obs.zarr")
    to_msv4(base_ms, out, with_pointing=False)
    return out


@pytest.fixture(scope="module")
def v4(processing_set):
    return msutils.msinfo(processing_set, level="data")


@pytest.fixture(scope="module")
def v2(base_ms):
    return msutils.msinfo(base_ms, level="data")


# --------------------------------------------------------------------------
# conversion

def test_convert_produces_a_processing_set(processing_set):
    import os
    assert os.path.isdir(processing_set)
    assert msutils.detect_format(processing_set) == "MSv4"


def test_convert_returns_msinfo(base_ms, tmp_path):
    from msutils.convert import to_msv4

    info = to_msv4(base_ms, str(tmp_path / "out.zarr"), with_pointing=False)
    assert info.format == "MSv4"
    assert info.nfields == 3


def test_convert_refuses_to_clobber(base_ms, tmp_path):
    from msutils.convert import to_msv4

    out = str(tmp_path / "twice.zarr")
    to_msv4(base_ms, out, with_pointing=False)
    with pytest.raises(FileExistsError, match="already exists"):
        to_msv4(base_ms, out)


def test_convert_rejects_unknown_partition_key(base_ms, tmp_path):
    from msutils.convert import to_msv4

    with pytest.raises(ValueError, match="unknown partition key"):
        to_msv4(base_ms, str(tmp_path / "x.zarr"),
                partition_scheme=["NOT_A_COLUMN"])


# --------------------------------------------------------------------------
# the two formats must agree

@pytest.mark.parametrize("attribute", [
    "nrows", "nantennas", "nfields", "nspws", "nscans", "nchan_total",
    "nbaselines", "nvisibilities", "nflagged", "start_utc", "end_utc",
    "integration_times",
])
def test_formats_agree(v2, v4, attribute):
    assert getattr(v4, attribute) == getattr(v2, attribute)


def test_time_is_converted_from_unix_to_mjd(v2, v4):
    """MSv4 stores unix seconds; the model uses MJD seconds throughout."""
    assert v4.start_utc.startswith("2024-01-01T00:00:00")
    assert v4.time_start == pytest.approx(v2.time_start)


def test_max_baseline_matches(v2, v4):
    assert v4.max_baseline == pytest.approx(v2.max_baseline)


def test_antennas_match(v2, v4):
    assert v4.antennas.names == v2.antennas.names
    for a4, a2 in zip(v4.antennas, v2.antennas):
        assert a4.position == pytest.approx(a2.position)
        assert a4.dish_diameter == a2.dish_diameter
        assert a4.latitude == pytest.approx(a2.latitude)


def test_observation_matches(v2, v4):
    assert v4.observation.telescope == v2.observation.telescope
    assert v4.observation.observer == v2.observation.observer
    assert v4.observation.project == v2.observation.project


def test_scan_durations_match(v2, v4):
    assert [s.duration for s in v4.scans] == [s.duration for s in v2.scans]


def test_phase_centres_match(v2, v4):
    for f4, f2 in zip(v4.fields, v2.fields):
        assert f4.phase_centre == pytest.approx(f2.phase_centre)
        assert f4.ra_hms == f2.ra_hms
        assert f4.dec_dms == f2.dec_dms


def test_intents_match(v2, v4):
    assert [f.intents for f in v4.fields] == [f.intents for f in v2.fields]


def test_polarizations_match(v2, v4):
    assert v4.polarizations[0].corr_labels == v2.polarizations[0].corr_labels
    assert v4.polarizations[0].num_corr == 4


def test_spw_frequencies_match(v2, v4):
    for s4, s2 in zip(v4.spws, v2.spws):
        assert s4.chan_freq == pytest.approx(s2.chan_freq)
        assert s4.chan_width == pytest.approx(s2.chan_width)
        assert s4.frame == s2.frame


# --------------------------------------------------------------------------
# MSv4-specific behaviour

def test_names_keep_xradios_disambiguating_suffix(v4):
    """xradio appends an index to field and SPW names; kept verbatim."""
    assert v4.fields[0].name.startswith("PKS1934-638")
    assert v4.spws[0].name.startswith("SPW0")


def test_data_descriptions_are_synthesised(v4):
    """MSv4 has no DATA_DESCRIPTION, so a 1:1 mapping is supplied."""
    assert len(v4.data_descriptions) == v4.nspws
    for dd in v4.data_descriptions:
        assert dd.spw_id == dd.id
        assert dd.pol_id == 0


def test_msv4_has_no_columns(v4):
    """MSv4 stores data variables, not casacore columns."""
    assert v4.column_names == []
    assert v4.subtables                        # the partitions are listed here


def test_meta_level_skips_the_data(processing_set):
    info = msutils.msinfo(processing_set, level="meta")
    assert info.nfields == 0 and info.nscans == 0    # needs the coords
    assert info.nantennas == 4                       # antenna_xds is cheap
    assert info.nflagged is None


def test_render_and_json(v4):
    text = v4.render(verbose=True)
    assert "MSv4" in text and "MEERKAT" in text
    import json
    payload = json.loads(v4.to_json())
    assert payload["format"] == "MSv4"
    assert payload["schema_version"]


def test_explicit_format_argument(processing_set):
    info = msutils.msinfo(processing_set, format="MSv4")
    assert info.format == "MSv4"


def test_msv2_detection_still_works(base_ms):
    assert msutils.msinfo(base_ms).format == "MSv2"


# --------------------------------------------------------------------------
# engines

def test_zarr_engine_needs_no_xradio(processing_set, monkeypatch):
    """Reading a stored processing set must work on plain xarray + zarr.

    xradio is a heavy dependency and is only needed to *write* MSv4; hiding
    it proves the read path does not reach for it.
    """
    import builtins

    real_import = builtins.__import__

    def no_xradio(name, *args, **kwargs):
        if name.startswith("xradio"):
            raise ImportError("xradio hidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_xradio)
    info = msutils.msinfo(processing_set)
    assert info.engine == "zarr"
    assert info.nfields == 3


def test_explicit_zarr_engine(processing_set):
    info = msutils.msinfo(processing_set, engine="zarr")
    assert info.engine == "zarr" and info.format == "MSv4"


def test_explicit_xradio_engine(processing_set):
    info = msutils.msinfo(processing_set, engine="xradio")
    assert info.engine == "xradio" and info.format == "MSv4"


def test_zarr_and_xradio_engines_agree(processing_set):
    a = msutils.msinfo(processing_set, engine="zarr", level="data")
    b = msutils.msinfo(processing_set, engine="xradio", level="data")
    for attribute in ("nrows", "nfields", "nspws", "nscans", "nflagged",
                      "start_utc", "nantennas"):
        assert getattr(a, attribute) == getattr(b, attribute), attribute


def test_unknown_engine_rejected(processing_set):
    with pytest.raises(ValueError, match="engine must be one of"):
        msutils.msinfo(processing_set, engine="telepathy")


# --------------------------------------------------------------------------
# xarray-ms: an MSv2 read through the MSv4 schema, without converting it

@pytest.fixture
def xarray_ms():
    return pytest.importorskip(
        "xarray_ms", reason="xarray-ms not installed (msutils[xarray-ms])")


def test_msv2_read_through_the_msv4_schema(base_ms, xarray_ms):
    info = msutils.msinfo(base_ms, engine="xarray-ms", level="data")
    # what is on disk is still an MSv2; only the view is MSv4-shaped
    assert info.format == "MSv2"
    assert info.engine == "xarray-ms"
    assert info.nfields == 3 and info.nspws == 2


def test_casacore_and_xarray_ms_engines_agree(base_ms, xarray_ms):
    native = msutils.msinfo(base_ms, level="data")
    viewed = msutils.msinfo(base_ms, engine="xarray-ms", level="data")

    assert native.engine == "casacore"
    for attribute in ("nrows", "nantennas", "nfields", "nspws", "nscans",
                      "nchan_total", "nvisibilities", "nflagged", "start_utc",
                      "end_utc", "integration_times"):
        assert getattr(viewed, attribute) == getattr(native, attribute), attribute
    assert viewed.max_baseline == pytest.approx(native.max_baseline)
    assert viewed.fields.names == native.fields.names
    assert viewed.antennas.names == native.antennas.names


def test_casacore_engine_is_the_default_for_msv2(base_ms):
    assert msutils.msinfo(base_ms).engine == "casacore"


def test_casacore_engine_tolerates_an_ms_stricter_readers_reject(tmp_path):
    """A malformed MS must still be inspectable.

    xarray-ms refuses an MS whose FEED subtable does not cover the antennas
    in use. That is a reasonable stance for a data reader and a bad one for a
    diagnostic tool -- the broken MS is exactly the one you need to look at --
    so the default engine stays casacore.
    """
    from casacore.tables import table
    from msfactory import make_ms

    path = make_ms(tmp_path / "no_feed.ms")
    with table(path + "::FEED", readonly=False, ack=False) as feed:
        feed.removerows(range(feed.nrows()))

    info = msutils.msinfo(path)               # must not raise
    assert info.nfields == 3 and info.nscans == 6

    pytest.importorskip("xarray_ms")
    with pytest.raises(Exception):
        msutils.msinfo(path, engine="xarray-ms")
