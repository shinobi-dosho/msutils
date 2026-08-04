"""The deprecated `summary()` still works, warns, and now returns correct numbers.

`summary()` is a compatibility adapter over `msinfo()`. It keeps the legacy
dict *shape* so existing pipelines keep running, but the values it reports are
the corrected ones -- these tests pin both halves of that contract.
"""

import json
import warnings

import pytest

import msutils


@pytest.fixture
def legacy(base_ms):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return msutils.summary(base_ms, display=False)


def test_summary_warns(base_ms):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msutils.summary(base_ms, display=False)
    assert any(issubclass(w.category, FutureWarning) for w in caught)
    assert any("msinfo" in str(w.message) for w in caught)


def test_legacy_keys_are_preserved(legacy):
    for key in ("FIELD", "SPW", "ANT", "MAXBL", "SCAN", "EXPOSURE", "NROW", "CORR", "NCOR"):
        assert key in legacy


def test_legacy_values(legacy):
    assert legacy["NROW"] == 216
    assert legacy["NCOR"] == 4
    assert legacy["CORR"]["NUM_CORR"] == 4
    assert legacy["CORR"]["CORR_TYPE"] == ["XX", "XY", "YX", "YY"]
    assert len(legacy["SPW"]["CHAN_FREQ"][0]) == 4
    # raw subtable dumps callers may have been indexing
    assert legacy["ANT"]["NAME"] == ["m000", "m001", "m002", "m003"]
    assert len(legacy["FIELD"]["PHASE_DIR"]) == 3


def test_legacy_scan_durations_are_corrected(legacy):
    # was max(TIME) - min(TIME) = 16 s; the scan really covers 24 s
    assert legacy["SCAN"]["0"]["1"] == pytest.approx(24.0)
    assert legacy["FIELD"]["PERIOD"] == [pytest.approx(48.0)] * 3


def test_legacy_state_id_comes_from_the_main_table(legacy):
    """One STATE row per field in the fixture, so field i uses state i."""
    assert legacy["FIELD"]["STATE_ID"] == [0, 1, 2]


def test_legacy_intents_are_per_field(legacy):
    # used to be the raw STATE::OBS_MODE column, identical for every field
    assert legacy["FIELD"]["INTENTS"][1] == ["CALIBRATE_PHASE#ON_SOURCE"]


def test_legacy_spw_gains_chan_width(legacy):
    # the old hard-coded column list omitted CHAN_WIDTH entirely
    assert legacy["SPW"]["CHAN_WIDTH"][0] == [10e6] * 4


def test_summary_writes_json(base_ms, tmp_path):
    outfile = tmp_path / "summary.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        msutils.summary(base_ms, outfile=str(outfile), display=False)
    assert json.loads(outfile.read_text())["NCOR"] == 4
