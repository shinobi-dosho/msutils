"""Behavioural tests for the core column operations, run against a synthetic MS.

These require the `simms`-backed `ms` fixture (see conftest.py); they skip when
simms is unavailable.
"""
import numpy
from casacore.tables import table

import msutils


def _getcol(msname, col):
    tab = table(msname)
    try:
        return tab.getcol(col)
    finally:
        tab.close()


def test_summary(ms):
    info = msutils.summary(ms, display=False)

    assert info["NROW"] > 0
    assert info["NCOR"] == 4
    assert info["CORR"]["NUM_CORR"] == 4
    assert info["CORR"]["CORR_TYPE"] == ["XX", "XY", "YX", "YY"]
    # one SPW with the 4 channels we asked for
    assert len(info["SPW"]["CHAN_FREQ"][0]) == 4


def test_summary_writes_json(ms, tmp_path):
    import json

    outfile = tmp_path / "summary.json"
    msutils.summary(ms, outfile=str(outfile), display=False)
    data = json.loads(outfile.read_text())
    assert data["NCOR"] == 4


def test_addcol_clone_and_idempotent(ms):
    assert msutils.addcol(ms, "MODEL_DATA", clone="DATA") == "added"
    # second call is a no-op
    assert msutils.addcol(ms, "MODEL_DATA", clone="DATA") == "exists"

    data = _getcol(ms, "DATA")
    model = _getcol(ms, "MODEL_DATA")
    assert model.shape == data.shape


def test_addcol_init_with(ms):
    msutils.addcol(ms, "IMAGING_WEIGHT_SPECTRUM", clone="DATA",
                   valuetype="float", init_with=1.0)
    col = _getcol(ms, "IMAGING_WEIGHT_SPECTRUM")
    assert numpy.allclose(col, 1.0)


def test_copycol(ms):
    msutils.copycol(ms, "DATA", "CORRECTED_DATA")
    assert numpy.array_equal(_getcol(ms, "DATA"), _getcol(ms, "CORRECTED_DATA"))


def test_sumcols(ms):
    msutils.copycol(ms, "DATA", "CORRECTED_DATA")
    msutils.sumcols(ms, col1="DATA", col2="CORRECTED_DATA", outcol="MODEL_DATA")
    data = _getcol(ms, "DATA")
    out = _getcol(ms, "MODEL_DATA")
    assert numpy.allclose(out, data + data)


def test_sumcols_subtract(ms):
    msutils.copycol(ms, "DATA", "CORRECTED_DATA")
    msutils.sumcols(ms, col1="DATA", col2="CORRECTED_DATA",
                    outcol="MODEL_DATA", subtract=True)
    out = _getcol(ms, "MODEL_DATA")
    assert numpy.allclose(out, 0)


def test_addnoise(ms):
    msutils.addcol(ms, "MODEL_DATA", clone="DATA")
    msutils.addnoise(ms, column="MODEL_DATA", noise=1.0)
    model = _getcol(ms, "MODEL_DATA")
    assert numpy.iscomplexobj(model)
    # noise was actually written (not left all-zero)
    assert numpy.abs(model).sum() > 0
