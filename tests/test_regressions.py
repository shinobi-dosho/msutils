"""Regression tests for bugs found in the 2.x audit.

Each test here failed before the fix. They are kept together so the failure
modes stay documented even if the surrounding code moves.
"""
import numpy
import pytest

import msutils
from casacore.tables import table


def _getcol(msname, col):
    tab = table(msname, ack=False)
    try:
        return tab.getcol(col)
    finally:
        tab.close()


# --------------------------------------------------------------------------
# addnoise: per-channel noise across multiple row chunks

def test_addnoise_per_channel_across_chunks(ms):
    """Was: ValueError, non-broadcastable operand on the second chunk.

    The noise array was reshaped *inside* the row loop, so it grew an extra
    axis for every chunk after the first.
    """
    msutils.addcol(ms, "MODEL_DATA", clone="DATA")
    noise = numpy.array([1.0, 2.0, 3.0, 4.0])
    # rowchunk well below the row count, to force several chunks
    msutils.addnoise(ms, column="MODEL_DATA", noise=noise, rowchunk=8)

    model = _getcol(ms, "MODEL_DATA")
    assert numpy.isfinite(model).all()
    # noise scales with channel, so channel 3 must be noisier than channel 0
    per_chan = numpy.abs(model).std(axis=(0, 2))
    assert per_chan[3] > per_chan[0]


def test_addnoise_rejects_mismatched_channel_count(ms):
    msutils.addcol(ms, "MODEL_DATA", clone="DATA")
    with pytest.raises(ValueError, match="noise has 3 channels"):
        msutils.addnoise(ms, column="MODEL_DATA", noise=numpy.ones(3))


def test_addnoise_scalar_still_works(ms):
    msutils.addcol(ms, "MODEL_DATA", clone="DATA")
    msutils.addnoise(ms, column="MODEL_DATA", noise=2.0, rowchunk=8)
    model = _getcol(ms, "MODEL_DATA")
    assert numpy.iscomplexobj(model)
    assert numpy.abs(model).sum() > 0


# --------------------------------------------------------------------------
# addcol: unbound data_desc and the ignored data_desc_type

def test_addcol_without_clone_or_shape_raises_clearly(ms):
    """Was: UnboundLocalError on `data_desc`."""
    with pytest.raises(ValueError, match="not enough information"):
        msutils.addcol(ms, "NOWHERE", clone=None)


def test_addcol_scalar_column(ms):
    """Was: UnboundLocalError -- data_desc_type was documented but never read.

    The code branched on `valuetype == 'scalar'`, conflating the value type
    with the cell layout, so a scalar column could not be created at all.
    """
    assert msutils.addcol(ms, "MY_FLAG", data_desc_type="scalar",
                          valuetype="bool", clone=None, init_with=False) == "added"
    col = _getcol(ms, "MY_FLAG")
    assert col.shape == (216,)
    assert col.dtype == bool


def test_addcol_scalar_float_column(ms):
    msutils.addcol(ms, "MY_SCALE", data_desc_type="scalar", valuetype="float",
                   clone=None, init_with=2.5)
    assert numpy.allclose(_getcol(ms, "MY_SCALE"), 2.5)


def test_addcol_explicit_shape(ms):
    msutils.addcol(ms, "MY_ARRAY", shape=[4, 4], valuetype="float",
                   clone=None, init_with=1.0)
    col = _getcol(ms, "MY_ARRAY")
    assert col.shape == (216, 4, 4)
    assert numpy.allclose(col, 1.0)


def test_addcol_rejects_bad_data_desc_type(ms):
    with pytest.raises(ValueError, match="data_desc_type must be"):
        msutils.addcol(ms, "X", data_desc_type="vector")


def test_addcol_missing_clone_column(ms):
    with pytest.raises(ValueError, match="cannot clone"):
        msutils.addcol(ms, "X", clone="NO_SUCH_COLUMN")


def test_addcol_requires_a_name(ms):
    with pytest.raises(ValueError, match="colname is required"):
        msutils.addcol(ms, None)
