"""Shared pytest fixtures.

Behavioural tests run against synthetic Measurement Sets built by
``tests/msfactory.py`` using python-casacore alone -- no ``simms``, no other
heavy simulation stack, and therefore no skipping. Anything that can
``import msutils`` can run the whole suite.
"""

import shutil

import pytest
from msfactory import make_ms


@pytest.fixture(scope="session")
def base_ms(tmp_path_factory):
    """A synthetic MS built once per session.

    3 fields x 2 scans x 3 times x 2 SPWs x 6 baselines = 216 rows, with
    distinct observing intents per field and every third row flagged.
    """
    return make_ms(tmp_path_factory.mktemp("base") / "synthetic.ms")


@pytest.fixture
def ms(base_ms, tmp_path):
    """A fresh, writable copy of the synthetic MS for each test."""
    dest = str(tmp_path / "test.ms")
    shutil.copytree(base_ms, dest)
    return dest


@pytest.fixture(scope="session")
def patterned_ms(tmp_path_factory):
    """An MS whose flags differ per correlation, per channel and per antenna.

    The synthetic MS flags whole rows, which makes every axis report the same
    percentage -- so it cannot tell a correct per-correlation breakdown from
    the old broken one that copied the total into every slot. This fixture
    lays down a pattern where each axis has a different, known answer:

    * correlation 1 (XY): fully flagged
    * channel 3: fully flagged
    * every baseline involving antenna 0: fully flagged
    * nothing else
    """
    import numpy as np
    from casacore.tables import table

    path = make_ms(tmp_path_factory.mktemp("patterned") / "patterned.ms", nspw=1, flag_every=None)
    with table(path, readonly=False, ack=False) as tab:
        flag = np.zeros_like(tab.getcol("FLAG"))
        flag[:, :, 1] = True  # correlation XY
        flag[:, 3, :] = True  # channel 3
        a1, a2 = tab.getcol("ANTENNA1"), tab.getcol("ANTENNA2")
        flag[(a1 == 0) | (a2 == 0)] = True  # antenna 0
        tab.putcol("FLAG", flag)
    return path


@pytest.fixture(scope="session")
def single_spw_ms(tmp_path_factory):
    """A minimal one-field, one-SPW, unflagged MS for edge-case checks."""
    return make_ms(
        tmp_path_factory.mktemp("single") / "single.ms",
        nspw=1,
        nfield=1,
        nscan_per_field=1,
        flag_every=None,
    )
