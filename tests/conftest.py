"""Shared pytest fixtures.

Behavioural tests need a real Measurement Set. We generate a tiny one with
`simms` (7-antenna kat-7 layout, a few times/channels). `simms` is a heavy,
optional test dependency (install it from git, see the `test` dependency group
in pyproject.toml); when it is not importable the MS-backed tests skip rather
than fail.
"""
import shutil

import pytest


def _make_ms(path):
    """Create a small MS at ``path`` using simms, or skip if simms is missing."""
    try:
        from simms.telescope.generate_ms import create_ms
    except ImportError as exc:  # simms (or a dep) not installed
        pytest.skip("simms not installed: {0}".format(exc))

    create_ms(
        ms=str(path),
        telescope_name="kat-7",           # 7 antennas -> small MS
        pointing_direction=["J2000", "0h0m0s", "-30d0m0s"],
        dtime=10,                          # seconds per integration
        ntimes=3,
        start_freq="1.4GHz",
        dfreq="10MHz",
        nchan=4,
        correlations=["XX", "XY", "YX", "YY"],
        row_chunks=10000,
        sefd=400.0,
        column="DATA",
    )
    return str(path)


@pytest.fixture(scope="session")
def base_ms(tmp_path_factory):
    """Build the synthetic MS once per test session."""
    path = tmp_path_factory.mktemp("base") / "synthetic.ms"
    return _make_ms(path)


@pytest.fixture
def ms(base_ms, tmp_path):
    """A fresh, writable copy of the synthetic MS for each test."""
    dest = str(tmp_path / "test.ms")
    shutil.copytree(base_ms, dest)
    return dest
