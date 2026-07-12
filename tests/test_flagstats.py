"""Test the matplotlib flag-stats plotting.

Needs the `flagstats` extra (dask-ms + matplotlib); skips otherwise. This
exercises the plotting directly with synthetic stats, so it doesn't need an MS.
"""
import pytest


def test_plot_flag_stats_writes_png(tmp_path):
    try:
        from msutils import flag_stats
    except ImportError as exc:
        pytest.skip("flagstats extra not installed: {0}".format(exc))

    def axis(names):
        return {i: {"name": n, "frac": 0.1 * (i + 1)} for i, n in enumerate(names)}

    outfile = tmp_path / "flags.png"
    flag_stats._plot_flag_stats(
        antenna_stats={"antennas": axis(["m000", "m001", "m002"])},
        scan_stats={"scans": axis(["1", "2"])},
        target_stats={"fields": axis(["PKS1934", "3C286"])},
        corr_stats={"corrs": axis(["XX", "XY", "YX", "YY"])},
        outfile=str(outfile),
    )

    assert outfile.exists()
    assert outfile.stat().st_size > 0
