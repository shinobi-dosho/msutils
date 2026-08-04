"""Tests for the TaQL flag statistics.

The `patterned_ms` fixture flags a specific correlation, a specific channel and
a specific antenna, so each axis has a different known answer. That is what
distinguishes a real per-axis breakdown from the 2.x implementation, whose
reduction kernel wrote the overall total into every correlation slot.
"""

import json

import pytest

from msutils.flagstats import FlagCount, flagstats


@pytest.fixture(scope="module")
def stats(patterned_ms):
    return flagstats(patterned_ms)


# --------------------------------------------------------------------------
# the axes the old implementation got wrong


def test_correlations_are_distinguished(stats):
    """XY is fully flagged; the other three are not.

    The 2.x kernel reported the same number for all four correlations.
    """
    by_name = {c.name: c for c in stats.by_correlation}
    assert set(by_name) == {"XX", "XY", "YX", "YY"}
    assert by_name["XY"].fraction == 1.0
    for name in ("XX", "YX", "YY"):
        assert 0.0 < by_name[name].fraction < 1.0
        assert by_name[name].fraction == pytest.approx(by_name["XX"].fraction)


def test_channels_are_distinguished(stats):
    """Channel 3 is fully flagged; channels 0-2 are not."""
    channels = stats.by_channel[0]
    assert len(channels) == 4
    assert channels[3].fraction == 1.0
    for ch in channels[:3]:
        assert 0.0 < ch.fraction < 1.0


def test_antennas_are_distinguished(stats):
    """Antenna 0 is fully flagged; the others only partly."""
    by_name = {a.name: a for a in stats.by_antenna}
    assert by_name["m000"].fraction == 1.0
    for name in ("m001", "m002", "m003"):
        assert 0.0 < by_name[name].fraction < 1.0


def test_baselines_are_reported(stats):
    # 4 antennas -> 6 cross-correlation baselines
    assert len(stats.by_baseline) == 6
    involving_zero = [b for b in stats.by_baseline if 0 in b.id]
    assert len(involving_zero) == 3
    assert all(b.fraction == 1.0 for b in involving_zero)


def test_field_selection_is_honoured(patterned_ms):
    """`fields=` used to be computed and then ignored for the antenna axis."""
    everything = flagstats(patterned_ms)
    one_field = flagstats(patterned_ms, fields=["PKS1934-638"])
    assert one_field.total.total < everything.total.total
    # antenna totals must shrink too, not stay at the all-fields value
    assert sum(a.total for a in one_field.by_antenna) < sum(a.total for a in everything.by_antenna)
    assert len(one_field.by_field) == 1


# --------------------------------------------------------------------------
# counting


def test_counts_are_internally_consistent(stats, patterned_ms):
    from casacore.tables import table

    with table(patterned_ms, ack=False) as tab:
        flag = tab.getcol("FLAG")

    assert stats.total.flagged == int(flag.sum())
    assert stats.total.total == int(flag.size)

    # each axis must partition the same total
    for axis in (stats.by_field, stats.by_scan, stats.by_correlation, stats.by_spw):
        assert sum(c.flagged for c in axis) == stats.total.flagged
        assert sum(c.total for c in axis) == stats.total.total

    # antennas double-count (each visibility belongs to two of them)
    assert sum(a.total for a in stats.by_antenna) == 2 * stats.total.total


def test_sums_are_not_inflated_by_group_count(stats):
    """2.x multiplied per-scan sums by the number of scans."""
    assert sum(s.flagged for s in stats.by_scan) == stats.total.flagged


def test_fraction_of_empty_bin_is_zero():
    assert FlagCount(id=0).fraction == 0.0
    assert FlagCount(id=0, flagged=1, total=4).percent == 25.0


# --------------------------------------------------------------------------
# selection and validation


def test_selection_by_id_and_name(patterned_ms):
    by_id = flagstats(patterned_ms, antennas=[0, 1])
    by_name = flagstats(patterned_ms, antennas=["m000", "m001"])
    assert by_id.total.flagged == by_name.total.flagged


def test_scan_selection(patterned_ms):
    one = flagstats(patterned_ms, scans=[1])
    assert len(one.by_scan) == 1
    assert one.by_scan[1].name == "1"


def test_spw_selection(ms):
    both = flagstats(ms)
    first = flagstats(ms, spws=[0])
    assert len(both.by_spw) == 2
    assert len(first.by_spw) == 1
    assert first.total.total == both.total.total / 2


def test_unknown_selection_raises(patterned_ms):
    with pytest.raises(ValueError, match="unknown field"):
        flagstats(patterned_ms, fields=["NO_SUCH_FIELD"])
    with pytest.raises(ValueError, match="unknown antenna id"):
        flagstats(patterned_ms, antennas=[99])


def test_unflagged_ms(single_spw_ms):
    stats = flagstats(single_spw_ms)
    assert stats.total.flagged == 0
    assert stats.total.fraction == 0.0
    assert all(c.fraction == 0.0 for c in stats.by_correlation)


# --------------------------------------------------------------------------
# output


def test_json_round_trip(stats, tmp_path):
    out = tmp_path / "flags.json"
    stats.save(str(out))
    payload = json.loads(out.read_text())
    assert payload["total"]["flagged"] == stats.total.flagged
    assert payload["by_correlation"][1]["name"] == "XY"
    assert payload["by_correlation"][1]["fraction"] == 1.0


def test_outfile_argument(patterned_ms, tmp_path):
    out = tmp_path / "direct.json"
    flagstats(patterned_ms, outfile=str(out))
    assert json.loads(out.read_text())["total"]["total"] > 0


def test_render_reports_every_axis(stats):
    text = stats.render()
    for expected in (
        "Flag statistics",
        "overall",
        "Field (3)",
        "Correlation (4)",
        "Antenna (4)",
        "Channel",
        "XY",
    ):
        assert expected in text
    assert str(stats) == text


def test_plot_statistics_writes_png(patterned_ms, tmp_path):
    pytest.importorskip("matplotlib")
    from msutils.flagstats import plot_statistics

    plot = tmp_path / "flags.png"
    result = plot_statistics(patterned_ms, plotfile=str(plot), outfile=str(tmp_path / "flags.json"))
    assert plot.exists() and plot.stat().st_size > 0
    assert result.total.total > 0


def test_save_statistics_entry_point(patterned_ms, tmp_path):
    from msutils.flagstats import save_statistics

    out = tmp_path / "saved.json"
    result = save_statistics(patterned_ms, outfile=str(out))
    assert out.exists()
    assert result.total.flagged > 0


def test_flagstats_needs_no_dask():
    """The dask-ms dependency is gone; importing must not pull it in.

    Checked in a subprocess: with the optional extras installed, other test
    modules will already have imported dask by the time this runs.
    """
    import os
    import subprocess
    import sys

    code = (
        "import sys, msutils.flagstats;"
        "print(','.join(m for m in ('dask', 'daskms') if m in sys.modules))"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.strip() == "", result.stdout
