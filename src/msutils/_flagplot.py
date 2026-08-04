"""matplotlib summary plot for :class:`~msutils.flagstats.FlagStats`.

Isolated in its own module so that importing ``msutils.flagstats`` -- and
computing statistics -- never needs matplotlib. Only :func:`plot_flagstats`
does, which is why the import lives inside the function.
"""
from __future__ import annotations

from typing import List, Sequence

from ._log import create_logger

__all__ = ["plot_flagstats"]

LOGGER = create_logger(__name__)

#: Bars beyond this many are thinned to the worst offenders, so the axis stays
#: readable on arrays with hundreds of baselines.
_MAX_BARS = 64


def plot_flagstats(stats, outfile: str) -> str:
    """Write a 3x2 bar-chart summary of ``stats`` to ``outfile``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Field", stats.by_field, False),
        ("Correlation", stats.by_correlation, False),
        ("Scan", stats.by_scan, True),
        ("Antenna", stats.by_antenna, True),
        ("Spectral window", stats.by_spw, False),
        ("Baseline", stats.by_baseline, True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for (title, counts, rotate), ax in zip(panels, axes.flatten()):
        names, percents, note = _bars(counts)
        positions = range(len(names))
        ax.bar(positions, percents, width=0.9)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(names, rotation=90 if rotate else 0, fontsize=8)
        ax.set_xlabel(title)
        ax.set_ylabel("Flagged data (%)")
        ax.set_title("{0} RFI summary{1}".format(title, note))
        ax.set_ylim(0, 100)

    fig.suptitle("{0} -- {1:.2f}% flagged overall".format(
        stats.path, stats.total.percent))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outfile)
    plt.close(fig)
    LOGGER.info("Wrote flag summary plot to %s", outfile)
    return outfile


def _bars(counts: Sequence) -> tuple:
    """Bar labels and percentages, thinned to the worst bins when crowded."""
    items: List = list(counts)
    if not items:
        return [], [], ""
    note = ""
    if len(items) > _MAX_BARS:
        items = sorted(items, key=lambda c: c.fraction, reverse=True)[:_MAX_BARS]
        note = " (worst {0} of {1})".format(_MAX_BARS, len(counts))
    return ([str(c.name) for c in items], [c.percent for c in items], note)
