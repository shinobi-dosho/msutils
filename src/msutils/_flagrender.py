"""Plain-text rendering of :class:`~msutils.flagstats.FlagStats`."""
from __future__ import annotations

from typing import List, Sequence

__all__ = ["render_flagstats"]

#: Bins per axis in the text report before it collapses to a summary line.
_MAX_ROWS = 24


def render_flagstats(stats) -> str:
    out: List[str] = ["Flag statistics: {0}".format(stats.path),
                      "  overall: {0:.2f}% flagged ({1:,} of {2:,} visibilities)".format(
                          stats.total.percent, stats.total.flagged, stats.total.total)]

    for title, counts in (("Field", stats.by_field),
                          ("Spectral window", stats.by_spw),
                          ("Correlation", stats.by_correlation),
                          ("Scan", stats.by_scan),
                          ("Antenna", stats.by_antenna)):
        out.append("")
        out.append("{0} ({1}):".format(title, len(counts)))
        out.extend(_bins(counts))

    if stats.by_channel:
        out.append("")
        out.append("Channel:")
        for spw_id, channels in sorted(stats.by_channel.items()):
            out.append("  SPW {0}: {1}".format(
                spw_id, " ".join("{0:.0f}%".format(c.percent) for c in channels)))

    return "\n".join(out)


def _bins(counts: Sequence) -> List[str]:
    """One line per bin, or a worst-offenders summary when there are many."""
    if not len(counts):
        return ["  (none)"]
    if len(counts) > _MAX_ROWS:
        worst = sorted(counts, key=lambda c: c.fraction, reverse=True)[:5]
        return ["  {0} bins; worst: {1}".format(
            len(counts), ", ".join("{0} {1:.1f}%".format(c.name, c.percent)
                                   for c in worst))]
    width = max(len(str(c.name)) for c in counts)
    return ["  {0:<{1}}  {2:6.2f}%  ({3:,} / {4:,})".format(
        c.name, width, c.percent, c.flagged, c.total) for c in counts]
