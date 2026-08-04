"""Plain-text rendering of :class:`~msutils.flagstats.FlagStats`."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["render_flagstats"]

#: Bins per axis in the text report before it collapses to a summary line.
_MAX_ROWS = 24


def render_flagstats(stats) -> str:
    out: list[str] = [
        f"Flag statistics: {stats.path}",
        f"  overall: {stats.total.percent:.2f}% flagged ({stats.total.flagged:,} of {stats.total.total:,} visibilities)",
    ]

    for title, counts in (
        ("Field", stats.by_field),
        ("Spectral window", stats.by_spw),
        ("Correlation", stats.by_correlation),
        ("Scan", stats.by_scan),
        ("Antenna", stats.by_antenna),
    ):
        out.append("")
        out.append(f"{title} ({len(counts)}):")
        out.extend(_bins(counts))

    if stats.by_channel:
        out.append("")
        out.append("Channel:")
        for spw_id, channels in sorted(stats.by_channel.items()):
            out.append(
                "  SPW {}: {}".format(spw_id, " ".join(f"{c.percent:.0f}%" for c in channels))
            )

    return "\n".join(out)


def _bins(counts: Sequence) -> list[str]:
    """One line per bin, or a worst-offenders summary when there are many."""
    if not len(counts):
        return ["  (none)"]
    if len(counts) > _MAX_ROWS:
        worst = sorted(counts, key=lambda c: c.fraction, reverse=True)[:5]
        return [
            "  {} bins; worst: {}".format(
                len(counts), ", ".join(f"{c.name} {c.percent:.1f}%" for c in worst)
            )
        ]
    width = max(len(str(c.name)) for c in counts)
    return [
        "  {0:<{1}}  {2:6.2f}%  ({3:,} / {4:,})".format(
            c.name, width, c.percent, c.flagged, c.total
        )
        for c in counts
    ]
