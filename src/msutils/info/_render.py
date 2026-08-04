"""Render an :class:`~msutils.info._model.MSInfo` as a human-readable report.

Deliberately plain text with no dependencies: this is what ``msutils info``
prints, and it needs to survive being piped into a log file. Structured
consumers should use :meth:`MSInfo.to_dict` instead of parsing this.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._model import MSInfo, format_duration

__all__ = ["format_bytes", "format_frequency", "render"]


def format_bytes(nbytes: float | None) -> str:
    """Bytes -> a human-readable binary size, e.g. ``271.4 MiB``."""
    if nbytes is None:
        return "-"
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.1f} TiB"         # pragma: no cover - unreachable


def format_frequency(hz: float | None) -> str:
    """Hz -> a readable frequency with an appropriate unit."""
    if hz is None:
        return "-"
    value = abs(float(hz))
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if value >= scale:
            return f"{float(hz) / scale:.6g} {unit}"
    return f"{float(hz):.6g} Hz"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
           indent: str = "  ") -> list[str]:
    """Lay out rows as an aligned text table; numeric-ish columns right-align."""
    if not rows:
        return [indent + "(none)"]
    cells = [[("-" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _numeric(index: int) -> bool:
        return all(_looks_numeric(row[index]) for row in cells)

    aligns = [_numeric(i) for i in range(len(headers))]
    out = [indent + "  ".join(
        h.rjust(widths[i]) if aligns[i] else h.ljust(widths[i])
        for i, h in enumerate(headers))]
    out.append(indent + "  ".join("-" * w for w in widths))
    for row in cells:
        out.append(indent + "  ".join(
            c.rjust(widths[i]) if aligns[i] else c.ljust(widths[i])
            for i, c in enumerate(row)).rstrip())
    return out


def _looks_numeric(text: str) -> bool:
    try:
        float(text.replace(",", "").rstrip("%"))
        return True
    except ValueError:
        return text == "-"


def _shorten(intents: Sequence[str], limit: int = 2) -> str:
    """Intents are long; show the first couple and count the rest."""
    if not intents:
        return "-"
    # Strip the '#ON_SOURCE' style qualifiers for the summary view.
    short = []
    for intent in intents:
        base = intent.split("#")[0]
        if base not in short:
            short.append(base)
    if len(short) <= limit:
        return ",".join(short)
    return "{},+{} more".format(",".join(short[:limit]), len(short) - limit)


def render(info: MSInfo, verbose: bool = False) -> str:
    """Build the report. ``verbose`` adds per-scan, per-antenna and column tables."""
    out: list[str] = []
    obs = info.observation

    out.append(f"Measurement Set: {info.path}")
    engine = "" if info.engine == "casacore" else f"  (via {info.engine})"
    out.append(f"  format      : {info.format}{engine}    size: {format_bytes(info.size_bytes)}    rows: {info.nrows:,}")
    out.append("  telescope   : {:<12} observer: {:<16} project: {}".format(
        obs.telescope or "-", obs.observer or "-", obs.project or "-"))
    if info.start_utc:
        out.append("  observed    : {} -> {}  ({})".format(
            info.start_utc, info.end_utc,
            format_duration(info.duration) if info.duration else "-"))
        # OBSERVATION::TIME_RANGE is frequently stale; only worth showing when
        # it disagrees with the data, and then as the secondary value.
        if verbose and obs.time_start is not None and (
                abs((obs.time_start or 0) - info.time_start) > 1.0
                or abs((obs.time_end or 0) - info.time_end) > 1.0):
            out.append(f"  (OBSERVATION::TIME_RANGE says {obs.start_utc} -> {obs.end_utc})")
    summary = "  totals      : {} antennas, {} baselines, {} fields, {} scans, {} SPWs, {} channels".format(
        info.nantennas, info.nbaselines or "?", info.nfields, info.nscans,
        info.nspws, info.nchan_total)
    out.append(summary)
    if info.integration_times:
        out.append("  integration : {}".format(
            ", ".join(f"{t:g}s" for t in info.integration_times)))
    if info.max_baseline is not None:
        line = f"  max baseline: {info.max_baseline:,.1f} m"
        if info.max_uv_distance is not None:
            line += f"   (max uv distance {info.max_uv_distance:,.1f} m)"
        out.append(line)
    if info.flagged_fraction is not None:
        out.append(f"  flagged     : {info.flagged_fraction * 100.0:.2f}% of {info.nvisibilities:,} visibilities")

    out.append("")
    out.append(f"Fields ({info.nfields}):")
    out.extend(_table(
        ["ID", "Name", "RA", "Dec", "Frame", "Intents", "Scans", "Rows"],
        [[f.id, f.name, f.ra_hms, f.dec_dms, f.ref_frame, _shorten(f.intents),
          f.nscans or "-", f"{f.nrows:,}" if f.nrows else "-"]
         for f in info.fields]))

    out.append("")
    out.append(f"Spectral windows ({info.nspws}):")
    out.extend(_table(
        ["ID", "Name", "#Chan", "Frame", "Centre", "Bandwidth", "Chan width"],
        [[s.id, s.name or "-", s.num_chan, s.frame,
          format_frequency(s.centre_freq), format_frequency(s.total_bandwidth),
          format_frequency(s.chan_width[0] if s.chan_width else None)]
         for s in info.spws]))

    out.append("")
    out.append(f"Polarization setups ({len(info.polarizations)}):")
    out.extend(_table(
        ["ID", "#Corr", "Correlations"],
        [[p.id, p.num_corr, ",".join(p.corr_labels)] for p in info.polarizations]))

    if verbose or _nontrivial_ddid(info):
        out.append("")
        out.append(f"Data descriptions ({len(info.data_descriptions)}):")
        out.extend(_table(
            ["DDID", "SPW", "Pol"],
            [[d.id, d.spw_id, d.pol_id] for d in info.data_descriptions]))

    if verbose:
        out.append("")
        out.append(f"Scans ({info.nscans}):")
        out.extend(_table(
            ["Scan", "Field", "Start (UTC)", "Duration", "SPWs", "Rows", "Intents"],
            [[s.number, s.field_name or s.field_id, s.start_utc,
              format_duration(s.duration),
              ",".join(str(i) for i in s.spw_ids), f"{s.nrows:,}",
              _shorten(s.intents)] for s in info.scans]))

        out.append("")
        out.append(f"Antennas ({info.nantennas}):")
        out.extend(_table(
            ["ID", "Name", "Station", "Mount", "Diameter", "Longitude", "Latitude", "Elevation"],
            [[a.id, a.name, a.station or "-", a.mount or "-",
              f"{a.dish_diameter:.1f} m",
              f"{a.longitude:.5f}", f"{a.latitude:.5f}",
              f"{a.elevation:.1f} m"] for a in info.antennas]))

        out.append("")
        out.append(f"Columns ({len(info.columns)}):")
        out.extend(_table(
            ["Name", "Type", "Shape", "Unit", "Manager", "Size"],
            [[c.name, c.dtype, c.shape_label, c.unit or "-",
              c.data_manager or "-", format_bytes(c.nbytes)]
             for c in info.columns]))
    else:
        out.append("")
        out.append("Columns ({}): {}".format(
            len(info.columns), ", ".join(info.column_names)))

    return "\n".join(out)


def _nontrivial_ddid(info: MSInfo) -> bool:
    """True when DATA_DESC_ID is not simply the SPW id -- worth showing explicitly."""
    return any(d.id != d.spw_id or d.pol_id != 0 for d in info.data_descriptions)
