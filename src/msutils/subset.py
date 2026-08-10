"""Write a selected -- and optionally averaged -- copy of an MS.

:func:`subset` is a TaQL selection deep-copied into a new MS. Passing
``time_bin``/``chan_bin`` averages that selection on the way out, by
delegating to :func:`average`, which in turn delegates the actual binning to
`codex-africanus <https://github.com/ratt-ru/codex-africanus>`_ rather than
reimplementing it; averaging correctly (weights, flags, exposure, uvw,
centroids) is subtle and africanus is the reference implementation.

**Indices are preserved by default, not renumbered.** CASA's ``split``
compacts field and SPW ids so the output starts at 0; these functions keep the
originals, so ids in a subset still match the parent MS. That is usually what
you want when subsetting for inspection, and it is the difference to watch for
when porting a ``split`` call. Pass ``reindex=True`` for the ``split``
behaviour: unused FIELD, SPECTRAL_WINDOW and DATA_DESCRIPTION rows are dropped
and what remains is renumbered from 0.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._log import create_logger
from ._tables import closing_table, open_table, query
from .info import msinfo

__all__ = ["average", "subset"]

LOGGER = create_logger(__name__)


def subset(
    msname: str,
    outms: str,
    fields: Sequence | None = None,
    spws: Sequence | None = None,
    scans: Sequence | None = None,
    antennas: Sequence | None = None,
    taql: str | None = None,
    time_bin: float | None = None,
    chan_bin: int | None = None,
    datacolumn: str = "DATA",
    reindex: bool = False,
    overwrite: bool = False,
) -> str:
    """Write the selected rows of ``msname`` to a new MS at ``outms``.

    Args:
        msname: Source MS.
        outms: Destination path.
        fields: Field ids or names to keep.
        spws: Spectral window ids or names to keep.
        scans: Scan numbers to keep.
        antennas: Antenna ids or names; a baseline is kept if *either*
            antenna is selected.
        taql: Extra TaQL predicate, ANDed with the rest.
        time_bin: Average the selection into bins this many seconds wide.
            ``None`` (the default) copies the rows unaveraged.
        chan_bin: Average this many channels together. ``None`` leaves the
            channel axis alone.
        datacolumn: Visibility column to average. Used only when averaging;
            an unaveraged subset copies every column.
        reindex: Renumber field and SPW ids from 0 and drop the subtable rows
            the output no longer uses, as CASA ``split`` does. Off by default,
            so ids keep matching the parent MS. If reindexing fails, no output
            is left behind.
        overwrite: Replace ``outms`` if it exists.

    Returns:
        The path written.

    Note:
        Averaging needs the ``average`` extra
        (``pip install 'msutils[average]'``); see :func:`average`.
    """
    if time_bin is not None or chan_bin is not None:
        return average(
            msname,
            outms,
            time_bin=1.0 if time_bin is None else time_bin,
            chan_bin=1 if chan_bin is None else chan_bin,
            fields=fields,
            spws=spws,
            scans=scans,
            antennas=antennas,
            taql=taql,
            datacolumn=datacolumn,
            reindex=reindex,
            overwrite=overwrite,
        )

    info = msinfo(msname, level="meta")
    where = _selection(info, fields=fields, spws=spws, scans=scans, antennas=antennas, taql=taql)
    _prepare_output(outms, overwrite)

    with open_table(msname) as tab:
        command = "SELECT FROM $1" + (" WHERE " + where if where else "")
        LOGGER.info("Subsetting %s -> %s (%s)", msname, outms, where or "all rows")
        with query(command, [tab]) as selection:
            nrows = selection.nrows()
            if nrows == 0:
                raise ValueError("selection matched no rows: {}".format(where or "all rows"))
            with closing_table(selection.copy(outms, deep=True)):
                pass

    if reindex:
        _reindex_output(outms)

    LOGGER.info("Wrote %d rows to %s", nrows, outms)
    return outms


def average(
    msname: str,
    outms: str,
    time_bin: float = 1.0,
    chan_bin: int = 1,
    fields: Sequence | None = None,
    spws: Sequence | None = None,
    scans: Sequence | None = None,
    antennas: Sequence | None = None,
    taql: str | None = None,
    datacolumn: str = "DATA",
    reindex: bool = False,
    overwrite: bool = False,
    rowchunk: int = 100000,
) -> str:
    """Time- and channel-average an MS into a new one.

    Averaging is done by :func:`africanus.averaging.time_and_channel`, which
    needs the ``average`` extra (``pip install 'msutils[average]'``). Rows are
    processed one ``(FIELD_ID, DATA_DESC_ID, SCAN_NUMBER)`` group at a time,
    which is the granularity africanus expects -- time bins must not span
    scans or spectral windows.

    Args:
        msname: Source MS.
        outms: Destination path.
        time_bin: Averaging interval in seconds. 1.0 (the default) leaves the
            time axis alone for typical integration times.
        chan_bin: Channels per output channel.
        fields, spws, scans, antennas, taql: Selection, as for :func:`subset`.
        datacolumn: Visibility column to average.
        reindex: Renumber field and SPW ids from 0, as for :func:`subset`.
        overwrite: Replace ``outms`` if it exists.
        rowchunk: Rows read per iteration within a group.

    Returns:
        The path written.
    """
    try:
        from africanus.averaging import time_and_channel
    except ImportError as exc:  # pragma: no cover - extra absent
        raise ImportError(
            "averaging needs codex-africanus. Install with: pip install 'msutils[average]'"
        ) from exc

    if chan_bin < 1:
        raise ValueError(f"chan_bin must be >= 1, got {chan_bin}")
    if time_bin <= 0:
        raise ValueError(f"time_bin must be > 0, got {time_bin}")

    info = msinfo(msname, level="meta")
    where = _selection(info, fields=fields, spws=spws, scans=scans, antennas=antennas, taql=taql)
    _prepare_output(outms, overwrite)

    with open_table(msname) as tab:
        if datacolumn not in tab.colnames():
            raise ValueError(f"{msname} has no column {datacolumn!r}")
        optional = [c for c in ("WEIGHT_SPECTRUM", "SIGMA_SPECTRUM") if c in tab.colnames()]
        groups = _groups(tab, where)
        if not groups:
            raise ValueError("selection matched no rows: {}".format(where or "all rows"))

        spw_of_ddid = {d.id: d.spw_id for d in info.data_descriptions}
        averaged_spw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        blocks: list[dict[str, Any]] = []

        for field_id, ddid, scan in groups:
            spw = info.spws[spw_of_ddid[ddid]]
            chan_freq = np.asarray(spw.chan_freq, dtype=np.float64)
            chan_width = np.asarray(spw.chan_width, dtype=np.float64)

            clauses = [
                f"FIELD_ID=={field_id:d}",
                f"DATA_DESC_ID=={ddid:d}",
                f"SCAN_NUMBER=={scan:d}",
            ]
            if where:
                clauses.append("(" + where + ")")
            command = (
                "SELECT FROM $1 WHERE "
                + " AND ".join(clauses)
                + " ORDERBY TIME, ANTENNA1, ANTENNA2"
            )

            with query(command, [tab]) as rows:
                if rows.nrows() == 0:
                    continue
                block = _average_group(
                    time_and_channel,
                    rows,
                    datacolumn,
                    optional,
                    chan_freq,
                    chan_width,
                    time_bin,
                    chan_bin,
                    rowchunk,
                )

            block.update(field_id=field_id, ddid=ddid, scan=scan)
            blocks.append(block)
            averaged_spw.setdefault(
                spw_of_ddid[ddid], (block.pop("chan_freq"), block.pop("chan_width"))
            )

        _write_output(msname, outms, info, blocks, averaged_spw, optional, datacolumn)

    if reindex:
        _reindex_output(outms, rowchunk=rowchunk)

    total = sum(len(b["time"]) for b in blocks)
    LOGGER.info(
        "Averaged %s -> %s: %d rows (time_bin=%gs, chan_bin=%d)",
        msname,
        outms,
        total,
        time_bin,
        chan_bin,
    )
    return outms


# --------------------------------------------------------------------------
# averaging internals


def _groups(tab, where: str) -> list[tuple[int, int, int]]:
    """The (field, ddid, scan) groups present, in a stable order.

    Time bins must not straddle scans or spectral windows, so this is the
    coarsest grouping that is safe to average within.
    """
    command = (
        "SELECT FIELD_ID, DATA_DESC_ID, SCAN_NUMBER FROM $1 "
        + ("WHERE " + where + " " if where else "")
        + "GROUPBY FIELD_ID, DATA_DESC_ID, SCAN_NUMBER"
    )
    with query(command, [tab]) as res:
        if res.nrows() == 0:
            return []
        return sorted(
            zip(
                (int(v) for v in res.getcol("FIELD_ID")),
                (int(v) for v in res.getcol("DATA_DESC_ID")),
                (int(v) for v in res.getcol("SCAN_NUMBER")),
                strict=False,
            )
        )


def _reconcile_flags(flag: np.ndarray, flag_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make FLAG and FLAG_ROW mutually consistent.

    africanus rejects inputs where ``flag_row`` disagrees with ``flag``
    ("flag_row and flag arrays mismatch"), but plenty of real MSs carry
    exactly that: rows flagged wholesale in FLAG with FLAG_ROW left at False,
    or the reverse. A row marked in FLAG_ROW means the whole row is bad, so
    fold that into FLAG, then derive FLAG_ROW back from it. Nothing flagged is
    ever unflagged by this.
    """
    flag = np.asarray(flag, dtype=bool)
    flag_row = np.asarray(flag_row, dtype=bool)
    if flag_row.any():
        flag = flag | flag_row[:, np.newaxis, np.newaxis]
    return flag, flag.all(axis=(1, 2))


def _average_group(
    time_and_channel,
    rows,
    datacolumn,
    optional,
    chan_freq,
    chan_width,
    time_bin,
    chan_bin,
    rowchunk,
) -> dict[str, Any]:
    """Average one (field, ddid, scan) group and return its output arrays."""
    flag, flag_row = _reconcile_flags(rows.getcol("FLAG"), rows.getcol("FLAG_ROW"))

    kwargs = {
        "time": rows.getcol("TIME"),
        "interval": rows.getcol("INTERVAL"),
        "antenna1": rows.getcol("ANTENNA1"),
        "antenna2": rows.getcol("ANTENNA2"),
        "time_centroid": rows.getcol("TIME_CENTROID"),
        "exposure": rows.getcol("EXPOSURE"),
        "flag_row": flag_row,
        "uvw": rows.getcol("UVW"),
        "weight": rows.getcol("WEIGHT"),
        "sigma": rows.getcol("SIGMA"),
        "chan_freq": chan_freq,
        "chan_width": chan_width,
        "effective_bw": chan_width,
        "resolution": chan_width,
        "visibilities": rows.getcol(datacolumn),
        "flag": flag,
        "time_bin_secs": float(time_bin),
        "chan_bin_size": int(chan_bin),
    }
    for column in optional:
        kwargs[column.lower()] = rows.getcol(column)

    result = time_and_channel(**kwargs)

    block = {
        "time": result.time,
        "interval": result.interval,
        "antenna1": result.antenna1,
        "antenna2": result.antenna2,
        "time_centroid": result.time_centroid,
        "exposure": result.exposure,
        "flag_row": result.flag_row,
        "uvw": result.uvw,
        "weight": result.weight,
        "sigma": result.sigma,
        "visibilities": result.visibilities,
        "flag": result.flag,
        "chan_freq": result.chan_freq,
        "chan_width": result.chan_width,
    }
    for column in optional:
        block[column] = getattr(result, column.lower())
    return block


def _write_output(
    msname: str, outms: str, info, blocks, averaged_spw, optional, datacolumn: str
) -> None:
    """Build the output MS: averaged main table, rebuilt SPECTRAL_WINDOW."""
    from casacore.tables import default_ms, makearrcoldesc, maketabdesc, table

    if not blocks:
        raise ValueError("nothing to write: no group produced any rows")

    ncorr = blocks[0]["visibilities"].shape[2]
    nchan = {b["visibilities"].shape[1] for b in blocks}
    # A fixed cell shape needs one channel count; mixed-width SPWs get a
    # variable-shaped column instead.
    coldescs = []
    if len(nchan) == 1:
        shape = [nchan.pop(), ncorr]
        coldescs.append(makearrcoldesc(datacolumn, 0 + 0j, shape=shape, valuetype="complex"))
        for column in optional:
            coldescs.append(makearrcoldesc(column, 0.0, shape=shape, valuetype="float"))
    else:
        coldescs.append(makearrcoldesc(datacolumn, 0 + 0j, ndim=2, valuetype="complex"))
        for column in optional:
            coldescs.append(makearrcoldesc(column, 0.0, ndim=2, valuetype="float"))
    default_ms(outms, maketabdesc(coldescs))

    # Subtables that averaging does not change are copied verbatim.
    for name in (
        "ANTENNA",
        "FIELD",
        "POLARIZATION",
        "DATA_DESCRIPTION",
        "OBSERVATION",
        "STATE",
        "FEED",
        "SOURCE",
        "HISTORY",
        "PROCESSOR",
        "POINTING",
    ):
        _copy_subtable(msname, outms, name)

    _write_averaged_spw(msname, outms, info, averaged_spw)

    with table(outms, readonly=False, ack=False) as out:
        row0 = 0
        for block in blocks:
            nrows = len(block["time"])
            out.addrows(nrows)
            out.putcol("TIME", block["time"], row0, nrows)
            out.putcol("TIME_CENTROID", block["time_centroid"], row0, nrows)
            out.putcol("INTERVAL", block["interval"], row0, nrows)
            out.putcol("EXPOSURE", block["exposure"], row0, nrows)
            out.putcol("ANTENNA1", block["antenna1"].astype(np.int32), row0, nrows)
            out.putcol("ANTENNA2", block["antenna2"].astype(np.int32), row0, nrows)
            out.putcol("UVW", block["uvw"], row0, nrows)
            out.putcol("WEIGHT", block["weight"], row0, nrows)
            out.putcol("SIGMA", block["sigma"], row0, nrows)
            out.putcol("FLAG", block["flag"], row0, nrows)
            out.putcol("FLAG_ROW", block["flag_row"], row0, nrows)
            out.putcol(datacolumn, block["visibilities"], row0, nrows)
            for column in optional:
                if block.get(column) is not None:
                    out.putcol(column, block[column], row0, nrows)

            for name, value in (
                ("FIELD_ID", block["field_id"]),
                ("DATA_DESC_ID", block["ddid"]),
                ("SCAN_NUMBER", block["scan"]),
            ):
                out.putcol(name, np.full(nrows, value, np.int32), row0, nrows)
            for name in (
                "ARRAY_ID",
                "OBSERVATION_ID",
                "PROCESSOR_ID",
                "FEED1",
                "FEED2",
                "STATE_ID",
            ):
                out.putcol(name, np.zeros(nrows, np.int32), row0, nrows)
            row0 += nrows


def _write_averaged_spw(msname: str, outms: str, info, averaged_spw) -> None:
    """Rewrite SPECTRAL_WINDOW with the averaged channel grid."""
    from casacore.tables import table

    _copy_subtable(msname, outms, "SPECTRAL_WINDOW")
    if not averaged_spw:
        return
    with table(outms + "::SPECTRAL_WINDOW", readonly=False, ack=False) as tab:
        for spw_id, (chan_freq, chan_width) in averaged_spw.items():
            if spw_id >= tab.nrows():  # pragma: no cover - malformed MS
                continue
            freqs = np.atleast_1d(chan_freq)
            widths = np.atleast_1d(chan_width)
            tab.putcell("NUM_CHAN", spw_id, len(freqs))
            tab.putcell("CHAN_FREQ", spw_id, freqs)
            tab.putcell("CHAN_WIDTH", spw_id, widths)
            tab.putcell("EFFECTIVE_BW", spw_id, widths)
            tab.putcell("RESOLUTION", spw_id, widths)
            tab.putcell("TOTAL_BANDWIDTH", spw_id, float(np.abs(widths).sum()))
            tab.putcell("REF_FREQUENCY", spw_id, float(freqs[0]))


def _copy_subtable(msname: str, outms: str, name: str) -> None:
    """Replace ``outms``'s subtable with the source's, keeping the keyword."""
    from casacore.tables import table

    source = os.path.join(msname, name)
    if not os.path.isdir(source):
        return
    target = os.path.join(outms, name)
    with open_table(source) as src:
        if os.path.exists(target):
            shutil.rmtree(target)
        with closing_table(src.copy(target, deep=True)):
            pass
    with table(outms, readonly=False, ack=False) as out:
        out.putkeyword(name, "Table: " + os.path.abspath(target))


# --------------------------------------------------------------------------
# reindexing

#: Subtables whose ``SPECTRAL_WINDOW_ID`` has to follow a renumbered
#: SPECTRAL_WINDOW. ``-1`` there means "every window" and is left as it is.
SPW_REFERENCES = ("FEED", "SOURCE", "SYSCAL", "FREQ_OFFSET", "CAL_DEVICE", "SYSPOWER")


def _reindex_output(outms: str, rowchunk: int = 100000) -> None:
    """Reindex ``outms`` in place, removing it if reindexing fails.

    The output is fully written before reindexing starts, so a failure would
    otherwise leave a valid but unreindexed MS behind -- and a retry would
    trip over it with ``FileExistsError``. Failure leaves nothing instead.
    """
    try:
        _reindex(outms, rowchunk=rowchunk)
    except Exception:
        shutil.rmtree(outms, ignore_errors=True)
        raise


def _reindex(outms: str, rowchunk: int = 100000) -> None:
    """Renumber ``outms``'s field and SPW ids from 0, CASA ``split`` style.

    The FIELD, SPECTRAL_WINDOW and DATA_DESCRIPTION rows the output does not
    use are dropped, the survivors keep their relative order -- so the lowest
    surviving id becomes 0 -- and the main table's index columns are rewritten
    to match. FEED and friends follow the renumbered SPWs.

    Subtables a row selection cannot invalidate are left whole: ANTENNA,
    POLARIZATION, STATE and OBSERVATION keep every row, so ``ANTENNA1``,
    ``STATE_ID`` and the rest still resolve. Antennas in particular are *not*
    pruned -- ``antennas=`` keeps a baseline if either end is selected, and
    dropping ANTENNA rows would invalidate FEED, POINTING and SYSCAL for no
    real gain. ``SOURCE_ID`` is likewise left alone: SOURCE is keyed by that
    *value* rather than by row number, so the surviving FIELD rows go on
    resolving without it being touched.
    """
    with open_table(outms) as tab:
        used_fields = _distinct(tab, "FIELD_ID")
        used_ddids = _distinct(tab, "DATA_DESC_ID")

    with open_table(os.path.join(outms, "DATA_DESCRIPTION")) as tab:
        spw_of_ddid = [int(v) for v in tab.getcol("SPECTRAL_WINDOW_ID")]
    dangling = [d for d in used_ddids if d >= len(spw_of_ddid)]
    if dangling:
        raise ValueError(f"cannot reindex: DATA_DESC_ID {dangling} has no DATA_DESCRIPTION row")
    used_spws = sorted({spw_of_ddid[d] for d in used_ddids})

    field_map = _prune(outms, "FIELD", used_fields, "FIELD_ID")
    spw_map = _prune(outms, "SPECTRAL_WINDOW", used_spws, "SPECTRAL_WINDOW_ID")
    ddid_map = _prune(outms, "DATA_DESCRIPTION", used_ddids, "DATA_DESC_ID")

    # DATA_DESCRIPTION now holds the kept rows in their original order, so its
    # SPECTRAL_WINDOW_ID column is the kept ddids' windows, renumbered.
    with open_table(os.path.join(outms, "DATA_DESCRIPTION"), readonly=False) as tab:
        tab.putcol(
            "SPECTRAL_WINDOW_ID",
            np.array([spw_map[spw_of_ddid[d]] for d in used_ddids], np.int32),
        )

    with open_table(outms, readonly=False) as tab:
        _remap_column(tab, "FIELD_ID", field_map, rowchunk)
        _remap_column(tab, "DATA_DESC_ID", ddid_map, rowchunk)

    for name in SPW_REFERENCES:
        _follow_spws(outms, name, spw_map)

    LOGGER.info(
        "Reindexed %s: %d field(s), %d SPW(s), %d data description(s)",
        outms,
        len(field_map),
        len(spw_map),
        len(ddid_map),
    )


def _distinct(tab, column: str) -> list[int]:
    """The distinct values of an index column, sorted, aggregated in TaQL."""
    with query(f"SELECT DISTINCT {column} FROM $1", [tab]) as res:
        if res.nrows() == 0:
            return []
        return sorted({int(v) for v in res.getcol(column)})


def _prune(outms: str, name: str, keep: Sequence[int], column: str) -> dict[int, int]:
    """Drop every row of subtable ``name`` bar ``keep``; return old id -> new id.

    Removal preserves row order, so a kept row's new id is its rank among the
    kept ones -- hence the sort.
    """
    keep = sorted(keep)
    with open_table(os.path.join(outms, name), readonly=False) as tab:
        nrows = tab.nrows()
        dangling = [row for row in keep if not 0 <= row < nrows]
        if dangling:
            raise ValueError(
                f"cannot reindex: {column} {dangling} has no row in {name}; "
                "msutils.check() reports such dangling references"
            )
        drop = sorted(set(range(nrows)) - set(keep))
        if drop:
            tab.removerows(drop)
    return {old: new for new, old in enumerate(keep)}


def _remap_column(tab, column: str, mapping: dict[int, int], rowchunk: int) -> None:
    """Rewrite an index column through ``mapping``, a chunk of rows at a time."""
    if not mapping or all(old == new for old, new in mapping.items()):
        return
    lookup = np.zeros(max(mapping) + 1, np.int32)
    for old, new in mapping.items():
        lookup[old] = new
    nrows = tab.nrows()
    for row0 in range(0, nrows, rowchunk):
        count = min(rowchunk, nrows - row0)
        tab.putcol(column, lookup[tab.getcol(column, row0, count)], row0, count)


def _follow_spws(outms: str, name: str, spw_map: dict[int, int]) -> None:
    """Drop a subtable's rows for removed SPWs and renumber the rest."""
    path = os.path.join(outms, name)
    if not os.path.isdir(path):
        return
    with open_table(path, readonly=False) as tab:
        if tab.nrows() == 0 or "SPECTRAL_WINDOW_ID" not in tab.colnames():
            return
        spws = [int(v) for v in tab.getcol("SPECTRAL_WINDOW_ID")]
        drop = {row for row, spw in enumerate(spws) if spw >= 0 and spw not in spw_map}
        if drop:
            tab.removerows(sorted(drop))
            spws = [spw for row, spw in enumerate(spws) if row not in drop]
        remapped = [spw_map[s] if s >= 0 else s for s in spws]
        if remapped != spws:
            tab.putcol("SPECTRAL_WINDOW_ID", np.array(remapped, np.int32))


# --------------------------------------------------------------------------
# selection


def _selection(info, fields=None, spws=None, scans=None, antennas=None, taql=None) -> str:
    """Build a TaQL WHERE predicate from the selection arguments."""
    clauses = []
    field_ids = _resolve(fields, info.fields, "field")
    if field_ids:
        clauses.append("FIELD_ID IN [{}]".format(",".join(map(str, field_ids))))

    spw_ids = _resolve(spws, info.spws, "spectral window")
    if spw_ids:
        # Select on DATA_DESC_ID, which is what the main table stores.
        ddids = [d.id for d in info.data_descriptions if d.spw_id in spw_ids]
        if not ddids:
            raise ValueError("no DATA_DESC_IDs map to the requested SPWs")
        clauses.append("DATA_DESC_ID IN [{}]".format(",".join(map(str, ddids))))

    if scans:
        clauses.append("SCAN_NUMBER IN [{}]".format(",".join(str(int(s)) for s in scans)))

    antenna_ids = _resolve(antennas, info.antennas, "antenna")
    if antenna_ids:
        listed = ",".join(map(str, antenna_ids))
        clauses.append(f"(ANTENNA1 IN [{listed}] OR ANTENNA2 IN [{listed}])")

    if taql:
        clauses.append("(" + taql + ")")

    return " AND ".join(clauses)


def _resolve(selection, registry, label: str) -> list[int] | None:
    """Map a mixed list of ids and names to ids."""
    if not selection:
        return None
    resolved = []
    for item in selection:
        if isinstance(item, str) and not item.lstrip("-").isdigit():
            try:
                resolved.append(registry[item].id)
            except KeyError:
                raise ValueError(f"unknown {label} {item!r}; have {registry.names}") from None
        else:
            ident = int(item)
            if ident not in registry.ids:
                raise ValueError(f"unknown {label} id {ident}; have {registry.ids}")
            resolved.append(ident)
    return resolved


def _prepare_output(outms: str, overwrite: bool) -> None:
    if os.path.exists(outms):
        if not overwrite:
            raise FileExistsError(f"{outms} already exists; pass overwrite=True to replace it")
        shutil.rmtree(outms)
