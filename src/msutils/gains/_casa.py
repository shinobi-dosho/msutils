"""Reading and writing CASA calibration tables.

A caltable is row-major -- one row per (time, antenna), each holding a
``(channel, correlation)`` array -- so reading is a fold into the dense
`(time, freq, antenna, correlation)` block the model defines, and writing is
the inverse. The fold is done per ``(FIELD_ID, SPECTRAL_WINDOW_ID)``, which
is what makes a table holding two calibrators (the shape CASA's own
``fluxscale`` requires) come back as two blocks rather than one ragged one.

**Writing goes through a copy of the source table**, never a table built from
scratch. A caltable carries keywords (``VisCal``, ``ParType``, ``MSName``)
and subtables that nothing here models, and inventing them is how a table
becomes unreadable by the tools that consume it. So the writer selects the
rows it is keeping, copies them, and puts new values into the copy -- every
column it does not understand survives untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from msutils._tables import open_table, table_exists

from ._model import GainBlock, GainTable

__all__ = ["is_casa_table", "read_casa", "write_casa"]

#: Fallback correlation labels, by count. A caltable records no
#: ``CORR_TYPE``, so the feed basis is not recoverable from the table alone;
#: these are labels for reporting, and nothing computes with them.
_CORR_LABELS = {1: ("I",), 2: ("0", "1"), 4: ("0", "1", "2", "3")}


def is_casa_table(path: str | Path) -> bool:
    """Whether `path` looks like a casacore table (as opposed to a zarr store)."""
    return (Path(path) / "table.dat").exists()


def _corr_labels(n_corr: int) -> tuple[str, ...]:
    return _CORR_LABELS.get(n_corr, tuple(str(i) for i in range(n_corr)))


def _channel_frequencies(path: str, spw_id: int, n_chan: int) -> np.ndarray:
    """The frequencies a block's channel axis stands for.

    A `G` solve writes one channel while its spectral window has hundreds,
    and the honest frequency for that single solution is the window's
    centre -- which is also the frequency CASA quotes a bootstrapped flux
    density at. A per-channel (`B`) solve maps one-to-one. Anything else is
    left as NaN rather than guessed at, so a consumer cannot mistake a
    fabricated axis for a real one.
    """
    if not table_exists(f"{path}::SPECTRAL_WINDOW"):
        return np.full(n_chan, np.nan)
    with open_table(f"{path}::SPECTRAL_WINDOW") as spw:
        if spw_id >= spw.nrows():
            return np.full(n_chan, np.nan)
        chan_freq = np.asarray(spw.getcell("CHAN_FREQ", spw_id), dtype=float)
    if n_chan == chan_freq.size:
        return chan_freq
    if n_chan == 1:
        return np.array([chan_freq.mean()])
    return np.full(n_chan, np.nan)


def _subtable_names(path: str, subtable: str) -> tuple[str, ...]:
    if not table_exists(f"{path}::{subtable}"):
        return ()
    with open_table(f"{path}::{subtable}") as tab:
        if "NAME" not in tab.colnames():
            return ()
        return tuple(str(n) for n in tab.getcol("NAME"))


def _antenna_label(names: tuple[str, ...], ant_id: int) -> str:
    """An antenna's name if the table records one, else its id as a string."""
    return names[ant_id] if 0 <= ant_id < len(names) and names[ant_id] else str(ant_id)


def read_casa(path: str | Path) -> GainTable:
    """Read a CASA calibration table into a :class:`GainTable`.

    Args:
        path: The caltable directory.

    Returns:
        A `GainTable` with one block per (field, spectral window) present.
        Solutions absent from the table -- an antenna with no row at some
        time -- come back flagged, not zero.
    """
    path = str(path)
    with open_table(path) as tab:
        keywords = tab.getkeywords()
        param_col = "CPARAM" if "CPARAM" in tab.colnames() else "FPARAM"
        times = np.asarray(tab.getcol("TIME"), dtype=float)
        field_ids = np.asarray(tab.getcol("FIELD_ID"), dtype=int)
        spw_ids = np.asarray(tab.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
        ant_ids = np.asarray(tab.getcol("ANTENNA1"), dtype=int)
        params = np.asarray(tab.getcol(param_col))
        flags = np.asarray(tab.getcol("FLAG"), dtype=bool)

    if params.ndim != 3:  # (row, channel, correlation)
        raise ValueError(f"{path}: expected a 3-D {param_col}, got shape {params.shape}")
    n_chan, n_corr = params.shape[1:]
    corrs = _corr_labels(n_corr)
    antenna_names = _subtable_names(path, "ANTENNA")
    field_names = _subtable_names(path, "FIELD")

    blocks: list[GainBlock] = []
    for field_id in sorted(set(field_ids.tolist())):
        for spw_id in sorted(set(spw_ids[field_ids == field_id].tolist())):
            rows = np.flatnonzero((field_ids == field_id) & (spw_ids == spw_id))
            block_times = np.unique(times[rows])
            block_ants = np.unique(ant_ids[rows])
            shape = (block_times.size, n_chan, block_ants.size, n_corr)
            gains = np.zeros(shape, dtype=complex)
            # Flagged by default: a (time, antenna) pair with no row must not
            # read as a valid zero gain once it is folded into a dense array.
            block_flags = np.ones(shape, dtype=bool)
            row_map = np.full((block_times.size, block_ants.size), -1, dtype=int)
            time_index = {t: i for i, t in enumerate(block_times.tolist())}
            ant_index = {a: i for i, a in enumerate(block_ants.tolist())}
            for row in rows.tolist():
                ti = time_index[times[row]]
                ai = ant_index[ant_ids[row]]
                gains[ti, :, ai, :] = params[row]
                block_flags[ti, :, ai, :] = flags[row]
                row_map[ti, ai] = row
            blocks.append(
                GainBlock(
                    gains=gains,
                    flags=block_flags,
                    times=block_times,
                    freqs=_channel_frequencies(path, spw_id, n_chan),
                    antennas=tuple(_antenna_label(antenna_names, a) for a in block_ants.tolist()),
                    corrs=corrs,
                    field_id=int(field_id),
                    spw_id=int(spw_id),
                    rows=row_map,
                )
            )

    return GainTable(
        blocks=blocks,
        term=str(keywords.get("VisCal", "")),
        gain_type=str(keywords.get("VisCal", "")),
        param_type="complex" if param_col == "CPARAM" else "float",
        format="casa",
        path=path,
        antenna_names=antenna_names,
        field_names={
            fid: field_names[fid]
            for fid in {b.field_id for b in blocks}
            if fid is not None and fid < len(field_names)
        },
    )


def write_casa(gains: GainTable, dest: str | Path, *, template: str | Path | None = None) -> str:
    """Write a :class:`GainTable` out as a CASA caltable.

    The output is a copy of `template` (the table `gains` was read from,
    unless another is given) holding **only the rows the blocks cover**, with
    their values replaced. Subsetting matters: CASA's own `fluxscale` leaves
    the reference calibrator's solutions in its output table, and applying
    that table without naming a field corrects the target with a blend of two
    calibrators' gains -- a bug caracal had to fix downstream. A table
    holding exactly what it claims to cannot be misapplied that way.

    Args:
        gains: What to write. Every block must carry a `rows` map, i.e. have
            come from a CASA table.
        dest: Output table path; must not exist.
        template: Table to copy structure and untouched columns from.
            Defaults to `gains.path`.

    Returns:
        The path written.
    """
    dest = str(dest)
    template = str(template or gains.path)
    if not template:
        raise ValueError("write_casa needs a template table (the gains carry no source path)")
    if Path(dest).exists():
        raise FileExistsError(f"{dest} already exists -- gain operations never overwrite in place")
    missing = [b.key for b in gains.blocks if b.rows is None]
    if missing:
        raise ValueError(
            f"cannot write blocks {missing} as a CASA table: they carry no source rows "
            "(they did not come from one). Converting between formats is a separate operation."
        )

    kept = np.unique(np.concatenate([b.rows[b.rows >= 0].ravel() for b in gains.blocks]))
    with open_table(template) as tab:
        selection = tab.selectrows(kept.tolist())
        try:
            selection.copy(dest, deep=True)
        finally:
            selection.close()

    # Row r of the copy is `kept[r]` of the source, so the blocks' own row
    # maps translate by position in `kept`.
    new_row = {int(old): int(new) for new, old in enumerate(kept.tolist())}
    with open_table(dest, readonly=False) as tab:
        param_col = "CPARAM" if "CPARAM" in tab.colnames() else "FPARAM"
        params = np.asarray(tab.getcol(param_col))
        flags = np.asarray(tab.getcol("FLAG"), dtype=bool)
        for block in gains.blocks:
            for ti in range(block.shape[0]):
                for ai in range(block.shape[2]):
                    source_row = int(block.rows[ti, ai])
                    if source_row < 0:
                        continue
                    row = new_row[source_row]
                    values = block.gains[ti, :, ai, :]
                    params[row] = values.real if param_col == "FPARAM" else values
                    flags[row] = block.flags[ti, :, ai, :]
        tab.putcol(param_col, params)
        tab.putcol("FLAG", flags)
    return dest
