"""Core Measurement Set column operations (numpy + python-casacore only)."""

from __future__ import annotations

import math
import traceback
from collections.abc import Sequence
from contextlib import closing
from typing import Any

import numpy as np
from casacore.tables import makearrcoldesc, makescacoldesc, maketabdesc

from ._compat import summary
from ._log import create_logger
from ._tables import open_table
from .info._model import STOKES_TYPES

__all__ = [
    "STOKES_TYPES",
    "addcol",
    "addnoise",
    "compute_vis_noise",
    "copycol",
    "delcol",
    "renamecol",
    "sumcols",
    "summary",
    "verify_antpos",
]

LOGGER = create_logger(__name__)


def _measures():
    """casacore.measures handle, created on first use.

    Importing ``casacore.measures`` costs ~140 ms and needs the measures data
    tables, but only :func:`verify_antpos` uses it -- so it stays out of the
    package import path.
    """
    import casacore.measures

    return casacore.measures.measures()


def addcol(
    msname: str,
    colname: str | None = None,
    shape: Sequence[int] | None = None,
    data_desc_type: str = "array",
    valuetype: str | None = None,
    init_with: Any = None,
    coldesc: dict | None = None,
    coldmi: dict | None = None,
    clone: str | None = "DATA",
    rowchunk: int | None = None,
    **kw,
) -> str:
    """Add a column to an MS.

    Args:
        msname: MS to add the column to.
        colname: Name of the new column.
        shape: Cell shape for an array column, e.g. ``[nchan, ncorr]``.
        data_desc_type: ``'array'`` (default) or ``'scalar'`` -- whether each
            cell holds an array or a single value.
        valuetype: casacore value type, e.g. ``'complex'``, ``'float'``,
            ``'int'``, ``'bool'``. Defaults to ``'complex'``.
        init_with: Value to initialise every cell with. ``None`` leaves the
            column unfilled.
        coldesc: A ready-made column description, overriding everything else.
        coldmi: Optional data manager info passed to ``addcols``.
        clone: Existing column whose shape and type the new column should
            match. This is the usual way to create a visibility column.
        rowchunk: Rows per write when initialising.

    Returns:
        ``'added'``, or ``'exists'`` if the column was already there.
    """
    if not colname:
        raise ValueError("colname is required")
    if data_desc_type not in ("array", "scalar"):
        raise ValueError(f"data_desc_type must be 'array' or 'scalar', got {data_desc_type!r}")

    with open_table(msname, readonly=False) as tab:
        if colname in tab.colnames():
            LOGGER.info("Column already exists")
            return "exists"

        LOGGER.info("Attempting to add %s column to %s", colname, msname)

        valuetype = valuetype or "complex"
        fill = init_with if init_with is not None else _zero_of(valuetype)

        if coldesc:
            data_desc = coldesc
            shape = coldesc.get("shape")
        elif shape:
            data_desc = maketabdesc(
                makearrcoldesc(colname, fill, shape=list(shape), valuetype=valuetype)
            )
        elif data_desc_type == "scalar":
            shape = []
            data_desc = maketabdesc(makescacoldesc(colname, fill, valuetype=valuetype))
        elif clone:
            if clone not in tab.colnames():
                raise ValueError(f"cannot clone {clone!r}: no such column in {msname}")
            element = tab.getcell(clone, 0)
            if np.ndim(element):  # array column
                shape = list(np.shape(element))
                data_desc = maketabdesc(
                    makearrcoldesc(
                        colname, np.asarray(element).flatten()[0], shape=shape, valuetype=valuetype
                    )
                )
            else:  # scalar column
                shape = []
                data_desc = maketabdesc(makescacoldesc(colname, element, valuetype=valuetype))
        else:
            # Previously fell through with `data_desc` unbound, raising an
            # opaque UnboundLocalError.
            raise ValueError(
                f"not enough information to create column {colname!r}: pass one of "
                "`shape`, `coldesc`, `clone`, or data_desc_type='scalar'"
            )

        colinfo = [data_desc, coldmi] if coldmi else [data_desc]
        tab.addcols(*colinfo)

        LOGGER.info("Column added successfuly.")

        if init_with is None:
            return "added"

        spwids = sorted(set(tab.getcol("DATA_DESC_ID")))
        for spw in spwids:
            LOGGER.info("Initialising %s column with %s. DDID is %d", colname, init_with, spw)
            with closing(tab.query(f"DATA_DESC_ID=={spw:d}")) as tab_spw:
                nrows = tab_spw.nrows()
                chunk = rowchunk or max(nrows // 10, 1)
                dshape = [0, *list(shape or [])]
                for row0 in range(0, nrows, chunk):
                    nr = min(chunk, nrows - row0)
                    dshape[0] = nr
                    LOGGER.info(
                        "Writing to column  %s (rows %d to %d)", colname, row0, row0 + nr - 1
                    )
                    tab_spw.putcol(
                        colname,
                        np.full(dshape, init_with, dtype=np.asarray(init_with).dtype),
                        row0,
                        nr,
                    )

    return "added"


#: Zero value of the right Python type for each casacore value type, used as
#: the placeholder element when building a column description.
_ZERO_VALUES = {
    "complex": 0 + 0j,
    "dcomplex": 0 + 0j,
    "float": 0.0,
    "double": 0.0,
    "int": 0,
    "uint": 0,
    "short": 0,
    "ushort": 0,
    "int64": 0,
    "bool": False,
    "boolean": False,
    "string": "",
}


def _zero_of(valuetype: str) -> Any:
    """A zero of ``valuetype``, for column descriptions with no init value."""
    return _ZERO_VALUES.get(str(valuetype).lower(), 0.0)


def _required_columns() -> frozenset:
    """Main-table columns the MSv2 standard requires."""
    from casacore.tables import required_ms_desc

    return frozenset(k for k in required_ms_desc() if not k.startswith("_"))


def delcol(msname: str, *colnames: str, force: bool = False) -> list[str]:
    """Remove columns from an MS.

    Reclaiming the space taken by ``CORRECTED_DATA``/``MODEL_DATA`` after
    calibration is one of the most common MS chores, and casacore's
    ``removecols`` is the only way to do it without rewriting the MS.

    Args:
        msname: MS to modify.
        colnames: Columns to remove. Missing ones are skipped with a warning.
        force: Allow removing columns the MSv2 standard requires. Off by
            default, since doing so produces an MS that other tools reject.

    Returns:
        The names actually removed.
    """
    if not colnames:
        raise ValueError("no columns given")

    with open_table(msname, readonly=False) as tab:
        present = set(tab.colnames())
        required = _required_columns()

        removable = []
        for name in colnames:
            if name not in present:
                LOGGER.warning("Column %s is not in %s; skipping", name, msname)
                continue
            if name in required and not force:
                raise ValueError(
                    f"{name} is required by the MSv2 standard; removing it will "
                    f"make {msname} unreadable to other tools. Pass force=True if "
                    "you really mean it."
                )
            removable.append(name)

        if not removable:
            return []

        LOGGER.info("Removing column(s) %s from %s", ", ".join(removable), msname)
        tab.removecols(removable)

    return removable


def renamecol(msname: str, fromcol: str, tocol: str) -> None:
    """Rename a column in place, without copying its data.

    Args:
        msname: MS to modify.
        fromcol: Existing column name.
        tocol: New name; must not already exist.
    """
    with open_table(msname, readonly=False) as tab:
        colnames = tab.colnames()
        if fromcol not in colnames:
            raise ValueError(f"{msname} has no column {fromcol!r}")
        if tocol in colnames:
            raise ValueError(f"{msname} already has a column {tocol!r}")
        if fromcol in _required_columns():
            raise ValueError(f"{fromcol} is required by the MSv2 standard and cannot be renamed")
        LOGGER.info("Renaming %s to %s in %s", fromcol, tocol, msname)
        tab.renamecol(fromcol, tocol)


def sumcols(
    msname: str,
    col1: str | None = None,
    col2: str | None = None,
    outcol: str | None = None,
    cols: Sequence[str] | None = None,
    subtract: bool = False,
) -> None:
    """Add col1 to col2, or sum columns in 'cols' list.
    If subtract, subtract col2 from col1
    """

    with open_table(msname, readonly=False) as tab:
        if outcol not in tab.colnames():
            LOGGER.info("outcol %s does not exist, will add it first.", outcol)
            addcol(msname, outcol, clone=col1 or cols[0])

        spws = set(tab.getcol("DATA_DESC_ID"))
        for spw in spws:
            with closing(tab.query(f"DATA_DESC_ID=={spw:d}")) as tab_spw:
                nrows = tab_spw.nrows()
                rowchunk = nrows // 10 if nrows > 10000 else nrows
                for row0 in range(0, nrows, rowchunk):
                    nr = min(rowchunk, nrows - row0)
                    LOGGER.info(
                        "Writing to column  %s (rows %d to %d)", outcol, row0, row0 + nr - 1
                    )
                    if subtract:
                        data = tab_spw.getcol(col1, row0, nr) - tab_spw.getcol(col2, row0, nr)
                    else:
                        cols = cols or [col1, col2]
                        data = 0
                        for col in cols:
                            data += tab_spw.getcol(col, row0, nr)

                    tab_spw.putcol(outcol, data, row0, nr)


def copycol(msname: str, fromcol: str, tocol: str) -> None:
    """
    Copy data from one column to another
    """

    with open_table(msname, readonly=False) as tab:
        if tocol not in tab.colnames():
            addcol(msname, tocol, clone=fromcol)

        spws = set(tab.getcol("DATA_DESC_ID"))
        for spw in spws:
            with closing(tab.query(f"DATA_DESC_ID=={spw:d}")) as tab_spw:
                nrows = tab_spw.nrows()
                rowchunk = nrows // 10 if nrows > 5000 else nrows
                for row0 in range(0, nrows, rowchunk):
                    nr = min(rowchunk, nrows - row0)
                    data = tab_spw.getcol(fromcol, row0, nr)
                    tab_spw.putcol(tocol, data, row0, nr)


def compute_vis_noise(msname: str, sefd: float, spw_id: int = 0) -> float:
    """Computes nominal per-visibility noise"""

    with open_table(msname) as tab, open_table(msname + "::SPECTRAL_WINDOW") as spwtab:
        freq0 = spwtab.getcol("CHAN_FREQ")[spw_id, 0]
        wavelength = 300e6 / freq0
        bw = spwtab.getcol("CHAN_WIDTH")[spw_id, 0]
        dt = tab.getcol("EXPOSURE", 0, 1)[0]
        dtf = (tab.getcol("TIME", tab.nrows() - 1, 1) - tab.getcol("TIME", 0, 1))[0]

    LOGGER.info(
        "%s freq %.2f MHz (lambda=%.2fm), bandwidth %.2g kHz, %.2fs integrations, %.2fh synthesis",
        msname,
        freq0 * 1e-6,
        wavelength,
        bw * 1e-3,
        dt,
        dtf / 3600,
    )
    noise = sefd / math.sqrt(abs(2 * bw * dt))
    LOGGER.info("SEFD of %.2f Jy gives per-visibility noise of %.2f mJy", sefd, noise * 1000)

    return noise


def verify_antpos(msname: str, fix: bool = False, hemisphere: int | None = None) -> None:
    """Verifies antenna Y positions in MS. If Y coordinate convention is wrong, either fixes the positions (fix=True) or
    raises an error. hemisphere=-1 makes it assume that the observatory is in the Western hemisphere, hemisphere=1
    in the Eastern, or else tries to find observatory name using MS and casacore.measures."""

    if not hemisphere:
        with open_table(msname + "::OBSERVATION") as obstab:
            obs = obstab.getcol("TELESCOPE_NAME")[0]
        LOGGER.info("observatory is %s", obs)
        try:
            hemisphere = 1 if _measures().observatory(obs)["m0"]["value"] > 0 else -1
        except Exception:
            traceback.print_exc()
            LOGGER.warning(
                "%s is unknown, or casacore.measures is missing. Will not verify antenna positions.",
                obs,
            )
            return
    LOGGER.info("antenna Y positions should be of sign %+d", hemisphere)

    with open_table(msname + "::ANTENNA", readonly=False) as anttab:
        pos = anttab.getcol("POSITION")
        wrong = pos[:, 1] < 0 if hemisphere > 0 else pos[:, 1] > 0
        nw = sum(wrong)

        if nw:
            if not fix:
                raise RuntimeError(
                    f"{msname}/ANTENNA has {nw} incorrect Y antenna positions. Check "
                    "your coordinate conversions (from UVFITS?), or run "
                    "verify_antpos(fix=True)"
                )
            pos[wrong, 1] *= -1
            anttab.putcol("POSITION", pos)
            LOGGER.warning(
                "%s/ANTENNA: %s incorrect antenna positions were adjusted (Y sign flipped)",
                msname,
                nw,
            )
        else:
            LOGGER.info("%s/ANTENNA: all antenna positions appear to have correct Y sign", msname)


def addnoise(
    msname: str,
    column: str = "MODEL_DATA",
    noise: float | Sequence[float] = 0,
    sefd: float | Sequence[float] = 551,
    rowchunk: int | None = None,
    addToCol: str | None = None,
    spw_id: int | None = None,
) -> None:
    """Add Gaussian noise to a visibility column.

    Args:
        msname: MS to write into.
        column: Column receiving the noisy visibilities.
        noise: Noise standard deviation. Either a scalar, or one value per
            channel, in which case it is broadcast along the frequency axis.
        sefd: SEFD in Jy, used to derive ``noise`` when ``noise`` is falsy.
        rowchunk: Rows per write.
        addToCol: Add the noise to this column's data instead of writing pure
            noise.
        spw_id: Spectral window used when deriving the noise from ``sefd``.
    """
    with open_table(msname, readonly=False) as tab:
        multi_chan_noise = hasattr(noise, "__iter__") or hasattr(sefd, "__iter__")
        if multi_chan_noise:
            # Shape once, up front. Reshaping inside the row loop appended a
            # fresh axis on every chunk, so any MS needing more than one chunk
            # died with a broadcasting error on the second one.
            noise = np.asarray(noise)[np.newaxis, :, np.newaxis]
        else:
            noise = noise or compute_vis_noise(msname, sefd=sefd, spw_id=spw_id or 0)

        spws = sorted(set(tab.getcol("DATA_DESC_ID")))
        for spw in spws:
            with closing(tab.query(f"DATA_DESC_ID=={spw:d}")) as tab_spw:
                nrows = tab_spw.nrows()
                nchan, ncor = tab_spw.getcell(column, 0).shape

                if multi_chan_noise and noise.shape[1] not in (1, nchan):
                    raise ValueError(
                        f"noise has {noise.shape[1]:d} channels but DATA_DESC_ID {spw:d} has "
                        f"{nchan:d}"
                    )

                chunk = rowchunk or max(nrows // 10, 1)

                for row0 in range(0, nrows, chunk):
                    nr = min(chunk, nrows - row0)
                    data = np.random.randn(nr, nchan, ncor) + 1j * np.random.randn(nr, nchan, ncor)
                    data *= noise

                    if addToCol:
                        data += tab_spw.getcol(addToCol, row0, nr)
                        LOGGER.info(
                            "%s + noise --> %s (rows %d to %d)",
                            addToCol,
                            column,
                            row0,
                            row0 + nr - 1,
                        )
                    else:
                        LOGGER.info(
                            "Adding noise to column %s (rows %d to %d)", column, row0, row0 + nr - 1
                        )

                    tab_spw.putcol(column, data, row0, nr)
