"""Format detection, and the one door in and out of the gain model.

Every operation calls :func:`read_gains` / :func:`write_gains` and never a
format module directly, so a new format is one entry in `_READERS` and
nothing else changes. Detection is by what is on disk -- a casacore table
has a ``table.dat``, a zarr store has a ``.zgroup`` -- rather than by
extension, because neither format's directory name is load-bearing.
"""

from __future__ import annotations

from pathlib import Path

from ._casa import is_casa_table, read_casa, write_casa
from ._model import GainTable
from ._quartical import is_quartical_store, read_quartical

__all__ = ["detect_format", "read_gains", "write_gains"]

#: Format name -> (detector, reader). Order matters only in that the first
#: match wins; the two detectors are mutually exclusive in practice.
_READERS = {
    "casa": (is_casa_table, read_casa),
    "quartical": (is_quartical_store, read_quartical),
}


def detect_format(path: str | Path) -> str:
    """The format of the gain table/store at `path`.

    Raises:
        FileNotFoundError: If nothing is there.
        ValueError: If it is neither a casacore table nor a zarr store.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no gain table or store at {path}")
    for name, (detector, _) in _READERS.items():
        if detector(path):
            return name
    raise ValueError(
        f"{path} is neither a CASA calibration table (no table.dat) nor a QuartiCal store (no .zgroup)"
    )


def read_gains(path: str | Path, *, format: str | None = None, term: str | None = None) -> GainTable:
    """Read gains from a CASA caltable or a QuartiCal store.

    Args:
        path: Table or store to read.
        format: Force ``"casa"`` or ``"quartical"`` instead of detecting.
        term: Which term to read; QuartiCal stores only, where it is
            required if the store holds more than one.

    Returns:
        A :class:`~msutils.gains.GainTable`.
    """
    format = format or detect_format(path)
    if format not in _READERS:
        raise ValueError(f"unknown gain format {format!r} (known: {sorted(_READERS)})")
    if format == "quartical":
        return read_quartical(path, term=term)
    if term is not None:
        raise ValueError("`term` selects a term within a QuartiCal store; a CASA caltable holds one term")
    return read_casa(path)


def write_gains(gains: GainTable, dest: str | Path, *, template: str | Path | None = None) -> str:
    """Write a :class:`~msutils.gains.GainTable` back out in its own format.

    Args:
        gains: What to write.
        dest: Where to write it; must not exist. Operations never write in
            place -- a half-rewritten calibration table is unrecoverable,
            and a pipeline that caches on declared outputs cannot see an
            in-place edit at all.
        template: Structure to copy (CASA only). Defaults to the table the
            gains were read from.

    Returns:
        The path written.
    """
    if gains.format == "casa":
        return write_casa(gains, dest, template=template)
    raise NotImplementedError(
        f"writing {gains.format!r} gains is not implemented yet -- only CASA caltables can be written. "
        "Reading a QuartiCal store and writing a caltable is a format conversion, which needs its own "
        "command rather than falling out of a write."
    )
