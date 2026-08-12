"""Reading QuartiCal gain stores.

A store is a zarr group per term, holding one dataset per (field,
data-description) that the solve chunked over::

    <store>.qc/
      G/
        G_0/    gains (gain_time, gain_freq, antenna, direction, correlation)
                gain_flags (gain_time, gain_freq, antenna, direction)
                attrs: FIELD_ID, DATA_DESC_ID, FIELD_NAME, NAME, TYPE
        G_1/    ...
      K/
        K_0/    ...

which is already the model's shape, so reading is a rename of axes and a
split on `direction` rather than a fold. Two details that are not obvious
from the arrays alone:

* **flags carry no correlation axis.** QuartiCal flags a solution, not a
  correlation of one, so the flag array is broadcast across correlations on
  the way in. A writer must undo that (and refuse to, if an operation has
  made the correlations disagree).
* **a store holds several terms**, unlike a caltable which holds one. So
  `read_quartical` takes a term, defaulting to the only one when the store
  holds one and refusing to guess when it holds more.

xarray and zarr are an optional extra (``msutils[gains]``): imported inside
the functions, never at module scope, so ``import msutils`` stays free of
them (``tests/test_import.py`` enforces that for the package as a whole).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._model import GainBlock, GainTable

__all__ = ["is_quartical_store", "quartical_terms", "read_quartical"]


def is_quartical_store(path: str | Path) -> bool:
    """Whether `path` looks like a QuartiCal gain store."""
    path = Path(path)
    if not (path / ".zgroup").exists():
        return False
    return any((child / ".zgroup").exists() for child in path.iterdir() if child.is_dir())


def quartical_terms(path: str | Path) -> tuple[str, ...]:
    """The term names a store holds (``("K", "G")``), in directory order."""
    path = Path(path)
    return tuple(sorted(child.name for child in path.iterdir() if child.is_dir() and (child / ".zgroup").exists()))


def _resolve_term(path: Path, term: str | None) -> str:
    terms = quartical_terms(path)
    if not terms:
        raise ValueError(f"{path}: no term groups found -- is this a QuartiCal gain store?")
    if term is not None:
        if term not in terms:
            raise ValueError(f"{path}: no term {term!r} in this store (it holds {list(terms)})")
        return term
    if len(terms) > 1:
        raise ValueError(
            f"{path} holds {len(terms)} terms {list(terms)} -- name the one you mean. "
            "A store written by a chain carries every term of that chain, and which one "
            "an operation should act on is not something to guess."
        )
    return terms[0]


def read_quartical(path: str | Path, *, term: str | None = None) -> GainTable:
    """Read one term of a QuartiCal gain store into a :class:`GainTable`.

    Args:
        path: The ``.qc`` store directory.
        term: Which term to read (``"G"``, ``"K"``, ...). Optional only when
            the store holds exactly one.

    Returns:
        A `GainTable` with one block per (field, data-description, direction).

    Raises:
        ImportError: If xarray/zarr are not installed (``msutils[gains]``).
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "reading a QuartiCal gain store needs xarray and zarr: pip install 'msutils[gains]'"
        ) from exc

    path = Path(path)
    term = _resolve_term(path, term)
    datasets = sorted(
        (child for child in (path / term).iterdir() if child.is_dir() and (child / ".zgroup").exists()),
        key=lambda p: p.name,
    )
    if not datasets:
        raise ValueError(f"{path}: term {term!r} holds no datasets")

    blocks: list[GainBlock] = []
    gain_type = ""
    field_names: dict[int, str] = {}
    antenna_names: tuple[str, ...] = ()
    for dataset_path in datasets:
        dataset = xr.open_zarr(str(dataset_path))
        attrs = dataset.attrs
        gain_type = str(attrs.get("TYPE", gain_type))
        field_id = int(attrs["FIELD_ID"]) if "FIELD_ID" in attrs else None
        spw_id = int(attrs["DATA_DESC_ID"]) if "DATA_DESC_ID" in attrs else None
        if field_id is not None and "FIELD_NAME" in attrs:
            field_names[field_id] = str(attrs["FIELD_NAME"])
        gains = np.asarray(dataset["gains"].values)
        # (time, freq, antenna, direction) -> broadcast over correlation, which
        # QuartiCal does not flag separately.
        flags = np.asarray(dataset["gain_flags"].values).astype(bool)
        flags = np.repeat(flags[..., None], gains.shape[-1], axis=-1)
        antennas = tuple(str(a) for a in np.asarray(dataset["antenna"].values))
        antenna_names = antenna_names or antennas
        corrs = tuple(str(c) for c in np.asarray(dataset["correlation"].values))
        times = np.asarray(dataset["gain_time"].values, dtype=float)
        freqs = np.asarray(dataset["gain_freq"].values, dtype=float)
        for index, direction in enumerate(np.asarray(dataset["direction"].values).tolist()):
            blocks.append(
                GainBlock(
                    gains=gains[:, :, :, index, :],
                    flags=flags[:, :, :, index, :],
                    times=times,
                    freqs=freqs,
                    antennas=antennas,
                    corrs=corrs,
                    field_id=field_id,
                    spw_id=spw_id,
                    direction=int(direction),
                )
            )
        dataset.close()

    return GainTable(
        blocks=blocks,
        term=term,
        gain_type=gain_type,
        param_type="complex",
        format="quartical",
        path=str(path),
        antenna_names=antenna_names,
        field_names=field_names,
    )
