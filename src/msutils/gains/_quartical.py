"""Reading and writing QuartiCal gain stores.

A store is a zarr group per term, holding one dataset per chunk the solve
was cut into::

    <store>.qc/
      G/
        G_0/    gains (gain_time, gain_freq, antenna, direction, correlation)
                gain_flags (gain_time, gain_freq, antenna, direction)
                attrs: FIELD_ID, DATA_DESC_ID, FIELD_NAME, NAME, TYPE
        G_1/    ...
      K/
        K_0/    ...

which is nearly the model's shape, so reading is a rename of axes rather
than a fold. Four details that are not obvious from the arrays alone, each
of which the model has to undo:

* **a field is spread over several datasets.** `G_0` and `G_1` can both
  carry `FIELD_ID: 0`, differing only in the times they cover -- QuartiCal
  chunks its solve, normally per scan. So (field, ddid) does not identify a
  dataset, and reading them separately would make a "block" a chunk rather
  than a field's solution: an operation asking for a two-hour window would
  silently get whatever one scan held. They are concatenated in time on the
  way in and split back on the way out (`GainBlock.chunks`).
* **flags carry no correlation axis.** QuartiCal flags a solution, not a
  correlation of one, so the flag array is broadcast across correlations on
  the way in and collapsed with `any` on the way out.
* **most terms are parameterised.** The `gains` array is not the whole
  solution for them -- see `UNPARAMETERISED_TYPES`, which is why writing is
  refused for most of what a real chain solves.
* **a store holds several terms**, unlike a caltable which holds one. So
  `read_quartical` takes a term, defaulting to the only one when the store
  holds one and refusing to guess when it holds more.

xarray and zarr are an optional extra (``msutils[gains]``): imported inside
the functions, never at module scope, so ``import msutils`` stays free of
them (``tests/test_import.py`` enforces that for the package as a whole).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from msutils._log import create_logger

from ._model import GainBlock, GainTable

__all__ = [
    "UNPARAMETERISED_TYPES",
    "is_quartical_store",
    "quartical_terms",
    "read_quartical",
    "write_quartical",
]

LOGGER = create_logger(__name__)


def _open_dataset(path: Path):
    """Open one dataset, consolidated metadata or not, without a warning.

    QuartiCal's own stores carry no `.zmetadata`, and xarray's default is to
    try for one and warn when it falls back -- a warning per dataset per
    read, about something the caller cannot act on.
    """
    import xarray as xr

    try:
        return xr.open_zarr(str(path), consolidated=True)
    except (KeyError, ValueError, OSError):
        return xr.open_zarr(str(path), consolidated=False)


def _is_zarr_group(path: Path) -> bool:
    """Whether `path` is a zarr group, in either format version.

    v2 writes `.zgroup` and v3 writes `zarr.json`; dask-ms writes v2 today,
    and a detector that knows only that would stop working on the day it
    does not.
    """
    return (path / ".zgroup").exists() or (path / "zarr.json").exists()


def is_quartical_store(path: str | Path) -> bool:
    """Whether `path` looks like a QuartiCal gain store."""
    path = Path(path)
    if not _is_zarr_group(path):
        return False
    return any(_is_zarr_group(child) for child in path.iterdir() if child.is_dir())


def quartical_terms(path: str | Path) -> tuple[str, ...]:
    """The term names a store holds (``("K", "G")``), in directory order."""
    path = Path(path)
    return tuple(sorted(child.name for child in path.iterdir() if child.is_dir() and _is_zarr_group(child)))


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
        import xarray  # noqa: F401  -- imported for the error, used via `_open_dataset`
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "reading a QuartiCal gain store needs xarray and zarr: pip install 'msutils[gains]'"
        ) from exc

    path = Path(path)
    term = _resolve_term(path, term)
    datasets = sorted(
        (child for child in (path / term).iterdir() if child.is_dir() and _is_zarr_group(child)),
        key=lambda p: p.name,
    )
    if not datasets:
        raise ValueError(f"{path}: term {term!r} holds no datasets")

    # A store splits one field's solutions across several datasets --
    # QuartiCal chunks its solve, normally per scan -- so the datasets are
    # grouped by (field, ddid) and concatenated in time. Without that a
    # "block" would be a chunk rather than a field's solution, and an
    # operation asking for a two-hour window would silently get whatever a
    # scan happened to hold.
    read: list[dict] = []
    gain_type = ""
    field_names: dict[int, str] = {}
    antenna_names: tuple[str, ...] = ()
    for dataset_path in datasets:
        dataset = _open_dataset(dataset_path)
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
        read.append(
            {
                "name": dataset_path.name,
                "field_id": field_id,
                "spw_id": spw_id,
                "gains": gains,
                "flags": flags,
                "antennas": antennas,
                "corrs": tuple(str(c) for c in np.asarray(dataset["correlation"].values)),
                "times": np.asarray(dataset["gain_time"].values, dtype=float),
                "freqs": np.asarray(dataset["gain_freq"].values, dtype=float),
                "directions": [int(d) for d in np.asarray(dataset["direction"].values).tolist()],
            }
        )
        dataset.close()

    groups: dict[tuple[int | None, int | None], list[dict]] = {}
    for entry in read:
        groups.setdefault((entry["field_id"], entry["spw_id"]), []).append(entry)

    blocks: list[GainBlock] = []
    for (field_id, spw_id), entries in groups.items():
        entries.sort(key=lambda e: float(e["times"].min()))
        first = entries[0]
        for other in entries[1:]:
            if (
                other["antennas"] != first["antennas"]
                or other["corrs"] != first["corrs"]
                or not np.array_equal(other["freqs"], first["freqs"])
                or other["directions"] != first["directions"]
            ):
                raise ValueError(
                    f"{path}: datasets {first['name']} and {other['name']} cover the same field and "
                    "spectral window but disagree on their antenna, correlation, frequency or direction "
                    "axes, so they cannot be read as one solution set"
                )
        chunks = tuple((entry["name"], int(entry["times"].size)) for entry in entries)
        times = np.concatenate([entry["times"] for entry in entries])
        for index, direction in enumerate(first["directions"]):
            blocks.append(
                GainBlock(
                    gains=np.concatenate([entry["gains"][:, :, :, index, :] for entry in entries]),
                    flags=np.concatenate([entry["flags"][:, :, :, index, :] for entry in entries]),
                    times=times,
                    freqs=first["freqs"],
                    antennas=first["antennas"],
                    corrs=first["corrs"],
                    field_id=field_id,
                    spw_id=spw_id,
                    direction=direction,
                    chunks=chunks,
                )
            )

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


#: Term types whose `gains` array is the whole solution. Everything else is
#: **parameterised**: QuartiCal stores the parameters (a delay in ns, an
#: amplitude, a phase) alongside the gains and *reconstructs* the gains from
#: them when the term is loaded, so a writer that changed `gains` and left
#: `params` alone would be silently overruled on the next apply.
#:
#: Read off QuartiCal 0.2.7's own `quartical.gains.TERM_TYPES` -- the types
#: with no `param_axes` -- rather than guessed at. Note `amplitude` and
#: `phase` are parameterised despite sounding elementary, which is exactly
#: the kind of thing not to assume.
UNPARAMETERISED_TYPES = ("complex", "diag_complex", "feed_flip", "leakage")


def write_quartical(gains: GainTable, dest: str | Path, *, template: str | Path | None = None) -> str:
    """Write a :class:`GainTable` back out as a QuartiCal gain store.

    The output is a **copy of the source store** with the term's `gains` and
    `gain_flags` replaced. Copying rather than building: a store carries
    per-term parameters, convergence counters, chunking metadata and
    dask-ms's own attributes, none of which this model represents, and a
    store missing them is not one QuartiCal will read.

    Args:
        gains: What to write; must have come from a QuartiCal store (its
            blocks are matched back to the source datasets by field, spw and
            direction).
        dest: Output store path; must not exist.
        template: Store to copy. Defaults to `gains.path`.

    Returns:
        The path written.

    Raises:
        ValueError: If the term is parameterised (see
            `UNPARAMETERISED_TYPES`), if a block cannot be matched to a
            source dataset, or if a block's shape no longer fits.
        FileExistsError: If `dest` is already there.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "writing a QuartiCal gain store needs xarray and zarr: pip install 'msutils[gains]'"
        ) from exc

    dest = Path(dest)
    template = Path(template or gains.path)
    if not str(template):
        raise ValueError("write_quartical needs a template store (the gains carry no source path)")
    if dest.exists():
        raise FileExistsError(f"{dest} already exists -- gain operations never overwrite in place")
    if gains.gain_type and gains.gain_type not in UNPARAMETERISED_TYPES:
        raise ValueError(
            f"term {gains.term!r} is of parameterised type {gains.gain_type!r}: QuartiCal rebuilds its gains "
            f"from the `params` array when the term is loaded, so writing gains alone would be silently "
            f"overruled on the next apply. Writable types are {list(UNPARAMETERISED_TYPES)}."
        )

    shutil.copytree(template, dest)
    missing = [block.key for block in gains.blocks if not block.chunks]
    if missing:
        raise ValueError(
            f"cannot write blocks {missing} as a QuartiCal store: they carry no source datasets "
            "(they did not come from one). Converting between formats is a separate operation."
        )

    written: set[str] = set()
    for block in gains.blocks:
        offset = 0
        for name, count in block.chunks:
            dataset_path = dest / gains.term / name
            if not dataset_path.exists():
                raise ValueError(f"{template} has no dataset {name!r} for term {gains.term!r} to write back to")
            dataset = _open_dataset(dataset_path)
            directions = [int(d) for d in np.asarray(dataset["direction"].values).tolist()]
            attrs = dataset.attrs
            field_id = int(attrs["FIELD_ID"]) if "FIELD_ID" in attrs else None
            spw_id = int(attrs["DATA_DESC_ID"]) if "DATA_DESC_ID" in attrs else None
            dataset.close()
            # Identity, not just shape: a store written for another field can
            # have datasets of the same name and the same dimensions, and
            # writing into it would produce a plausible, wrong store.
            if (field_id, spw_id) != (block.field_id, block.spw_id):
                raise ValueError(
                    f"dataset {name!r} in {template} holds field {field_id}, spw {spw_id}, but the block "
                    f"being written is field {block.field_id}, spw {block.spw_id} -- these gains did not "
                    "come from this store"
                )
            if block.direction not in directions:
                raise ValueError(f"dataset {name!r} holds directions {directions}, not {block.direction}")
            index = directions.index(block.direction)

            values = zarr.open_array(str(dataset_path / "gains"), mode="r+")
            flags = zarr.open_array(str(dataset_path / "gain_flags"), mode="r+")
            gain_chunk = values[...]
            flag_chunk = flags[...]
            expected = gain_chunk[:, :, :, index, :].shape
            piece = block.gains[offset : offset + count]
            if piece.shape != expected:
                raise ValueError(
                    f"block (field {block.field_id}, spw {block.spw_id}, direction {block.direction}) "
                    f"contributes {piece.shape} to dataset {name!r}, which expects {expected}"
                )
            gain_chunk[:, :, :, index, :] = piece
            # QuartiCal flags a solution, not a correlation of one, so the
            # correlation axis has to collapse. `any` rather than `all`: a
            # solution one of whose correlations is unusable is not a
            # solution to hand to an apply.
            block_flags = block.flags[offset : offset + count]
            collapsed = block_flags.any(axis=-1)
            if collapsed.sum() * block.shape[-1] != block_flags.sum():
                LOGGER.warning(
                    "field %s spw %s direction %s: flags differ between correlations and this format "
                    "cannot express that -- flagging the solution wherever any correlation was flagged",
                    block.field_id,
                    block.spw_id,
                    block.direction,
                )
            flag_chunk[:, :, :, index] = collapsed
            values[...] = gain_chunk
            flags[...] = flag_chunk
            written.add(name)
            offset += count
        if offset != block.shape[0]:
            raise ValueError(
                f"block (field {block.field_id}, spw {block.spw_id}, direction {block.direction}) has "
                f"{block.shape[0]} times but its source datasets account for {offset} -- it has been "
                "reshaped in time, which this writer cannot put back"
            )

    remaining = {
        child.name
        for child in (dest / gains.term).iterdir()
        if child.is_dir() and _is_zarr_group(child) and child.name not in written
    }
    if remaining:
        raise ValueError(
            f"{sorted(remaining)} in {template} were not covered by the gains being written -- they hold "
            "fewer datasets than the store does, so the output would be part stale"
        )
    return str(dest)
