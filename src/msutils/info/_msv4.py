"""Read anything shaped like the MSv4 schema into the shared :class:`MSInfo`.

MSv4 is a different shape of thing from MSv2: a *processing set* is a
collection of xarray datasets, one per partition, each already split so that a
partition holds a single spectral window, polarization setup and field. There
are no rows -- visibilities are an n-dimensional
``(time, baseline, frequency, polarization)`` array.

This module reads the *schema*, not one particular library's output, because
three different sources produce the same tree and only differ in how it is
opened (:data:`ENGINES`):

* ``zarr`` -- a stored MSv4 processing set, opened with plain ``xarray``. No
  xradio needed to *read* one; xradio is only required to *write* one (see
  :mod:`msutils.convert`).
* ``xradio`` -- xradio's ``open_processing_set``. Handles cloud stores and
  netcdf, so it is the fallback when plain zarr cannot open the path.
* ``xarray-ms`` -- an **MSv2 on disk**, presented through the MSv4 schema
  without any conversion. Useful when you want the n-dimensional view of a
  Measurement Set you already have.

Two mapping details worth knowing:

* **Time.** The MSv4 schema stores unix epoch seconds; MSv2 (and this model)
  use MJD seconds. Converted on the way in, so ``start_utc`` agrees whichever
  engine read the data.
* **Rows.** ``nrows`` is the equivalent MSv2 row count, ``time x baseline``
  summed over partitions -- an MSv4 has no rows of its own.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from msutils._log import create_logger

from ._model import (
    STOKES_TYPES,
    Antenna,
    DataDescription,
    Field,
    MSInfo,
    Observation,
    Polarization,
    Registry,
    Scan,
    SpectralWindow,
)

LOGGER = create_logger(__name__)

#: Seconds between the MJD epoch (1858-11-17) and the unix epoch (1970-01-01).
#: 40587 days.
_MJD_UNIX_OFFSET = 40587 * 86400

#: Correlation label -> casacore Stokes code, for round-tripping MSv4's
#: string polarization labels back into the shared model.
_STOKES_CODES = {label: code for code, label in STOKES_TYPES.items()}

#: Ways of obtaining an MSv4-schema tree, in the order :func:`read` tries them
#: when no engine is named.
ENGINES = ("zarr", "xradio", "xarray-ms")


def read(path: str, level: str = "full", engine: str | None = None,
         format: str = "MSv4") -> MSInfo:
    """Read MSv4-schema metadata at ``path``.

    Args:
        path: Path to a processing set, or to an MSv2 when
            ``engine="xarray-ms"``.
        level: ``"meta"``, ``"full"`` or ``"data"``, as for the MSv2 reader.
            Only ``"data"`` touches the visibility-sized FLAG arrays.
        engine: One of :data:`ENGINES`, or ``None`` to try them in order.
        format: Value for ``MSInfo.format``. ``"MSv2"`` when reading a
            Measurement Set through ``xarray-ms``, since what is on disk is
            still an MSv2.
    """
    from ._msv2 import LEVELS, _directory_size, _max_antenna_separation
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    partitions, names, used_engine = _open(path, engine)

    info = MSInfo(path=os.path.abspath(path), format=format, engine=used_engine)
    info.size_bytes = _directory_size(path)
    info.subtables = list(names)
    if not partitions:
        return info

    info.antennas = Registry(_read_antennas(partitions[0]))
    info.observation = _read_observation(partitions[0], info)

    spws: dict[str, SpectralWindow] = {}
    pols: dict[str, Polarization] = {}
    fields: dict[str, Field] = {}
    scans: dict[int, Scan] = {}
    intervals = set()
    nflagged = nvis = None
    nrows = 0

    for partition in partitions:
        dataset = partition.ds
        spw = _spw(dataset, spws)
        _polarization(dataset, pols)           # registers the setup in `pols`
        ntime = dataset.sizes.get("time", 0)
        nbaseline = dataset.sizes.get("baseline_id", 0)
        nrows += ntime * nbaseline

        times = _mjd_seconds(np.asarray(dataset.time.values))
        interval = _integration_time(dataset)
        intervals.add(round(interval, 6))

        if level != "meta":
            _fold_partition(partition, times, interval, spw, fields, scans,
                            nbaseline)

        if level == "data":
            flagged, total = _count_flags(dataset)
            nflagged = (nflagged or 0) + flagged
            nvis = (nvis or 0) + total

    info.nrows = nrows
    info.spws = Registry(sorted(spws.values(), key=lambda s: s.id))
    info.polarizations = Registry(sorted(pols.values(), key=lambda p: p.id))
    # MSv4 partitions already carry one SPW and one polarization each, so
    # there is no DATA_DESCRIPTION indirection to represent; synthesise the
    # 1:1 mapping so consumers can treat both formats alike.
    info.data_descriptions = Registry(
        [DataDescription(id=s.id, spw_id=s.id, pol_id=0) for s in info.spws])
    info.fields = Registry(sorted(fields.values(), key=lambda f: f.id))
    info.scans = Registry([scans[n] for n in sorted(scans)], key="number")
    info.integration_times = sorted(intervals)
    info.nflagged = nflagged
    info.nvisibilities = nvis
    info.nbaselines = max(
        (p.ds.sizes.get("baseline_id", 0) for p in partitions), default=0)
    info.max_baseline = _max_antenna_separation(
        info.antennas, [(a.id, b.id) for a in info.antennas
                        for b in info.antennas if a.id < b.id])

    if len(info.scans):
        info.time_start = min(s.time_start for s in info.scans)
        info.time_end = max(s.time_end + s.interval for s in info.scans)
        if info.observation.time_start is None:
            info.observation.time_start = info.time_start
            info.observation.time_end = info.time_end
    return info


def _open(path: str, engine: str | None) -> tuple:
    """Open ``path`` as MSv4-schema partitions: (nodes, names, engine used)."""
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")

    candidates = [engine] if engine else list(ENGINES)
    openers = {"zarr": _open_zarr, "xradio": _open_xradio,
               "xarray-ms": _open_xarray_ms}
    failures = []
    for name in candidates:
        try:
            nodes, labels = openers[name](path)
        except ImportError as exc:
            failures.append(f"{name}: {exc}")
            continue
        except Exception as exc:               # unreadable by this engine
            failures.append("{}: {}".format(name, str(exc).split("\n")[0]))
            continue
        if nodes:
            LOGGER.debug("opened %s with the %s engine", path, name)
            return nodes, labels, name
        failures.append(f"{name}: no partitions found")

    raise ValueError(
        "could not read {!r} as an MSv4-schema dataset. Tried:\n  {}\n"
        "Reading MSv4 needs 'pip install msutils[msv4]'; reading an MSv2 "
        "through the MSv4 schema needs 'pip install msutils[xarray-ms]'."
        .format(path, "\n  ".join(failures)))


def _open_zarr(path: str) -> tuple:
    """A stored processing set, via plain xarray. Needs no xradio.

    A processing set is a directory of per-partition zarr groups, so opening
    each subdirectory is all "opening the set" amounts to.
    """
    import xarray

    names = sorted(entry for entry in os.listdir(path)
                   if os.path.isdir(os.path.join(path, entry)))
    nodes, labels = [], []
    for name in names:
        child = os.path.join(path, name)
        if not any(os.path.exists(os.path.join(child, marker))
                   for marker in (".zgroup", ".zattrs", "zarr.json")):
            continue
        nodes.append(xarray.open_datatree(child, engine="zarr"))
        labels.append(name)
    return nodes, labels


def _open_xradio(path: str) -> tuple:
    """xradio's own opener -- handles cloud stores and netcdf backends."""
    from xradio.measurement_set import open_processing_set

    processing_set = open_processing_set(str(path))
    names = sorted(processing_set.children)
    return [processing_set[name] for name in names], names


def _open_xarray_ms(path: str) -> tuple:
    """An MSv2 on disk, presented through the MSv4 schema, no conversion.

    xarray-ms registers itself as an xarray backend for Measurement Sets, so
    this is just ``open_datatree`` on the MS itself.
    """
    import xarray
    import xarray_ms  # noqa: F401  (registers the backend)

    tree = xarray.open_datatree(str(path))
    names = sorted(tree.children)
    return [tree[name] for name in names], names


def _fold_partition(partition, times, interval, spw, fields, scans,
                    nbaseline) -> None:
    """Accumulate this partition's fields and scans into the shared model."""
    dataset = partition.ds
    field_names = _coord_strings(dataset, "field_name", len(times))
    scan_names = _coord_strings(dataset, "scan_name", len(times))
    intents = _intents(dataset)
    phase_centre, frame = _phase_centre(partition)

    for index, timestamp in enumerate(times):
        name = field_names[index] if index < len(field_names) else ""
        field = fields.get(name)
        if field is None:
            field = fields[name] = Field(
                id=len(fields), name=name, phase_centre=phase_centre,
                ref_frame=frame, intents=list(intents))
            field.time_start = timestamp
            field.time_end = timestamp
        field.time_start = min(field.time_start, timestamp)
        field.time_end = max(field.time_end, timestamp)
        field.nrows += nbaseline
        field.exposure += interval * nbaseline
        if spw.id not in field.spw_ids:
            field.spw_ids.append(spw.id)

        scan_number = _scan_number(scan_names, index)
        scan = scans.get(scan_number)
        if scan is None:
            scan = scans[scan_number] = Scan(
                number=scan_number, field_id=field.id, field_name=name,
                time_start=timestamp, time_end=timestamp, interval=interval,
                intents=list(intents))
        scan.time_start = min(scan.time_start, timestamp)
        scan.time_end = max(scan.time_end, timestamp)
        scan.nrows += nbaseline
        if spw.id not in scan.spw_ids:
            scan.spw_ids.append(spw.id)
        if scan_number not in fields[name].scan_numbers:
            fields[name].scan_numbers.append(scan_number)


def _read_antennas(partition) -> list[Antenna]:
    """Antennas from a partition's ``antenna_xds`` sub-dataset."""
    try:
        antenna_xds = partition["antenna_xds"].ds
    except KeyError:                           # pragma: no cover - malformed set
        return []

    names = _values(antenna_xds, "antenna_name")
    stations = _values(antenna_xds, "station_name")
    mounts = _values(antenna_xds, "mount")
    positions = np.atleast_2d(np.asarray(antenna_xds.ANTENNA_POSITION.values))
    diameters = (np.asarray(antenna_xds.ANTENNA_DISH_DIAMETER.values)
                 if "ANTENNA_DISH_DIAMETER" in antenna_xds else None)

    antennas = []
    for i, name in enumerate(names):
        antennas.append(Antenna(
            id=i,
            name=str(name),
            station=str(stations[i]) if i < len(stations) else "",
            mount=str(mounts[i]) if i < len(mounts) else "",
            position=tuple(float(v) for v in positions[i]),
            dish_diameter=(float(diameters[i]) if diameters is not None
                           and i < len(diameters) else 0.0),
        ))
    return antennas


def _read_observation(partition, info: MSInfo) -> Observation:
    observation = Observation()
    try:
        antenna_xds = partition["antenna_xds"].ds
        telescopes = _values(antenna_xds, "telescope_name")
        if len(telescopes):
            observation.telescope = str(telescopes[0])
    except KeyError:                           # pragma: no cover
        pass

    obs_info = partition.ds.attrs.get("observation_info") or {}
    observer = obs_info.get("observer")
    if isinstance(observer, (list, tuple)):
        observer = observer[0] if observer else ""
    observation.observer = str(observer or "")
    observation.project = str(obs_info.get("project_UID") or "")
    return observation


def _spw(dataset, spws: dict[str, SpectralWindow]) -> SpectralWindow:
    """The spectral window this partition covers, created on first sight."""
    frequency = dataset.frequency
    attrs = dict(frequency.attrs)
    name = str(attrs.get("spectral_window_name", "SPW"))
    if name in spws:
        return spws[name]

    channels = np.atleast_1d(np.asarray(frequency.values, dtype=float))
    width = abs(float(_quantity(attrs.get("channel_width"), default=0.0)))
    spw = SpectralWindow(
        id=len(spws),
        name=name,
        num_chan=len(channels),
        chan_freq=[float(f) for f in channels],
        chan_width=[width] * len(channels),
        ref_frequency=float(_quantity(attrs.get("reference_frequency"),
                                      default=channels[0] if len(channels) else 0.0)),
        total_bandwidth=width * len(channels),
        meas_freq_ref=_frame_code(attrs.get("observer")),
    )
    spws[name] = spw
    return spw


def _polarization(dataset, pols: dict[str, Polarization]) -> Polarization:
    labels = [str(v) for v in np.atleast_1d(dataset.polarization.values)]
    key = ",".join(labels)
    if key not in pols:
        pols[key] = Polarization(
            id=len(pols), num_corr=len(labels),
            corr_type=[_STOKES_CODES.get(label, 0) for label in labels])
    return pols[key]


def _phase_centre(partition) -> tuple:
    """((ra, dec) in radians, frame) from the field_and_source sub-dataset.

    The sub-node name depends on the data group; ``base`` is the usual one,
    but ephemeris datasets carry a differently-named variant.
    """
    for name in ("field_and_source_base_xds", "field_and_source_xds"):
        if name not in partition.children:
            continue
        xds = partition[name].ds
        if "FIELD_PHASE_CENTER_DIRECTION" not in xds:
            continue
        direction = xds.FIELD_PHASE_CENTER_DIRECTION
        values = np.atleast_2d(np.asarray(direction.values, dtype=float))
        frame = str(dict(direction.attrs).get("frame", "fk5"))
        if values.size >= 2:
            return (float(values[0][0]), float(values[0][1])), frame
    return (0.0, 0.0), "fk5"


def _count_flags(dataset) -> tuple:
    """Flagged and total visibility counts for one partition."""
    if "FLAG" not in dataset:
        return 0, 0
    flag = dataset.FLAG.data
    total = int(np.prod(dataset.FLAG.shape))
    flagged = flag.sum()
    if hasattr(flagged, "compute"):            # dask-backed
        flagged = flagged.compute()
    return int(flagged), total


def _integration_time(dataset) -> float:
    attrs = dict(dataset.time.attrs)
    value = _quantity(attrs.get("integration_time"), default=0.0)
    return float(value)


def _quantity(value: Any, default: float = 0.0) -> float:
    """Unwrap an MSv4 quantity, which may be a dict or a bare number."""
    if value is None:
        return default
    if isinstance(value, dict):
        return float(np.atleast_1d(value.get("data", default)).reshape(-1)[0])
    return float(np.atleast_1d(value).reshape(-1)[0])


def _frame_code(observer: Any) -> int:
    """Spectral frame name -> the MEAS_FREQ_REF code the model expects."""
    from ._model import FREQ_FRAMES
    name = str(observer or "TOPO").upper()
    for code, label in FREQ_FRAMES.items():
        if label.upper() == name:
            return code
    return 5                                   # TOPO


def _intents(dataset) -> list[str]:
    """Observing intents, which MSv4 hangs off the ``scan_name`` coordinate."""
    intents = []
    if "scan_name" in dataset.coords:
        intents = dataset.scan_name.attrs.get("scan_intents") or []
    if not intents:
        intents = dataset.attrs.get("scan_intents") or []
    if isinstance(intents, str):
        intents = [intents]
    return [str(i) for i in intents]


def _coord_strings(dataset, name: str, length: int) -> list[str]:
    if name not in dataset.coords:
        return [""] * length
    values = np.atleast_1d(np.asarray(dataset[name].values))
    return [str(v) for v in values]


def _values(dataset, name: str):
    if name not in dataset.coords and name not in dataset:
        return []
    return np.atleast_1d(np.asarray(dataset[name].values))


def _scan_number(scan_names: list[str], index: int) -> int:
    """MSv4 scan names are strings; keep the number when there is one."""
    if index >= len(scan_names):
        return 0
    name = scan_names[index]
    try:
        return int(name)
    except (TypeError, ValueError):
        return abs(hash(name)) % (10 ** 6)


def _mjd_seconds(times: np.ndarray) -> np.ndarray:
    """MSv4 unix epoch seconds -> the MJD seconds the model uses."""
    return np.asarray(times, dtype=float) + _MJD_UNIX_OFFSET
