"""Materialise the correlated-interferometer part of MSv4 as an MSv2 table.

MSv4 is deliberately more expressive than MSv2.  In particular, weights are
per channel and processing sets may contain single-dish or mixed-polarisation
data.  This writer is intentionally a narrow adapter: it writes the common
correlated-interferometer profile losslessly, and says no before creating an
output for content which cannot be represented by an MSv2 main table.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np
from casacore.tables import default_ms, makearrcoldesc, maketabdesc

from ._tables import open_table
from .info import _msv4
from .info._model import STOKES_TYPES

_STOKES_CODES = {label: code for code, label in STOKES_TYPES.items()}
_MJD_UNIX_OFFSET = 40587 * 86400


@dataclass(frozen=True)
class _Partition:
    """Validated, representable view of one processing-set partition."""

    node: object
    field_names: tuple[str, ...]
    scan_numbers: tuple[int, ...]
    intent: str
    field_direction: tuple[float, float]
    field_frame: str
    spw_key: tuple
    pol_labels: tuple[str, ...]
    data_name: str
    flag_name: str | None
    weight_name: str | None
    uvw_name: str
    time_centroid_name: str | None
    exposure_name: str | None
    antenna1: np.ndarray
    antenna2: np.ndarray
    interval: float

    @property
    def ds(self):
        return self.node.ds


def to_msv2(
    msv4: str,
    outpath: str,
    *,
    overwrite: bool = False,
    weight_spectrum: bool = True,
    rowchunk: int = 64,
):
    """Write ``msv4`` to a new MSv2 table and return its :class:`MSInfo`.

    The source is opened only with the plain zarr reader.  This keeps the
    reverse direction useful in the lightweight ``msv4`` extra and, more
    importantly, makes its accepted on-disk profile explicit.
    """
    if not isinstance(rowchunk, int) or isinstance(rowchunk, bool) or rowchunk < 1:
        raise ValueError("rowchunk must be a positive integer")
    if os.path.exists(outpath) and not overwrite:
        raise FileExistsError(f"{outpath} already exists; pass overwrite=True to replace it")

    try:
        import xarray  # noqa: F401  (needed by _msv4._open_zarr)
    except ImportError as exc:  # pragma: no cover - optional extra absent
        raise ImportError(
            "MSv4 materialisation needs xarray and zarr. Install with: pip install 'msutils[msv4]'"
        ) from exc

    try:
        nodes, _, _ = _msv4._open(msv4, "zarr")
        partitions = _validated_partitions(nodes)
        antennas = _antennas(nodes)

        # All unsupported content is rejected above: do not leave a half-built
        # destination merely because the source had an unusual partition late
        # in its directory order.
        _remove_output(outpath, overwrite)
        try:
            _create_ms(outpath, weight_spectrum)
            _write_subtables(outpath, partitions, antennas)
            _write_main(outpath, partitions, antennas, weight_spectrum, rowchunk)
        except Exception:
            _remove_output(outpath, overwrite=True)
            raise
    finally:
        for node in locals().get("nodes", []):
            close = getattr(node, "close", None)
            if close is not None:
                close()

    from .info import msinfo

    return msinfo(outpath, level="full")


def _remove_output(outpath: str, overwrite: bool) -> None:
    if not os.path.exists(outpath):
        return
    if not overwrite:
        raise FileExistsError(f"{outpath} already exists; pass overwrite=True to replace it")
    if os.path.isdir(outpath):
        shutil.rmtree(outpath)
    else:
        os.unlink(outpath)


def _validated_partitions(nodes) -> list[_Partition]:
    if not nodes:
        raise ValueError("MSv4 processing set has no partitions")
    parts = [_partition(node) for node in nodes]
    if not any(
        part.ds.sizes.get("time", 0) and part.ds.sizes.get("baseline_id", 0) for part in parts
    ):
        raise ValueError("MSv4 processing set contains no correlated visibility rows")
    return parts


def _partition(node) -> _Partition:
    ds = node.ds
    if str(ds.attrs.get("type", "visibility")) != "visibility":
        raise ValueError("only correlated-interferometer MSv4 visibility data can be materialised")
    groups = ds.attrs.get("data_groups") or {}
    base = groups.get("base") or {}
    data_name = str(base.get("correlated_data") or "VISIBILITY")
    flag_name = base.get("flag", "FLAG")
    weight_name = base.get("weight", "WEIGHT")
    uvw_name = str(base.get("uvw") or "UVW")
    for name in (data_name, uvw_name):
        if name not in ds:
            raise ValueError(f"MSv4 partition has no required {name!r} variable")
    for name, expected in (
        (data_name, ("time", "baseline_id", "frequency", "polarization")),
        (uvw_name, ("time", "baseline_id", "uvw_label")),
    ):
        if tuple(ds[name].dims) != expected:
            raise ValueError(
                f"MSv4 {name} must have dimensions {expected}, got {tuple(ds[name].dims)}"
            )
    if ds[uvw_name].sizes.get("uvw_label") != 3:
        raise ValueError("MSv4 UVW must have exactly three coordinates")
    for name in (flag_name, weight_name):
        if name is not None and name in ds and tuple(ds[name].dims) != tuple(ds[data_name].dims):
            raise ValueError(f"MSv4 {name} must have the same dimensions as {data_name}")
    flag_name = str(flag_name) if flag_name in ds else None
    weight_name = str(weight_name) if weight_name in ds else None

    for coord in (
        "time",
        "baseline_id",
        "frequency",
        "polarization",
        "baseline_antenna1_name",
        "baseline_antenna2_name",
    ):
        if coord not in ds.coords:
            raise ValueError(f"MSv4 partition has no required {coord!r} coordinate")
    if "polarization_mixed" in ds.coords:
        raise ValueError(
            "mixed-polarization MSv4 data cannot be represented by one MSv2 POLARIZATION row"
        )

    labels = tuple(str(label) for label in np.asarray(ds.polarization.values).reshape(-1))
    if not labels or any(label not in _STOKES_CODES for label in labels):
        raise ValueError(f"MSv4 has unsupported polarization labels {labels!r}")
    a1_names = np.asarray(ds.baseline_antenna1_name.values).reshape(-1)
    a2_names = np.asarray(ds.baseline_antenna2_name.values).reshape(-1)
    if len(a1_names) != ds.sizes["baseline_id"] or len(a2_names) != ds.sizes["baseline_id"]:
        raise ValueError("MSv4 baseline antenna coordinates do not match baseline_id")

    field_names = _coord_values(ds, "field_name", ds.sizes["time"])
    scan_names = _coord_values(ds, "scan_name", ds.sizes["time"])
    intent_values = _msv4._intents(ds)
    intent = ",".join(intent_values)
    direction, frame = _msv4._phase_centre(node)

    frequency = np.asarray(ds.frequency.values, dtype=float).reshape(-1)
    if not len(frequency):
        raise ValueError("MSv4 partition has no frequency channels")
    attrs = dict(ds.frequency.attrs)
    spw_name = str(attrs.get("spectral_window_name", "SPW"))
    width = abs(float(_msv4._quantity(attrs.get("channel_width"), default=0.0)))
    reference = float(_msv4._quantity(attrs.get("reference_frequency"), default=frequency[0]))
    frame_name = str(attrs.get("observer", "TOPO"))
    spw_key = (spw_name, tuple(frequency.tolist()), width, reference, frame_name)

    centroid = "TIME_CENTROID" if "TIME_CENTROID" in ds else None
    exposure = "EFFECTIVE_INTEGRATION_TIME" if "EFFECTIVE_INTEGRATION_TIME" in ds else None
    for name in (centroid, exposure):
        if name is not None and tuple(ds[name].dims) != ("time", "baseline_id"):
            raise ValueError(f"MSv4 {name} must have dimensions ('time', 'baseline_id') for MSv2")
    interval = float(_msv4._integration_time(ds))
    if interval < 0:
        raise ValueError("MSv4 integration_time must not be negative")
    return _Partition(
        node=node,
        field_names=tuple(field_names),
        scan_numbers=tuple(_scan_number(name) for name in scan_names),
        intent=intent,
        field_direction=direction,
        field_frame=frame,
        spw_key=spw_key,
        pol_labels=labels,
        data_name=data_name,
        flag_name=flag_name,
        weight_name=weight_name,
        uvw_name=uvw_name,
        time_centroid_name=centroid,
        exposure_name=exposure,
        antenna1=np.asarray(a1_names, dtype=str),
        antenna2=np.asarray(a2_names, dtype=str),
        interval=interval,
    )


def _coord_values(ds, name: str, length: int) -> list[str]:
    if name not in ds.coords:
        return [""] * length
    values = np.asarray(ds[name].values).reshape(-1)
    if len(values) != length:
        raise ValueError(f"MSv4 {name!r} coordinate does not match time")
    return [str(value) for value in values]


def _scan_number(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"MSv4 scan name {value!r} is not an integer MSv2 scan number") from exc


def _antennas(nodes):
    antennas = _msv4._read_antennas(nodes[0])
    if not antennas:
        raise ValueError("MSv4 processing set has no antenna_xds metadata")
    names = [antenna.name for antenna in antennas]
    if len(names) != len(set(names)):
        raise ValueError("MSv4 antenna names must be unique to materialise MSv2")
    return antennas


def _create_ms(outpath: str, weight_spectrum: bool) -> None:
    columns = [makearrcoldesc("DATA", 0j, ndim=2, valuetype="complex")]
    if weight_spectrum:
        columns.append(makearrcoldesc("WEIGHT_SPECTRUM", 0.0, ndim=2, valuetype="float"))
    default_ms(outpath, maketabdesc(columns))


def _write_subtables(outpath: str, parts: list[_Partition], antennas) -> None:
    spws = {key: index for index, key in enumerate(dict.fromkeys(part.spw_key for part in parts))}
    pols = {
        labels: index
        for index, labels in enumerate(dict.fromkeys(part.pol_labels for part in parts))
    }
    fields = {
        name: index
        for index, name in enumerate(
            dict.fromkeys(name for part in parts for name in part.field_names)
        )
    }
    states = {
        intent: index for index, intent in enumerate(dict.fromkeys(part.intent for part in parts))
    }

    _write_antenna(outpath, antennas)
    _write_spectral_windows(outpath, spws)
    _write_polarizations(outpath, pols)
    _write_data_descriptions(outpath, parts, spws, pols)
    _write_fields(outpath, parts, fields)
    _write_states(outpath, states)
    _write_feeds(outpath, antennas, spws, pols, parts)
    _write_observation(outpath, parts)


def _write_antenna(outpath, antennas) -> None:
    with open_table(outpath + "::ANTENNA", readonly=False) as tab:
        tab.addrows(len(antennas))
        tab.putcol("NAME", [a.name for a in antennas])
        tab.putcol("STATION", [a.station for a in antennas])
        tab.putcol("MOUNT", [a.mount for a in antennas])
        tab.putcol("TYPE", [a.type or "GROUND-BASED" for a in antennas])
        tab.putcol("POSITION", np.asarray([a.position for a in antennas], dtype=float))
        tab.putcol("DISH_DIAMETER", np.asarray([a.dish_diameter for a in antennas], dtype=float))
        tab.putcol("FLAG_ROW", np.asarray([a.flagged for a in antennas], dtype=bool))


def _write_spectral_windows(outpath, spws) -> None:
    with open_table(outpath + "::SPECTRAL_WINDOW", readonly=False) as tab:
        tab.addrows(len(spws))
        for (name, frequencies, width, reference, frame), row in spws.items():
            freq = np.asarray(frequencies, dtype=float)
            widths = np.full(len(freq), width, dtype=float)
            tab.putcell("NUM_CHAN", row, len(freq))
            tab.putcell("CHAN_FREQ", row, freq)
            tab.putcell("CHAN_WIDTH", row, widths)
            tab.putcell("EFFECTIVE_BW", row, widths)
            tab.putcell("RESOLUTION", row, widths)
            tab.putcell("REF_FREQUENCY", row, reference)
            tab.putcell("TOTAL_BANDWIDTH", row, width * len(freq))
            tab.putcell("NAME", row, name)
            tab.putcell("MEAS_FREQ_REF", row, _msv4._frame_code(frame))
            tab.putcell("FREQ_GROUP_NAME", row, "")


def _write_polarizations(outpath, pols) -> None:
    with open_table(outpath + "::POLARIZATION", readonly=False) as tab:
        tab.addrows(len(pols))
        for labels, row in pols.items():
            products = _corr_products(labels)
            tab.putcell("NUM_CORR", row, len(labels))
            tab.putcell("CORR_TYPE", row, np.asarray([_STOKES_CODES[label] for label in labels]))
            tab.putcell("CORR_PRODUCT", row, products)


def _corr_products(labels: tuple[str, ...]) -> np.ndarray:
    """Return MSv2 feed-receptor pairs for ordinary X/Y or R/L products."""
    receptors = sorted({char for label in labels if len(label) == 2 for char in label})
    if not receptors or any(
        len(label) != 2 or char not in "XYRL" for label in labels for char in label
    ):
        raise ValueError(f"MSv4 polarization {labels!r} has no MSv2 feed-receptor representation")
    if set(receptors) not in ({"X", "Y"}, {"R", "L"}):
        raise ValueError(f"MSv4 polarization {labels!r} mixes linear and circular receptors")
    index = {receptor: number for number, receptor in enumerate(receptors)}
    return np.asarray([[index[label[0]], index[label[1]]] for label in labels], dtype=np.int32).T


def _write_data_descriptions(outpath, parts, spws, pols) -> None:
    pairs = list(dict.fromkeys((spws[part.spw_key], pols[part.pol_labels]) for part in parts))
    with open_table(outpath + "::DATA_DESCRIPTION", readonly=False) as tab:
        tab.addrows(len(pairs))
        tab.putcol("SPECTRAL_WINDOW_ID", np.asarray([pair[0] for pair in pairs], dtype=np.int32))
        tab.putcol("POLARIZATION_ID", np.asarray([pair[1] for pair in pairs], dtype=np.int32))
        tab.putcol("FLAG_ROW", np.zeros(len(pairs), dtype=bool))


def _write_fields(outpath, parts, fields) -> None:
    by_name = {name: part for part in parts for name in part.field_names}
    with open_table(outpath + "::FIELD", readonly=False) as tab:
        tab.addrows(len(fields))
        for name, row in fields.items():
            part = by_name[name]
            direction = np.asarray([part.field_direction], dtype=float)
            tab.putcell("NAME", row, name)
            tab.putcell("CODE", row, "")
            for column in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
                tab.putcell(column, row, direction)
            tab.putcell("SOURCE_ID", row, row)
            tab.putcell("TIME", row, 0.0)
            tab.putcell("NUM_POLY", row, 0)


def _write_states(outpath, states) -> None:
    with open_table(outpath + "::STATE", readonly=False) as tab:
        tab.addrows(len(states))
        tab.putcol("OBS_MODE", list(states))
        tab.putcol("SIG", np.ones(len(states), dtype=bool))
        tab.putcol("REF", np.zeros(len(states), dtype=bool))
        tab.putcol("FLAG_ROW", np.zeros(len(states), dtype=bool))


def _write_feeds(outpath, antennas, spws, pols, parts) -> None:
    # MSv4 records baseline labels, not feeds.  A conventional feed 0 for
    # every antenna/SPW satisfies MSv2's referential integrity; correlations
    # themselves retain the precise receptor products in POLARIZATION.
    labels_by_spw = {}
    for part in parts:
        spw = spws[part.spw_key]
        previous = labels_by_spw.setdefault(spw, part.pol_labels)
        if set("".join(previous)) != set("".join(part.pol_labels)):
            raise ValueError("one MSv4 spectral window has incompatible feed receptor bases")
    with open_table(outpath + "::FEED", readonly=False) as tab:
        tab.addrows(len(antennas) * len(spws))
        row = 0
        for spw in spws.values():
            labels = labels_by_spw[spw]
            receptors = sorted({char for label in labels for char in label})
            for antenna in antennas:
                tab.putcell("ANTENNA_ID", row, antenna.id)
                tab.putcell("FEED_ID", row, 0)
                tab.putcell("SPECTRAL_WINDOW_ID", row, spw)
                tab.putcell("TIME", row, 0.0)
                tab.putcell("INTERVAL", row, 1e30)
                tab.putcell("NUM_RECEPTORS", row, len(receptors))
                tab.putcell("BEAM_ID", row, -1)
                tab.putcell("BEAM_OFFSET", row, np.zeros((len(receptors), 2)))
                tab.putcell("POLARIZATION_TYPE", row, receptors)
                tab.putcell("POL_RESPONSE", row, np.eye(len(receptors), dtype=np.complex64))
                tab.putcell("POSITION", row, np.zeros(3))
                tab.putcell("RECEPTOR_ANGLE", row, np.zeros(len(receptors)))
                row += 1


def _write_observation(outpath, parts) -> None:
    node = parts[0].node
    info = node.ds.attrs.get("observation_info") or {}
    observer = info.get("observer") or ""
    if isinstance(observer, (list, tuple)):
        observer = observer[0] if observer else ""
    times = [(np.asarray(part.ds.time.values, dtype=float), part.interval) for part in parts]
    nonempty = [(values, interval) for values, interval in times if len(values)]
    start = (
        float(min(np.min(values) for values, _ in nonempty) + _MJD_UNIX_OFFSET) if nonempty else 0.0
    )
    end = (
        float(max(np.max(values) + interval for values, interval in nonempty) + _MJD_UNIX_OFFSET)
        if nonempty
        else 0.0
    )
    with open_table(outpath + "::OBSERVATION", readonly=False) as tab:
        tab.addrows(1)
        tab.putcell("TELESCOPE_NAME", 0, _msv4._read_observation(node, None).telescope)
        tab.putcell("OBSERVER", 0, str(observer))
        tab.putcell("PROJECT", 0, str(info.get("project_UID") or ""))
        tab.putcell("TIME_RANGE", 0, np.asarray([start, end]))


def _write_main(outpath, parts, antennas, weight_spectrum: bool, rowchunk: int) -> None:
    spws = {key: index for index, key in enumerate(dict.fromkeys(part.spw_key for part in parts))}
    pols = {
        labels: index
        for index, labels in enumerate(dict.fromkeys(part.pol_labels for part in parts))
    }
    fields = {
        name: index
        for index, name in enumerate(
            dict.fromkeys(name for part in parts for name in part.field_names)
        )
    }
    states = {
        intent: index for index, intent in enumerate(dict.fromkeys(part.intent for part in parts))
    }
    dds = {
        pair: index
        for index, pair in enumerate(
            dict.fromkeys((spws[p.spw_key], pols[p.pol_labels]) for p in parts)
        )
    }
    antenna_ids = {antenna.name: antenna.id for antenna in antennas}

    with open_table(outpath, readonly=False) as tab:
        row0 = 0
        for part in parts:
            try:
                a1 = np.asarray([antenna_ids[name] for name in part.antenna1], dtype=np.int32)
                a2 = np.asarray([antenna_ids[name] for name in part.antenna2], dtype=np.int32)
            except KeyError as exc:
                raise ValueError(
                    f"MSv4 baseline refers to unknown antenna {exc.args[0]!r}"
                ) from None
            for start in range(0, part.ds.sizes["time"], rowchunk):
                stop = min(start + rowchunk, part.ds.sizes["time"])
                count = (stop - start) * len(a1)
                tab.addrows(count)
                data = _chunk(part.ds[part.data_name], start, stop).astype(np.complex64, copy=False)
                flag = (
                    _chunk(part.ds[part.flag_name], start, stop).astype(bool, copy=False)
                    if part.flag_name
                    else np.zeros(data.shape, dtype=bool)
                )
                weight = (
                    _chunk(part.ds[part.weight_name], start, stop).astype(np.float32, copy=False)
                    if part.weight_name
                    else np.ones(data.shape, dtype=np.float32)
                )
                uvw = _chunk(part.ds[part.uvw_name], start, stop).astype(float, copy=False)
                ntime, nbase, nchan, ncorr = data.shape
                if (nbase, nchan, ncorr) != (
                    len(a1),
                    part.ds.sizes["frequency"],
                    len(part.pol_labels),
                ):
                    raise ValueError("MSv4 visibility array shape does not match its coordinates")
                times = np.asarray(part.ds.time.isel(time=slice(start, stop)).values, dtype=float)
                centroid = (
                    _chunk(part.ds[part.time_centroid_name], start, stop)
                    if part.time_centroid_name
                    else np.broadcast_to(times[:, None], (ntime, nbase))
                )
                exposure = (
                    _chunk(part.ds[part.exposure_name], start, stop)
                    if part.exposure_name
                    else np.full((ntime, nbase), part.interval)
                )
                tab.putcol("TIME", np.repeat(times + _MJD_UNIX_OFFSET, nbase), row0, count)
                tab.putcol(
                    "TIME_CENTROID",
                    np.asarray(centroid).reshape(-1) + _MJD_UNIX_OFFSET,
                    row0,
                    count,
                )
                tab.putcol("ANTENNA1", np.tile(a1, ntime), row0, count)
                tab.putcol("ANTENNA2", np.tile(a2, ntime), row0, count)
                tab.putcol(
                    "DATA_DESC_ID",
                    np.full(
                        count, dds[(spws[part.spw_key], pols[part.pol_labels])], dtype=np.int32
                    ),
                    row0,
                    count,
                )
                field_ids = np.asarray(
                    [fields[name] for name in part.field_names[start:stop]], dtype=np.int32
                )
                scans = np.asarray(part.scan_numbers[start:stop], dtype=np.int32)
                tab.putcol("FIELD_ID", np.repeat(field_ids, nbase), row0, count)
                tab.putcol("SCAN_NUMBER", np.repeat(scans, nbase), row0, count)
                tab.putcol(
                    "STATE_ID", np.full(count, states[part.intent], dtype=np.int32), row0, count
                )
                tab.putcol("EXPOSURE", np.asarray(exposure, dtype=float).reshape(-1), row0, count)
                tab.putcol("INTERVAL", np.full(count, part.interval), row0, count)
                tab.putcol("UVW", uvw.reshape(-1, 3), row0, count)
                tab.putcol("DATA", data.reshape(count, nchan, ncorr), row0, count)
                tab.putcol("FLAG", flag.reshape(count, nchan, ncorr), row0, count)
                tab.putcol("FLAG_ROW", np.all(flag, axis=(2, 3)).reshape(-1), row0, count)
                tab.putcol("WEIGHT", np.mean(weight, axis=2).reshape(count, ncorr), row0, count)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma = 1.0 / np.sqrt(np.mean(weight, axis=2))
                tab.putcol("SIGMA", sigma.astype(np.float32).reshape(count, ncorr), row0, count)
                if weight_spectrum:
                    tab.putcol("WEIGHT_SPECTRUM", weight.reshape(count, nchan, ncorr), row0, count)
                for column in ("ARRAY_ID", "OBSERVATION_ID", "PROCESSOR_ID", "FEED1", "FEED2"):
                    tab.putcol(column, np.zeros(count, dtype=np.int32), row0, count)
                row0 += count


def _chunk(array, start: int, stop: int) -> np.ndarray:
    """Read at most ``rowchunk`` time samples, computing dask only here."""
    data = array.isel(time=slice(start, stop)).data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)
