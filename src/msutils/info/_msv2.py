"""Read an MSv2 into an :class:`~msutils.info._model.MSInfo`, using TaQL.

The whole point of this module is that **main-table metadata is gathered in a
single streaming pass**. The old ``summary()`` ran one ``table.query()`` per
field and another per scan, so its cost grew as O(nfields x nscans) full table
scans -- 0.55 s for a 595k-row MS with 6 scans, 1.53 s for the same MS with 60
scans. One ``GROUPBY`` does the same work in casacore's C++ layer in 0.14 s,
flat in the number of scans.

Everything derived from the main table therefore comes from
:func:`_aggregate`, which groups by ``(FIELD_ID, SCAN_NUMBER, DATA_DESC_ID,
STATE_ID)`` -- the finest grouping anything here needs -- and folds the result
up into scans, fields and MS-wide totals.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy

from .._log import create_logger
from .._tables import open_table, query, subtable_names
from ._model import (Antenna, Column, DataDescription, Field, MSInfo,
                     Observation, Polarization, Registry, Scan, SpectralWindow)

LOGGER = create_logger(__name__)

#: Detail levels accepted by :func:`read`, cheapest first. Each names the cost
#: tier by what it has to read.
LEVELS = ("meta", "full", "data")

# Aggregates gathered in the single main-table pass. Kept as a list so the
# SELECT list and the column extraction cannot drift apart.
#
# Deliberately index columns only. TaQL re-reads a column for every expression
# mentioning it, so each added UVW/FLAG term costs a further pass over the
# bulk data: on a 595k-row MS this select is 0.17 s, and one `SUMSQR(UVW)`
# term takes it to 0.64 s. Those live at level="data".
_AGGREGATES = [
    ("T0", "GMIN(TIME)"),
    ("T1", "GMAX(TIME)"),
    ("NROW", "GCOUNT()"),
    ("DT", "GMIN(INTERVAL)"),
    ("EXPOSURE", "GSUM(EXPOSURE)"),
]
_DATA_AGGREGATES = [
    ("MAXUVW", "GMAX(SUMSQR(UVW))"),
    ("NFLAG", "GNTRUE(FLAG)"),
    ("NVIS", "GNFALSE(FLAG) + GNTRUE(FLAG)"),
]

_GROUP_KEYS = ("FIELD_ID", "SCAN_NUMBER", "DATA_DESC_ID", "STATE_ID")


def read(path: str, level: str = "full") -> MSInfo:
    """Read MS metadata at ``path``.

    Args:
        path: Path to the Measurement Set.
        level: How much work to do.

            * ``"meta"`` -- subtables only. No main-table scan at all, so the
              cost is independent of MS size. Scans and per-field row counts
              are left empty; ``max_baseline`` still comes out, since it is
              geometry from the ANTENNA table.
            * ``"full"`` (default) -- one aggregate pass over the main table's
              *index* columns, giving scans, time ranges and row counts.
            * ``"data"`` -- also read the bulk UVW and FLAG columns, for uv
              coverage and flag statistics. Several times more expensive.
    """
    if level not in LEVELS:
        raise ValueError("level must be one of {0}, got {1!r}".format(LEVELS, level))

    with open_table(path) as tab:
        info = MSInfo(path=os.path.abspath(path), format="MSv2",
                      engine="casacore", nrows=tab.nrows())
        info.size_bytes = _directory_size(path)
        info.subtables = subtable_names(tab)
        info.columns = Registry(_read_columns(tab), key="name")

        info.antennas = Registry(_read_antennas(path))
        info.spws = Registry(_read_spws(path))
        info.polarizations = Registry(_read_polarizations(path))
        info.data_descriptions = Registry(_read_data_descriptions(path, info))
        info.observation = _read_observation(path)
        intents_by_state = _read_state_intents(path)
        fields = _read_fields(path)

        if level == "meta" or info.nrows == 0:
            info.fields = Registry(fields)
            info.max_baseline = _max_antenna_separation(
                info.antennas, [(a.id, b.id) for a in info.antennas
                                for b in info.antennas if a.id < b.id])
            _fill_observation_span(info)
            return info

        groups = _aggregate(tab, with_data=(level == "data"))
        _fold_groups(info, groups, fields, intents_by_state)
        baselines = _read_baselines(tab)
        info.nbaselines = len(baselines)
        # Max baseline is the longest *physical* antenna separation among the
        # pairs actually present -- exact, and free, unlike deriving it from
        # per-row UVW. `max_uv_distance` (the projected uv extent, which is
        # what limits image resolution) does need UVW, so it is level="data".
        info.max_baseline = _max_antenna_separation(info.antennas, baselines)

    _fill_observation_span(info)
    return info


def _aggregate(tab, with_data: bool = False) -> Dict[str, numpy.ndarray]:
    """One GROUPBY pass over the main table; returns column-name -> array."""
    aggregates = _AGGREGATES + (_DATA_AGGREGATES if with_data else [])
    selects = list(_GROUP_KEYS) + ["{0} AS {1}".format(expr, name)
                                   for name, expr in aggregates]
    command = "SELECT {0} FROM $1 GROUPBY {1}".format(
        ", ".join(selects), ", ".join(_GROUP_KEYS))
    LOGGER.debug("aggregating main table: %s", command)

    wanted = list(_GROUP_KEYS) + [name for name, _ in aggregates]
    with query(command, [tab]) as res:
        return {name: numpy.asarray(res.getcol(name)) for name in wanted}


def _fold_groups(info: MSInfo, groups: Dict[str, numpy.ndarray],
                 fields: List[Field], intents_by_state: Dict[int, List[str]]) -> None:
    """Collapse the per-(field, scan, ddid, state) aggregates into the model."""
    ddid_to_spw = {dd.id: dd.spw_id for dd in info.data_descriptions}
    field_by_id = {f.id: f for f in fields}

    scans: Dict[int, Scan] = {}
    per_field: Dict[int, Dict[str, Any]] = {}
    max_uvw = None
    intervals = set()
    nflag = nvis = None

    for i in range(len(groups["FIELD_ID"])):
        fid = int(groups["FIELD_ID"][i])
        scan_no = int(groups["SCAN_NUMBER"][i])
        ddid = int(groups["DATA_DESC_ID"][i])
        state_id = int(groups["STATE_ID"][i])
        t0, t1 = float(groups["T0"][i]), float(groups["T1"][i])
        nrow = int(groups["NROW"][i])
        interval = float(groups["DT"][i])
        exposure = float(groups["EXPOSURE"][i])
        spw_id = ddid_to_spw.get(ddid, ddid)
        intents = intents_by_state.get(state_id, [])

        intervals.add(round(interval, 6))
        if "MAXUVW" in groups:
            # aggregated as squared magnitude; one sqrt here beats one per row
            max_uvw = _max(max_uvw, groups["MAXUVW"][i])
            nflag = int(groups["NFLAG"][i]) + (nflag or 0)
            nvis = int(groups["NVIS"][i]) + (nvis or 0)

        scan = scans.get(scan_no)
        if scan is None:
            scan = scans[scan_no] = Scan(
                number=scan_no, field_id=fid,
                field_name=field_by_id[fid].name if fid in field_by_id else "",
                time_start=t0, time_end=t1, interval=interval)
        scan.time_start = min(scan.time_start, t0)
        scan.time_end = max(scan.time_end, t1)
        scan.interval = min(scan.interval, interval)
        scan.nrows += nrow
        _extend(scan.spw_ids, spw_id)
        for intent in intents:
            _extend(scan.intents, intent)

        stats = per_field.setdefault(fid, {"nrows": 0, "t0": t0, "t1": t1,
                                           "scans": [], "spws": [],
                                           "intents": [], "exposure": 0.0})
        stats["nrows"] += nrow
        stats["exposure"] += exposure
        stats["t0"] = min(stats["t0"], t0)
        stats["t1"] = max(stats["t1"], t1)
        _extend(stats["scans"], scan_no)
        _extend(stats["spws"], spw_id)
        for intent in intents:
            _extend(stats["intents"], intent)

    for fid, stats in per_field.items():
        target = field_by_id.get(fid)
        if target is None:                    # FIELD_ID with no FIELD row
            target = field_by_id[fid] = Field(id=fid, name="FIELD_{0:d}".format(fid))
            fields.append(target)
        target.nrows = stats["nrows"]
        target.time_start = stats["t0"]
        target.time_end = stats["t1"]
        target.scan_numbers = sorted(stats["scans"])
        target.spw_ids = sorted(stats["spws"])
        target.intents = sorted(stats["intents"])
        target.exposure = stats["exposure"]

    info.fields = Registry(sorted(fields, key=lambda f: f.id))
    info.scans = Registry([scans[n] for n in sorted(scans)], key="number")
    if scans:
        info.time_start = min(s.time_start for s in scans.values())
        info.time_end = max(s.time_end + s.interval for s in scans.values())
    info.max_uv_distance = math.sqrt(max_uvw) if max_uvw is not None else None
    info.integration_times = sorted(intervals)
    info.nflagged = nflag
    info.nvisibilities = nvis


def _read_baselines(tab) -> List[Tuple[int, int]]:
    """The distinct antenna pairs actually present in the data."""
    with query("SELECT DISTINCT ANTENNA1, ANTENNA2 FROM $1", [tab]) as res:
        if res.nrows() == 0:
            return []
        return [(int(a), int(b))
                for a, b in zip(res.getcol("ANTENNA1"), res.getcol("ANTENNA2"))]


def _max_antenna_separation(antennas, baselines: List[Tuple[int, int]]) -> Optional[float]:
    """Longest physical separation, in metres, over the given antenna pairs."""
    if not baselines or not len(antennas):
        return None
    position = {a.id: numpy.asarray(a.position, dtype=float) for a in antennas}
    longest = 0.0
    for p, q in baselines:
        if p in position and q in position and p != q:
            longest = max(longest, float(numpy.linalg.norm(position[p] - position[q])))
    return longest or None


def _read_columns(tab) -> List[Column]:
    """Describe the main-table columns, including their data managers."""
    manager_of = {}
    try:
        for spec in tab.getdminfo().values():
            for name in spec.get("COLUMNS", []):
                manager_of[name] = spec.get("TYPE", "")
    except RuntimeError:                       # pragma: no cover - exotic tables
        LOGGER.debug("could not read data manager info for %s", tab.name())

    columns = []
    nrows = tab.nrows()
    for name in tab.colnames():
        desc = tab.getcoldesc(name)
        ndim = int(desc.get("ndim", 0) or 0)
        shape = desc.get("shape")
        shape = [int(s) for s in shape] if shape is not None else None
        if shape is None and ndim and nrows:
            # Variable-shaped array column: sample the first cell. Unfilled
            # cells (FLAG_CATEGORY in most MSs) raise, and stay shapeless.
            try:
                sampled = [int(s) for s in numpy.shape(tab.getcell(name, 0))]
                shape = sampled or None
            except RuntimeError:
                shape = None
        unit = ""
        keywords = desc.get("keywords") or {}
        units = keywords.get("QuantumUnits")
        if units is not None and len(units):
            unit = str(units[0])
        columns.append(Column(
            name=name,
            dtype=str(desc.get("valueType", "")),
            ndim=ndim,
            shape=shape,
            unit=unit,
            data_manager=manager_of.get(name, ""),
            nbytes=_column_nbytes(desc.get("valueType", ""), shape, nrows),
        ))
    return columns


#: Bytes per element for the casacore value types we expect in an MS.
_ITEMSIZE = {
    "bool": 1, "boolean": 1, "uchar": 1, "short": 2, "ushort": 2,
    "int": 4, "uint": 4, "int64": 8, "float": 4, "double": 8,
    "complex": 8, "dcomplex": 16,
}


def _column_nbytes(valuetype: str, shape: Optional[List[int]],
                   nrows: int) -> Optional[int]:
    """Logical size of a column: rows x cell size. Not on-disk size."""
    itemsize = _ITEMSIZE.get(str(valuetype).lower())
    if itemsize is None or not nrows:
        return None
    cells = 1
    for dim in shape or []:
        cells *= int(dim)
    return int(itemsize * cells * nrows)


def _read_antennas(path: str) -> List[Antenna]:
    with open_table(path + "::ANTENNA") as tab:
        if tab.nrows() == 0:
            return []
        names = _strings(tab, "NAME", tab.nrows())
        stations = _strings(tab, "STATION", tab.nrows())
        mounts = _strings(tab, "MOUNT", tab.nrows())
        types = _strings(tab, "TYPE", tab.nrows())
        positions = tab.getcol("POSITION")
        diameters = _optional_col(tab, "DISH_DIAMETER")
        flags = _optional_col(tab, "FLAG_ROW")
        return [Antenna(id=i, name=names[i], station=stations[i], mount=mounts[i],
                        type=types[i],
                        dish_diameter=float(diameters[i]) if diameters is not None else 0.0,
                        position=tuple(float(v) for v in positions[i]),
                        flagged=bool(flags[i]) if flags is not None else False)
                for i in range(tab.nrows())]


def _read_spws(path: str) -> List[SpectralWindow]:
    with open_table(path + "::SPECTRAL_WINDOW") as tab:
        if tab.nrows() == 0:
            return []
        names = _strings(tab, "NAME", tab.nrows())
        groups = _strings(tab, "FREQ_GROUP_NAME", tab.nrows())
        spws = []
        for i in range(tab.nrows()):
            spws.append(SpectralWindow(
                id=i,
                name=names[i],
                num_chan=int(tab.getcell("NUM_CHAN", i)),
                chan_freq=[float(f) for f in numpy.atleast_1d(tab.getcell("CHAN_FREQ", i))],
                chan_width=[float(w) for w in numpy.atleast_1d(tab.getcell("CHAN_WIDTH", i))],
                ref_frequency=float(tab.getcell("REF_FREQUENCY", i)),
                total_bandwidth=float(tab.getcell("TOTAL_BANDWIDTH", i)),
                meas_freq_ref=int(tab.getcell("MEAS_FREQ_REF", i))
                if "MEAS_FREQ_REF" in tab.colnames() else 0,
                freq_group_name=groups[i],
            ))
        return spws


def _read_polarizations(path: str) -> List[Polarization]:
    with open_table(path + "::POLARIZATION") as tab:
        return [Polarization(id=i,
                             num_corr=int(tab.getcell("NUM_CORR", i)),
                             corr_type=[int(c) for c in
                                        numpy.atleast_1d(tab.getcell("CORR_TYPE", i))])
                for i in range(tab.nrows())]


def _read_data_descriptions(path: str, info: MSInfo) -> List[DataDescription]:
    with open_table(path + "::DATA_DESCRIPTION") as tab:
        if tab.nrows() == 0:
            # Degenerate MS: assume a 1:1 DDID -> SPW mapping.
            return [DataDescription(id=i, spw_id=i, pol_id=0)
                    for i in range(len(info.spws))]
        spw_ids = tab.getcol("SPECTRAL_WINDOW_ID")
        pol_ids = tab.getcol("POLARIZATION_ID")
        flags = _optional_col(tab, "FLAG_ROW")
        return [DataDescription(id=i, spw_id=int(spw_ids[i]), pol_id=int(pol_ids[i]),
                                flagged=bool(flags[i]) if flags is not None else False)
                for i in range(tab.nrows())]


def _read_fields(path: str) -> List[Field]:
    with open_table(path + "::FIELD") as tab:
        if tab.nrows() == 0:
            return []
        names = _strings(tab, "NAME", tab.nrows())
        codes = _strings(tab, "CODE", tab.nrows())
        source_ids = _optional_col(tab, "SOURCE_ID")
        frame = _direction_frame(tab, "PHASE_DIR")
        fields = []
        for i in range(tab.nrows()):
            centre = numpy.atleast_2d(tab.getcell("PHASE_DIR", i))[0]
            fields.append(Field(
                id=i, name=names[i], code=codes[i],
                source_id=int(source_ids[i]) if source_ids is not None else -1,
                phase_centre=(float(centre[0]), float(centre[1])),
                ref_frame=frame,
            ))
        return fields


def _read_observation(path: str) -> Observation:
    with open_table(path + "::OBSERVATION") as tab:
        if tab.nrows() == 0:
            return Observation()
        span = numpy.atleast_1d(tab.getcell("TIME_RANGE", 0))
        return Observation(
            telescope=str(tab.getcell("TELESCOPE_NAME", 0)),
            observer=str(tab.getcell("OBSERVER", 0)),
            project=str(tab.getcell("PROJECT", 0)),
            time_start=float(span[0]) if len(span) else None,
            time_end=float(span[1]) if len(span) > 1 else None,
        )


def _read_state_intents(path: str) -> Dict[int, List[str]]:
    """STATE_ID -> observing intents.

    ``OBS_MODE`` holds comma-separated intents such as
    ``CALIBRATE_BANDPASS#ON_SOURCE``. Resolving them per ``STATE_ID`` is what
    makes intents correct per scan; the old code dumped the raw column and
    labelled it as per-field, which only worked when the STATE and FIELD
    tables happened to be parallel.
    """
    try:
        with open_table(path + "::STATE") as tab:
            modes = _strings(tab, "OBS_MODE", tab.nrows())
    except RuntimeError:
        return {}
    return {i: [part.strip() for part in modes[i].split(",") if part.strip()]
            for i in range(len(modes))}


def _fill_observation_span(info: MSInfo) -> None:
    """Fall back to OBSERVATION::TIME_RANGE when there is no observed span.

    ``MSInfo.time_start``/``time_end`` describe the data itself and are the
    values worth trusting; the OBSERVATION subtable is often stale. At
    ``level="meta"`` there are no scans to derive a span from, so the
    subtable is all we have.
    """
    if info.time_start is not None:
        return
    obs = info.observation
    if obs.time_start is not None and obs.time_end is not None:
        info.time_start, info.time_end = obs.time_start, obs.time_end


def _direction_frame(tab, column: str) -> str:
    """Direction reference frame ('J2000', 'ICRS', ...) from a column's MEASINFO."""
    try:
        measinfo = tab.getcoldesc(column).get("keywords", {}).get("MEASINFO", {})
        return str(measinfo.get("Ref", "J2000"))
    except (RuntimeError, AttributeError):     # pragma: no cover
        return "J2000"


def _strings(tab, column: str, nrows: int) -> List[str]:
    """Read a string column, tolerating its absence."""
    if column not in tab.colnames() or nrows == 0:
        return [""] * nrows
    return [str(v) for v in tab.getcol(column)]


def _optional_col(tab, column: str):
    """Read a column, or return ``None`` if it is absent or unfilled."""
    if column not in tab.colnames():
        return None
    try:
        return tab.getcol(column)
    except RuntimeError:
        return None


def _max(current: Optional[float], candidate: Any) -> float:
    value = float(candidate)
    return value if current is None else max(current, value)


def _extend(target: List[Any], value: Any) -> None:
    """Append ``value`` to ``target`` if not already present (order-preserving)."""
    if value not in target:
        target.append(value)


def _directory_size(path: str) -> Optional[int]:
    """Total bytes occupied by an MS directory tree."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    except OSError:                            # pragma: no cover
        return None
    return total
