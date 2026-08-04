"""Typed, format-agnostic description of a Measurement Set.

These dataclasses are the contract between the readers (:mod:`._msv2` on
casacore/TaQL, :mod:`._msv4` on anything shaped like the MSv4 schema) and
everything that consumes MS metadata -- the renderer, the JSON dump, the CLI,
and callers.

Design notes:

* **Addressable by name and by id.** Real work says "the bandpass calibrator",
  not "row 1", so collections are :class:`Registry` objects supporting
  ``info.fields[0]``, ``info.fields["PKS1934-638"]`` and iteration.
* **Derived values are properties, not stored fields.** ``freq_start`` is
  computed from ``chan_freq``; nothing can drift out of sync, and the JSON
  dump stays a faithful record of what was read.
* **JSON is a stable, versioned schema.** :data:`SCHEMA_VERSION` is emitted in
  every dump so downstream pipelines can branch on it.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field as _field, fields as _fields
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy

__all__ = [
    "SCHEMA_VERSION",
    "STOKES_TYPES",
    "FREQ_FRAMES",
    "Registry",
    "Antenna",
    "SpectralWindow",
    "Polarization",
    "DataDescription",
    "Scan",
    "Field",
    "Observation",
    "Column",
    "MSInfo",
    "mjd_seconds_to_datetime",
    "format_ra",
    "format_dec",
    "format_duration",
    "itrf_to_geodetic",
    "geodetic_to_itrf",
]

#: Version of the JSON structure produced by :meth:`MSInfo.to_dict`. Bumped
#: on any breaking change to key names or nesting.
SCHEMA_VERSION = "1.0"

#: Correlation-type code -> label, per the casacore ``Stokes`` enumeration.
STOKES_TYPES = {
    0: "Undefined", 1: "I", 2: "Q", 3: "U", 4: "V",
    5: "RR", 6: "RL", 7: "LR", 8: "LL",
    9: "XX", 10: "XY", 11: "YX", 12: "YY",
    13: "RX", 14: "RY", 15: "LX", 16: "LY",
    17: "XR", 18: "XL", 19: "YR", 20: "YL",
    21: "PP", 22: "PQ", 23: "QP", 24: "QQ",
    25: "RCircular", 26: "LCircular", 27: "Linear",
    28: "Ptotal", 29: "Plinear", 30: "PFtotal", 31: "PFlinear", 32: "Pangle",
}

#: ``SPECTRAL_WINDOW::MEAS_FREQ_REF`` code -> spectral reference frame.
FREQ_FRAMES = {
    0: "REST", 1: "LSRK", 2: "LSRD", 3: "BARY", 4: "GEO",
    5: "TOPO", 6: "GALACTO", 7: "LGROUP", 8: "CMB", 64: "Undefined",
}

# MJD zero point: 1858-11-17 00:00:00 UTC. MS TIME columns are seconds since
# this epoch. Leap seconds are ignored, as they are throughout casacore's own
# listobs-style output.
_MJD0 = datetime.datetime(1858, 11, 17, tzinfo=datetime.timezone.utc)


def mjd_seconds_to_datetime(seconds: float) -> datetime.datetime:
    """Convert an MS ``TIME`` value (MJD seconds, UTC) to a datetime."""
    return _MJD0 + datetime.timedelta(seconds=float(seconds))


def _isot(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    return mjd_seconds_to_datetime(seconds).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def _sexagesimal(value: float, decimals: int) -> Tuple[int, int, float]:
    """Split ``value`` (hours or degrees) into whole/minutes/seconds.

    Rounds to the display precision *before* splitting. Doing it the other way
    round produces carry artefacts: -30 degrees arrives as -29.999999999999996
    and formats as ``-29d59m60.00``.
    """
    total = round(value * 3600.0, decimals)
    whole, remainder = divmod(total, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return int(whole), int(minutes), seconds


def format_ra(radians: float) -> str:
    """Right ascension in radians -> ``HHhMMmSS.sss``."""
    hh, mm, ss = _sexagesimal(math.degrees(radians) / 15.0 % 24.0, 3)
    return "{0:02d}h{1:02d}m{2:06.3f}".format(hh % 24, mm, ss)


def format_dec(radians: float) -> str:
    """Declination in radians -> ``+DDdMMmSS.ss``."""
    degrees = math.degrees(radians)
    sign = "-" if degrees < 0 else "+"
    dd, mm, ss = _sexagesimal(abs(degrees), 2)
    return "{0}{1:02d}d{2:02d}m{3:05.2f}".format(sign, dd, mm, ss)


def format_duration(seconds: float) -> str:
    """Seconds -> a compact ``1h23m45s`` style string."""
    seconds = float(seconds)
    if seconds < 60:
        return "{0:.1f}s".format(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return "{0:d}h{1:02d}m{2:02.0f}s".format(hours, minutes, sec)
    return "{0:d}m{1:02.0f}s".format(minutes, sec)


# WGS84 ellipsoid, as used for ITRF antenna positions.
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F ** 2


def itrf_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ITRF/ECEF metres -> (longitude, latitude, height) in radians, radians, metres.

    Uses Bowring's closed-form approximation, accurate to well under a
    millimetre for terrestrial heights.
    """
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p == 0.0:                                  # on the polar axis
        return lon, math.copysign(math.pi / 2, z), abs(z) - _WGS84_B
    ep2 = (_WGS84_A ** 2 - _WGS84_B ** 2) / _WGS84_B ** 2
    theta = math.atan2(z * _WGS84_A, p * _WGS84_B)
    lat = math.atan2(z + ep2 * _WGS84_B * math.sin(theta) ** 3,
                     p - _WGS84_E2 * _WGS84_A * math.cos(theta) ** 3)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    height = p / math.cos(lat) - n
    return lon, lat, height


def geodetic_to_itrf(lon: float, lat: float, height: float) -> Tuple[float, float, float]:
    """(longitude, latitude, height) in radians/metres -> ITRF/ECEF metres."""
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    return ((n + height) * math.cos(lat) * math.cos(lon),
            (n + height) * math.cos(lat) * math.sin(lon),
            ((_WGS84_B ** 2 / _WGS84_A ** 2) * n + height) * math.sin(lat))


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and dataclasses to plain JSON types."""
    if isinstance(value, Registry):
        return [_jsonify(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, numpy.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class _Record:
    """Mixin giving dataclasses a JSON-safe ``to_dict`` including properties."""

    #: Property names to include in ``to_dict`` alongside the declared fields.
    _derived: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        out = {f.name: _jsonify(getattr(self, f.name)) for f in _fields(self)}
        for name in self._derived:
            out[name] = _jsonify(getattr(self, name))
        return out


class Registry(Sequence):
    """An ordered collection addressable by integer id *or* by name.

    Integer keys are looked up by the record's ``id``/``number`` attribute
    rather than by position, because MS ids are not always contiguous (a
    subset MS can keep the original field ids). Names must be unique to be
    addressable; duplicates resolve to the first match.
    """

    def __init__(self, items: Sequence[Any], key: str = "id", name: str = "name"):
        self._items = list(items)
        self._key = key
        self._name = name
        self._by_id = {getattr(i, key): i for i in reversed(self._items)}
        self._by_name = {}
        for item in reversed(self._items):
            label = getattr(item, name, None)
            if label:
                self._by_name[str(label)] = item

    def __getitem__(self, key: Union[int, str, slice]) -> Any:
        if isinstance(key, slice):
            return self._items[key]
        if isinstance(key, str):
            try:
                return self._by_name[key]
            except KeyError:
                raise KeyError("no entry named {0!r}; have {1}".format(
                    key, sorted(self._by_name))) from None
        try:
            return self._by_id[key]
        except KeyError:
            raise KeyError("no entry with {0}={1!r}; have {2}".format(
                self._key, key, sorted(self._by_id))) from None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __repr__(self) -> str:
        return "Registry({0!r})".format(self._items)

    @property
    def ids(self) -> List[Any]:
        """The id of every record, in order."""
        return [getattr(i, self._key) for i in self._items]

    @property
    def names(self) -> List[str]:
        """The name of every record, in order."""
        return [getattr(i, self._name, None) for i in self._items]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [i.to_dict() for i in self._items]


@dataclass
class Antenna(_Record):
    """One row of the ANTENNA subtable."""

    id: int
    name: str
    station: str = ""
    mount: str = ""
    type: str = ""
    dish_diameter: float = 0.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    flagged: bool = False

    _derived = ("longitude", "latitude", "elevation")

    @property
    def _geodetic(self) -> Tuple[float, float, float]:
        return itrf_to_geodetic(*self.position)

    @property
    def longitude(self) -> float:
        """Geodetic longitude in degrees."""
        return math.degrees(self._geodetic[0])

    @property
    def latitude(self) -> float:
        """Geodetic latitude in degrees."""
        return math.degrees(self._geodetic[1])

    @property
    def elevation(self) -> float:
        """Height above the WGS84 ellipsoid, in metres."""
        return self._geodetic[2]


@dataclass
class SpectralWindow(_Record):
    """One row of the SPECTRAL_WINDOW subtable."""

    id: int
    name: str = ""
    num_chan: int = 0
    chan_freq: List[float] = _field(default_factory=list)
    chan_width: List[float] = _field(default_factory=list)
    ref_frequency: float = 0.0
    total_bandwidth: float = 0.0
    meas_freq_ref: int = 0
    freq_group_name: str = ""

    _derived = ("frame", "freq_start", "freq_end", "centre_freq")

    @property
    def frame(self) -> str:
        """Spectral reference frame name, e.g. ``TOPO``."""
        return FREQ_FRAMES.get(int(self.meas_freq_ref), "Unknown")

    @property
    def freq_start(self) -> Optional[float]:
        """Lower edge of the first channel, in Hz."""
        if not len(self.chan_freq):
            return None
        return float(self.chan_freq[0]) - abs(float(self.chan_width[0])) / 2.0

    @property
    def freq_end(self) -> Optional[float]:
        """Upper edge of the last channel, in Hz."""
        if not len(self.chan_freq):
            return None
        return float(self.chan_freq[-1]) + abs(float(self.chan_width[-1])) / 2.0

    @property
    def centre_freq(self) -> Optional[float]:
        """Midpoint of the band, in Hz."""
        if not len(self.chan_freq):
            return None
        return (self.freq_start + self.freq_end) / 2.0


@dataclass
class Polarization(_Record):
    """One row of the POLARIZATION subtable."""

    id: int
    num_corr: int = 0
    corr_type: List[int] = _field(default_factory=list)

    _derived = ("corr_labels", "name")

    @property
    def corr_labels(self) -> List[str]:
        """Correlation codes decoded to labels, e.g. ``["XX", "XY", ...]``."""
        return [STOKES_TYPES.get(int(c), "Unknown") for c in self.corr_type]

    @property
    def name(self) -> str:
        """A readable setup name, e.g. ``XX,XY,YX,YY`` (also the Registry key)."""
        return ",".join(self.corr_labels)


@dataclass
class DataDescription(_Record):
    """One row of DATA_DESCRIPTION: the ``DATA_DESC_ID`` -> (SPW, POL) mapping.

    The main table keys off ``DATA_DESC_ID``, which is *not* interchangeable
    with ``SPECTRAL_WINDOW_ID`` in general -- conflating the two is a common
    source of quietly wrong results.
    """

    id: int
    spw_id: int = 0
    pol_id: int = 0
    flagged: bool = False

    _derived = ("name",)

    @property
    def name(self) -> str:
        return "DD{0:d}".format(self.id)


@dataclass
class Scan(_Record):
    """A single scan, as observed: one field, one contiguous stretch of time."""

    number: int
    field_id: int
    field_name: str = ""
    spw_ids: List[int] = _field(default_factory=list)
    time_start: float = 0.0
    time_end: float = 0.0
    interval: float = 0.0
    nrows: int = 0
    intents: List[str] = _field(default_factory=list)

    _derived = ("name", "duration", "start_utc", "end_utc")

    @property
    def name(self) -> str:
        return str(self.number)

    @property
    def duration(self) -> float:
        """Elapsed time in seconds, **including** the final integration.

        ``TIME`` holds integration midpoints, so ``max - min`` understates the
        scan by one integration; the missing ``interval`` is added back here.
        """
        return float(self.time_end - self.time_start) + float(self.interval)

    @property
    def start_utc(self) -> Optional[str]:
        return _isot(self.time_start)

    @property
    def end_utc(self) -> Optional[str]:
        return _isot(self.time_end)


@dataclass
class Field(_Record):
    """One row of the FIELD subtable, plus what the main table says about it."""

    id: int
    name: str
    code: str = ""
    source_id: int = -1
    phase_centre: Tuple[float, float] = (0.0, 0.0)
    ref_frame: str = "J2000"
    intents: List[str] = _field(default_factory=list)
    scan_numbers: List[int] = _field(default_factory=list)
    spw_ids: List[int] = _field(default_factory=list)
    nrows: int = 0
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    exposure: float = 0.0

    _derived = ("ra", "dec", "ra_hms", "dec_dms", "nscans", "start_utc", "end_utc")

    @property
    def ra(self) -> float:
        """Phase-centre right ascension in degrees."""
        return math.degrees(self.phase_centre[0]) % 360.0

    @property
    def dec(self) -> float:
        """Phase-centre declination in degrees."""
        return math.degrees(self.phase_centre[1])

    @property
    def ra_hms(self) -> str:
        return format_ra(self.phase_centre[0])

    @property
    def dec_dms(self) -> str:
        return format_dec(self.phase_centre[1])

    @property
    def nscans(self) -> int:
        return len(self.scan_numbers)

    @property
    def start_utc(self) -> Optional[str]:
        return _isot(self.time_start)

    @property
    def end_utc(self) -> Optional[str]:
        return _isot(self.time_end)


@dataclass
class Observation(_Record):
    """OBSERVATION subtable content plus the observed time span."""

    telescope: str = ""
    observer: str = ""
    project: str = ""
    time_start: Optional[float] = None
    time_end: Optional[float] = None

    _derived = ("start_utc", "end_utc", "duration")

    @property
    def start_utc(self) -> Optional[str]:
        return _isot(self.time_start)

    @property
    def end_utc(self) -> Optional[str]:
        return _isot(self.time_end)

    @property
    def duration(self) -> Optional[float]:
        """Wall-clock span of the observation in seconds."""
        if self.time_start is None or self.time_end is None:
            return None
        return float(self.time_end - self.time_start)


@dataclass
class Column(_Record):
    """A main-table column: what it is, and roughly what it costs.

    ``shape`` is the shape of a single cell, or ``None`` for scalar columns
    *and* for variable-shaped array columns whose first cell is unfilled --
    ``ndim`` distinguishes the two. ``nbytes`` is the logical size (rows x
    cell size), not on-disk bytes, which depend on the storage manager.
    """

    name: str
    dtype: str = ""
    ndim: int = 0
    shape: Optional[List[int]] = None
    unit: str = ""
    data_manager: str = ""
    nbytes: Optional[int] = None

    _derived = ("is_scalar", "shape_label")

    @property
    def is_scalar(self) -> bool:
        return self.ndim == 0

    @property
    def shape_label(self) -> str:
        """Cell shape as ``4x4``, ``scalar``, or ``3-D (variable)``."""
        if self.shape:
            return "x".join(str(s) for s in self.shape)
        if self.ndim:
            return "{0:d}-D (variable)".format(self.ndim)
        return "scalar"


@dataclass
class MSInfo(_Record):
    """Everything :func:`msutils.msinfo` knows about one Measurement Set."""

    path: str
    format: str = "MSv2"
    engine: str = "casacore"
    nrows: int = 0
    size_bytes: Optional[int] = None
    observation: Observation = _field(default_factory=Observation)
    antennas: Registry = _field(default_factory=lambda: Registry([]))
    fields: Registry = _field(default_factory=lambda: Registry([]))
    spws: Registry = _field(default_factory=lambda: Registry([]))
    polarizations: Registry = _field(default_factory=lambda: Registry([]))
    data_descriptions: Registry = _field(default_factory=lambda: Registry([]))
    scans: Registry = _field(default_factory=lambda: Registry([], key="number"))
    columns: Registry = _field(default_factory=lambda: Registry([], key="name"))
    subtables: List[str] = _field(default_factory=list)
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    nbaselines: int = 0
    max_baseline: Optional[float] = None
    max_uv_distance: Optional[float] = None
    integration_times: List[float] = _field(default_factory=list)
    nflagged: Optional[int] = None
    nvisibilities: Optional[int] = None
    schema_version: str = SCHEMA_VERSION

    _derived = ("nantennas", "nfields", "nspws", "nscans", "nchan_total",
                "column_names", "flagged_fraction", "start_utc", "end_utc",
                "duration")

    @property
    def start_utc(self) -> Optional[str]:
        """First timestamp actually present in the data."""
        return _isot(self.time_start)

    @property
    def end_utc(self) -> Optional[str]:
        """Last timestamp actually present in the data."""
        return _isot(self.time_end)

    @property
    def duration(self) -> Optional[float]:
        """Elapsed time spanned by the data, in seconds.

        Taken from the scans rather than ``OBSERVATION::TIME_RANGE``, which is
        frequently stale or wrong in MSs written by real telescopes.
        """
        if self.time_start is None or self.time_end is None:
            return None
        return float(self.time_end - self.time_start)

    @property
    def nantennas(self) -> int:
        return len(self.antennas)

    @property
    def nfields(self) -> int:
        return len(self.fields)

    @property
    def nspws(self) -> int:
        return len(self.spws)

    @property
    def nscans(self) -> int:
        return len(self.scans)

    @property
    def nchan_total(self) -> int:
        """Channels summed over all spectral windows."""
        return int(sum(s.num_chan for s in self.spws))

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @property
    def flagged_fraction(self) -> Optional[float]:
        """Fraction of visibilities flagged, or ``None`` if not computed."""
        if self.nflagged is None or not self.nvisibilities:
            return None
        return float(self.nflagged) / float(self.nvisibilities)

    def render(self, verbose: bool = False) -> str:
        """A human-readable, ``listobs``-style report."""
        from ._render import render
        return render(self, verbose=verbose)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialise to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str, indent: Optional[int] = 2) -> str:
        """Write :meth:`to_json` to ``path`` and return the path."""
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(self.to_json(indent=indent))
        return path

    def __str__(self) -> str:
        return self.render()
