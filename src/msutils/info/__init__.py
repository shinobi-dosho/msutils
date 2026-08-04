"""Structured Measurement Set metadata.

:func:`msinfo` is the entry point. It returns an :class:`MSInfo` -- a typed,
JSON-serialisable description of an MS whose collections are addressable by
name as well as by id::

    import msutils

    info = msutils.msinfo("obs.ms")
    print(info.render())                       # listobs-style report

    info.fields["PKS1934-638"].scan_numbers    # [1, 2]
    info.spws[0].chan_width[0]                 # 10000000.0
    info.antennas["m003"].latitude             # -30.71...
    info.to_dict()                             # stable, versioned JSON

This replaces the older :func:`msutils.summary`, which returned an unstructured
dict of ALL-CAPS keys, misreported observing intents and scan durations, and
scanned the main table once per field and once per scan.
"""
from __future__ import annotations

import os
from typing import Optional

from ._model import (SCHEMA_VERSION, STOKES_TYPES, Antenna, Column,
                     DataDescription, Field, MSInfo, Observation, Polarization,
                     Registry, Scan, SpectralWindow, format_dec, format_duration,
                     format_ra, geodetic_to_itrf, itrf_to_geodetic,
                     mjd_seconds_to_datetime)
from ._render import format_bytes, format_frequency, render

__all__ = [
    "msinfo",
    "detect_format",
    "MSInfo",
    "Observation",
    "Antenna",
    "Field",
    "Scan",
    "SpectralWindow",
    "Polarization",
    "DataDescription",
    "Column",
    "Registry",
    "SCHEMA_VERSION",
    "STOKES_TYPES",
    "render",
    "format_bytes",
    "format_frequency",
    "format_ra",
    "format_dec",
    "format_duration",
    "itrf_to_geodetic",
    "geodetic_to_itrf",
    "mjd_seconds_to_datetime",
]


def detect_format(path: str) -> str:
    """Return ``"MSv2"`` or ``"MSv4"`` for the dataset at ``path``.

    MSv2 is a casacore table (a directory containing ``table.dat``); MSv4
    processing sets are zarr hierarchies (``.zgroup``/``zarr.json``, possibly
    one level down, since a processing set is a directory of MSv4 datasets).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.isfile(os.path.join(path, "table.dat")):
        return "MSv2"
    for marker in (".zgroup", ".zattrs", "zarr.json"):
        if os.path.exists(os.path.join(path, marker)):
            return "MSv4"
    try:
        for entry in sorted(os.listdir(path)):
            child = os.path.join(path, entry)
            if os.path.isdir(child) and any(
                    os.path.exists(os.path.join(child, m))
                    for m in (".zgroup", ".zattrs", "zarr.json")):
                return "MSv4"
    except OSError:                            # pragma: no cover
        pass
    raise ValueError(
        "{0!r} is neither a casacore MSv2 table nor a zarr MSv4 processing "
        "set".format(path))


def msinfo(path: str, level: str = "full", outfile: Optional[str] = None,
           format: Optional[str] = None, engine: Optional[str] = None) -> MSInfo:
    """Collect structured metadata for the dataset at ``path``.

    Args:
        path: Path to an MSv2 table or an MSv4 processing set.
        level: Detail level -- ``"meta"`` (subtables only, no main-table scan),
            ``"full"`` (default; one pass over the index columns), or
            ``"data"`` (also read UVW and FLAG).
        outfile: If given, also write the JSON dump to this path.
        format: Force ``"MSv2"`` or ``"MSv4"`` instead of auto-detecting.
        engine: How to read the data.

            MSv2 defaults to ``"casacore"``: TaQL aggregation, no dependencies
            beyond the base install, and it will open a Measurement Set that
            stricter readers reject -- which matters, because a malformed MS
            is exactly the one you need to inspect.

            Pass ``"xarray-ms"`` to read an MSv2 through the MSv4 schema
            instead (needs the ``xarray-ms`` extra). MSv4 sets are read with
            plain ``xarray``/``zarr``; ``"xradio"`` forces xradio's opener,
            which also handles cloud stores and netcdf.

    Returns:
        An :class:`MSInfo`.
    """
    fmt = format or detect_format(path)
    if fmt == "MSv2" and engine in (None, "casacore"):
        from . import _msv2
        info = _msv2.read(path, level=level)
    elif fmt == "MSv2":
        # An MSv2 on disk, read through the MSv4 schema. `format` stays MSv2
        # because that is what is actually stored.
        from . import _msv4
        info = _msv4.read(path, level=level, engine=engine, format="MSv2")
    elif fmt == "MSv4":
        from . import _msv4
        info = _msv4.read(path, level=level, engine=engine)
    else:
        raise ValueError("unknown format {0!r}; expected 'MSv2' or 'MSv4'".format(fmt))

    if outfile:
        info.save(outfile)
    return info
