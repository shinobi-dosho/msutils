"""Convert MSv2 Measurement Sets to MSv4 processing sets.

A thin wrapper over :func:`xradio.measurement_set.convert_msv2_to_processing_set`
-- xradio owns the conversion, this owns the ergonomics: argument validation,
overwrite handling, and returning an :class:`~msutils.info.MSInfo` for the
result so the output can be inspected with the same code as the input.

Needs the ``convert`` extra (``pip install 'msutils[convert]'``). Note that
*reading* an MSv4 set needs only ``msutils[msv4]`` (xarray + zarr); xradio is
required for writing one.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from typing import Any

from ._log import create_logger
from .info import MSInfo, msinfo

__all__ = ["PARTITION_KEYS", "to_msv4"]

LOGGER = create_logger(__name__)

#: Extra keys ``partition_scheme`` accepts. MSv4 always partitions by spectral
#: window, polarization setup and observation mode; these subdivide further.
PARTITION_KEYS = ("FIELD_ID", "SCAN_NUMBER", "STATE_ID", "ANTENNA1")


def to_msv4(
    msname: str,
    outpath: str,
    partition_scheme: Sequence[str] | None = None,
    overwrite: bool = False,
    with_pointing: bool = True,
    storage_backend: str = "zarr",
    **kwargs: Any,
) -> MSInfo:
    """Convert the MSv2 at ``msname`` to an MSv4 processing set at ``outpath``.

    Args:
        msname: Source MSv2.
        outpath: Destination processing set (a zarr directory).
        partition_scheme: Extra columns to partition on, from
            :data:`PARTITION_KEYS`. MSv4 already splits by spectral window,
            polarization setup and observation mode; adding ``FIELD_ID`` or
            ``SCAN_NUMBER`` splits more finely still.
        overwrite: Replace ``outpath`` if it exists.
        with_pointing: Carry the POINTING subtable across. Turning this off
            is worthwhile when POINTING is large and unused.
        storage_backend: ``"zarr"`` (default) or ``"netcdf"``.
        kwargs: Passed through to xradio unchanged.

    Returns:
        An :class:`~msutils.info.MSInfo` describing the converted set.
    """
    try:
        from xradio.measurement_set import convert_msv2_to_processing_set
    except ImportError as exc:  # pragma: no cover - extra absent
        raise ImportError(
            "MSv4 conversion needs xradio. Install with: pip install 'msutils[convert]'"
        ) from exc

    scheme: list[str] = list(partition_scheme or [])
    unknown = [key for key in scheme if key not in PARTITION_KEYS]
    if unknown:
        raise ValueError(
            f"unknown partition key(s) {unknown}; expected any of {list(PARTITION_KEYS)}"
        )

    for existing in (outpath, outpath + _XRADIO_SUFFIX):
        if not os.path.exists(existing):
            continue
        if not overwrite:
            raise FileExistsError(f"{existing} already exists; pass overwrite=True to replace it")
        shutil.rmtree(existing)

    LOGGER.info(
        "Converting %s -> %s (MSv4, partition_scheme=%s)", msname, outpath, scheme or "default"
    )
    convert_msv2_to_processing_set(
        in_file=str(msname),
        out_file=str(outpath),
        partition_scheme=scheme,
        with_pointing=with_pointing,
        storage_backend=storage_backend,
        **kwargs,
    )

    _honour_requested_path(outpath)
    info = msinfo(outpath, level="full", format="MSv4")
    LOGGER.info(
        "Wrote %d MSv4 partition(s) covering %d field(s) and %d SPW(s)",
        len(info.subtables),
        info.nfields,
        info.nspws,
    )
    return info


#: Suffix xradio appends to ``out_file`` when it does not already end in it.
_XRADIO_SUFFIX = ".ps.zarr"


def _honour_requested_path(outpath: str) -> None:
    """Move xradio's output to the path the caller actually asked for.

    xradio appends ``.ps.zarr`` unless ``out_file`` already ends that way, so
    ``convert(..., "obs.zarr")`` silently writes ``obs.zarr.ps.zarr``. A
    processing set is a plain directory and opens under any name, so it is
    moved back rather than surprising the caller with a path they did not
    choose.
    """
    if os.path.exists(outpath):
        return
    written = outpath + _XRADIO_SUFFIX
    if os.path.exists(written):
        LOGGER.debug("xradio wrote %s; moving to the requested %s", written, outpath)
        shutil.move(written, outpath)
