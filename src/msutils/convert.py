"""Convert MSv2 to MSv4 and materialise a bounded MSv2 representation.

A thin wrapper over :func:`xradio.measurement_set.convert_msv2_to_processing_set`
-- xradio owns the conversion, this owns the ergonomics: argument validation,
overwrite handling, and returning an :class:`~msutils.info.MSInfo` for the
result so the output can be inspected with the same code as the input.

The mapped writer uses MSv4 arrays and derives MSv2 row weights. The opt-in
exact-native writer restores a data-only native preservation bundle (schema
``msutils-native-preservation/v2``: a manifest plus a Zarr v3 payload) that
is bound to the MSv4 tree by its :func:`~msutils.logical_id`, so a lossless
rechunk or recompression of either tree keeps the pair valid. It does not
consume MSv4 array values. Writing MSv4 needs ``msutils[convert]``.
Exact-native restoration needs ``msutils[exact-native]``;
:func:`native_logical_id` needs only the base install.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from typing import Any

from .info import MSInfo, msinfo

__all__ = [
    "PARTITION_KEYS",
    "NativePreservationIds",
    "NativePreservationRefusal",
    "capture_native_preservation",
    "native_logical_id",
    "to_msv2",
    "to_msv4",
    "verify_native_preservation",
]

from ._native_preservation import NativePreservationIds, NativePreservationRefusal

LOGGER = logging.getLogger(__name__)

#: Extra keys ``partition_scheme`` accepts. MSv4 always partitions by spectral
#: window, polarization setup and observation mode; these subdivide further.
PARTITION_KEYS = ("FIELD_ID", "SCAN_NUMBER", "STATE_ID", "ANTENNA1")


def to_msv2(
    msv4: str,
    outpath: str,
    *,
    overwrite: bool = False,
    weight_spectrum: bool | None = None,
    rowchunk: int | None = None,
    fidelity: str = "mapped",
    preservation: str | None = None,
) -> MSInfo:
    """Materialise a mapped MSv2 or restore a native preservation bundle.

    This supports correlated-interferometer processing sets.  MSv4 content
    with no MSv2 equivalent is rejected rather than being silently omitted.
    It needs the ``msv4`` extra (plain xarray + zarr), not xradio or
    xarray-ms.

    Args:
        msv4: Source MSv4 Zarr processing set.
        outpath: New MSv2 table to create.
        overwrite: Replace a pre-existing destination.
        weight_spectrum: In mapped mode, write MSv4's per-channel weights as
            ``WEIGHT_SPECTRUM`` (default true). MSv2 ``WEIGHT`` is always the
            channel mean. Not accepted in exact mode, even when explicitly
            set to its mapped default.
        rowchunk: In mapped mode, time samples loaded from each partition at
            a time (default 64). Not accepted in exact mode, even when
            explicitly set to its mapped default.
        fidelity: ``"mapped"`` retains the MSv4 array conversion, including
            its derived channel-mean ``WEIGHT`` and ``SIGMA``.  Use
            ``"exact-native-v1"`` with a native preservation bundle to
            restore exact MSv2 rows and metadata from that bundle. The exact
            writer requires the Zarr tree's logical ID to equal the one the
            bundle recorded (any lossless re-layout of either tree is
            accepted) but does not consume its array values. The fidelity
            name describes the guarantee, which is unchanged; the bundle
            format is ``msutils-native-preservation/v2``, and bundles written
            before that are refused with ``bundle-version``.
        preservation: Bundle from :func:`capture_native_preservation`.
    """
    if fidelity == "exact-native-v1":
        if preservation is None:
            raise ValueError("exact-native-v1 requires preservation")
        if overwrite:
            raise ValueError("exact-native-v1 only publishes a fresh destination")
        if weight_spectrum is not None or rowchunk is not None:
            raise ValueError("weight_spectrum and rowchunk options apply only to mapped fidelity")
        from ._native_preservation import materialize_exact_native

        return materialize_exact_native(msv4, outpath, preservation)
    if fidelity != "mapped":
        raise ValueError(f"unknown fidelity mode {fidelity!r}")
    if preservation is not None:
        raise ValueError("preservation requires fidelity='exact-native-v1'")
    weight_spectrum = True if weight_spectrum is None else weight_spectrum
    rowchunk = 64 if rowchunk is None else rowchunk
    from ._msv4convert import to_msv2 as materialise

    return materialise(
        msv4,
        outpath,
        overwrite=overwrite,
        weight_spectrum=weight_spectrum,
        rowchunk=rowchunk,
    )


def capture_native_preservation(
    source_ms: str, msv4: str, bundle: str, *, block_rows: int | None = None
):
    """Capture a versioned, data-only exact-native bundle for one MSv4 state.

    Fixed-shape, fully-defined or wholly-undefined columns are supported;
    mixed undefinedness and ragged cells are refused before publication.

    The bundle is a new directory holding ``manifest.json`` and
    ``native.zarr``, a Zarr v3 hierarchy with one group per table and one
    array per defined column in the column's exact casacore type. It records
    three logical IDs -- the MSv4 tree's, the payload's and the native MS's
    (:func:`native_logical_id`) -- and is read back and re-hashed before it
    is published. Needs ``msutils[msv4]`` (zarr).

    Args:
        source_ms: The native MSv2 the MSv4 tree was exported from.
        msv4: The MSv4 Zarr v3 tree to bind the bundle to.
        bundle: New directory to create (refused if it exists).
        block_rows: Rows per payload chunk. ``None`` (default) sizes chunks
            to about 64 MiB decoded (4096 rows for strings). It is a storage
            choice only: it does not change any logical ID, and a chunk above
            2 GiB decoded is refused with ``chunk-size``.
    """
    from ._native_preservation import capture_native_preservation as capture

    return capture(source_ms, msv4, bundle, block_rows=block_rows)


def native_logical_id(ms: str) -> str:
    """Native logical ID of a live MSv2 under the exact-native profile.

    Equal to the ``native_logical_id`` of a bundle captured from ``ms`` and to
    that of an MS faithfully restored from such a bundle, so it re-validates
    a materialised MS against the state it came from. Refuses content outside
    the profile exactly as :func:`capture_native_preservation` does. Needs
    only the base install; holds read locks while the MS's files are hashed
    twice (to detect concurrent writers) and every defined cell is read once.
    """
    from ._native_preservation import native_logical_id as native

    return native(ms)


def verify_native_preservation(msv4: str, bundle: str) -> NativePreservationIds:
    """Check a preservation bundle against its MSv4 tree without writing anything.

    Performs every check exact-native restoration performs before writing:
    bundle version and structure, the payload's reading policy and layout,
    every column digest, and the recomputed payload, native and MSv4 logical
    IDs. Returns those IDs; refuses with :class:`NativePreservationRefusal`.
    Costs one full read of the payload and one of the MSv4 tree. Needs
    ``msutils[msv4]`` (zarr).
    """
    from ._native_preservation import verify_native_preservation as verify

    return verify(msv4, bundle)


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
