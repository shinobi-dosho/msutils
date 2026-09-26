MSv2, MSv4, and reader engines
==============================

:func:`~msutils.msinfo` reads two on-disk formats and returns the same
:class:`~msutils.MSInfo` for both, so code written against one works against
the other.

.. code-block:: python

    msutils.msinfo("obs.ms")                          # MSv2, via casacore/TaQL
    msutils.msinfo("obs.zarr")                        # MSv4, via xarray + zarr
    msutils.msinfo("obs.ms", engine="xarray-ms")      # MSv2 through the MSv4 schema

The two formats are shaped very differently. An MSv2 is a casacore table: rows,
each carrying one visibility spectrum, with subtables hanging off it. An MSv4
*processing set* is a collection of xarray datasets, one per partition, already
split so that a partition holds a single spectral window, polarization setup
and field — there are no rows at all, just an n-dimensional
``(time, baseline, frequency, polarization)`` array.

Folding the second back into the shared model means two conversions worth
knowing about:

**Time.** The MSv4 schema stores unix epoch seconds; MSv2 and this model use
MJD seconds. The reader converts on the way in, so ``start_utc`` agrees
whichever format the data came from.

**Rows.** ``MSInfo.nrows`` for an MSv4 is the equivalent MSv2 row count —
``time × baseline`` summed over partitions. An MSv4 has no rows of its own.

Engines
-------

``_msv4.py`` reads the MSv4 *schema*, not one particular library's output,
because three different sources produce the same tree:

``zarr``
    A stored processing set, opened with plain xarray. This is why *reading*
    MSv4 needs only ``msutils[msv4]`` — xradio is required to *write* a
    processing set, not to read one.

``xradio``
    xradio's own ``open_processing_set``. Handles cloud stores and netcdf, so
    it is the fallback when plain zarr cannot open the path.

``xarray-ms``
    An MSv2 on disk, presented through the MSv4 schema with no conversion
    step. Useful when you want the n-dimensional view of an MS you already
    have.

Why casacore stays the MSv2 default
-----------------------------------

``engine="casacore"`` — TaQL aggregation — is the default for MSv2 for three
reasons, in increasing order of importance:

1. It needs nothing beyond the base install.
2. It is several times faster for metadata: 0.37 s against 1.01 s on a
   595k-row MS.
3. **It opens Measurement Sets that stricter readers reject.** xarray-ms
   refuses an MS whose ``FEED`` subtable does not cover the antennas in use,
   which is a reasonable stance for a data reader and a bad one for a
   diagnostic tool. The malformed MS is exactly the one you need to inspect,
   and :func:`~msutils.check` exists to tell you what is wrong with it.

Converting
----------

.. code-block:: console

    $ msutils convert obs.ms obs.zarr

:func:`msutils.convert.to_msv4` wraps xradio's conversion (the ``convert``
extra), validates its arguments, refuses to clobber an existing output, and
returns an :class:`~msutils.MSInfo` for the result so the output can be
inspected with the same code as the input.

Materialising MSv2
-------------------

``to_msv2(msv4, outpath)`` uses the mapped MSv4 schema. It retains the
existing convention: MSv2 ``WEIGHT`` is the per-channel mean, and ``SIGMA``
is derived from that mean. It cannot restore native rows, subtables, column
descriptors or keywords that an MSv4 export omitted.

For exact native restoration, capture a preservation bundle while the source
MSv2 still exists, then keep it with the exported Zarr state::

    from msutils import (capture_native_preservation, native_logical_id, to_msv2,
                         verify_native_preservation)

    capture_native_preservation("source.ms", "state.zarr", "state.native")
    to_msv2("state.zarr", "restored.ms", fidelity="exact-native-v1",
             preservation="state.native")

    # Re-validate the restored MS against the state it came from.
    ids = verify_native_preservation("state.zarr", "state.native")
    assert native_logical_id("restored.ms") == ids.native_logical_id

This opt-in path uses ``msutils[exact-native]`` (dask-ms 0.2.32). Its versioned
bundle (schema ``msutils-native-preservation/v2``) contains typed descriptors,
managers, keywords, subtable relationships and row counts in a manifest, and
the fixed-shape native cells as a Zarr v3 payload in their exact casacore
types. In this first profile the native cells are preserved in full, so the
bundle can be large. The exact writer restores those bundle payloads; it does
not read the MSv4 visibility arrays into the target. It binds the Zarr tree by
its logical ID (:func:`msutils.logical_id`), so lossless rechunking or
recompression of either tree keeps the bundle valid, but this API does not
prove that the tree was exported from the captured MSv2. Callers must
establish that provenance during export. The bundle and Zarr together are the
reusable state; a Zarr hierarchy alone is insufficient for exact native
restoration. :func:`msutils.native_logical_id` gives the native MS's own
identity, which the bundle records and a restored MS reproduces; see
:doc:`reconstruction` and :doc:`logical_identity`.

Capture and verification read every defined cell, in row ranges through
``getcol``, and capture also inventories and hashes the source tree before
and after reading it. Both runtime and bundle size therefore scale with the
complete native MS, not just its metadata; plan capacity and storage before
using this mode on a large MS.

The exact mode accepts fixed-shape columns whose cells are all defined or all
undefined. It refuses mixed definedness, ragged cells, ambiguous empty typed
keyword arrays, external table references and untested storage managers with
a structured ``NativePreservationRefusal`` identifying the table, column and
row where possible. The baseline experiment's ``CUSTOM_VARIABLE`` therefore
still refuses. References must belong to the subtables named by the MAIN
table, and table, column and keyword names must be identifiers (letters,
digits and underscores, starting with a letter or underscore). Typed metadata
arrays are limited to 64 MiB and eight dimensions. This profile does not
discover nested referenced tables. A future
native-cell adapter can widen that boundary.

Reconstruction writes to a private sibling, reopens every table with
python-casacore, checks native descriptors, keywords, managers, definedness
and cell bytes, and publishes a new destination with an atomic no-replace
rename. Any failure removes the private candidate. Exact mode refuses an
existing output even if ``overwrite=True`` was supplied; mapped mode keeps
its existing overwrite behavior. The mapped ``weight_spectrum`` and
``rowchunk`` options are not accepted in exact mode. Atomic no-replace
publication currently requires Linux ``renameat2`` with
``RENAME_NOREPLACE`` support; unsupported systems refuse before publishing.

Manager comparison preserves and checks the raw ``getdminfo()`` record except
for ``StandardStMan.SPEC.IndexLength``. Casacore exposes that field as the
serialized byte length of its bucket index, but recomputes it from physical
index layout and write history instead of accepting it as reconstruction
configuration. It can therefore differ between logically identical tables.
Exact-native identity excludes only that derived field; storage-manager type,
name, column bindings, bucket size, cache configuration and every other
manager specification remain part of the equality gate. The unnormalised
``IndexLength`` remains in the preservation bundle as physical-layout
evidence. Byte-for-byte table-file fidelity would require preserving the
original casacore files and is outside this data-only profile.
