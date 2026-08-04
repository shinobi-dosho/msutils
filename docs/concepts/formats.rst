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
