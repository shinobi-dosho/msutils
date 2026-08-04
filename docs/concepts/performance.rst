Reading big Measurement Sets
============================

Measurement Sets do not fit in memory, and the interesting ones have hundreds
of scans. Two design choices follow, and they are worth understanding if you
are calling msutils in a pipeline.

Aggregate in TaQL, not in Python
--------------------------------

The 2.x ``summary()`` ran one ``table.query()`` per field and another per
scan, so its cost grew as O(fields × scans) full table scans — 0.55 s for a
595k-row MS with 6 scans, and 1.53 s for the *same MS* with 60. The work was
not in casacore; it was in doing casacore's work from Python, repeatedly.

:func:`~msutils.msinfo` issues a single ``GROUPBY`` and folds the result:
0.37 s on that MS, and flat in the number of scans.

The same change drives :func:`~msutils.flagstats`, where one
``GNTRUES(FLAG)`` aggregation per axis returns a
``(group, channel, correlation)`` count array in one streaming pass — three
queries cover every axis, replacing an implementation that made a full pass
over the MS *per antenna*.

Levels: read only what you need
-------------------------------

Most questions about an MS do not require reading the MS. ``msinfo`` takes a
``level``:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``level``
     - Reads
     - Gives you
   * - ``"meta"``
     - subtables only
     - fields, SPWs, antennas, columns, max baseline
   * - ``"full"`` *(default)*
     - + one pass over the index columns
     - scans, time ranges, row counts
   * - ``"data"``
     - + UVW and FLAG
     - uv coverage, flag statistics

``"meta"`` is independent of MS size — about 10 ms whether the MS is 200 rows
or 200 million — because it never touches the main table. Use it when you want
field names, channel widths or antenna positions.

The split between ``"full"`` and ``"data"`` exists because **TaQL re-reads a
column for every expression that mentions it.** On a 595k-row MS the
index-column aggregate takes 0.17 s; adding a single ``SUMSQR(UVW)`` term
takes it to 0.64 s, and a second UVW term to 1.54 s. Bulk-column work
therefore lives at ``"data"`` and is not paid for by default.

Geometry beats data
-------------------

``MSInfo.max_baseline`` is computed from the ``ANTENNA`` table — the longest
physical separation among the antenna pairs actually present — not from the
per-row ``UVW`` column. It is exact, it is free, and it is available even at
``level="meta"``.

``max_uv_distance`` (the projected *uv* extent, which is what limits image
resolution) genuinely needs ``UVW``, so it is a ``"data"``-level value. The
2.x ``MAXBL`` conflated the two, reporting a uv distance under a
baseline-length name.
