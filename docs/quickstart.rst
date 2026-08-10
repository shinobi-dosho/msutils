Quickstart
==========

Look inside an MS
-----------------

:func:`~msutils.msinfo` returns an :class:`~msutils.MSInfo`. Its collections
are addressable by name as well as by id, because real work says "the bandpass
calibrator", not "row 1":

.. code-block:: python

    import msutils

    info = msutils.msinfo("obs.ms")

    info.observation.telescope           # 'MEERKAT'
    info.fields["DEEP_2"].intents        # ['OBSERVE_TARGET#ON_SOURCE']
    info.fields["DEEP_2"].ra_hms         # '04h23m33.635'
    info.scans[7].duration               # 24.0  (includes the final integration)
    info.spws["SPW0"].centre_freq        # 1415000000.0
    info.data_descriptions[1].spw_id     # DDID -> SPW, explicitly
    info.columns["DATA"].shape           # [4, 4]

``render()`` gives a ``listobs``-style report, and ``to_dict()`` a stable,
versioned JSON structure:

.. code-block:: python

    print(info.render(verbose=True))
    info.save("obs.json")

On the command line:

.. code-block:: console

    $ msutils info obs.ms -v
    $ msutils info obs.ms --json-stdout | jq .fields[0]

Take a piece of it
------------------

.. code-block:: python

    msutils.subset("obs.ms", "target.ms", fields=["DEEP_2"], spws=[0])
    msutils.average("obs.ms", "avg.ms", time_bin=8.0, chan_bin=4)

    # select and average in one pass (hands off to average, so needs [average])
    msutils.subset("obs.ms", "target.ms", fields=["DEEP_2"], time_bin=8.0, chan_bin=4)

:func:`~msutils.subset` keeps the original field and SPW ids rather than
renumbering them the way CASA ``split`` does, so ids in a subset still match
the parent MS. Pass ``reindex=True`` for ``split``'s behaviour:

.. code-block:: python

    msutils.subset("obs.ms", "target.ms", fields=["DEEP_2"], reindex=True)
    msutils.msinfo("target.ms").fields.ids            # [0], not [2]

Reindexing drops the FIELD, SPECTRAL_WINDOW and DATA_DESCRIPTION rows the
output no longer uses and renumbers the survivors from 0, keeping their
original order; FEED and SOURCE follow the renumbered windows. ANTENNA,
POLARIZATION, STATE and OBSERVATION keep every row, so the ids referring to
them stay valid -- antennas in particular are never pruned, since
``antennas=`` keeps a baseline if either end is selected.

Columns
-------

.. code-block:: python

    msutils.addcol("obs.ms", "MODEL_DATA", clone="DATA")
    msutils.copycol("obs.ms", "DATA", "CORRECTED_DATA")
    msutils.sumcols("obs.ms", cols=["DATA", "MODEL_DATA"], outcol="CORRECTED_DATA")
    msutils.delcol("obs.ms", "CORRECTED_DATA")          # reclaim the disk

Flags
-----

.. code-block:: python

    stats = msutils.flagstats("obs.ms")
    stats.by_correlation["XY"].percent
    stats.by_channel[0]                                 # per-channel, per SPW

    msutils.flag_backup("obs.ms", name="pre-rfi")
    # ... flag ...
    msutils.flag_restore("obs.ms", "pre-rfi")

Diagnostics
-----------

.. code-block:: python

    msutils.du("obs.ms")                                # where the bytes went
    msutils.check("obs.ms")                             # MSv2 conformance
    msutils.taql("SELECT DISTINCT FIELD_ID FROM $1", "obs.ms")

.. warning::

   :func:`~msutils.taql` evaluates whatever you give it. TaQL can write and
   delete like SQL can — do not build a query by concatenating untrusted
   input. See ``SECURITY.md``.
