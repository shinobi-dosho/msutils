API reference
=============

Generated from the source docstrings. See :doc:`../concepts/formats` and
:doc:`../concepts/performance` for the design behind these modules.

Top-level package
-----------------

Everything below is re-exported on the package root, so
``msutils.msinfo(...)`` and ``msutils.info.msinfo(...)`` are the same
function. The sections that follow document each definition once, at the
module that defines it.

.. automodule:: msutils

Metadata
--------

.. automodule:: msutils.info
   :members: msinfo, detect_format

.. automodule:: msutils.info._model
   :members: MSInfo, Observation, Field, Scan, SpectralWindow, Polarization,
             DataDescription, Antenna, Column, Registry

Readers
~~~~~~~

.. automodule:: msutils.info._msv2
   :members: read, LEVELS

.. automodule:: msutils.info._msv4
   :members: read, ENGINES

Column operations
-----------------

.. automodule:: msutils._ms
   :members: addcol, delcol, renamecol, copycol, sumcols, addnoise,
             compute_vis_noise, verify_antpos

Subsetting and averaging
------------------------

.. automodule:: msutils.subset
   :members: subset, average

Flags
-----

.. automodule:: msutils.flagstats
   :members: flagstats, FlagStats, FlagCount, save_statistics, plot_statistics

.. automodule:: msutils.flags
   :members: flag_backup, flag_restore, flag_versions, flag_delete, FlagVersion

Diagnostics
-----------

.. automodule:: msutils.diagnostics
   :members: du, check, taql, DiskUsage, StorageGroup, CheckReport

Format conversion
-----------------

.. automodule:: msutils.convert
   :members: to_msv4, PARTITION_KEYS

Deprecated
----------

.. automodule:: msutils._compat
   :members: summary
