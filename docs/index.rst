msutils
=======

*Everyday Measurement Set operations.*

**msutils** is a library and CLI for the routine things you do to a
Measurement Set: look inside it, take a piece of it, average it, add and drop
columns, and keep track of flags. It does not calibrate or image — those
belong in the tools built for them.

The entry point is :func:`~msutils.msinfo`, which returns a typed, addressable
description of an MS:

.. code-block:: python

    import msutils

    info = msutils.msinfo("obs.ms")
    print(info.render())

    info.fields["PKS1934-638"].scan_numbers   # [1, 2]
    info.spws[0].chan_width[0]                # 10000000.0
    info.antennas["m003"].latitude            # -30.71053

.. code-block:: console

    $ msutils info obs.ms
    $ msutils subset obs.ms target.ms --field DEEP_2
    $ msutils flagstats obs.ms --plot flags.png

The same call reads MSv2 tables and MSv4 processing sets, returning the same
:class:`~msutils.MSInfo` either way — see :doc:`concepts/formats`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/formats
   concepts/performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
