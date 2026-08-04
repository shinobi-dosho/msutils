Installation
============

msutils requires Python 3.11 or newer. ``python-casacore`` is pulled in
automatically — its wheels bundle the casacore libraries, so there is nothing
to build.

.. code-block:: console

    $ pip install msutils

That base install covers :func:`~msutils.msinfo`, every column operation,
:func:`~msutils.subset`, flag versions, :func:`~msutils.flagstats` and the
diagnostics — everything that runs on casacore and TaQL.

Optional extras
---------------

Heavier stacks are opt-in, and each is imported only by the function that
needs it:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Brings in
     - Enables
   * - ``plots``
     - matplotlib
     - PNG summary plots for :func:`~msutils.flagstats`
   * - ``average``
     - codex-africanus
     - :func:`~msutils.average`
   * - ``msv4``
     - xarray, zarr
     - reading MSv4 processing sets
   * - ``xarray-ms``
     - xarray-ms
     - reading an MSv2 through the MSv4 schema
   * - ``convert``
     - xradio
     - writing MSv2 → MSv4

.. code-block:: console

    $ pip install "msutils[plots]"
    $ pip install "msutils[all]"        # everything

Note that *reading* MSv4 needs only ``msv4`` (xarray + zarr); xradio is
required just to *write* a processing set.

From source
-----------

.. code-block:: console

    $ git clone https://github.com/SpheMakh/msutils.git
    $ cd msutils
    $ uv sync --all-extras --group dev
    $ uv run pytest

See :doc:`contributing` for the full development setup.
