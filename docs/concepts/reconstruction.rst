MSv2 reconstruction
====================

``msutils`` exposes mapped MSv4-to-MSv2 materialisation and an opt-in,
bounded exact-native profile. The earlier executable probes against arcae
0.5.4 remain capability evidence for the longer-term writer. They establish
which native table operations can support the reconstruction design in
`issue #47 <https://github.com/shinobi-dosho/msutils/issues/47>`_, and which
cases must still refuse.

The intended capability is deliberately narrower than a generic format
conversion:

.. code-block:: text

   native MSv2
       -> normalized MSv4/Zarr + a versioned native-preservation manifest
       -> fresh native MSv2

Only an unmodified bundle produced by the matching exporter is in the first
profile.  The preservation manifest is required because normalized MSv4 does
not retain all native row identities, columns, cell states, descriptors and
subtable relationships.

Pinned probe environment
------------------------

The Phase 0 tests are in ``tests/test_arcae_reconstruction_probes.py`` and run
when the ``reconstruction`` extra is installed:

.. code-block:: console

   $ uv sync --extra reconstruction --group dev
   $ uv run --extra reconstruction pytest -q \
       tests/test_arcae_reconstruction_probes.py -rxX

The compatibility baseline is:

* arcae 0.5.4 (``ff117d4ae4816914a475e2d9997f4502190cecd9``);
* Python 3.11–3.13, matching the project test matrix;
* local directory-backed Measurement Sets.

Capability matrix
-----------------

``supported`` means the pinned executable probe demonstrates the primitive;
it does not by itself broaden the public exact-native profile. ``gated`` means
that profile must refuse the case until a later probe and implementation
demonstrate it.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Capability
     - Status
     - Evidence or boundary
   * - Fresh MAIN creation
     - supported
     - ``Table.ms_from_descriptor`` creates MAIN and the required default
       subtables.
   * - Optional subtable creation
     - supported
     - A WEATHER table can be created, linked and reopened through ``::``.
   * - Bounded row allocation
     - supported
     - Repeated positive ``addrows`` calls preserve the requested row count.
   * - One-call allocation above a C ``int``
     - gated
     - The pinned Cython binding converts the count to ``int``; reconstruction
       must split larger counts into bounded calls.
   * - Indexed scalar and fixed-shape writes
     - supported
     - Non-contiguous row selections round-trip without changing destination
       row order.
   * - Boolean, string, complex and special floats
     - supported
     - Value and descriptor probes cover flags, scalar strings, complex64
       data, signed zero, NaN and infinities. Arcae 0.5.4 exposes a native
       BOOLEAN column to NumPy as ``uint8`` rather than ``bool``.
   * - Explicit descriptors and data managers
     - supported
     - Fixed-shape tiled columns reopen with their declared descriptors and
       manager group.
   * - Bounded storage-manager cache request
     - supported
     - Fresh creation accepts the pinned ``cache_size`` API while performing
       fixed-shape writes.
   * - Detecting undefined variable-shape cells
     - supported for refusal
     - ``row_shapes`` reports the undefined cell as null, so the first profile
       can reject it without reading fabricated data.
   * - Reconstructing undefined or ragged cells
     - gated
     - ``putcol`` accepts rectangular NumPy arrays; there is no general
       null-aware/ragged writer in the pinned API.
   * - Zero-length array cells
     - gated
     - Kept as a strict expected-failure probe until both creation and reopen
       preserve the cell shape.
   * - Empty typed keyword arrays
     - gated
     - Descriptor JSON loses the element type of an empty array.  The probe
       requires independent native type verification before support can be
       claimed.
   * - Relocating a staged MS with linked optional subtables
     - supported
     - After a parent-directory rename, the optional-subtable keyword resolves
       to the published path and the table reopens through ``::``.
   * - Arbitrary source data-manager records
     - gated
     - The first profile must use an allowlisted output policy; virtual and
       plugin-specific managers may depend on unavailable source state.

The strict expected-failure probes are intentional capability sentinels.  An
unexpected pass fails the suite so that the matrix and reconstruction boundary
must be reviewed rather than silently becoming broader.

Dependency boundary
-------------------

The ``exact-native`` extra currently pins dask-ms 0.2.32 as a temporary,
established writer backend. Capture and the mandatory independent read-back
verification use python-casacore, already a base dependency of ``msutils``.
The reconstruction plan and preservation bundle remain backend-neutral so the
writer can move to xarray-ms once its MSv2 writing support is ready; exact-mode
semantics must not depend on dask-ms-specific objects.

The separate ``reconstruction`` extra pins arcae only for the Phase 0
capability probes above. Installing it does not select the public exact-native
writer.
