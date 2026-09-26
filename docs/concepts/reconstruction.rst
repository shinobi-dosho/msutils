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

Exact-native bundle (schema v2)
-------------------------------

:func:`~msutils.capture_native_preservation` writes a new directory holding
exactly two entries:

.. code-block:: text

   <bundle>/
     manifest.json        descriptors, info, managers, keywords, references,
                          column records and the three logical IDs
     native.zarr/         Zarr v3: one group per table, one array per defined column
       MAIN/DATA/...
       ANTENNA/...        a group even when the table has no rows

Nothing else may sit inside the bundle; keep sidecars next to it. Each
defined column is one array of shape ``[rows, *cell_shape]`` in the column's
own casacore type:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - casacore ``valueType``
     - Payload ``data_type``
     - Note
   * - ``boolean``
     - ``bool``
     - never widened to bytes
   * - ``uchar``
     - ``uint8``
     - python-casacore reads it as ``uint16``; capture narrows it, refusing a
       value above 255 (``cell-dtype``)
   * - ``short``, ``int``, ``uint``, ``int64``
     - ``int16``, ``int32``, ``uint32``, ``int64``
     -
   * - ``float``, ``double``
     - ``float32``, ``float64``
     - NaN payloads and signed zeros kept bit for bit
   * - ``complex``, ``dcomplex``
     - ``complex64``, ``complex128``
     - written with ``write_empty_chunks`` so signed zeros survive
   * - ``string``
     - ``string`` (``vlen-utf8``)
     - variable length, including embedded NULs
   * - anything else
     - --
     - refused before any cell is read (``column-type``)

An undefined column has no array, only its manifest record; a table with no
rows is an empty group. Payload arrays carry no attributes or dimension
names -- the manifest is the only authority for metadata. ``block_rows`` is
now just the payload's chunking: ``None`` (the default) sizes chunks to about
64 MiB decoded (4096 rows for strings), and a chunk above 2 GiB decoded is
refused with ``chunk-size``. Capture reads cells in row ranges with
``getcol``, finds definedness with one TaQL ``ISDEFINED`` query per table,
checks shape uniformity with ``getcolshapestring``, and reads the staged
payload back and compares its digests with those streamed from casacore
before it publishes anything (``payload-readback``).

Binding by logical identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The manifest records three IDs, all ``msutils-logical-hash/v1`` (see
:doc:`logical_identity`):

``msv4.logical_id``
    :func:`~msutils.logical_id` of the MSv4 tree. It is a pairing binding
    only: the exact writer never reads MSv4 values.
``payload.logical_id``
    :func:`~msutils.logical_id` of ``native.zarr``, with one array digest per
    column recorded in the column's ``digest``.
``native_logical_id``
    The logical hash of the ``msutils-native-model/v1`` virtual tree of the
    native MS: one group per table whose attributes are its rows, descriptor,
    info, storage managers (without ``StandardStMan`` ``IndexLength``),
    keywords, references and column records, and one array per defined
    column. :func:`~msutils.native_logical_id` computes it from a live MSv2
    with only the base install, so it is equal for the source at capture, for
    the bundle, and for a faithfully materialised target.

Because the bindings are logical, **either tree may be rechunked,
recompressed, resharded or re-encoded losslessly** (any layout the reading
policy admits) and the pair stays valid. Changing one value, attribute,
keyword or structural detail changes an ID and refuses.

Planning and restoring recompute everything, cheapest checks first: the
bundle root and manifest; the version; the two permitted root entries; the
manifest structure; the payload's reading policy and exact structure
(``payload-structure``); every payload column digest (``payload-changed``)
and the payload and native IDs (``bundle-integrity``); and finally the MSv4
tree's ID (``zarr-changed``, or ``zarr-empty`` for a tree without arrays).
:func:`~msutils.verify_native_preservation` runs exactly these checks without
writing anything and returns the three IDs.

Verification then compares the target's cells with the digests *planned*
from the payload, not with a fresh read of it, so a payload modified between
planning and writing is refused (``verify-payload-changed``), and a differing
cell is named by table, column and row (``verify-cell``). The target's own
native logical ID must equal the bundle's (``verify-native-id``), a final
guard against the structural checks and the native model drifting apart.

New and renamed refusal codes: ``bundle-version``, ``bundle-extra-entry``,
``payload-structure``, ``payload-changed``, ``payload-readback``,
``bundle-integrity``, ``column-type``, ``chunk-size`` (was ``block-size``),
``verify-payload-changed``, ``verify-native-id``, and every ``zarr-*`` code
of the reading policy, prefixed ``msv4:`` or ``payload:`` in the reason.
``zarr-changed`` now means the MSv4 tree's *logical* content changed.

Cost
~~~~

Planning reads the whole payload once and the whole MSv4 tree once. The
writer reads the payload again; verification reads the target once and
re-reads the payload only to name a failure. Capture and
:func:`~msutils.native_logical_id` hold read locks while hashing every file
of the source MS twice (before and after, to detect concurrent writers) and
reading every defined cell once.

Version 1 bundles
~~~~~~~~~~~~~~~~~

Bundles written by pre-release ``main`` (schema
``msutils-native-preservation/v1``: per-block ``.npy`` files bound by a byte
ledger of the Zarr files) are refused with ``bundle-version``; recapture them
from the source MS. Schema v1 was never part of a release. The public
fidelity name, ``exact-native-v1``, and the profile,
``fixed-shape-defined-or-empty/v1``, are unchanged: what is guaranteed and
what is accepted did not change, only the storage format.

Both trees must be Zarr v3. :func:`~msutils.to_msv4` (xradio 1.2) and
xarray-ms exports write Zarr v3 on the supported stack; a Zarr v2 MSv4 tree is
refused with ``zarr-format`` and must be re-exported with ``zarr_format=3``
before a bundle can be bound to it.

Dependency boundary
-------------------

The ``exact-native`` extra currently pins dask-ms 0.2.32 as a temporary,
established writer backend. Capture and the mandatory independent read-back
verification use python-casacore, already a base dependency of ``msutils``;
the payload and the logical IDs of Zarr trees need zarr from the ``msv4``
extra (``zarr>=3.1``), which ``exact-native`` includes.
:func:`~msutils.native_logical_id` needs only the base install.
The reconstruction plan and preservation bundle remain backend-neutral so the
writer can move to xarray-ms once its MSv2 writing support is ready; exact-mode
semantics must not depend on dask-ms-specific objects.

The separate ``reconstruction`` extra pins arcae only for the Phase 0
capability probes above. Installing it does not select the public exact-native
writer.
