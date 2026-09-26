Logical identity of Zarr trees
==============================

:func:`msutils.logical_id` computes ``msutils-logical-hash/v1``, an identity
for the *logical content* of a Zarr v3 hierarchy that does not depend on how
that content is stored. It binds an exact-native preservation bundle to its
MSv4 tree (see :doc:`reconstruction`), and the same algorithm, applied to a
virtual tree, gives :func:`msutils.native_logical_id` for a native MSv2.

.. code-block:: python

    import msutils

    msutils.logical_id("state.zarr")
    # 'msutils-logical-hash/v1:4a23766e...'

This page is the normative specification. An independent implementation
that follows it reproduces the golden vectors below; ``tests/test_logical.py``
freezes them so that any drift fails the build.

Scope and meaning
-----------------

The ID identifies the **raw logical content of a Zarr v3 hierarchy**:

* the tree of node names;
* each node's kind (group or array);
* each node's ``attributes``;
* for arrays, the logical dtype, the shape, the dimension names, and every
  element value after codec decoding, *before* any CF or xarray decoding.

The ID is **independent of**:

* the chunk grid, sharding, codecs and their configuration, byte order and
  ``transpose``;
* the chunk key encoding;
* which chunks are physically present (a missing chunk reads as
  ``fill_value``);
* ``fill_value`` itself;
* consolidated metadata;
* the physical representation of strings (``string``, or
  ``fixed_length_utf32`` of any width);
* the zarr-python version.

It is **not** an identity over xarray's decoded view:

1. Decoding is reader policy, and it depends on the xarray version (masking,
   scaling, time units).
2. The raw content is where exactness lives.
3. A change that looks cosmetic at the xarray level can alter CF encoding
   attributes. An xarray round trip with ``encoding`` cleared drops the
   self-referential ``coordinates`` attribute from coordinate variables: a
   real change to the tree's contents, and the ID changing is the correct
   response. A re-encode that keeps the attributes, such as an xarray rewrite
   that drops only the chunk and codec encoding keys, keeps the ID.

``fill_value`` is excluded because, for ``zarr_format`` 3, xarray does not use
it as a mask (``use_zarr_fill_value_as_mask`` defaults to false for v3) and
records ``_FillValue`` as an ordinary attribute, which *is* hashed. For Zarr
v2 the default is the opposite, which would make ``fill_value`` logical
content. That is one reason v2 trees are refused.

Nothing is excluded from attributes. Two exports of the same MS that record
different ``creation_date`` attributes have different IDs; deduplicating
across exports is not what this ID is for.

Primitives
----------

* ``H`` is SHA-256.
* ``u64(n)`` is the 8-byte little-endian unsigned integer.
* ``DOMAIN = b"msutils-logical-hash/v1\x00"``.
* ``frame(tag)`` is a SHA-256 state initialised with
  ``DOMAIN || tag || b"\x00"``, where ``tag`` is ASCII ``block``, ``array`` or
  ``group``.
* ``BLOCK_ELEMENTS = 262144`` (2\ :sup:`18`), fixed by the algorithm version
  (:data:`msutils.logical.BLOCK_ELEMENTS`).

Canonical value encoding
------------------------

Attributes and headers are encoded with a small type-length-value scheme
rather than canonical JSON, because float formatting and string escaping are
hard to reproduce exactly across implementations.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Encoding
   * - null
     - ``N``
   * - false / true
     - ``F`` / ``T``
   * - integer
     - ``I`` ‖ u64(len) ‖ ASCII decimal: no leading zeros, ``-`` for
       negatives, ``0`` for zero, arbitrary precision
   * - float
     - ``D`` ‖ 8 bytes of IEEE-754 binary64, little-endian
   * - string
     - ``S`` ‖ u64(len) ‖ UTF-8. Strict: a lone surrogate is refused
       (``zarr-metadata``).
   * - list
     - ``L`` ‖ u64(count) ‖ the items in order
   * - object
     - ``O`` ‖ u64(count) ‖ one ``u64(len(key_utf8)) ‖ key_utf8 ‖ value`` per
       key, in ascending order of the keys' UTF-8 bytes

JSON parsing rules for metadata:

* A number is an integer if and only if its lexical form has no ``.``, ``e``
  or ``E``.
* Any other number becomes binary64 with round-to-nearest; an overflow
  becomes ±inf.
* The non-standard tokens ``NaN``, ``Infinity`` and ``-Infinity`` are
  accepted; ``NaN`` has the bits ``0x7ff8000000000000``.
* Duplicate object keys are refused (``zarr-metadata``).
* Floats keep their sign and payload bits, so ``-0.0`` differs from ``0.0``,
  and ``1`` differs from ``1.0``. Object key order does not matter.

Array digest
------------

A Zarr ``data_type`` maps to a logical tag:

* ``bool``, ``int8``, ``int16``, ``int32``, ``int64``, ``uint8``, ``uint16``,
  ``uint32``, ``uint64``, ``float16``, ``float32``, ``float64``,
  ``complex64`` and ``complex128`` map to the same name;
* ``string`` and ``{"name": "fixed_length_utf32", ...}`` map to ``string``;
* anything else is refused (``zarr-dtype``): raw bits ``r*``, fixed- or
  variable-length bytes, datetimes and timedeltas, structured types, and
  v2's object dtype.

Elements are encoded over the C-order flattened logical array:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tag
     - Bytes per element
   * - bool
     - one byte, ``0x00`` or ``0x01`` (any nonzero stored byte gives ``0x01``)
   * - int*, uint*
     - two's complement or unsigned at the tag's width, little-endian
   * - float16/32/64
     - IEEE-754 bits, little-endian, **unchanged**: NaN payloads, signalling
       NaNs and signed zero are preserved
   * - complex64/128
     - real part then imaginary part, each as the matching float
   * - string
     - ``u64(len(utf8)) ‖ UTF-8``, strict. For ``fixed_length_utf32`` the
       element is numpy's ``str`` value, so trailing NULs are stripped.

The digest:

.. code-block:: text

   size     = product(shape)              (1 for rank 0)
   nblocks  = ceil(size / BLOCK_ELEMENTS)  (0 when size == 0)
   block_k  = frame("block") ‖ u64(k) ‖ u64(count_k) ‖ elements[k*B : k*B + count_k]
   dims     = null if dimension_names is absent or every entry is null,
              else a list of (string | null)
   header   = TLV({"attributes": attributes or {}, "block_elements": 262144,
                   "dims": dims, "dtype": tag, "shape": [extents...]})
   array    = frame("array") ‖ u64(len(header)) ‖ header ‖ u64(nblocks)
              ‖ block_0 ‖ ... ‖ block_{n-1}

Each ``block_k`` contributes its 32-byte digest. Blocks are cut at fixed
element offsets, never at chunk boundaries, which is what makes the ID
chunk-independent; they are also what lets the digests be computed in
parallel.

Group digest and the ID
-----------------------

.. code-block:: text

   header = TLV(attributes or {})
   group  = frame("group") ‖ u64(len(header)) ‖ header ‖ u64(nchildren)
            ‖ for each child in ascending UTF-8 byte order of its name:
                u64(len(name_utf8)) ‖ name_utf8 ‖ (b"A" | b"G") ‖ child_digest
   ID     = "msutils-logical-hash/v1:" + lowercase hex(group digest of the root)

The root must be a group. An empty group is valid content.

Golden vectors
--------------

.. list-table::
   :header-rows: 1
   :widths: 6 44 50

   * - Vector
     - Tree
     - ID hex
   * - V1
     - empty root group
     - ``b31fb78faf8af3773d2ad097eb301b9a988fe2928be42e36d28f40165e5a5fce``
   * - V2
     - root attributes ``{"a": 1, "b": [1.0, "x", null, true]}``
     - ``29fff581653c3accdc0ef72da73a75700e97efb999fb51d4cd09833c9b2c60a4``
   * - V3
     - ``/x`` int32 ``[1, -2, 3]``, dims ``["row"]``
     - ``df60e0a83ad2fbfc85f8c242592ccb8833e74f459b4e9757deec81419fc765fe``
   * - V4
     - ``/f`` float64 ``[0.0, -0.0, NaN(0x7ff8000000000000), +inf]``,
       attributes ``{"units": "Jy"}``
     - ``ff8b5491b187e184152d092bb3f26515a2b3cf8eebf8b6394ed43818c798b82f``
   * - V5
     - ``/s`` string ``["", "é𝄞", "a\x00"]``
     - ``635957fa858a02f486125e219690e19cd8a7b50fe0f54f3554a862f20301dec5``
   * - V6
     - ``/b`` bool ``[[T, F], [F, T]]``, ``/c`` complex64 scalar
       ``(-0.0+1.5j)``
     - ``238bf1f19702e9267c9f173d3f94855a1e545a18a24ee7cba2579df6cca9dc3b``
   * - V7
     - ``/g`` (attributes ``{"k": "v"}``) / ``y`` uint8
       ``arange(262145) % 251``, dims ``["n"]`` (two blocks)
     - ``e420de3773b19f3049f3e2ddc0ebfba94cbf927c063f5a8bb2a1a38ae1e8cc68``

Intermediate values for V3:

.. code-block:: text

   header = 4f05000000000000000a00000000000000617474726962757465734f00000000000000000e0000000000000062
            6c6f636b5f656c656d656e7473490600000000000000323632313434040000000000000064696d734c010000
            0000000000530300000000000000726f7705000000000000006474797065530500000000000000696e7433320500
            00000000000073686170654c010000000000000049010000000000000033
   block0 = d40268dd983687025cf15a17c14defeb674f18d9862fed430d61ea9fed6a98fc
   array  = 9c418c5a4e38ae1879a8a700dac041461e4be03414be91ded0eaa9bb25af3bfa
   root   = df60e0a83ad2fbfc85f8c242592ccb8833e74f459b4e9757deec81419fc765fe

Reading policy
--------------

The policy below is **not part of the digest**: it decides which trees are
read at all, and fails closed. Widening it in a later msutils release (for
example to admit another lossless codec) changes no ID. A refusal raises
:class:`~msutils.LogicalIdRefusal` with a stable ``code``, the offending
``path`` and a ``reason``.

:func:`~msutils.logical_id` first walks the tree with ``lstat``, **without
following links**, and parses every ``zarr.json`` with msutils' own parser,
before zarr-python opens anything.

1. **Root.** A real directory, not a symlink, holding a ``zarr.json`` with
   ``zarr_format`` 3 and ``node_type`` ``group`` (``zarr-root``). Any
   ``.zgroup``, ``.zarray``, ``.zattrs`` or ``.zmetadata`` anywhere is
   ``zarr-format``: Zarr v2 is not supported; re-export with
   ``zarr_format=3``.
2. **Entry types.** Symlinks (``zarr-symlink``); FIFOs, sockets and devices
   (``zarr-special-file``) -- a FIFO would otherwise block a read.
3. **Structure.** A group directory holds only ``zarr.json`` and
   subdirectories that are nodes; a directory without ``zarr.json`` (an
   implicit node) or any other file is ``zarr-extra-entry``. An array
   directory holds only ``zarr.json`` and chunk keys valid for its
   ``chunk_key_encoding`` (``default``: ``c`` then indices, separated by
   ``/`` or ``.``; ``v2``: bare indices) that lie inside the chunk grid.
   Stale out-of-grid chunks, dotfiles and nested nodes are
   ``zarr-extra-entry``.
4. **Names.** Valid UTF-8, not starting with ``__`` (``zarr-name``).
5. **Metadata documents.** At most 64 MiB, strict JSON with the rules above,
   an object at the top level. An unknown top-level key is refused
   (``zarr-metadata``) unless its value is an object with
   ``"must_understand": false``, in which case it is ignored.
6. **Arrays.** ``shape`` has at most 32 entries, each an integer from 0 to
   2\ :sup:`53` − 1, and at most 2\ :sup:`62` elements in all; ``data_type``
   maps to a tag; ``chunk_grid`` is ``regular`` with a positive chunk shape
   of the same rank; ``storage_transformers`` is absent or empty;
   ``dimension_names`` is absent, or a same-rank list of strings or nulls;
   ``attributes`` is an object.
7. **Codec allowlist (lossless only)**, anything else being ``zarr-codec``:

   * array to array: ``transpose``, whose order must be a permutation;
   * array to bytes: ``bytes`` (endian ``little`` or ``big``, required unless
     the item size is 1); ``vlen-utf8`` (only for ``string``);
     ``sharding_indexed``, one level deep, with allowlisted inner codecs,
     index codecs ``bytes(little)`` optionally followed by ``crc32c``, index
     location ``start`` or ``end``, and an inner chunk shape dividing the
     shard shape;
   * bytes to bytes: ``zstd``, ``gzip`` (levels 0--9), ``blosc`` (``lz4``,
     ``lz4hc``, ``blosclz``, ``zstd`` or ``zlib``) and ``crc32c``.

   This refuses the whole ``numcodecs.*`` namespace, including the lossy
   ``bitround``, ``quantize`` and ``fixedscaleoffset`` and anything that
   might unpickle.
8. **Ceilings** (``zarr-chunk-size``). A decoded (inner) chunk of a
   fixed-width type is at most 2 GiB; a ``string`` chunk holds at most
   2\ :sup:`24` elements; an unsharded chunk file is at most 2 GiB on disk.
9. **Consolidated metadata.** Any inline ``consolidated_metadata`` must
   describe exactly the nodes below its group, each entry equal to that
   node's own ``zarr.json`` (ignoring nested ``consolidated_metadata``);
   otherwise ``zarr-consolidated-stale``. xarray and xradio read consolidated
   metadata by default, so a stale copy would show them attributes that were
   never hashed. Arrays are then opened individually, never through the
   consolidated copy.
10. **Time of check and use.** After an array is hashed its ``zarr.json`` is
    re-read; a change is ``zarr-changed-during-read``. Any error while
    decoding (a corrupt chunk, a checksum mismatch, truncated data) is
    ``zarr-unreadable``.

Reads are aligned to the axis-0 chunk grid and bounded at 256 MiB per slab,
so each chunk is decoded once and memory stays bounded. Block digests are
computed on ``workers`` threads (default ``min(8, cpus)``); the ID does not
depend on the worker count.

What the ID detects, and what it does not
-----------------------------------------

zarr-python's default ``write_empty_chunks=False`` skips writing a chunk that
"equals" ``fill_value``, and that comparison is only bitwise for real floats
compared with ``0.0``. A chunk of complex signed zeros, or of NaNs carrying a
payload against a NaN fill, is silently dropped and reads back as the fill
value. A third-party rechunk with those defaults can therefore change stored
values. The ID is computed over the materialised values, so such a loss
**changes the ID** and a bundle bound to the original refuses; it is never
hidden. msutils' own payload writer sets ``write_empty_chunks=True`` and
re-hashes what it wrote before publishing.

The ID is an integrity check, not authentication: whoever can rewrite both a
tree and the record of its ID can make them agree.

Residual risks:

* A ``string`` chunk can decompress into much more memory than its file
  size suggests; the element ceiling and the file-size ceiling bound it, but
  a full fix needs a bound on zstd output.
* The preflight walk and the reads are not atomic. A local process could
  swap in a symlink between them; zarr's ``LocalStore`` does not open with
  ``O_NOFOLLOW``. The trees are assumed to be local and owned by the caller;
  re-reading ``zarr.json`` narrows the window for metadata only.
* Zero-size arrays have no blocks: a ``(0, 3)`` array and a ``(3, 0)`` array
  differ only through ``shape`` in the header, as intended.
