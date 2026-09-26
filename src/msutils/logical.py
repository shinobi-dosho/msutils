"""Chunk-, shard- and codec-independent logical identity of a Zarr v3 hierarchy.

``msutils-logical-hash/v1`` identifies the *raw logical content* of a Zarr v3
tree: its node names and kinds, every node's attributes, and for arrays the
logical dtype, shape, dimension names and every element value after codec
decoding (before any CF or xarray decoding). It is a SHA-256 Merkle tree over
fixed blocks of :data:`BLOCK_ELEMENTS` elements, so chunking, sharding,
codecs, byte order, chunk-key encoding, ``fill_value``, consolidated metadata
and the physical string representation do not enter it. The normative
specification, golden vectors and the reading policy are in
``docs/concepts/logical_identity.rst``; this module is the only
implementation, shared by the exact-native bundle (the MSv4 binding, the
native payload and the native virtual tree of a live MSv2).

Reading fails closed. Before zarr-python opens anything, the tree is walked
with ``lstat`` (never following links) and every ``zarr.json`` is parsed by
this module: Zarr v2 metadata, symlinks, special files, unknown entries,
non-allowlisted (lossy or unknown) codecs and dtypes, stale consolidated
metadata and chunks above the decoded-size ceiling are refused with a
:class:`LogicalIdRefusal`. The allowlist is reading policy, not identity:
widening it later changes no ID.

Only numpy is imported at module level; zarr (``msutils[msv4]``) is imported
when a tree is actually read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = ["BLOCK_ELEMENTS", "LOGICAL_HASH", "LogicalIdRefusal", "logical_id"]

#: Algorithm name and ID prefix. Bumped only if the digest definition changes.
LOGICAL_HASH: Final = "msutils-logical-hash/v1"
#: Elements per leaf block. Fixed by the algorithm version.
BLOCK_ELEMENTS: Final = 262144

_DOMAIN = LOGICAL_HASH.encode("ascii") + b"\x00"
_ID = re.compile(re.escape(LOGICAL_HASH) + r":[0-9a-f]{64}\Z")

# Reading policy (not identity). Module constants so tests can lower them.
_MAX_ZARR_METADATA_BYTES = 64 * 1024 * 1024
_MAX_CHUNK_BYTES = 2 * 1024 * 1024 * 1024
_MAX_STRING_CHUNK_ELEMENTS = 1 << 24
_READ_BYTES = 256 * 1024 * 1024
_MAX_RANK = 32
_MAX_EXTENT = (1 << 53) - 1
_MAX_ELEMENTS = 1 << 62
# A read slab of strings is sized as if each element took this many bytes.
_STRING_READ_ITEMSIZE = 64

#: Logical tag -> the little-endian numpy dtype whose bytes are hashed.
_NUMERIC = {
    "bool": np.dtype("|b1"),
    "int8": np.dtype("|i1"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "uint8": np.dtype("|u1"),
    "uint16": np.dtype("<u2"),
    "uint32": np.dtype("<u4"),
    "uint64": np.dtype("<u8"),
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "complex64": np.dtype("<c8"),
    "complex128": np.dtype("<c16"),
}
_TAGS = frozenset((*_NUMERIC, "string"))
_V2_MARKERS = frozenset((".zgroup", ".zarray", ".zattrs", ".zmetadata"))
_GROUP_KEYS = frozenset(("zarr_format", "node_type", "attributes", "consolidated_metadata"))
_ARRAY_KEYS = frozenset(
    (
        "zarr_format",
        "node_type",
        "shape",
        "data_type",
        "chunk_grid",
        "chunk_key_encoding",
        "fill_value",
        "codecs",
        "attributes",
        "dimension_names",
        "storage_transformers",
    )
)
_ARRAY_REQUIRED = frozenset(
    ("shape", "data_type", "chunk_grid", "chunk_key_encoding", "fill_value", "codecs")
)
_BLOSC_CNAMES = frozenset(("lz4", "lz4hc", "blosclz", "zstd", "zlib"))
_BLOSC_SHUFFLES = frozenset(("noshuffle", "shuffle", "bitshuffle"))
_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")
# The quiet NaN every JSON ``NaN`` token decodes to, stated by its bits.
_JSON_NAN = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]


class LogicalIdRefusal(ValueError):
    """A Zarr tree outside the reading policy, refused before or while hashing.

    Attributes:
        code: Stable refusal code, e.g. ``zarr-format`` or ``zarr-codec``.
        path: The offending node or file, relative to the tree root, where known.
        reason: Human-readable detail.
    """

    def __init__(self, code: str, path: str | None, reason: str):
        self.code, self.path, self.reason = code, path, reason
        location = f" at {path or '/'}" if path is not None else ""
        super().__init__(f"{code}{location}: {reason}")


# --------------------------------------------------------------------------
# Canonical encoding (spec section 4.3)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _frame(tag: bytes) -> Any:
    return hashlib.sha256(_DOMAIN + tag + b"\x00")


def _encode_tlv(value: Any) -> bytes:
    """Canonical binary encoding of a JSON-model value."""
    out = bytearray()
    _tlv_into(out, value)
    return bytes(out)


def _utf8(text: str, what: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        raise LogicalIdRefusal(
            "zarr-metadata", None, f"{what} is not valid Unicode (lone surrogate)"
        ) from None


def _tlv_into(out: bytearray, value: Any) -> None:
    if value is None:
        out += b"N"
    elif value is True:
        out += b"T"
    elif value is False:
        out += b"F"
    elif isinstance(value, int):
        digits = str(value).encode("ascii")
        out += b"I" + _u64(len(digits)) + digits
    elif isinstance(value, float):
        out += b"D" + struct.pack("<d", value)
    elif isinstance(value, str):
        data = _utf8(value, "string")
        out += b"S" + _u64(len(data)) + data
    elif isinstance(value, (list, tuple)):
        out += b"L" + _u64(len(value))
        for item in value:
            _tlv_into(out, item)
    elif isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise LogicalIdRefusal("zarr-metadata", None, "object keys must be strings")
        items = sorted((_utf8(key, "object key"), item) for key, item in value.items())
        out += b"O" + _u64(len(items))
        for key, item in items:
            out += _u64(len(key)) + key
            _tlv_into(out, item)
    else:
        raise LogicalIdRefusal(
            "zarr-metadata", None, f"{type(value).__name__} is not a JSON-model value"
        )


def _format_id(root_digest: bytes) -> str:
    return f"{LOGICAL_HASH}:{root_digest.hex()}"


def _group_digest(
    attributes: Mapping[str, Any], children: Mapping[str, tuple[bytes, bytes]]
) -> bytes:
    """Digest of a group from its attributes and ``{name: (b"A"|b"G", digest)}``."""
    header = _encode_tlv(dict(attributes or {}))
    digest = _frame(b"group")
    digest.update(_u64(len(header)) + header + _u64(len(children)))
    for key, (kind, child) in sorted(
        ((_utf8(name, "node name"), value) for name, value in children.items())
    ):
        digest.update(_u64(len(key)) + key + kind + child)
    return digest.digest()


# --------------------------------------------------------------------------
# Array digest (spec section 4.4)


class _Pool:
    """Block-digest scheduling: serial for one worker, else a bounded thread window."""

    def __init__(self, workers: int):
        self.window = 2 * workers
        self.executor = ThreadPoolExecutor(workers) if workers > 1 else None

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


def _default_workers(workers: int | None) -> int:
    if workers is None:
        return min(8, os.cpu_count() or 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return workers


@contextmanager
def _pool(workers: int | None = None) -> Iterator[_Pool]:
    pool = _Pool(_default_workers(workers))
    try:
        yield pool
    finally:
        pool.close()


def _canonical(values: Any, tag: str) -> np.ndarray:
    """Flatten ``values`` (C order) into the representation a block hashes.

    Numeric values must already have the tag's kind and width -- this never
    converts a value, it only byte-swaps -- so a caller handing float64 data
    to a float32 array fails loudly instead of hashing rounded numbers.
    """
    array = np.asarray(values)
    if tag == "string":
        if array.dtype.kind not in "OUT":
            raise TypeError(f"{array.dtype} values for a string array")
        return array.reshape(-1).astype(object)
    want = _NUMERIC[tag]
    if array.dtype.kind != want.kind or array.dtype.itemsize != want.itemsize:
        raise TypeError(f"{array.dtype} values for a {tag} array")
    flat = array.reshape(-1)
    order = flat.dtype.byteorder
    if order == ">" or (order == "=" and sys.byteorder == "big"):
        flat = flat.byteswap().view(flat.dtype.newbyteorder("<"))
    return np.ascontiguousarray(flat)


def _block_digest(tag: str, index: int, block: np.ndarray) -> bytes:
    digest = _frame(b"block")
    digest.update(_u64(index) + _u64(block.size))
    if tag == "string":
        out = bytearray()
        for item in block:
            if not isinstance(item, str):
                raise TypeError(f"{type(item).__name__} element in a string array")
            data = item.encode("utf-8")
            out += _u64(len(data)) + data
        digest.update(out)
    elif tag == "bool":
        digest.update(np.not_equal(block.view(np.uint8), 0).view(np.uint8))
    else:
        digest.update(block.view(np.uint8))
    return digest.digest()


class _ArrayHasher:
    """Streaming array digest: feed C-order values of any run length, in order.

    The digest does not depend on how the values are split between calls or
    on the worker count; blocks are cut at fixed element offsets.
    """

    def __init__(
        self,
        tag: str,
        shape: tuple[int, ...] | list[int],
        *,
        dims: list[str | None] | None = None,
        attributes: Mapping[str, Any] | None = None,
        pool: _Pool | None = None,
    ):
        if tag not in _TAGS:
            raise ValueError(f"unknown logical dtype {tag!r}")
        shape = [int(extent) for extent in shape]
        if dims is not None and all(name is None for name in dims):
            dims = None
        self._tag = tag
        self._size = math.prod(shape)
        self._nblocks = -(-self._size // BLOCK_ELEMENTS)
        header = _encode_tlv(
            {
                "attributes": dict(attributes or {}),
                "block_elements": BLOCK_ELEMENTS,
                "dims": None if dims is None else list(dims),
                "dtype": tag,
                "shape": shape,
            }
        )
        self._top = _frame(b"array")
        self._top.update(_u64(len(header)) + header + _u64(self._nblocks))
        self._pool = pool if pool is not None and pool.executor is not None else None
        self._window: deque[Any] = deque()
        self._pending: list[np.ndarray] = []
        self._pending_count = 0
        self._seen = 0
        self._index = 0
        self._digest: bytes | None = None

    def update(self, values: Any) -> None:
        if self._digest is not None:
            raise RuntimeError("digest already finalised")
        flat = _canonical(values, self._tag)
        self._seen += flat.size
        if self._seen > self._size:
            raise ValueError(f"more than the {self._size} elements the shape declares")
        offset = 0
        if self._pending_count:
            take = min(BLOCK_ELEMENTS - self._pending_count, flat.size)
            self._pending.append(flat[:take])
            self._pending_count += take
            offset = take
            if self._pending_count < BLOCK_ELEMENTS:
                return
            self._emit(np.concatenate(self._pending))
            self._pending, self._pending_count = [], 0
        while flat.size - offset >= BLOCK_ELEMENTS:
            self._emit(flat[offset : offset + BLOCK_ELEMENTS])
            offset += BLOCK_ELEMENTS
        if offset < flat.size:
            # Copy the tail so a large read slab is not kept alive by it.
            self._pending, self._pending_count = [flat[offset:].copy()], flat.size - offset

    def _emit(self, block: np.ndarray) -> None:
        index = self._index
        self._index += 1
        if self._pool is None:
            self._top.update(_block_digest(self._tag, index, block))
            return
        self._window.append(self._pool.executor.submit(_block_digest, self._tag, index, block))
        while len(self._window) > self._pool.window:
            self._top.update(self._window.popleft().result())

    def digest(self) -> bytes:
        if self._digest is None:
            if self._seen != self._size:
                raise ValueError(f"{self._seen} of {self._size} elements supplied")
            if self._pending_count:
                self._emit(np.concatenate(self._pending))
                self._pending, self._pending_count = [], 0
            while self._window:
                self._top.update(self._window.popleft().result())
            if self._index != self._nblocks:  # pragma: no cover - internal invariant
                raise AssertionError("block count mismatch")
            self._digest = self._top.digest()
        return self._digest

    def hexdigest(self) -> str:
        return self.digest().hex()


# --------------------------------------------------------------------------
# Preflight: the tree as msutils' own parser sees it (spec section 4.6)


@dataclass(frozen=True)
class _Array:
    path: str
    tag: str
    shape: tuple[int, ...]
    dims: list[str | None] | None
    attributes: dict[str, Any]
    #: The (inner) chunk shape a read slab is aligned to.
    chunks: tuple[int, ...]
    itemsize: int
    meta: dict[str, Any] = field(repr=False)
    raw: bytes = field(repr=False)


@dataclass(frozen=True)
class _Group:
    path: str
    attributes: dict[str, Any]
    children: dict[str, _Group | _Array]
    meta: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class TreeDigests:
    """Result of hashing one tree: its ID, parsed nodes and per-array digests."""

    root_id: str
    tree: _Group
    arrays: dict[str, str]


def _join(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


def _refuse(code: str, path: str | None, reason: str) -> LogicalIdRefusal:
    return LogicalIdRefusal(code, path, reason)


def _listing(directory: Path, relative: str) -> dict[str, os.DirEntry]:
    """Directory entries with name, link and file-type policy applied."""
    entries = {}
    with os.scandir(directory) as scan:
        for entry in scan:
            name = entry.name
            where = _join(relative, name)
            try:
                name.encode("utf-8")
            except UnicodeEncodeError:
                raise _refuse("zarr-name", relative, f"{name!r} is not valid UTF-8") from None
            if entry.is_symlink():
                raise _refuse("zarr-symlink", where, "symbolic links are not followed")
            mode = entry.stat(follow_symlinks=False).st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise _refuse("zarr-special-file", where, "not a regular file or directory")
            entries[name] = entry
    return entries


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_constant(name: str) -> float:
    return {"NaN": _JSON_NAN, "Infinity": math.inf, "-Infinity": -math.inf}[name]


def _read_metadata_bytes(path: Path, relative: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _refuse("zarr-metadata", relative, f"cannot open zarr.json: {exc}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise _refuse("zarr-metadata", relative, "zarr.json is not a regular file")
        if info.st_size > _MAX_ZARR_METADATA_BYTES:
            raise _refuse("zarr-metadata", relative, "zarr.json exceeds the metadata ceiling")
        raw = stream.read(_MAX_ZARR_METADATA_BYTES + 1)
    if len(raw) > _MAX_ZARR_METADATA_BYTES:
        raise _refuse("zarr-metadata", relative, "zarr.json exceeds the metadata ceiling")
    return raw


def _parse_metadata(raw: bytes, relative: str) -> dict[str, Any]:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_json_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise _refuse("zarr-metadata", relative, f"invalid zarr.json: {exc}") from exc
    if not isinstance(document, dict):
        raise _refuse("zarr-metadata", relative, "zarr.json must hold an object")
    if document.get("zarr_format") != 3:
        raise _refuse(
            "zarr-format",
            relative,
            f"zarr_format {document.get('zarr_format')!r} is not supported; "
            "re-export with zarr_format=3",
        )
    return document


def _check_extensions(document: dict[str, Any], known: frozenset[str], relative: str) -> None:
    for key, value in document.items():
        if key in known:
            continue
        if not (isinstance(value, dict) and value.get("must_understand") is False):
            raise _refuse("zarr-metadata", relative, f"unknown required metadata key {key!r}")


def _attributes(document: dict[str, Any], relative: str) -> dict[str, Any]:
    attributes = document.get("attributes", {})
    if not isinstance(attributes, dict):
        raise _refuse("zarr-metadata", relative, "attributes must be an object")
    _encode_tlv(attributes)  # refuses lone surrogates and non-JSON values now
    return attributes


def _positive_ints(value: Any) -> bool:
    return isinstance(value, list) and all(
        type(item) is int and 0 < item <= _MAX_EXTENT for item in value
    )


def _dtype(data_type: Any, relative: str) -> tuple[str, int | None]:
    """Logical tag and fixed item size (``None`` for variable-length strings)."""
    if isinstance(data_type, str):
        if data_type in _NUMERIC:
            return data_type, _NUMERIC[data_type].itemsize
        if data_type == "string":
            return "string", None
    elif isinstance(data_type, dict) and data_type.get("name") == "fixed_length_utf32":
        config = data_type.get("configuration")
        length = config.get("length_bytes") if isinstance(config, dict) else None
        if (
            set(data_type) <= {"name", "configuration"}
            and isinstance(config, dict)
            and set(config) == {"length_bytes"}
            and type(length) is int
            and 0 < length <= _MAX_CHUNK_BYTES
            and length % 4 == 0
        ):
            return "string", length
        raise _refuse("zarr-metadata", relative, "invalid fixed_length_utf32 configuration")
    raise _refuse("zarr-dtype", relative, f"data_type {data_type!r} is not supported")


def _codec_config(codec: dict[str, Any], allowed: set[str], relative: str) -> dict[str, Any]:
    config = codec.get("configuration", {})
    if not isinstance(config, dict) or set(config) - allowed:
        raise _refuse("zarr-codec", relative, f"unsupported {codec['name']} configuration")
    return config


def _validate_codecs(
    codecs: Any,
    *,
    relative: str,
    data_type: Any,
    itemsize: int | None,
    rank: int,
    chunk_shape: list[int],
    nested: bool = False,
) -> list[int]:
    """Validate a codec chain against the lossless allowlist; return the inner chunk shape."""
    if not isinstance(codecs, list) or not codecs:
        raise _refuse("zarr-codec", relative, "codecs must be a non-empty list")
    stage = "array"
    inner = chunk_shape
    for codec in codecs:
        if (
            not isinstance(codec, dict)
            or not isinstance(codec.get("name"), str)
            or set(codec) - {"name", "configuration"}
        ):
            raise _refuse("zarr-codec", relative, f"malformed codec {codec!r}")
        name = codec["name"]
        if name == "transpose":
            config = _codec_config(codec, {"order"}, relative)
            order = config.get("order")
            if stage != "array" or not (
                isinstance(order, list)
                and all(type(axis) is int for axis in order)
                and sorted(order) == list(range(rank))
            ):
                raise _refuse("zarr-codec", relative, "transpose must permute the array axes")
        elif name in ("bytes", "vlen-utf8", "sharding_indexed"):
            if stage != "array":
                raise _refuse("zarr-codec", relative, "more than one array-to-bytes codec")
            stage = "bytes"
            if name == "bytes":
                config = _codec_config(codec, {"endian"}, relative)
                endian = config.get("endian")
                if data_type == "string" or not (
                    endian in ("little", "big") or (endian is None and itemsize == 1)
                ):
                    raise _refuse("zarr-codec", relative, "invalid bytes codec for this dtype")
            elif name == "vlen-utf8":
                _codec_config(codec, set(), relative)
                if data_type != "string":
                    raise _refuse("zarr-codec", relative, "vlen-utf8 needs data_type string")
            else:
                if nested:
                    raise _refuse("zarr-codec", relative, "nested sharding is not supported")
                config = _codec_config(
                    codec, {"chunk_shape", "codecs", "index_codecs", "index_location"}, relative
                )
                shard_inner = config.get("chunk_shape")
                if (
                    not _positive_ints(shard_inner)
                    or len(shard_inner) != rank
                    or any(
                        outer % part for outer, part in zip(chunk_shape, shard_inner, strict=True)
                    )
                ):
                    raise _refuse(
                        "zarr-codec", relative, "shard chunk shape must divide the shard shape"
                    )
                if config.get("index_location", "end") not in ("start", "end"):
                    raise _refuse("zarr-codec", relative, "invalid shard index_location")
                index_codecs = config.get("index_codecs")
                if not isinstance(index_codecs, list) or [
                    (item.get("name"), item.get("configuration"))
                    if isinstance(item, dict) and set(item) <= {"name", "configuration"}
                    else None
                    for item in index_codecs
                ] not in (
                    [("bytes", {"endian": "little"})],
                    [("bytes", {"endian": "little"}), ("crc32c", None)],
                    [("bytes", {"endian": "little"}), ("crc32c", {})],
                ):
                    raise _refuse("zarr-codec", relative, "unsupported shard index codecs")
                inner = _validate_codecs(
                    config.get("codecs"),
                    relative=relative,
                    data_type=data_type,
                    itemsize=itemsize,
                    rank=rank,
                    chunk_shape=shard_inner,
                    nested=True,
                )
        elif name in ("zstd", "gzip", "blosc", "crc32c"):
            if stage != "bytes":
                raise _refuse("zarr-codec", relative, f"{name} before the array-to-bytes codec")
            if name == "zstd":
                config = _codec_config(codec, {"level", "checksum"}, relative)
                if type(config.get("level")) is not int or not isinstance(
                    config.get("checksum", False), bool
                ):
                    raise _refuse("zarr-codec", relative, "invalid zstd configuration")
            elif name == "gzip":
                config = _codec_config(codec, {"level"}, relative)
                if type(config.get("level")) is not int or not 0 <= config["level"] <= 9:
                    raise _refuse("zarr-codec", relative, "invalid gzip configuration")
            elif name == "blosc":
                config = _codec_config(
                    codec, {"cname", "clevel", "shuffle", "typesize", "blocksize"}, relative
                )
                if (
                    config.get("cname") not in _BLOSC_CNAMES
                    or config.get("shuffle", "noshuffle") not in _BLOSC_SHUFFLES
                    or any(
                        key in config and type(config[key]) is not int
                        for key in ("clevel", "typesize", "blocksize")
                    )
                ):
                    raise _refuse("zarr-codec", relative, "invalid blosc configuration")
            else:
                _codec_config(codec, set(), relative)
        else:
            raise _refuse(
                "zarr-codec", relative, f"codec {name!r} is not on the lossless allowlist"
            )
    if stage != "bytes":
        raise _refuse("zarr-codec", relative, "missing array-to-bytes codec")
    return inner


def _chunk_key_rules(document: dict[str, Any], relative: str) -> tuple[str, str]:
    encoding = document["chunk_key_encoding"]
    if not isinstance(encoding, dict) or set(encoding) - {"name", "configuration"}:
        raise _refuse("zarr-metadata", relative, "invalid chunk_key_encoding")
    name = encoding.get("name")
    config = encoding.get("configuration", {})
    if name not in ("default", "v2") or not isinstance(config, dict) or set(config) - {"separator"}:
        raise _refuse("zarr-metadata", relative, f"unsupported chunk_key_encoding {name!r}")
    separator = config.get("separator", "/" if name == "default" else ".")
    if separator not in ("/", "."):
        raise _refuse("zarr-metadata", relative, "chunk key separator must be '/' or '.'")
    return name, separator


def _key_indices(parts: list[str], encoding: str, rank: int) -> list[str] | None:
    """Strip the encoding's prefix from key components; ``None`` if malformed."""
    if encoding == "default":
        if not parts or parts[0] != "c":
            return None
        return parts[1:]
    if rank == 0:
        return [] if parts == ["0"] else None
    return parts


def _check_chunks(
    directory: Path, relative: str, document: dict[str, Any], shape: list[int], sharded: bool
) -> None:
    """Every entry of an array directory must be an in-grid chunk key."""
    encoding, separator = _chunk_key_rules(document, relative)
    chunk_shape = document["chunk_grid"]["configuration"]["chunk_shape"]
    grid = [-(-extent // chunk) for extent, chunk in zip(shape, chunk_shape, strict=True)]
    rank = len(shape)

    def valid(indices: list[str] | None, complete: bool) -> bool:
        # A file is a complete key; a directory is a strict prefix of one.
        if indices is None or (len(indices) != rank if complete else len(indices) >= rank):
            return False
        return all(
            _INDEX.match(index) and int(index) < grid[axis] for axis, index in enumerate(indices)
        )

    def walk(folder: Path, parts: list[str]) -> None:
        for name, entry in sorted(_listing(folder, _join(relative, "/".join(parts))).items()):
            if not parts and name == "zarr.json":
                continue
            where = _join(relative, "/".join((*parts, name)))
            if entry.is_dir(follow_symlinks=False):
                prefix = [*parts, name]
                if separator != "/" or not valid(_key_indices(prefix, encoding, rank), False):
                    raise _refuse("zarr-extra-entry", where, "not a chunk key of this array")
                walk(folder / name, prefix)
                continue
            key = [*parts, name] if separator == "/" else name.split(".")
            if (separator != "/" and parts) or not valid(_key_indices(key, encoding, rank), True):
                raise _refuse("zarr-extra-entry", where, "not a chunk key of this array")
            if not sharded and entry.stat(follow_symlinks=False).st_size > _MAX_CHUNK_BYTES:
                raise _refuse("zarr-chunk-size", where, "chunk file exceeds the size ceiling")

    walk(directory, [])


def _scan_array(directory: Path, relative: str, document: dict[str, Any], raw: bytes) -> _Array:
    _check_extensions(document, _ARRAY_KEYS, relative)
    missing = _ARRAY_REQUIRED - set(document)
    if missing:
        raise _refuse("zarr-metadata", relative, f"missing array metadata {sorted(missing)}")
    shape = document["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) > _MAX_RANK
        or any(type(extent) is not int or not 0 <= extent <= _MAX_EXTENT for extent in shape)
        or math.prod(shape) > _MAX_ELEMENTS
    ):
        raise _refuse("zarr-metadata", relative, "invalid array shape")
    rank = len(shape)
    tag, itemsize = _dtype(document["data_type"], relative)
    grid = document["chunk_grid"]
    config = grid.get("configuration") if isinstance(grid, dict) else None
    chunk_shape = config.get("chunk_shape") if isinstance(config, dict) else None
    if (
        not isinstance(grid, dict)
        or grid.get("name") != "regular"
        or set(grid) - {"name", "configuration"}
        or not isinstance(config, dict)
        or set(config) != {"chunk_shape"}
        or not _positive_ints(chunk_shape)
        or len(chunk_shape) != rank
    ):
        raise _refuse("zarr-metadata", relative, "chunk_grid must be regular and match the rank")
    if document.get("storage_transformers", []) != []:
        raise _refuse("zarr-metadata", relative, "storage transformers are not supported")
    dims = document.get("dimension_names")
    if dims is not None and (
        not isinstance(dims, list)
        or len(dims) != rank
        or any(name is not None and not isinstance(name, str) for name in dims)
    ):
        raise _refuse("zarr-metadata", relative, "invalid dimension_names")
    if dims is not None:
        _encode_tlv(dims)
        if all(name is None for name in dims):
            dims = None
    attributes = _attributes(document, relative)
    codecs = document["codecs"]
    inner = _validate_codecs(
        codecs,
        relative=relative,
        data_type=document["data_type"],
        itemsize=itemsize,
        rank=rank,
        chunk_shape=chunk_shape,
    )
    elements = math.prod(inner)
    if (itemsize is None and elements > _MAX_STRING_CHUNK_ELEMENTS) or (
        itemsize is not None and elements * itemsize > _MAX_CHUNK_BYTES
    ):
        raise _refuse("zarr-chunk-size", relative, "decoded chunk exceeds the size ceiling")
    sharded = any(
        isinstance(codec, dict) and codec.get("name") == "sharding_indexed" for codec in codecs
    )
    _check_chunks(directory, relative, document, shape, sharded)
    return _Array(
        path=relative,
        tag=tag,
        shape=tuple(shape),
        dims=dims,
        attributes=attributes,
        chunks=tuple(inner),
        itemsize=itemsize or _STRING_READ_ITEMSIZE,
        meta=document,
        raw=raw,
    )


def _descendants(group: _Group, prefix: str = "") -> Iterator[tuple[str, _Group | _Array]]:
    for name, child in group.children.items():
        key = _join(prefix, name)
        yield key, child
        if isinstance(child, _Group):
            yield from _descendants(child, key)


def _without_consolidated(document: Any) -> Any:
    if isinstance(document, dict):
        return {key: value for key, value in document.items() if key != "consolidated_metadata"}
    return document


def _check_consolidated(group: _Group) -> None:
    """Consolidated metadata must describe exactly, and only, the nodes below it."""
    consolidated = group.meta.get("consolidated_metadata")
    if consolidated is None:
        return
    if (
        not isinstance(consolidated, dict)
        or consolidated.get("kind") != "inline"
        or consolidated.get("must_understand") is not False
        or not isinstance(consolidated.get("metadata"), dict)
        or set(consolidated) - {"kind", "must_understand", "metadata"}
    ):
        raise _refuse("zarr-metadata", group.path, "unsupported consolidated_metadata")
    entries = consolidated["metadata"]
    nodes = dict(_descendants(group))
    if set(entries) != set(nodes):
        raise _refuse(
            "zarr-consolidated-stale",
            group.path,
            "consolidated metadata does not list exactly the nodes on disk",
        )
    for key, entry in entries.items():
        if not isinstance(entry, dict) or _encode_tlv(_without_consolidated(entry)) != _encode_tlv(
            _without_consolidated(nodes[key].meta)
        ):
            raise _refuse(
                "zarr-consolidated-stale",
                _join(group.path, key),
                "consolidated metadata differs from the node's zarr.json",
            )


def _scan_node(directory: Path, relative: str) -> _Group | _Array:
    entries = _listing(directory, relative)
    marker = sorted(_V2_MARKERS & set(entries))
    if marker:
        raise _refuse(
            "zarr-format",
            _join(relative, marker[0]),
            "zarr_format 2 is not supported; re-export with zarr_format=3",
        )
    entry = entries.pop("zarr.json", None)
    if entry is None:
        if not relative:
            raise _refuse("zarr-root", relative, "no zarr.json at the tree root")
        raise _refuse("zarr-extra-entry", relative, "directory is not a Zarr node (no zarr.json)")
    if not entry.is_file(follow_symlinks=False):
        raise _refuse("zarr-metadata", relative, "zarr.json is not a regular file")
    raw = _read_metadata_bytes(directory / "zarr.json", relative)
    document = _parse_metadata(raw, relative)
    kind = document.get("node_type")
    if kind == "array":
        return _scan_array(directory, relative, document, raw)
    if kind != "group":
        raise _refuse("zarr-metadata", relative, f"unknown node_type {kind!r}")
    _check_extensions(document, _GROUP_KEYS, relative)
    attributes = _attributes(document, relative)
    children: dict[str, _Group | _Array] = {}
    for name, child in sorted(entries.items()):
        where = _join(relative, name)
        if name.startswith("__"):
            raise _refuse("zarr-name", where, "names starting with '__' are reserved")
        if not child.is_dir(follow_symlinks=False):
            raise _refuse("zarr-extra-entry", where, "unexpected file in a group")
        children[name] = _scan_node(directory / name, where)
    group = _Group(path=relative, attributes=attributes, children=children, meta=document)
    _check_consolidated(group)
    return group


def _scan(root: str | os.PathLike) -> _Group:
    """Preflight a whole tree without opening it through zarr-python."""
    root = Path(root)
    try:
        info = os.lstat(root)
    except OSError as exc:
        raise _refuse("zarr-root", None, f"cannot stat the tree root: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise _refuse("zarr-symlink", "", "the tree root is a symbolic link")
    if not stat.S_ISDIR(info.st_mode):
        raise _refuse("zarr-root", "", "the tree root is not a directory")
    node = _scan_node(root, "")
    if not isinstance(node, _Group):
        raise _refuse("zarr-root", "", "the tree root must be a group, not an array")
    return node


# --------------------------------------------------------------------------
# Reading and hashing


def _slabs(shape: tuple[int, ...], chunks: tuple[int, ...], itemsize: int) -> Iterator[tuple]:
    """Index tuples covering the array in C order, aligned to the chunk grid.

    A slab spans whole chunks along the axis it slices, so each chunk is
    decoded once, unless a single index of that axis already exceeds the
    read budget, in which case the next axis is sliced instead.
    """
    if not shape:
        yield ()
        return
    if 0 in shape:
        return
    yield from _axis_slabs(shape, chunks, itemsize, (), 0)


def _axis_slabs(shape, chunks, itemsize, prefix, axis) -> Iterator[tuple]:
    row = itemsize * math.prod(shape[axis + 1 :])
    if row > _READ_BYTES and axis < len(shape) - 1:
        for index in range(shape[axis]):
            yield from _axis_slabs(shape, chunks, itemsize, (*prefix, index), axis + 1)
        return
    fit = max(1, _READ_BYTES // row)
    step = chunks[axis] * (fit // chunks[axis]) if fit >= chunks[axis] else fit
    for start in range(0, shape[axis], step):
        yield (*prefix, slice(start, min(start + step, shape[axis])))


def _import_zarr() -> Any:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - extra absent
        raise ImportError(
            "logical_id needs zarr. Install with: pip install 'msutils[msv4]'"
        ) from exc
    return zarr


@contextmanager
def _quiet_zarr() -> Iterator[None]:
    """Silence zarr's warning that fixed_length_utf32 has no stable v3 spec.

    The warning concerns forward compatibility of that representation, which
    the logical ID is deliberately independent of.
    """
    with warnings.catch_warnings():
        try:
            from zarr.errors import UnstableSpecificationWarning

            warnings.simplefilter("ignore", UnstableSpecificationWarning)
        except ImportError:  # pragma: no cover - older or newer zarr
            pass
        yield


def _hash_array(zarr: Any, store: Any, root: Path, node: _Array, pool: _Pool) -> bytes:
    try:
        with _quiet_zarr():
            array = zarr.open_array(store=store, path=node.path, mode="r", zarr_format=3)
            if tuple(array.shape) != node.shape:
                raise _refuse("zarr-changed-during-read", node.path, "shape changed while reading")
            hasher = _ArrayHasher(
                node.tag, node.shape, dims=node.dims, attributes=node.attributes, pool=pool
            )
            for index in _slabs(node.shape, node.chunks, node.itemsize):
                hasher.update(np.asarray(array[index]))
            digest = hasher.digest()
    except LogicalIdRefusal:
        raise
    except Exception as exc:
        raise _refuse("zarr-unreadable", node.path, f"{type(exc).__name__}: {exc}") from exc
    if _read_metadata_bytes(root / node.path / "zarr.json", node.path) != node.raw:
        raise _refuse("zarr-changed-during-read", node.path, "zarr.json changed while reading")
    return digest


def _tree_digests(
    root: str | os.PathLike,
    *,
    workers: int | None = None,
    check: Callable[[_Group], None] | None = None,
) -> TreeDigests:
    """Preflight, optionally ``check`` the parsed tree, then hash every array.

    ``check`` runs after the whole tree has passed the reading policy and
    before any array is opened; exceptions it raises propagate unchanged.
    """
    workers = _default_workers(workers)
    root = Path(os.path.abspath(root))
    try:
        tree = _scan(root)
    except LogicalIdRefusal:
        raise
    except Exception as exc:
        raise _refuse("zarr-unreadable", None, f"{type(exc).__name__}: {exc}") from exc
    if check is not None:
        check(tree)
    zarr = _import_zarr()
    from zarr.storage import LocalStore

    store = LocalStore(str(root), read_only=True)
    arrays: dict[str, str] = {}

    def digest(node: _Group | _Array, pool: _Pool) -> tuple[bytes, bytes]:
        if isinstance(node, _Array):
            value = _hash_array(zarr, store, root, node, pool)
            arrays[node.path] = value.hex()
            return b"A", value
        children = {name: digest(child, pool) for name, child in node.children.items()}
        return b"G", _group_digest(node.attributes, children)

    with _pool(workers) as pool:
        _, top = digest(tree, pool)
    return TreeDigests(root_id=_format_id(top), tree=tree, arrays=arrays)


def logical_id(zarr_root: str | os.PathLike, *, workers: int | None = None) -> str:
    """Chunk-, shard- and codec-independent logical ID of a Zarr v3 hierarchy.

    Returns ``"msutils-logical-hash/v1:<64 hex>"``. Two trees get the same ID
    exactly when they hold the same node names and kinds, attributes, array
    dtypes, shapes, dimension names and element values -- regardless of chunk
    grid, sharding, codecs, byte order, chunk-key encoding, which chunks are
    physically present, ``fill_value``, consolidated metadata, fixed-width
    versus variable-length strings or the zarr-python version. The digest
    covers raw stored values, not xarray's decoded view.

    Before any array is opened the tree is preflighted by msutils' own
    parser, which refuses (:class:`LogicalIdRefusal`) Zarr v2 metadata,
    symbolic links, special files, unknown entries, non-allowlisted codecs
    and dtypes, stale consolidated metadata and chunks above the
    decoded-size ceiling. Needs ``msutils[msv4]`` (zarr).

    Not bounded: the decoded size of variable-length string chunks cannot be
    known before decoding, and a small compressed chunk can expand into a
    very large amount of memory. Hashing an untrusted tree can therefore
    exhaust memory instead of refusing.

    Args:
        zarr_root: Root group of the hierarchy (a local directory).
        workers: Threads computing block digests (default ``min(8, cpus)``).
            The ID does not depend on it.
    """
    return _tree_digests(zarr_root, workers=workers).root_id
