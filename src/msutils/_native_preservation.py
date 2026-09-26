"""Data-only native MSv2 preservation for exact, bounded reconstruction.

The bundle records every supported native table component.  It deliberately
does not contain casacore table files: a target is built afresh, with a
writer adapter that can later be replaced without changing the bundle.

Bundle schema ``msutils-native-preservation/v2`` is a directory holding
exactly ``manifest.json`` (descriptors, info, managers, keywords, references,
definedness, digests) and ``native.zarr``, a Zarr v3 hierarchy with one group
per table and one array per defined column. Both the payload and the
companion MSv4 tree are bound by :mod:`msutils.logical` IDs rather than by
file bytes, so a lossless rechunk or recompression of either tree keeps the
bundle valid while any change of value, keyword or structure refuses.

Three identities are recorded and recomputed on every plan:

* ``msv4.logical_id`` -- the MSv4 tree, a pairing binding only (the exact
  writer never reads MSv4 values);
* ``payload.logical_id`` -- ``native.zarr`` as stored;
* ``native_logical_id`` -- the logical hash of a *virtual* tree of the
  native MSv2 (``msutils-native-model/v1``: one group per table whose
  attributes are the canonical native metadata, one array per defined
  column). It is computed identically from a live MSv2, from the bundle and
  from a materialised target, which is what lets a caller re-validate a
  restored MS against the state it came from.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from casacore.tables import table

from . import _tables
from .logical import (
    _ID,
    _MAX_CHUNK_BYTES,
    _MAX_STRING_CHUNK_ELEMENTS,
    _READ_BYTES,
    LOGICAL_HASH,
    LogicalIdRefusal,
    TreeDigests,
    _Array,
    _ArrayHasher,
    _format_id,
    _Group,
    _group_digest,
    _Pool,
    _pool,
    _tree_digests,
)

SCHEMA = "msutils-native-preservation/v2"
PROFILE = "fixed-shape-defined-or-empty/v1"
NATIVE_MODEL = "msutils-native-model/v1"
_SCHEMA_V1 = "msutils-native-preservation/v1"
_PAYLOAD = "native.zarr"
_MANAGER_TYPES = {"StandardStMan", "IncrementalStMan"}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_METADATA_RANK = 8
#: Target decoded size of one payload chunk when ``block_rows`` is ``None``.
_AUTO_CHUNK_BYTES = 64 * 1024 * 1024
#: Rows per payload chunk for string columns when ``block_rows`` is ``None``.
_AUTO_STRING_ROWS = 4096
#: Rows per ``getcolshapestring`` call when checking shape uniformity.
_SHAPE_ROWS = 65536
_MANIFEST_KEYS = {
    "schema",
    "profile",
    "hash",
    "native_model",
    "msv4",
    "payload",
    "native_logical_id",
    "tables",
    "coverage",
    "created_by",
}
_COLUMN_KEYS = {"name", "defined", "shape", "dtype", "digest"}
_TABLE_KEYS = {
    "id",
    "rows",
    "descriptor",
    "info",
    "managers",
    "keywords",
    "references",
    "columns",
    "components",
}

#: casacore ``valueType`` -> logical tag and Zarr ``data_type`` of the payload.
#: Anything else (``ushort``, ``char``, records, tables) is refused before any
#: cell is read: python-casacore cannot even read ``ushort`` cells.
_VALUE_TYPES = {
    "boolean": "bool",
    "uchar": "uint8",
    "short": "int16",
    "int": "int32",
    "uint": "uint32",
    "int64": "int64",
    "float": "float32",
    "double": "float64",
    "complex": "complex64",
    "dcomplex": "complex128",
    "string": "string",
}
_NUMPY = {tag: np.dtype(tag) for tag in _VALUE_TYPES.values() if tag != "string"}
# Planning estimate for one string element when sizing a read batch.
_STRING_ITEM_BYTES = 64


class NativePreservationRefusal(ValueError):
    """A structured, preflight refusal for content outside the exact profile."""

    def __init__(
        self,
        code: str,
        reason: str,
        *,
        table: str | None = None,
        column: str | None = None,
        row: int | None = None,
    ):
        self.code, self.table, self.column, self.row = code, table, column, row
        self.reason = reason
        location = "/".join(str(value) for value in (table, column, row) if value is not None)
        super().__init__(f"{code}{f' at {location}' if location else ''}: {reason}")


@dataclass(frozen=True)
class NativePreservationIds:
    """The three logical IDs a verified preservation bundle binds.

    Attributes:
        msv4_logical_id: :func:`~msutils.logical_id` of the paired MSv4 tree.
        payload_logical_id: :func:`~msutils.logical_id` of the bundle's
            ``native.zarr`` payload.
        native_logical_id: :func:`~msutils.native_logical_id` of the native
            MSv2 the bundle restores, independent of both representations.
    """

    msv4_logical_id: str
    payload_logical_id: str
    native_logical_id: str


def _publish_new(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a concurrent owner."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NativePreservationRefusal("atomic-publish", "renameat2 is unavailable")
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        if error in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise NativePreservationRefusal(
                "atomic-publish", "filesystem or kernel lacks renameat2 no-replace"
            )
        raise OSError(error, os.strerror(error), destination)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(root: Path) -> list[tuple[str, int, str]]:
    """Content inventory of the complete contained native tree under read locks."""
    state = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise NativePreservationRefusal("source-symlink", str(path))
        if path.is_file():
            state.append((path.relative_to(root).as_posix(), path.stat().st_size, _hash_file(path)))
    return state


def _manager_identity(managers: dict[str, Any]) -> dict[str, Any]:
    """Return storage-manager state relevant to canonical native identity.

    ``StandardStMan.SPEC.IndexLength`` is the byte length of casacore's
    serialized bucket index.  Casacore derives it from the physical index
    structure and write history, reports it through ``getdminfo()``, and
    recomputes it when writing rather than accepting it as configuration.
    Preserve the raw manager record in the bundle, but exclude this one
    physical-layout statistic from equality.  Manager type, name, column
    bindings and every other specification field remain exact.
    """
    identity: dict[str, Any] = {}
    for name, manager in managers.items():
        normalized = dict(manager)
        spec = dict(normalized.get("SPEC", {}))
        if normalized.get("TYPE") == "StandardStMan":
            spec.pop("IndexLength", None)
        normalized["SPEC"] = spec
        identity[name] = normalized
    return identity


def _refuse_contained_bundle(
    source: Path,
    zarr: Path,
    destination: Path,
    source_input: str | os.PathLike,
    zarr_input: str | os.PathLike,
) -> None:
    """Reject lexical and symlink-resolved containment before any mkdir."""
    lexical_target = Path(os.path.abspath(destination))
    canonical_target = destination.resolve(strict=False)
    for target, roots in (
        (lexical_target, (Path(os.path.abspath(source_input)), Path(os.path.abspath(zarr_input)))),
        (canonical_target, (source, zarr)),
    ):
        if any(target == root or root in target.parents for root in roots):
            raise NativePreservationRefusal(
                "bundle-contained", "bundle must be outside source MS and Zarr trees"
            )


def _require_identifier(name: str, kind: str, *, table_id: str, column: str | None = None) -> None:
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
        raise NativePreservationRefusal(
            "unsupported-name",
            f"{kind} {name!r} is outside the exact-native profile",
            table=table_id,
            column=column,
        )


def _require_node_names(names: list[str], kind: str, *, table_id: str) -> None:
    """Names that become Zarr nodes: not reserved, and distinct ignoring case.

    Zarr v3 reserves names starting with ``__``; a case-insensitive collision
    would make the payload unreadable once copied to a case-insensitive
    filesystem.
    """
    seen: dict[str, str] = {}
    for name in names:
        column = name if kind == "column" else None
        if name.startswith("__"):
            raise NativePreservationRefusal(
                "unsupported-name",
                f"{kind} {name!r} starts with '__', which Zarr reserves",
                table=table_id,
                column=column,
            )
        folded = name.casefold()
        if folded in seen:
            raise NativePreservationRefusal(
                "unsupported-name",
                f"{kind} {name!r} collides with {seen[folded]!r} ignoring case",
                table=table_id,
                column=column,
            )
        seen[folded] = name


def _encode(value: Any) -> Any:
    """Encode every value in an explicit tagged envelope.

    Ordinary dictionaries are always ``dict`` envelopes, even if their keys
    happen to be ``kind`` or ``hex``.  No user metadata can impersonate a
    codec tag on decode.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufcUS":
            raise NativePreservationRefusal("metadata-dtype", f"unsupported dtype {value.dtype}")
        _array_layout(
            {"dtype": value.dtype.str, "shape": list(value.shape)}, text=value.dtype.kind == "U"
        )
        if value.dtype.kind == "U":
            return {
                "$type": "text-array",
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "items": value.ravel().tolist(),
            }
        data = np.ascontiguousarray(value)
        return {
            "$type": "array",
            "dtype": data.dtype.str,
            "shape": list(data.shape),
            "base64": base64.b64encode(data.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return {"$type": "scalar", "dtype": value.dtype.str, "value": _encode(np.asarray(value))}
    if value is None:
        return {"$type": "none"}
    if isinstance(value, str):
        return {"$type": "str", "value": value}
    if isinstance(value, bool):
        return {"$type": "bool", "value": value}
    if isinstance(value, int):
        return {"$type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"$type": "float", "hex": value.hex()}
    if isinstance(value, complex):
        return {"$type": "complex", "real": value.real.hex(), "imag": value.imag.hex()}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise NativePreservationRefusal(
                "metadata-key", "metadata dictionary keys must be strings"
            )
        return {"$type": "dict", "items": [[key, _encode(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return {"$type": "list", "items": [_encode(item) for item in value]}
    raise NativePreservationRefusal("metadata-type", f"unsupported type {type(value).__name__}")


def _array_layout(value: dict[str, Any], *, text: bool) -> tuple[np.dtype, tuple[int, ...], int]:
    """Validate allocation size before decoding attacker-controlled metadata."""
    dtype_name, shape = value.get("dtype"), value.get("shape")
    if not isinstance(dtype_name, str) or len(dtype_name) > 32:
        raise NativePreservationRefusal("metadata-dtype", "invalid array dtype")
    try:
        dtype = np.dtype(dtype_name)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativePreservationRefusal("metadata-dtype", "invalid array dtype") from exc
    allowed = "U" if text else "biufcS"
    if dtype.kind not in allowed or dtype.fields is not None or dtype.subdtype is not None:
        raise NativePreservationRefusal("metadata-dtype", "unsupported array dtype")
    if dtype.itemsize < 1 or dtype.itemsize > _MAX_METADATA_BYTES:
        raise NativePreservationRefusal("metadata-size", "invalid array item size")
    if not isinstance(shape, list) or len(shape) > _MAX_METADATA_RANK:
        raise NativePreservationRefusal("metadata-shape", "invalid array rank")
    count = 1
    for extent in shape:
        if type(extent) is not int or not 0 <= extent <= _MAX_METADATA_BYTES:
            raise NativePreservationRefusal("metadata-shape", "invalid array dimension")
        if count and extent > _MAX_METADATA_BYTES // dtype.itemsize // count:
            raise NativePreservationRefusal("metadata-size", "array exceeds metadata ceiling")
        count *= extent
    size = count * dtype.itemsize
    if size > _MAX_METADATA_BYTES:
        raise NativePreservationRefusal("metadata-size", "array exceeds metadata ceiling")
    return dtype, tuple(shape), size


def _decode(value: Any) -> Any:
    if not isinstance(value, dict) or not isinstance(value.get("$type"), str):
        raise NativePreservationRefusal("metadata-codec", "missing typed envelope")
    kind = value["$type"]
    if kind == "array":
        dtype, shape, size = _array_layout(value, text=False)
        payload = value.get("base64")
        if not isinstance(payload, str) or len(payload) != 4 * ((size + 2) // 3):
            raise NativePreservationRefusal("metadata-size", "binary payload size mismatch")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise NativePreservationRefusal("metadata-codec", "invalid base64 payload") from exc
        if len(raw) != size:
            raise NativePreservationRefusal("metadata-size", "binary payload size mismatch")
        return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
    if kind == "text-array":
        dtype, shape, size = _array_layout(value, text=True)
        items = value.get("items")
        count = size // dtype.itemsize
        if (
            not isinstance(items, list)
            or len(items) != count
            or any(not isinstance(item, str) or len(item) > dtype.itemsize // 4 for item in items)
        ):
            raise NativePreservationRefusal("metadata-codec", "invalid text array items")
        return np.asarray(items, dtype=dtype).reshape(shape)
    if kind == "scalar":
        return _decode(value["value"])[()]
    if kind == "none":
        return None
    if kind == "str":
        return value["value"]
    if kind == "bool":
        return value["value"]
    if kind == "int":
        return int(value["value"])
    if kind == "float":
        return float.fromhex(value["hex"])
    if kind == "complex":
        return complex(float.fromhex(value["real"]), float.fromhex(value["imag"]))
    if kind == "dict":
        return {key: _decode(item) for key, item in value["items"]}
    if kind == "list":
        return [_decode(item) for item in value["items"]]
    if kind == "tuple":
        return tuple(_decode(item) for item in value["items"])
    raise NativePreservationRefusal("metadata-codec", f"unknown typed envelope {kind!r}")


def _references(keywords: dict[str, Any], owner: Path, source: Path, table_id: str):
    ordinary, references = {}, {}
    for name, value in keywords.items():
        _require_identifier(name, "keyword/reference", table_id=table_id)
        if isinstance(value, str) and value.startswith("Table: "):
            raw = Path(value[7:])
            target = (raw if raw.is_absolute() else owner / raw).resolve(strict=True)
            try:
                relative = target.relative_to(source)
            except ValueError:
                raise NativePreservationRefusal(
                    "external-reference", str(target), table=table_id
                ) from None
            if len(relative.parts) != 1:
                raise NativePreservationRefusal("nested-reference", str(relative), table=table_id)
            references[name] = relative.as_posix()
        else:
            ordinary[name] = _encode(value)
    return ordinary, references


# --------------------------------------------------------------------------
# Logical identities


def _logical_or_refuse(root: Path, where: str, *, check=None) -> TreeDigests:
    """Hash a tree, reporting a reading-policy refusal in the bundle's shape."""
    try:
        return _tree_digests(root, check=check)
    except LogicalIdRefusal as exc:
        raise NativePreservationRefusal(
            exc.code, f"{where}: {exc.path or '/'}: {exc.reason}"
        ) from exc


def _native_tree_id(tables: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    """Logical ID of the ``msutils-native-model/v1`` virtual tree of table records.

    Each record is a manifest table entry whose defined columns carry their
    array digest. Manager identity excludes ``IndexLength`` (see
    :func:`_manager_identity`); the profile is deliberately not part of the
    model, so a profile bump that changes no content changes no native ID.
    """
    children = {}
    for item in tables:
        arrays = {
            col["name"]: (b"A", bytes.fromhex(col["digest"]))
            for col in item["columns"]
            if col["defined"]
        }
        attributes = {
            "rows": item["rows"],
            "descriptor": item["descriptor"],
            "info": item["info"],
            "managers": _encode(_manager_identity(_decode(item["managers"]))),
            "keywords": item["keywords"],
            "references": item["references"],
            "columns": [
                {key: col[key] for key in ("name", "defined", "shape", "dtype")}
                for col in item["columns"]
            ],
        }
        children[item["id"]] = (b"G", _group_digest(attributes, arrays))
    return _format_id(_group_digest({"msutils_native_model": NATIVE_MODEL}, children))


# --------------------------------------------------------------------------
# Reading native tables


def _closure(source: Path) -> dict[str, Path]:
    """MAIN plus the subtables its keywords reference, by table id."""
    with table(str(source), readonly=True, ack=False) as main:
        _, references = _references(main.getkeywords(), source, source, "MAIN")
    paths = {"MAIN": source}
    for name, relative in references.items():
        _require_identifier(name, "referenced table", table_id="MAIN")
        _require_identifier(relative, "referenced table path", table_id="MAIN")
        if name != relative:
            raise NativePreservationRefusal("reference-alias", relative, table="MAIN")
        paths[name] = source / relative
    _require_node_names(list(paths), "table", table_id="MAIN")
    return paths


@contextmanager
def _read_locked(paths: dict[str, Path]) -> Iterator[dict[str, Any]]:
    """Hold casacore user read locks on the whole closure while reading it.

    The source content inventory (:func:`_source_state`) additionally catches
    writers that bypass casacore locking.
    """
    with ExitStack() as locks:
        handles = {}
        for table_id, path in paths.items():
            handle = locks.enter_context(
                table(str(path), readonly=True, ack=False, lockoptions="user")
            )
            handle.lock(write=False)
            locks.callback(handle.unlock)
            handles[table_id] = handle
        yield handles


def _table_metadata(tab: Any, path: Path, table_id: str, source: Path) -> dict[str, Any]:
    """Canonical native metadata of one table, with the profile's name/type refusals.

    Shared by capture and by verification's native-ID rebuild, so the two
    cannot extract metadata differently.
    """
    _require_identifier(table_id, "table", table_id=table_id)
    desc = tab.getdesc()
    info = tab.info()
    keywords, refs = _references(tab.getkeywords(), path, source, table_id)
    desc.pop("_keywords_", None)
    desc.pop("_private_keywords_", None)
    names = tab.colnames()
    for name in names:
        _require_identifier(name, "column", table_id=table_id, column=name)
        for keyword, value in desc[name].get("keywords", {}).items():
            _require_identifier(keyword, "column keyword", table_id=table_id, column=name)
            if isinstance(value, (list, np.ndarray)) and len(value) == 0:
                raise NativePreservationRefusal(
                    "empty-typed-keyword",
                    f"{keyword} has an ambiguous empty type",
                    table=table_id,
                    column=name,
                )
    _require_node_names(names, "column", table_id=table_id)
    managers = tab.getdminfo()
    for manager in managers.values():
        if manager.get("TYPE") not in _MANAGER_TYPES:
            raise NativePreservationRefusal(
                "storage-manager", str(manager.get("TYPE")), table=table_id
            )
    types = {}
    for name in names:
        value_type = desc[name].get("valueType")
        if value_type not in _VALUE_TYPES:
            raise NativePreservationRefusal(
                "column-type",
                f"casacore valueType {value_type!r} is outside the exact-native profile",
                table=table_id,
                column=name,
            )
        types[name] = _VALUE_TYPES[value_type]
    return {
        "id": table_id,
        "rows": tab.nrows(),
        "descriptor": _encode(desc),
        "info": _encode(info),
        "managers": _encode(managers),
        "keywords": keywords,
        "references": refs,
        "names": names,
        "types": types,
    }


def _definedness(tab: Any, names: list[str]) -> dict[str, np.ndarray]:
    """Per-row definedness of every column in one TaQL pass that reads no cells."""
    if not names or not tab.nrows():
        return {name: np.zeros(tab.nrows(), dtype=bool) for name in names}
    # Names are identifiers (checked by the caller); the backslash escapes
    # TaQL keywords such as FROM or F used as column names.
    columns = ", ".join(f"ISDEFINED(\\{name}) AS D{index}" for index, name in enumerate(names))
    with _tables.query(f"SELECT {columns} FROM $1", [tab]) as result:
        return {
            name: np.asarray(result.getcol(f"D{index}"), dtype=bool)
            for index, name in enumerate(names)
        }


def _cell_shape(tab: Any, name: str, nrows: int, table_id: str) -> tuple[int, ...]:
    """The common C-order cell shape; ragged or zero-length cells refuse.

    Reports the same row v1's per-cell scan did: a zero-length cell before a
    ragged one within a row, and the first row that differs from row 0.
    """
    if tab.isscalarcol(name):
        return ()
    first = tab.getcell(name, 0)
    shape = tuple(first["shape"]) if isinstance(first, dict) else np.asarray(first).shape
    if 0 in shape:
        raise NativePreservationRefusal(
            "zero-length-cell", "zero-length array cell", table=table_id, column=name, row=0
        )
    reference = None
    for start in range(0, nrows, _SHAPE_ROWS):
        count = min(_SHAPE_ROWS, nrows - start)
        shapes = tab.getcolshapestring(name, start, count)
        if len(shapes) == 1 and count > 1:
            break  # a FixedShape column reports its one shape, not one per row
        for offset, text in enumerate(shapes):
            if reference is None:
                reference = text
                continue
            if text != reference:
                extents = [int(part) for part in re.findall(r"\d+", text)]
                zero = 0 in extents
                raise NativePreservationRefusal(
                    "zero-length-cell" if zero else "ragged-cell",
                    "zero-length array cell" if zero else f"expected shape {reference}, got {text}",
                    table=table_id,
                    column=name,
                    row=start + offset,
                )
    return shape


def _chunk_rows(
    tag: str,
    shape: tuple[int, ...],
    nrows: int,
    block_rows: int | None,
    table_id: str,
    column: str,
) -> int:
    """Rows per payload chunk, bounded by the decoded-chunk ceilings."""
    cells = math.prod(shape)
    if tag == "string":
        if block_rows is None:
            rows = max(1, min(_AUTO_STRING_ROWS, _MAX_STRING_CHUNK_ELEMENTS // cells))
        else:
            rows = block_rows
        rows = min(rows, nrows)
        if rows * cells > _MAX_STRING_CHUNK_ELEMENTS:
            raise NativePreservationRefusal(
                "chunk-size",
                f"{rows * cells} strings per chunk; limit is {_MAX_STRING_CHUNK_ELEMENTS}",
                table=table_id,
                column=column,
                row=0,
            )
        return rows
    row_bytes = cells * _NUMPY[tag].itemsize
    rows = max(1, _AUTO_CHUNK_BYTES // row_bytes) if block_rows is None else block_rows
    rows = min(rows, nrows)
    if rows * row_bytes > _MAX_CHUNK_BYTES:
        raise NativePreservationRefusal(
            "chunk-size",
            f"decoded chunk is {rows * row_bytes} bytes; limit is {_MAX_CHUNK_BYTES}",
            table=table_id,
            column=column,
            row=0,
        )
    return rows


def _column_values(
    tab: Any, name: str, tag: str, shape: tuple[int, ...], start: int, count: int, table_id: str
) -> np.ndarray:
    """Rows ``[start, start + count)`` of a column in the payload's exact dtype."""
    raw = tab.getcol(name, start, count)
    want = (count, *shape)
    if tag == "string":
        if isinstance(raw, dict):
            items, got = raw.get("array"), tuple(raw.get("shape", ()))
        else:
            items, got = raw, (len(raw),) if isinstance(raw, list) else None
        if got != want or not isinstance(items, list) or len(items) != math.prod(want):
            raise NativePreservationRefusal(
                "ragged-cell",
                f"expected shape {want}, got {got}",
                table=table_id,
                column=name,
                row=start,
            )
        cells = max(1, math.prod(shape))
        for index, item in enumerate(items):
            if not isinstance(item, str):
                raise NativePreservationRefusal(
                    "cell-string",
                    f"{type(item).__name__} in a string column",
                    table=table_id,
                    column=name,
                    row=start + index // cells,
                )
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                raise NativePreservationRefusal(
                    "cell-string",
                    "string is not valid UTF-8",
                    table=table_id,
                    column=name,
                    row=start + index // cells,
                ) from None
        values = np.empty(len(items), dtype=object)
        values[:] = items
        return values.reshape(want)
    array = np.asarray(raw)
    if array.shape != want:
        raise NativePreservationRefusal(
            "ragged-cell",
            f"expected shape {want}, got {array.shape}",
            table=table_id,
            column=name,
            row=start,
        )
    expected = _NUMPY[tag]
    if tag == "uint8" and array.dtype == np.uint16:
        # python-casacore widens uchar to uint16; narrowing is lossless once
        # every value is known to fit.
        over = np.flatnonzero((array > 255).reshape(count, -1).any(axis=1))
        if over.size:
            raise NativePreservationRefusal(
                "cell-dtype",
                "uchar value above 255",
                table=table_id,
                column=name,
                row=start + int(over[0]),
            )
        return array.astype(np.uint8)
    if array.dtype != expected:
        raise NativePreservationRefusal(
            "cell-dtype",
            f"python-casacore returned {array.dtype} for {tag}",
            table=table_id,
            column=name,
            row=start,
        )
    return array


class _PayloadSink:
    """Writes ``native.zarr`` with explicit, version-independent metadata."""

    def __init__(self, root: Path):
        import zarr
        from zarr.storage import LocalStore

        self._zarr = zarr
        self._store = LocalStore(str(root))
        zarr.create_group(store=self._store, zarr_format=3, attributes={})

    def table(self, table_id: str) -> None:
        self._zarr.create_group(store=self._store, path=table_id, zarr_format=3, attributes={})

    def column(self, table_id: str, name: str, tag: str, shape: tuple[int, ...], rows: int):
        from zarr.codecs import BytesCodec, VLenUTF8Codec, ZstdCodec

        # write_empty_chunks: with zarr's default, a chunk "equal" to the fill
        # value is not written, and that equality is not bitwise for complex
        # signed zeros or NaN payloads -- the values would silently change.
        return self._zarr.create_array(
            store=self._store,
            name=f"{table_id}/{name}",
            shape=shape,
            dtype=str if tag == "string" else tag,
            chunks=(rows, *shape[1:]),
            serializer=VLenUTF8Codec() if tag == "string" else BytesCodec(endian="little"),
            compressors=[ZstdCodec(level=3, checksum=True)],
            config={"write_empty_chunks": True},
            zarr_format=3,
            attributes={},
        )


def _capture_table(
    tab: Any,
    path: Path,
    table_id: str,
    source: Path,
    sink: _PayloadSink | None,
    block_rows: int | None,
    pool: _Pool,
) -> dict[str, Any]:
    # Borrow the already user-locked handle; the caller's ExitStack owns unlock/close.
    with nullcontext(tab) as tab:
        meta = _table_metadata(tab, path, table_id, source)
        names, types, nrows = meta.pop("names"), meta.pop("types"), meta["rows"]
        components = [
            f"table/{table_id}/descriptor",
            f"table/{table_id}/info",
            f"table/{table_id}/managers",
            f"table/{table_id}/rows",
        ]
        components.extend(f"table/{table_id}/keyword/{name}" for name in meta["keywords"])
        components.extend(f"table/{table_id}/reference/{name}" for name in meta["references"])
        components.extend(f"table/{table_id}/column/{name}" for name in names)
        if sink is not None:
            sink.table(table_id)
        defined = _definedness(tab, names)
        columns = []
        for name in names:
            tag, present = types[name], defined[name]
            if nrows and present.any() and not present.all():
                raise NativePreservationRefusal(
                    "mixed-definedness",
                    "mixed defined and undefined cells",
                    table=table_id,
                    column=name,
                    row=int(np.argmin(present)),
                )
            if not (nrows and present.all()):
                columns.append(
                    {"name": name, "defined": False, "shape": None, "dtype": tag, "digest": None}
                )
                continue
            shape = _cell_shape(tab, name, nrows, table_id)
            if sink is None:
                # Identity only: the chunk ceilings bound what the payload
                # stores, not what an MS may contain, so they do not apply.
                rows = _batch_rows([{"dtype": tag, "shape": list(shape)}], nrows)
            else:
                rows = _chunk_rows(tag, shape, nrows, block_rows, table_id, name)
            full = (nrows, *shape)
            hasher = _ArrayHasher(tag, full, pool=pool)
            array = sink.column(table_id, name, tag, full, rows) if sink is not None else None
            for start in range(0, nrows, rows):
                count = min(rows, nrows - start)
                values = _column_values(tab, name, tag, shape, start, count, table_id)
                if array is not None and tag == "string":
                    size = sum(len(item.encode("utf-8")) for item in values.reshape(-1))
                    if size > _MAX_CHUNK_BYTES:
                        raise NativePreservationRefusal(
                            "chunk-size",
                            f"string chunk is {size} bytes; limit is {_MAX_CHUNK_BYTES}",
                            table=table_id,
                            column=name,
                            row=start,
                        )
                hasher.update(values)
                if array is not None:
                    array[start : start + count] = values
            columns.append(
                {
                    "name": name,
                    "defined": True,
                    "shape": list(shape),
                    "dtype": tag,
                    "digest": hasher.hexdigest(),
                }
            )
        return {**meta, "columns": columns, "components": sorted(components)}


def _read_native(
    handles: dict[str, Any],
    paths: dict[str, Path],
    source: Path,
    sink: _PayloadSink | None,
    block_rows: int | None,
) -> list[dict[str, Any]]:
    """Table records for the closure, MAIN first, streaming cells to ``sink``."""
    with _pool() as pool:
        tables = [
            _capture_table(handles[table_id], path, table_id, source, sink, block_rows, pool)
            for table_id, path in sorted(
                paths.items(), key=lambda item: (item[0] != "MAIN", item[0])
            )
        ]
    for item in tables:
        if any(target not in paths for target in item["references"].values()):
            raise NativePreservationRefusal(
                "reference-target", "reference outside root closure", table=item["id"]
            )
    return tables


def _require_array(digests: TreeDigests) -> None:
    if not digests.arrays:
        raise NativePreservationRefusal("zarr-empty", "Zarr hierarchy contains no arrays")


def _check_payload_structure(tree: _Group, tables: list[dict[str, Any]]) -> None:
    """The payload holds exactly one group per table and one array per defined column."""

    def refuse(reason: str, table_id: str | None = None, column: str | None = None):
        return NativePreservationRefusal("payload-structure", reason, table=table_id, column=column)

    if tree.attributes:
        raise refuse("payload root carries attributes")
    if set(tree.children) != {item["id"] for item in tables}:
        raise refuse(
            f"groups {sorted(tree.children)} differ from tables {[t['id'] for t in tables]}"
        )
    for item in tables:
        group = tree.children[item["id"]]
        if not isinstance(group, _Group) or group.attributes:
            raise refuse("table node must be a group without attributes", item["id"])
        columns = {col["name"]: col for col in item["columns"] if col["defined"]}
        if set(group.children) != set(columns):
            raise refuse(
                f"arrays {sorted(group.children)} differ from defined columns {sorted(columns)}",
                item["id"],
            )
        for name, col in columns.items():
            node = group.children[name]
            if not isinstance(node, _Array):
                raise refuse("column node must be an array", item["id"], name)
            if node.tag != col["dtype"] or list(node.shape) != [item["rows"], *col["shape"]]:
                raise refuse(
                    f"array is {node.tag}{list(node.shape)}, manifest has "
                    f"{col['dtype']}{[item['rows'], *col['shape']]}",
                    item["id"],
                    name,
                )
            if node.attributes or node.dims is not None:
                raise refuse("array carries attributes or dimension names", item["id"], name)


def _created_by() -> dict[str, str]:
    from importlib.metadata import version

    import zarr

    from ._package import __version__

    return {
        "msutils": __version__,
        "numpy": np.__version__,
        "python-casacore": version("python-casacore"),
        "zarr": zarr.__version__,
    }


def native_logical_id(ms: str | os.PathLike) -> str:
    """Native logical ID of a live MSv2 under the exact-native profile.

    The ID is :func:`msutils.logical_id`'s algorithm applied to the
    ``msutils-native-model/v1`` virtual tree of the MS: one group per table
    in MAIN's referenced closure, whose attributes are the canonical native
    metadata (rows, descriptor, info, storage managers without
    ``StandardStMan`` ``IndexLength``, keywords, references and column
    records), and one array per defined column. It equals the
    ``native_logical_id`` a preservation bundle captured from this MS
    records, and that of an MS faithfully restored from such a bundle.

    Needs only the base install. Content outside the exact-native profile is
    refused exactly as :func:`~msutils.capture_native_preservation` refuses
    it, with :class:`~msutils.NativePreservationRefusal`.

    Cost: read locks are held on the closure while its complete file content
    is hashed twice (before and after, to detect concurrent writers) and
    every defined cell is read once.
    """
    source = Path(ms).resolve(strict=True)
    if not source.is_dir():
        raise NativePreservationRefusal("source-ms", "expected an MSv2 directory")
    paths = _closure(source)
    with _read_locked(paths) as handles:
        predecessor = _source_state(source)
        tables = _read_native(handles, paths, source, None, None)
        if _source_state(source) != predecessor:
            raise NativePreservationRefusal("source-changed", "source MS changed during read")
    return _native_tree_id(tables)


def capture_native_preservation(
    source_ms: str | os.PathLike,
    msv4: str | os.PathLike,
    bundle: str | os.PathLike,
    *,
    block_rows: int | None = None,
) -> Path:
    """Capture a complete fixed-shape native bundle, bound to an MSv4 logical ID.

    The bundle is a new directory; neither source is changed.  A refusal
    leaves no bundle.  Every supported native component has one preservation
    entry, including empty subtables and all-undefined columns.
    """
    if block_rows is not None and (
        isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1
    ):
        raise ValueError("block_rows must be None or a positive integer")
    source, zarr, destination = (
        Path(source_ms).resolve(strict=True),
        Path(msv4).resolve(strict=True),
        Path(bundle),
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not source.is_dir():
        raise NativePreservationRefusal("source-ms", "expected an MSv2 directory")
    if source in (zarr, destination) or zarr == destination:
        raise ValueError("source, Zarr and bundle must be separate")
    _refuse_contained_bundle(source, zarr, destination, source_ms, msv4)
    msv4_digests = _logical_or_refuse(zarr, "msv4")
    _require_array(msv4_digests)
    paths = _closure(source)
    with _read_locked(paths) as handles:
        predecessor = _source_state(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            sink = _PayloadSink(stage / _PAYLOAD)
            tables = _read_native(handles, paths, source, sink, block_rows)
            if _source_state(source) != predecessor:
                raise NativePreservationRefusal(
                    "source-changed", "source MS changed during capture"
                )
            components = [component for item in tables for component in item["components"]]
            if len(components) != len(set(components)):
                raise NativePreservationRefusal(
                    "duplicate-component", "native component appears twice"
                )
            # Read the staged payload back through the same policy and hasher
            # plan uses: a write that zarr silently altered must not publish.
            readback = _logical_or_refuse(
                stage / _PAYLOAD,
                "payload",
                check=lambda tree: _check_payload_structure(tree, tables),
            )
            for item in tables:
                for col in item["columns"]:
                    if (
                        col["defined"]
                        and readback.arrays.get(f"{item['id']}/{col['name']}") != col["digest"]
                    ):
                        raise NativePreservationRefusal(
                            "payload-readback",
                            "stored payload differs from the values read from casacore",
                            table=item["id"],
                            column=col["name"],
                        )
            manifest = {
                "schema": SCHEMA,
                "profile": PROFILE,
                "hash": LOGICAL_HASH,
                "native_model": NATIVE_MODEL,
                "msv4": {"logical_id": msv4_digests.root_id},
                "payload": {"path": _PAYLOAD, "zarr_format": 3, "logical_id": readback.root_id},
                "native_logical_id": _native_tree_id(tables),
                "tables": tables,
                "coverage": [
                    {"component": component, "source": "preserved-native"}
                    for component in sorted(components)
                ],
                "created_by": _created_by(),
            }
            manifest_bytes = _json_bytes(manifest) + b"\n"
            if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
                raise NativePreservationRefusal(
                    "metadata-size", "manifest exceeds metadata ceiling"
                )
            (stage / "manifest.json").write_bytes(manifest_bytes)
            _publish_new(stage, destination)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return destination


# --------------------------------------------------------------------------
# Planning


@dataclass(frozen=True)
class ReconstructionPlan:
    """Validated, backend-neutral reconstruction inputs."""

    bundle: Path
    #: ``bundle / "native.zarr"``, validated and hashed.
    payload: Path
    tables: tuple[dict[str, Any], ...]
    msv4_logical_id: str
    payload_logical_id: str
    native_logical_id: str
    #: ``(table, column) -> hex`` array digests recomputed from the payload.
    column_digests: Mapping[tuple[str, str], str]


def plan_reconstruction(msv4: str | os.PathLike, bundle: str | os.PathLike) -> ReconstructionPlan:
    """Validate every bundle component and recompute every logical ID before writing."""
    try:
        return _plan_reconstruction(msv4, bundle)
    except (NativePreservationRefusal, ImportError):
        raise
    except Exception as exc:
        raise NativePreservationRefusal(
            "bundle-invalid", f"malformed or unreadable bundle: {exc}"
        ) from exc


def verify_native_preservation(
    msv4: str | os.PathLike, bundle: str | os.PathLike
) -> NativePreservationIds:
    """Verify a bundle against its MSv4 tree and return the IDs it binds.

    Performs every check reconstruction planning performs -- bundle
    structure and version, the payload's reading policy and structure, every
    payload column digest, and the recomputed payload, native and MSv4
    logical IDs -- without writing anything. Refusals are
    :class:`NativePreservationRefusal`. A string payload chunk's decoded size
    cannot be bounded in advance, so an untrusted bundle can exhaust memory
    rather than refuse. Cost: one full read of the payload
    and one of the MSv4 tree.
    """
    plan = plan_reconstruction(msv4, bundle)
    return NativePreservationIds(
        msv4_logical_id=plan.msv4_logical_id,
        payload_logical_id=plan.payload_logical_id,
        native_logical_id=plan.native_logical_id,
    )


def _v1_refusal() -> NativePreservationRefusal:
    return NativePreservationRefusal(
        "bundle-version",
        f"bundle is {_SCHEMA_V1} (per-block .npy payload bound by Zarr file bytes), which "
        "this msutils no longer reads; recapture it from the source MS with "
        f"capture_native_preservation() to produce a {SCHEMA.rsplit('/', 1)[1]} bundle. "
        "v1 was never part of a release.",
    )


def _check_manifest_header(manifest: dict[str, Any], root: Path) -> None:
    if manifest.get("schema") == _SCHEMA_V1 or os.path.lexists(root / "cells"):
        raise _v1_refusal()
    for key, wanted in (
        ("schema", SCHEMA),
        ("profile", PROFILE),
        ("hash", LOGICAL_HASH),
        ("native_model", NATIVE_MODEL),
    ):
        if manifest.get(key) != wanted:
            raise NativePreservationRefusal(
                "bundle-version", f"unsupported {key} {manifest.get(key)!r}; expected {wanted!r}"
            )
    entries = sorted(os.listdir(root))
    if (
        entries != ["manifest.json", _PAYLOAD]
        or (root / _PAYLOAD).is_symlink()
        or not (root / _PAYLOAD).is_dir()
    ):
        raise NativePreservationRefusal(
            "bundle-extra-entry",
            f"bundle must hold exactly manifest.json and a {_PAYLOAD} directory, found {entries}",
        )
    if set(manifest) != _MANIFEST_KEYS:
        raise NativePreservationRefusal(
            "bundle-manifest", f"manifest keys differ: {sorted(set(manifest) ^ _MANIFEST_KEYS)}"
        )
    ids = (
        manifest["msv4"].get("logical_id") if isinstance(manifest["msv4"], dict) else None,
        manifest["payload"].get("logical_id") if isinstance(manifest["payload"], dict) else None,
        manifest["native_logical_id"],
    )
    if (
        not isinstance(manifest["msv4"], dict)
        or set(manifest["msv4"]) != {"logical_id"}
        or not isinstance(manifest["payload"], dict)
        or set(manifest["payload"]) != {"path", "zarr_format", "logical_id"}
        or manifest["payload"]["path"] != _PAYLOAD
        or type(manifest["payload"]["zarr_format"]) is not int
        or manifest["payload"]["zarr_format"] != 3
        or any(not isinstance(value, str) or not _ID.fullmatch(value) for value in ids)
    ):
        raise NativePreservationRefusal("bundle-manifest", "invalid logical ID binding")
    created_by = manifest["created_by"]
    if (
        not isinstance(created_by, dict)
        or len(created_by) > 16
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(key) > 64
            or len(value) > 64
            for key, value in created_by.items()
        )
    ):
        raise NativePreservationRefusal("bundle-manifest", "invalid created_by record")


def _check_columns(item: dict[str, Any], desc: dict[str, Any]) -> None:
    rows = item["rows"]
    for col in item["columns"]:
        if (
            set(col) != _COLUMN_KEYS
            or not isinstance(col.get("defined"), bool)
            or not (col.get("shape") is None or isinstance(col["shape"], list))
        ):
            raise NativePreservationRefusal(
                "bundle-column", "invalid column fields", table=item["id"], column=col["name"]
            )
        if col["shape"] is not None and any(
            type(size) is not int or size <= 0 for size in col["shape"]
        ):
            raise NativePreservationRefusal(
                "bundle-column",
                "invalid or zero-length cell shape",
                table=item["id"],
                column=col["name"],
            )
        if (col["defined"] and (rows == 0 or col["shape"] is None)) or (
            not col["defined"] and col["shape"] is not None
        ):
            raise NativePreservationRefusal(
                "bundle-column",
                "definedness and shape disagree",
                table=item["id"],
                column=col["name"],
            )
        column_desc = desc.get(col["name"])
        value_type = column_desc.get("valueType") if isinstance(column_desc, dict) else None
        if not isinstance(value_type, str) or _VALUE_TYPES.get(value_type) != col["dtype"]:
            raise NativePreservationRefusal(
                "bundle-column",
                f"dtype {col['dtype']!r} does not match valueType {value_type!r}",
                table=item["id"],
                column=col["name"],
            )
        digest = col["digest"]
        if (
            not (isinstance(digest, str) and _SHA256.fullmatch(digest))
            if col["defined"]
            else digest is not None
        ):
            raise NativePreservationRefusal(
                "bundle-column",
                "a defined column needs a digest and an undefined one none",
                table=item["id"],
                column=col["name"],
            )


def _plan_reconstruction(msv4: str | os.PathLike, bundle: str | os.PathLike) -> ReconstructionPlan:
    raw_root = Path(bundle).absolute()
    if raw_root.is_symlink():
        raise NativePreservationRefusal("bundle-symlink", "bundle root is a symlink")
    root = raw_root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES
    ):
        raise NativePreservationRefusal("bundle-manifest", "manifest missing, linked or oversized")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise NativePreservationRefusal("bundle-manifest", "manifest must be an object")
    _check_manifest_header(manifest, root)
    tables = manifest.get("tables")
    if (
        not isinstance(tables, list)
        or not tables
        or not isinstance(tables[0], dict)
        or tables[0].get("id") != "MAIN"
    ):
        raise NativePreservationRefusal("bundle-tables", "missing MAIN table")
    if any(not isinstance(item, dict) for item in tables):
        raise NativePreservationRefusal("bundle-tables", "table records must be objects")
    names = [item.get("id") for item in tables]
    if len(names) != len(set(names)) or any(
        not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in names
    ):
        raise NativePreservationRefusal("bundle-tables", "invalid table ids")
    expected = []
    for item in tables:
        table_id = item["id"]
        if set(item) != _TABLE_KEYS:
            raise NativePreservationRefusal(
                "bundle-table",
                f"table record keys differ: {sorted(set(item) ^ _TABLE_KEYS)}",
                table=table_id,
            )
        if (
            not isinstance(item.get("keywords"), dict)
            or not isinstance(item.get("references"), dict)
            or not isinstance(item.get("columns"), list)
            or not isinstance(item.get("components"), list)
        ):
            raise NativePreservationRefusal("bundle-table", "invalid table fields", table=table_id)
        if any(
            not isinstance(name, str) or not _IDENTIFIER.fullmatch(name)
            for name in (*item["keywords"], *item["references"])
        ):
            raise NativePreservationRefusal(
                "bundle-table", "invalid keyword or reference name", table=table_id
            )
        if any(
            not isinstance(col, dict)
            or not isinstance(col.get("name"), str)
            or not _IDENTIFIER.fullmatch(col["name"])
            for col in item["columns"]
        ):
            raise NativePreservationRefusal(
                "bundle-column", "invalid column record", table=table_id
            )
        derived = [
            f"table/{table_id}/{part}" for part in ("descriptor", "info", "managers", "rows")
        ]
        derived.extend(f"table/{table_id}/keyword/{name}" for name in item["keywords"])
        derived.extend(f"table/{table_id}/reference/{name}" for name in item["references"])
        derived.extend(f"table/{table_id}/column/{col['name']}" for col in item["columns"])
        if sorted(derived) != sorted(item.get("components", [])):
            raise NativePreservationRefusal(
                "coverage", "table component list is incomplete", table=table_id
            )
        expected.extend(derived)
    coverage = manifest.get("coverage")
    if not isinstance(coverage, list) or any(
        not isinstance(item, dict)
        or item.get("source") != "preserved-native"
        or not isinstance(item.get("component"), str)
        for item in coverage
    ):
        raise NativePreservationRefusal("coverage", "invalid coverage rows")
    covered = [item["component"] for item in coverage]
    if (
        len(expected) != len(set(expected))
        or sorted(expected) != sorted(covered)
        or len(covered) != len(set(covered))
    ):
        raise NativePreservationRefusal(
            "coverage", "missing, duplicate or unsupported native component"
        )
    for item in tables:
        rows = item["rows"]
        if type(rows) is not int or rows < 0:
            raise NativePreservationRefusal("row-count", "invalid row count", table=item["id"])
        if any(
            not isinstance(target, str) or target not in names
            for target in item["references"].values()
        ):
            raise NativePreservationRefusal("reference-target", "missing table", table=item["id"])
        desc = _decode(item["descriptor"])
        managers = _decode(item["managers"])
        if (
            not isinstance(desc, dict)
            or not isinstance(managers, dict)
            or any(
                not isinstance(manager, dict) or manager.get("TYPE") not in _MANAGER_TYPES
                for manager in managers.values()
            )
        ):
            raise NativePreservationRefusal(
                "table-schema", "invalid descriptor or manager", table=item["id"]
            )
        info = _decode(item["info"])
        if not isinstance(info, dict):
            raise NativePreservationRefusal(
                "table-info", "table info must be a dictionary", table=item["id"]
            )
        column_names = [col["name"] for col in item["columns"]]
        if len(column_names) != len(set(column_names)) or set(desc) - {
            "_define_hypercolumn_"
        } != set(column_names):
            raise NativePreservationRefusal(
                "column-coverage", "descriptor/column mismatch", table=item["id"]
            )
        # The descriptor lists columns in colnames() order, and verification
        # requires the target to match the records' order; refuse a reordered
        # manifest here rather than after the target has been written.
        if [name for name in desc if name != "_define_hypercolumn_"] != column_names:
            raise NativePreservationRefusal(
                "column-coverage",
                "column records are not in descriptor (colnames) order",
                table=item["id"],
            )
        _check_columns(item, desc)
    payload = root / _PAYLOAD
    stored = _logical_or_refuse(
        payload, "payload", check=lambda tree: _check_payload_structure(tree, tables)
    )
    digests = {}
    for item in tables:
        for col in item["columns"]:
            if not col["defined"]:
                continue
            digest = stored.arrays[f"{item['id']}/{col['name']}"]
            if digest != col["digest"]:
                raise NativePreservationRefusal(
                    "payload-changed",
                    "payload values differ from the captured digest",
                    table=item["id"],
                    column=col["name"],
                )
            digests[(item["id"], col["name"])] = digest
    if stored.root_id != manifest["payload"]["logical_id"]:
        raise NativePreservationRefusal("bundle-integrity", "payload logical ID differs")
    native_id = _native_tree_id(tables)
    if native_id != manifest["native_logical_id"]:
        raise NativePreservationRefusal("bundle-integrity", "native logical ID differs")
    msv4_digests = _logical_or_refuse(Path(msv4).resolve(strict=True), "msv4")
    _require_array(msv4_digests)
    if msv4_digests.root_id != manifest["msv4"]["logical_id"]:
        raise NativePreservationRefusal("zarr-changed", "MSv4 logical content differs from capture")
    return ReconstructionPlan(
        bundle=root,
        payload=payload,
        tables=tuple(tables),
        msv4_logical_id=msv4_digests.root_id,
        payload_logical_id=stored.root_id,
        native_logical_id=native_id,
        column_digests=MappingProxyType(digests),
    )


# --------------------------------------------------------------------------
# Writing and verification


def _row_bytes(col: dict[str, Any]) -> int:
    itemsize = _STRING_ITEM_BYTES if col["dtype"] == "string" else _NUMPY[col["dtype"]].itemsize
    return max(1, itemsize * math.prod(col["shape"]))


def _batch_rows(columns: list[dict[str, Any]], rows: int) -> int:
    """Rows per write, verify or identity-read batch, keeping one batch bounded."""
    return max(1, min(rows, _READ_BYTES // sum(_row_bytes(col) for col in columns)))


def _open_payload(plan: ReconstructionPlan) -> tuple[Any, Any]:
    import zarr
    from zarr.storage import LocalStore

    return zarr, LocalStore(str(plan.payload), read_only=True)


def _payload_rows(zarr: Any, store: Any, table_id: str, col: dict[str, Any]):
    array = zarr.open_array(store=store, path=f"{table_id}/{col['name']}", mode="r", zarr_format=3)

    def read(start: int, stop: int) -> np.ndarray:
        values = np.asarray(array[start:stop])
        # dask-ms rejects numpy's StringDType, which zarr returns for vlen strings.
        return values.astype(object) if col["dtype"] == "string" else values

    return read


class DaskMsWriter:
    """Write a validated plan into a private, newly created MSv2 table."""

    def __init__(self):
        try:
            from importlib.metadata import version

            installed = version("dask-ms")
        except ImportError as exc:  # pragma: no cover
            raise ImportError("exact-native MSv2 writing needs msutils[exact-native]") from exc
        if installed != "0.2.32":
            raise NativePreservationRefusal(
                "dask-ms-version", f"tested version 0.2.32, got {installed}"
            )

    def write(self, plan: ReconstructionPlan, destination: Path) -> None:
        import dask.array as da
        import xarray as xr
        from daskms import xds_to_table

        zarr, store = _open_payload(plan)
        for item in plan.tables:
            table_id = item["id"]
            path = destination if table_id == "MAIN" else destination / table_id
            desc = _decode(item["descriptor"])
            managers = _decode(item["managers"])
            with table(
                str(path), desc, nrow=item["rows"], readonly=False, dminfo=managers, ack=False
            ):
                pass
        for item in plan.tables:
            table_id = item["id"]
            path = destination if table_id == "MAIN" else destination / table_id
            with table(str(path), readonly=False, ack=False) as tab:
                wanted_info = _decode(item["info"])
                if tab.info() != wanted_info:
                    # casacore putinfo appends one newline to readme.
                    write_info = dict(wanted_info)
                    readme = write_info.get("readme")
                    if isinstance(readme, str) and readme.endswith("\n"):
                        write_info["readme"] = readme[:-1]
                    tab.putinfo(write_info)
                for name, encoded in item["keywords"].items():
                    tab.putkeyword(name, _decode(encoded))
                for name, target in item["references"].items():
                    tab.putkeyword(name, f"Table: {destination / target}")
            columns = [col for col in item["columns"] if col["defined"]]
            if not columns:
                continue
            readers = {col["name"]: _payload_rows(zarr, store, table_id, col) for col in columns}
            batch = _batch_rows(columns, item["rows"])
            for start in range(0, item["rows"], batch):
                stop = min(start + batch, item["rows"])
                variables = {
                    "ROWID": (
                        ("row",),
                        da.from_array(np.arange(start, stop, dtype=np.int64), chunks=stop - start),
                    )
                }
                for col in columns:
                    values = readers[col["name"]](start, stop)
                    dims = ("row", *(f"{col['name']}_axis{i}" for i in range(values.ndim - 1)))
                    variables[col["name"]] = (dims, da.from_array(values, chunks=values.shape))
                dataset = xr.Dataset(variables)
                writes = xds_to_table(dataset, str(path), columns=[col["name"] for col in columns])
                for write in writes:
                    write.compute()


def _same_cell(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape:
        return False
    if actual.dtype.kind in "OUST" and expected.dtype.kind in "OUST":
        return bool(np.array_equal(actual.astype(object), expected.astype(object)))
    return (
        actual.dtype.str == expected.dtype.str
        and np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes()
    )


def _verify_column(
    tab: Any, plan: ReconstructionPlan, item: dict[str, Any], col: dict[str, Any], pool: _Pool
) -> str:
    """Digest a target column with the capture cast and compare it with the plan.

    On success the target column is read once. Only on a mismatch is the
    payload column re-read (to tell a post-planning payload change from a bad
    write) and then both columns streamed again to name the first bad row.
    """
    table_id, name, tag = item["id"], col["name"], col["dtype"]
    shape, rows = tuple(col["shape"]), item["rows"]
    batch = _batch_rows([col], rows)

    def target(start: int, count: int) -> np.ndarray:
        try:
            return _column_values(tab, name, tag, shape, start, count, table_id)
        except NativePreservationRefusal as exc:
            raise NativePreservationRefusal(
                "verify-cell", exc.reason, table=table_id, column=name, row=exc.row
            ) from exc

    hasher = _ArrayHasher(tag, (rows, *shape), pool=pool)
    for start in range(0, rows, batch):
        hasher.update(target(start, min(batch, rows - start)))
    digest = hasher.hexdigest()
    planned = plan.column_digests[(table_id, name)]
    if digest == planned:
        return digest
    # Name the failure: first whether the payload itself moved after planning,
    # then the first target row that differs from it.
    zarr, store = _open_payload(plan)
    payload = _payload_rows(zarr, store, table_id, col)
    check = _ArrayHasher(tag, (rows, *shape), pool=pool)
    for start in range(0, rows, batch):
        check.update(payload(start, min(start + batch, rows)))
    if check.hexdigest() != planned:
        raise NativePreservationRefusal(
            "verify-payload-changed",
            "payload changed after planning",
            table=table_id,
            column=name,
        )
    for start in range(0, rows, batch):
        count = min(batch, rows - start)
        actual, expected = target(start, count), payload(start, start + count)
        for offset in range(count):
            if not _same_cell(actual[offset], expected[offset]):
                raise NativePreservationRefusal(
                    "verify-cell",
                    f"native bytes differ: source={expected[offset]!r}, target={actual[offset]!r}",
                    table=table_id,
                    column=name,
                    row=start + offset,
                )
    raise NativePreservationRefusal(  # pragma: no cover - digests differ, cells equal
        "verify-cell", "column digest differs", table=table_id, column=name
    )


def _verify_target(plan: ReconstructionPlan, destination: Path) -> None:
    """Independent casacore read-back of every native component and cell."""
    records = []
    with _pool() as pool:
        for item in plan.tables:
            table_id = item["id"]
            path = destination if table_id == "MAIN" else destination / table_id
            with table(str(path), readonly=True, ack=False) as tab:
                if tab.nrows() != item["rows"]:
                    raise NativePreservationRefusal(
                        "verify-rows", "row count differs", table=table_id
                    )
                if tab.colnames() != [col["name"] for col in item["columns"]]:
                    raise NativePreservationRefusal(
                        "verify-column-order", "column order differs", table=table_id
                    )
                desc = tab.getdesc()
                desc.pop("_keywords_", None)
                desc.pop("_private_keywords_", None)
                if _json_bytes(_encode(desc)) != _json_bytes(item["descriptor"]):
                    expected = _decode(item["descriptor"])
                    different = sorted(
                        key
                        for key in set(desc) | set(expected)
                        if _json_bytes(_encode(desc.get(key)))
                        != _json_bytes(_encode(expected.get(key)))
                    )
                    raise NativePreservationRefusal(
                        "verify-descriptor",
                        f"descriptor differs: {different}; first source={expected[different[0]]!r}; target={desc[different[0]]!r}",
                        table=table_id,
                    )
                # Keep the raw bundle record for reconstruction and evidence, but
                # compare the documented canonical identity. StandardStMan's
                # IndexLength is derived physical bookkeeping, not MS state.
                expected_managers = _decode(item["managers"])
                if _json_bytes(_encode(_manager_identity(tab.getdminfo()))) != _json_bytes(
                    _encode(_manager_identity(expected_managers))
                ):
                    raise NativePreservationRefusal(
                        "verify-manager", "storage manager differs", table=table_id
                    )
                if _json_bytes(_encode(tab.info())) != _json_bytes(item["info"]):
                    raise NativePreservationRefusal(
                        "verify-info",
                        f"table info differs: source={_decode(item['info'])!r}, target={tab.info()!r}",
                        table=table_id,
                    )
                keywords, references = _references(tab.getkeywords(), path, destination, table_id)
                if keywords != item["keywords"] or references != item["references"]:
                    raise NativePreservationRefusal(
                        "verify-keywords", "keywords or references differ", table=table_id
                    )
                # Rebuilt from the target itself (its descriptor, ISDEFINED and
                # the digests below) with capture's own metadata extraction,
                # as a final guard against the structural checks above and
                # the native model drifting apart.
                meta = _table_metadata(tab, path, table_id, destination)
                names, types = meta.pop("names"), meta.pop("types")
                defined = _definedness(tab, names)
                columns = []
                for col in item["columns"]:
                    name, present = col["name"], defined[col["name"]]
                    if col["defined"] and not present.all():
                        raise NativePreservationRefusal(
                            "verify-definedness",
                            "cell undefined",
                            table=table_id,
                            column=name,
                            row=int(np.argmin(present)),
                        )
                    if not col["defined"] and present.any():
                        raise NativePreservationRefusal(
                            "verify-definedness",
                            "unexpected defined cell",
                            table=table_id,
                            column=name,
                            row=int(np.argmax(present)),
                        )
                    # A defined target column is read at the planned shape;
                    # any other shape refuses inside _verify_column.
                    columns.append(
                        {
                            "name": name,
                            "defined": col["defined"],
                            "shape": col["shape"],
                            "dtype": types[name],
                            "digest": _verify_column(tab, plan, item, col, pool)
                            if col["defined"]
                            else None,
                        }
                    )
                records.append({**meta, "columns": columns})
    if _native_tree_id(records) != plan.native_logical_id:
        raise NativePreservationRefusal(
            "verify-native-id", "target native logical ID differs from the bundle"
        )


def materialize_exact_native(
    msv4: str | os.PathLike,
    outpath: str | os.PathLike,
    preservation: str | os.PathLike,
    *,
    writer=None,
    expected_native_logical_id: str | None = None,
):
    """Publish a fresh MSv2 only after complete independent native verification.

    Existing destinations are refused, even when mapped mode permits overwrite:
    a single rename can then publish the verified private table atomically.

    ``expected_native_logical_id`` binds the restoration to a native logical
    ID the caller holds independently of the bundle: a mismatch with the
    bundle's verified ID refuses with ``native-id-mismatch`` before anything
    is written. Without it, a manifest re-edited consistently with a tampered
    payload is outside what the bundle alone can detect.
    """
    if expected_native_logical_id is not None and not isinstance(expected_native_logical_id, str):
        raise TypeError("expected_native_logical_id must be a string")
    destination = Path(outpath)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    plan = plan_reconstruction(msv4, preservation)
    if (
        expected_native_logical_id is not None
        and plan.native_logical_id != expected_native_logical_id
    ):
        raise NativePreservationRefusal(
            "native-id-mismatch",
            f"bundle restores {plan.native_logical_id}, expected {expected_native_logical_id}",
        )
    writer = DaskMsWriter() if writer is None else writer
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    stage = stage_root / "candidate.ms"
    try:
        writer.write(plan, stage)
        _verify_target(plan, stage)
        from .info import msinfo

        result = msinfo(os.fspath(stage), level="full")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        _publish_new(stage, destination)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    result.path = os.fspath(destination)
    return result
