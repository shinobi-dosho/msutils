"""Data-only native MSv2 preservation for exact, bounded reconstruction.

The bundle records every supported native table component.  It deliberately
does not contain casacore table files: a target is built afresh, with a
writer adapter that can later be replaced without changing the bundle.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from casacore.tables import table

SCHEMA = "msutils-native-preservation/v1"
PROFILE = "fixed-shape-defined-or-empty/v1"
_MANAGER_TYPES = {"StandardStMan", "IncrementalStMan"}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_BLOCK_BYTES = 2 * 1024 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_METADATA_RANK = 8


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


def _zarr_files(root: Path) -> list[dict[str, str]]:
    if not root.is_dir() or not ((root / ".zgroup").is_file() or (root / "zarr.json").is_file()):
        raise NativePreservationRefusal("zarr-root", "expected a Zarr hierarchy root")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise NativePreservationRefusal("zarr-symlink", "Zarr hierarchy contains a symlink")
    try:
        import zarr

        group = zarr.open_group(str(root), mode="r")
        arrays = []
        if hasattr(group, "visititems"):
            group.visititems(
                lambda name, item: arrays.append(item) if isinstance(item, zarr.Array) else None
            )
        else:
            pending = [group]
            while pending:
                for _, item in pending.pop().members():
                    if isinstance(item, zarr.Array):
                        arrays.append(item)
                    else:
                        pending.append(item)
        if not arrays:
            raise NativePreservationRefusal("zarr-empty", "Zarr hierarchy contains no arrays")
        for array in arrays:
            if array.ndim:
                blocks = tuple(
                    (extent + chunk - 1) // chunk
                    for extent, chunk in zip(array.shape, array.chunks, strict=True)
                )
                for index in np.ndindex(*blocks):
                    np.asarray(array.blocks[index])
            else:
                np.asarray(array[()])
    except NativePreservationRefusal:
        raise
    except Exception as exc:
        raise NativePreservationRefusal("zarr-unreadable", str(exc)) from exc
    files = []
    for path in paths:
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(), "sha256": _hash_file(path)})
    if not files:
        raise NativePreservationRefusal("zarr-empty", "Zarr hierarchy is empty")
    return files


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


def _capture_table(
    tab: Any, path: Path, table_id: str, source: Path, stage: Path, block_rows: int
) -> dict[str, Any]:
    # Borrow the already user-locked handle; capture's ExitStack owns unlock/close.
    with nullcontext(tab) as tab:
        _require_identifier(table_id, "table", table_id=table_id)
        desc = tab.getdesc()
        info = tab.info()
        keywords, refs = _references(tab.getkeywords(), path, source, table_id)
        desc.pop("_keywords_", None)
        desc.pop("_private_keywords_", None)
        for name in tab.colnames():
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
        managers = tab.getdminfo()
        for manager in managers.values():
            if manager.get("TYPE") not in _MANAGER_TYPES:
                raise NativePreservationRefusal(
                    "storage-manager", str(manager.get("TYPE")), table=table_id
                )
        components = [
            f"table/{table_id}/descriptor",
            f"table/{table_id}/info",
            f"table/{table_id}/managers",
            f"table/{table_id}/rows",
        ]
        components.extend(f"table/{table_id}/keyword/{name}" for name in keywords)
        components.extend(f"table/{table_id}/reference/{name}" for name in refs)
        columns = []
        for column_index, name in enumerate(tab.colnames()):
            component = f"table/{table_id}/column/{name}"
            components.append(component)
            nrows = tab.nrows()
            defined = [bool(tab.iscelldefined(name, row)) for row in range(nrows)]
            if any(defined) and not all(defined):
                row = next(row for row, present in enumerate(defined) if not present)
                raise NativePreservationRefusal(
                    "mixed-definedness",
                    "mixed defined and undefined cells",
                    table=table_id,
                    column=name,
                    row=row,
                )
            blocks = []
            shape = None
            if nrows and all(defined):
                for start in range(0, nrows, block_rows):
                    stop = min(start + block_rows, nrows)
                    values = [np.asarray(tab.getcell(name, row)) for row in range(start, stop)]
                    if shape is None:
                        shape = values[0].shape
                    for offset, value in enumerate(values):
                        if any(extent == 0 for extent in value.shape):
                            raise NativePreservationRefusal(
                                "zero-length-cell",
                                "zero-length array cell",
                                table=table_id,
                                column=name,
                                row=start + offset,
                            )
                        if value.shape != shape:
                            raise NativePreservationRefusal(
                                "ragged-cell",
                                f"expected shape {shape}, got {value.shape}",
                                table=table_id,
                                column=name,
                                row=start + offset,
                            )
                    array = np.stack(values)
                    if array.dtype.kind not in "biufcUS":
                        raise NativePreservationRefusal(
                            "cell-dtype", str(array.dtype), table=table_id, column=name, row=start
                        )
                    filename = (
                        f"cells/{len(table_id):02d}-{table_id}-{column_index:04d}-{start:012d}.npy"
                    )
                    target = stage / filename
                    target.parent.mkdir(exist_ok=True)
                    np.save(target, array, allow_pickle=False)
                    size = target.stat().st_size
                    if size > _MAX_BLOCK_BYTES:
                        raise NativePreservationRefusal(
                            "block-size",
                            f"encoded block is {size} bytes; limit is {_MAX_BLOCK_BYTES}",
                            table=table_id,
                            column=name,
                            row=start,
                        )
                    blocks.append(
                        {
                            "path": filename,
                            "rows": [start, stop],
                            "size": size,
                            "sha256": _hash_file(target),
                        }
                    )
            columns.append(
                {
                    "name": name,
                    "defined": bool(nrows and all(defined)),
                    "shape": list(shape) if shape is not None else None,
                    "blocks": blocks,
                }
            )
        return {
            "id": table_id,
            "rows": tab.nrows(),
            "descriptor": _encode(desc),
            "info": _encode(info),
            "managers": _encode(managers),
            "keywords": keywords,
            "references": refs,
            "columns": columns,
            "components": sorted(components),
        }


def capture_native_preservation(
    source_ms: str | os.PathLike,
    msv4: str | os.PathLike,
    bundle: str | os.PathLike,
    *,
    block_rows: int = 64,
) -> Path:
    """Capture a complete fixed-shape native bundle, bound to immutable Zarr bytes.

    The bundle is a new directory; neither source is changed.  A refusal
    leaves no bundle.  Every supported native component has one preservation
    entry, including empty subtables and all-undefined columns.
    """
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("block_rows must be a positive integer")
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
    zarr_files = _zarr_files(zarr)
    with table(str(source), readonly=True, ack=False) as main:
        _, references = _references(main.getkeywords(), source, source, "MAIN")
    paths = {"MAIN": source}
    for name, relative in references.items():
        _require_identifier(name, "referenced table", table_id="MAIN")
        _require_identifier(relative, "referenced table path", table_id="MAIN")
        if name != relative:
            raise NativePreservationRefusal("reference-alias", relative, table="MAIN")
        paths[name] = source / relative
    # Hold read locks on the full referenced closure throughout capture.
    # The content inventory also catches writers bypassing casacore locking.
    with ExitStack() as locks:
        handles = {}
        for table_id, path in paths.items():
            handle = locks.enter_context(
                table(str(path), readonly=True, ack=False, lockoptions="user")
            )
            handle.lock(write=False)
            locks.callback(handle.unlock)
            handles[table_id] = handle
        predecessor = _source_state(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            tables = [
                _capture_table(handles[table_id], path, table_id, source, stage, block_rows)
                for table_id, path in sorted(
                    paths.items(), key=lambda item: (item[0] != "MAIN", item[0])
                )
            ]
            for item in tables:
                if any(target not in paths for target in item["references"].values()):
                    raise NativePreservationRefusal(
                        "reference-target", "reference outside root closure", table=item["id"]
                    )
            if _source_state(source) != predecessor:
                raise NativePreservationRefusal(
                    "source-changed", "source MS changed during capture"
                )
            components = [component for item in tables for component in item["components"]]
            if len(components) != len(set(components)):
                raise NativePreservationRefusal(
                    "duplicate-component", "native component appears twice"
                )
            manifest = {
                "schema": SCHEMA,
                "profile": PROFILE,
                "block_rows": block_rows,
                "zarr_files": zarr_files,
                "tables": tables,
                "coverage": [
                    {"component": component, "source": "preserved-native"}
                    for component in sorted(components)
                ],
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


@dataclass(frozen=True)
class ReconstructionPlan:
    """Validated, backend-neutral reconstruction inputs."""

    bundle: Path
    tables: tuple[dict[str, Any], ...]
    block_rows: int


def plan_reconstruction(msv4: str | os.PathLike, bundle: str | os.PathLike) -> ReconstructionPlan:
    """Validate every bundle component and Zarr byte before writing a target."""
    try:
        return _plan_reconstruction(msv4, bundle)
    except NativePreservationRefusal:
        raise
    except Exception as exc:
        raise NativePreservationRefusal(
            "bundle-invalid", f"malformed or unreadable bundle: {exc}"
        ) from exc


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
    if manifest.get("schema") != SCHEMA or manifest.get("profile") != PROFILE:
        raise NativePreservationRefusal("bundle-version", "unsupported schema or profile")
    block_rows = manifest.get("block_rows")
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1:
        raise NativePreservationRefusal("block-rows", "invalid block row count")
    zarr_files = manifest.get("zarr_files")
    if not isinstance(zarr_files, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        or not _SHA256.fullmatch(item["sha256"])
        for item in zarr_files
    ):
        raise NativePreservationRefusal("zarr-ledger", "invalid Zarr file ledger")
    if _zarr_files(Path(msv4).resolve(strict=True)) != zarr_files:
        raise NativePreservationRefusal("zarr-changed", "Zarr hierarchy differs from capture")
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
        for col in item["columns"]:
            if (
                not isinstance(col.get("defined"), bool)
                or not isinstance(col.get("blocks"), list)
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
            cursor = 0
            for block in col["blocks"]:
                if (
                    not isinstance(block, dict)
                    or not isinstance(block.get("path"), str)
                    or not isinstance(block.get("rows"), list)
                    or len(block["rows"]) != 2
                    or any(type(row) is not int for row in block["rows"])
                    or type(block.get("size")) is not int
                    or not 0 < block["size"] <= _MAX_BLOCK_BYTES
                    or not isinstance(block.get("sha256"), str)
                    or not _SHA256.fullmatch(block["sha256"])
                ):
                    raise NativePreservationRefusal(
                        "block-schema", "invalid block record", table=item["id"], column=col["name"]
                    )
                relative = block["path"]
                path = root / relative
                if (
                    len(Path(relative).parts) != 2
                    or Path(relative).parts[0] != "cells"
                    or Path(relative).name in (".", "..")
                    or not relative.endswith(".npy")
                    or (root / "cells").is_symlink()
                    or path.is_symlink()
                    or path.resolve().parent != (root / "cells").resolve()
                ):
                    raise NativePreservationRefusal(
                        "block-path", relative, table=item["id"], column=col["name"]
                    )
                start, stop = block["rows"]
                if (
                    not path.is_file()
                    or path.stat().st_size != block["size"]
                    or start != cursor
                    or stop <= start
                    or stop > rows
                    or _hash_file(path) != block["sha256"]
                ):
                    raise NativePreservationRefusal(
                        "block-integrity", relative, table=item["id"], column=col["name"]
                    )
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                if array.shape[0] != stop - start or list(array.shape[1:]) != col["shape"]:
                    raise NativePreservationRefusal(
                        "block-shape", relative, table=item["id"], column=col["name"]
                    )
                if array.dtype.kind not in "biufcUS" or array.nbytes > _MAX_BLOCK_BYTES:
                    raise NativePreservationRefusal(
                        "block-dtype", relative, table=item["id"], column=col["name"]
                    )
                cursor = stop
            if col["defined"] and cursor != rows:
                raise NativePreservationRefusal(
                    "block-coverage", "missing rows", table=item["id"], column=col["name"]
                )
            if not col["defined"] and col["blocks"]:
                raise NativePreservationRefusal(
                    "block-coverage",
                    "undefined column has payload",
                    table=item["id"],
                    column=col["name"],
                )
    return ReconstructionPlan(root, tuple(tables), block_rows)


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
            for start in range(0, item["rows"], plan.block_rows):
                stop = min(start + plan.block_rows, item["rows"])
                variables = {
                    "ROWID": (
                        ("row",),
                        da.from_array(np.arange(start, stop, dtype=np.int64), chunks=stop - start),
                    )
                }
                for col in columns:
                    block = next(
                        (block for block in col["blocks"] if block["rows"] == [start, stop]), None
                    )
                    if block is None:
                        raise NativePreservationRefusal(
                            "block-coverage", "missing block", table=table_id, column=col["name"]
                        )
                    array = np.load(plan.bundle / block["path"], allow_pickle=False)
                    dims = ("row", *(f"{col['name']}_axis{i}" for i in range(array.ndim - 1)))
                    variables[col["name"]] = (dims, da.from_array(array, chunks=array.shape))
                dataset = xr.Dataset(variables)
                writes = xds_to_table(dataset, str(path), columns=[col["name"] for col in columns])
                for write in writes:
                    write.compute()


def _verify_target(plan: ReconstructionPlan, destination: Path) -> None:
    """Independent casacore read-back of every native component and cell."""
    for item in plan.tables:
        table_id = item["id"]
        path = destination if table_id == "MAIN" else destination / table_id
        with table(str(path), readonly=True, ack=False) as tab:
            if tab.nrows() != item["rows"]:
                raise NativePreservationRefusal("verify-rows", "row count differs", table=table_id)
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
            for col in item["columns"]:
                name = col["name"]
                if name not in tab.colnames():
                    raise NativePreservationRefusal(
                        "verify-column", "missing column", table=table_id, column=name
                    )
                for block in col["blocks"]:
                    array = np.load(plan.bundle / block["path"], allow_pickle=False)
                    start, stop = block["rows"]
                    for row in range(start, stop):
                        if not tab.iscelldefined(name, row):
                            raise NativePreservationRefusal(
                                "verify-definedness",
                                "cell undefined",
                                table=table_id,
                                column=name,
                                row=row,
                            )
                        actual = np.asarray(tab.getcell(name, row))
                        expected = array[row - start]
                        same = (
                            np.array_equal(actual, expected)
                            if actual.dtype.kind in "US" and expected.dtype.kind in "US"
                            else actual.dtype.str == expected.dtype.str
                            and np.ascontiguousarray(actual).tobytes()
                            == np.ascontiguousarray(expected).tobytes()
                        )
                        if actual.shape != expected.shape or not same:
                            raise NativePreservationRefusal(
                                "verify-cell",
                                f"native bytes differ: source={expected!r}, target={actual!r}",
                                table=table_id,
                                column=name,
                                row=row,
                            )
                if not col["defined"]:
                    for row in range(item["rows"]):
                        if tab.iscelldefined(name, row):
                            raise NativePreservationRefusal(
                                "verify-definedness",
                                "unexpected defined cell",
                                table=table_id,
                                column=name,
                                row=row,
                            )


def materialize_exact_native(
    msv4: str | os.PathLike,
    outpath: str | os.PathLike,
    preservation: str | os.PathLike,
    *,
    writer=None,
):
    """Publish a fresh MSv2 only after complete independent native verification.

    Existing destinations are refused, even when mapped mode permits overwrite:
    a single rename can then publish the verified private table atomically.
    """
    destination = Path(outpath)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    plan = plan_reconstruction(msv4, preservation)
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
