"""msutils-logical-hash/v1: frozen vectors, invariance, sensitivity and reading policy.

The golden vectors pin the specification in docs/concepts/logical_identity.rst:
a change to any digest input -- the TLV encoding, block framing, element
encoding or group ordering -- fails here, on every supported zarr version.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import struct
import warnings

import numpy as np
import pytest

from msutils import logical
from msutils.logical import (
    BLOCK_ELEMENTS,
    LOGICAL_HASH,
    LogicalIdRefusal,
    _ArrayHasher,
    _encode_tlv,
    _frame,
    _group_digest,
    _u64,
    logical_id,
)


def _nan64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _f32(bits: int) -> np.float32:
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


# A tree is {"attributes": {...}, "children": {name: tree | leaf}}, where a
# leaf is (values, dims, attributes). Values carry their own dtype; strings
# are StringDType (or object) arrays.
def _leaf(values, dims=None, attributes=None):
    return (np.asarray(values), dims, attributes or {})


def _tag(values: np.ndarray) -> str:
    return "string" if values.dtype.kind in "OUT" else values.dtype.name


def _memory_id(tree) -> str:
    def node(item):
        if isinstance(item, tuple):
            values, dims, attributes = item
            hasher = _ArrayHasher(_tag(values), values.shape, dims=dims, attributes=attributes)
            hasher.update(values)
            return b"A", hasher.digest()
        children = {name: node(child) for name, child in item.get("children", {}).items()}
        return b"G", _group_digest(item.get("attributes", {}), children)

    return f"{LOGICAL_HASH}:{node(tree)[1].hex()}"


NAN = _nan64(0x7FF8000000000000)
VECTORS = {
    "V1": ({}, "b31fb78faf8af3773d2ad097eb301b9a988fe2928be42e36d28f40165e5a5fce"),
    "V2": (
        {"attributes": {"a": 1, "b": [1.0, "x", None, True]}},
        "29fff581653c3accdc0ef72da73a75700e97efb999fb51d4cd09833c9b2c60a4",
    ),
    "V3": (
        {"children": {"x": _leaf(np.array([1, -2, 3], np.int32), ["row"])}},
        "df60e0a83ad2fbfc85f8c242592ccb8833e74f459b4e9757deec81419fc765fe",
    ),
    "V4": (
        {"children": {"f": _leaf(np.array([0.0, -0.0, NAN, np.inf]), attributes={"units": "Jy"})}},
        "ff8b5491b187e184152d092bb3f26515a2b3cf8eebf8b6394ed43818c798b82f",
    ),
    "V5": (
        {"children": {"s": _leaf(np.array(["", "é𝄞", "a\x00"], dtype=np.dtypes.StringDType()))}},
        "635957fa858a02f486125e219690e19cd8a7b50fe0f54f3554a862f20301dec5",
    ),
    "V6": (
        {
            "children": {
                "b": _leaf(np.array([[True, False], [False, True]])),
                "c": _leaf(np.array(complex(-0.0, 1.5), np.complex64)),
            }
        },
        "238bf1f19702e9267c9f173d3f94855a1e545a18a24ee7cba2579df6cca9dc3b",
    ),
    "V7": (
        {
            "children": {
                "g": {
                    "attributes": {"k": "v"},
                    "children": {"y": _leaf((np.arange(262145) % 251).astype(np.uint8), ["n"])},
                }
            }
        },
        "e420de3773b19f3049f3e2ddc0ebfba94cbf927c063f5a8bb2a1a38ae1e8cc68",
    ),
}


@pytest.fixture
def zarr_lib():
    return pytest.importorskip("zarr", reason="logical_id needs msutils[msv4]")


def _write(zarr, root, tree, *, layout=None):
    """Write ``tree`` to a fresh Zarr v3 store under one layout."""
    from zarr.codecs import BytesCodec, TransposeCodec

    layout = layout or {}
    shutil.rmtree(root, ignore_errors=True)
    store = zarr.storage.LocalStore(str(root))
    zarr.create_group(store=store, zarr_format=3, attributes=tree.get("attributes", {}))

    def walk(item, prefix):
        for name, child in item.get("children", {}).items():
            path = f"{prefix}/{name}" if prefix else name
            if not isinstance(child, tuple):
                zarr.create_group(
                    store=store, path=path, zarr_format=3, attributes=child.get("attributes", {})
                )
                walk(child, path)
                continue
            values, dims, attributes = child
            string = values.dtype.kind in "OUT"
            chunking = layout.get("chunks", lambda shape: tuple(min(3, s) for s in shape))
            chunks = tuple(max(1, c) for c in chunking(values.shape))
            kwargs = {}
            dtype = str if string else values.dtype
            if string and layout.get("fixed_strings") and not any("\x00" in s for s in values.flat):
                dtype = np.dtype(f"<U{max(1, *(len(s) for s in values.flat))}")
            if values.ndim and not string and layout.get("shards"):
                kwargs["shards"] = tuple(2 * c for c in chunks)
            if values.ndim > 1 and not string and layout.get("transpose"):
                kwargs["filters"] = [TransposeCodec(order=tuple(reversed(range(values.ndim))))]
            if not string and dtype.itemsize > 1 and layout.get("endian"):
                kwargs["serializer"] = BytesCodec(endian=layout["endian"])
            if "compressors" in layout:
                kwargs["compressors"] = layout["compressors"]
            if "chunk_key_encoding" in layout:
                kwargs["chunk_key_encoding"] = layout["chunk_key_encoding"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                array = zarr.create_array(
                    store=store,
                    name=path,
                    shape=values.shape,
                    dtype=dtype,
                    chunks=chunks,
                    dimension_names=dims,
                    attributes=attributes,
                    zarr_format=3,
                    config={"write_empty_chunks": layout.get("write_empty_chunks", True)},
                    **kwargs,
                )
                array[...] = values

    walk(tree, "")
    if layout.get("consolidated"):
        zarr.consolidate_metadata(store)
    return root


def _meta(path):
    return json.loads((path / "zarr.json").read_text())


def _put_meta(path, document):
    (path / "zarr.json").write_text(json.dumps(document))


# --------------------------------------------------------------------------
# Frozen specification


def test_v3_intermediate_values_are_frozen():
    header = _encode_tlv(
        {
            "attributes": {},
            "block_elements": BLOCK_ELEMENTS,
            "dims": ["row"],
            "dtype": "int32",
            "shape": [3],
        }
    )
    assert header.hex() == (
        "4f05000000000000000a00000000000000617474726962757465734f00000000000000000e0000000000000062"
        "6c6f636b5f656c656d656e7473490600000000000000323632313434040000000000000064696d734c010000"
        "0000000000530300000000000000726f7705000000000000006474797065530500000000000000696e7433320500"
        "00000000000073686170654c010000000000000049010000000000000033"
    )
    block = _frame(b"block")
    block.update(_u64(0) + _u64(3) + np.array([1, -2, 3], "<i4").tobytes())
    block = block.digest()
    assert block.hex() == "d40268dd983687025cf15a17c14defeb674f18d9862fed430d61ea9fed6a98fc"
    array = _frame(b"array")
    array.update(_u64(len(header)) + header + _u64(1) + block)
    array = array.digest()
    assert array.hex() == "9c418c5a4e38ae1879a8a700dac041461e4be03414be91ded0eaa9bb25af3bfa"
    root = _group_digest({}, {"x": (b"A", array)})
    assert root.hex() == "df60e0a83ad2fbfc85f8c242592ccb8833e74f459b4e9757deec81419fc765fe"


@pytest.mark.parametrize("name", sorted(VECTORS))
def test_golden_vectors_in_memory(name):
    tree, expected = VECTORS[name]
    assert _memory_id(tree) == f"{LOGICAL_HASH}:{expected}"


@pytest.mark.parametrize("name", sorted(VECTORS))
def test_golden_vectors_through_a_store(zarr_lib, tmp_path, name):
    tree, expected = VECTORS[name]
    chunks = (lambda shape: tuple(min(100000, s) for s in shape)) if name == "V7" else None
    root = _write(zarr_lib, tmp_path / "v.zarr", tree, layout={"chunks": chunks} if chunks else {})
    assert logical_id(root) == f"{LOGICAL_HASH}:{expected}"


def test_tlv_canonicalises_objects_and_keeps_number_types():
    assert _encode_tlv({"b": 1, "a": 2}) == _encode_tlv({"a": 2, "b": 1})
    assert _encode_tlv(1) != _encode_tlv(1.0)
    assert _encode_tlv(0.0) != _encode_tlv(-0.0)
    assert _encode_tlv(2**70) == b"I" + _u64(22) + str(2**70).encode()
    with pytest.raises(LogicalIdRefusal) as exc:
        _encode_tlv("\ud800")
    assert exc.value.code == "zarr-metadata"


def test_hasher_is_independent_of_run_boundaries_and_workers():
    values = (np.arange(3 * BLOCK_ELEMENTS + 17) * 7).astype(np.int64)

    def digest(pieces, workers):
        with logical._pool(workers) as pool:
            hasher = _ArrayHasher("int64", values.shape, pool=pool)
            for piece in pieces:
                hasher.update(piece)
            return hasher.digest()

    whole = digest([values], 1)
    split = np.split(values, [1, BLOCK_ELEMENTS - 1, BLOCK_ELEMENTS + 5, 2 * BLOCK_ELEMENTS])
    assert digest(split, 1) == whole
    assert digest(split, 4) == whole
    big_endian = values.astype(">i8")
    assert digest([big_endian], 3) == whole


def test_hasher_refuses_value_conversion_and_wrong_counts():
    hasher = _ArrayHasher("float32", (2,))
    with pytest.raises(TypeError):
        hasher.update(np.zeros(2, np.float64))
    with pytest.raises(ValueError):
        hasher.update(np.zeros(3, np.float32))
    short = _ArrayHasher("int32", (2,))
    short.update(np.zeros(1, np.int32))
    with pytest.raises(ValueError):
        short.digest()


# --------------------------------------------------------------------------
# Invariance and sensitivity


def _rich_tree():
    """Every allowed dtype with the values a lossy path would disturb."""
    float32 = np.array(
        [0.0, -0.0, np.inf, -np.inf, _f32(0x7FC00001), _f32(0xFFC12345), _f32(0x7F800001), 1.5],
        dtype=np.float32,
    ).reshape(2, 4)
    float64 = np.array(
        [0.0, -0.0, _nan64(0x7FF8000000000001), -np.inf, 1e300, 5e-324], dtype=np.float64
    )
    complex64 = np.array(
        [complex(-0.0, 1.5), complex(_f32(0x7FC00001), -0.0), complex(1, 2), complex(np.inf, 0)],
        dtype=np.complex64,
    )
    return {
        "attributes": {"float": -0.0, "int": 1, "big": 2**70, "nested": {"list": [1.5, None]}},
        "children": {
            "flags": _leaf(np.array([[True, False, True], [False, False, True]]), ["row", "corr"]),
            "i8": _leaf(np.array([-128, 0, 127], np.int8)),
            "i16": _leaf(np.array([-(2**15), 0, 2**15 - 1], np.int16)),
            "i32": _leaf(np.array([-(2**31), 0, 0, 0, 2**31 - 1], np.int32)),
            "i64": _leaf(np.array([-(2**63), 0, 2**63 - 1], np.int64)),
            "u8": _leaf(np.array([0, 0, 0, 255], np.uint8)),
            "u16": _leaf(np.array([0, 2**16 - 1], np.uint16)),
            "u32": _leaf(np.array([0, 2**32 - 1], np.uint32)),
            "u64": _leaf(np.array([0, 2**64 - 1], np.uint64)),
            "f16": _leaf(np.array([-0.0, 65504.0], np.float16)),
            "f32": _leaf(float32, ["row", "chan"], {"units": "Jy", "scale": 1.0}),
            "f64": _leaf(float64),
            "c64": _leaf(complex64),
            "c128": _leaf(np.array([complex(-0.0, 3.0), complex(1.0, -0.0)], np.complex128)),
            "s_nul": _leaf(np.array(["a\x00", "\x00lead", "mid\x00dle"], dtype=object)),
            "s_plain": _leaf(np.array(["", "é𝄞", "x" * 70], dtype=object)),
            "scalar": _leaf(np.array(7, np.int32)),
            "empty": _leaf(np.zeros((0, 3), np.float64), ["row", None]),
            "sub": {
                "attributes": {"k": "v"},
                "children": {"deep": {"children": {"z": _leaf(np.arange(10, dtype=np.int16))}}},
            },
        },
    }


LAYOUTS = {
    "chunks-of-1": {"chunks": lambda shape: (1,) * len(shape)},
    "whole-array": {"chunks": lambda shape: shape},
    "gzip": {"compressors": "gzip"},
    "blosc-crc32c": {"compressors": "blosc"},
    "uncompressed": {"compressors": None},
    "sharded": {"shards": True, "chunks": lambda shape: (1,) * len(shape)},
    "transpose": {"transpose": True},
    "big-endian": {"endian": "big"},
    "v2-keys": {"chunk_key_encoding": {"name": "v2", "separator": "."}},
    "empty-chunks-elided": {"write_empty_chunks": False, "chunks": lambda shape: (1,) * len(shape)},
    "consolidated": {"consolidated": True},
    "fixed-width-strings": {"fixed_strings": True},
}


def _compressors(name):
    from zarr.codecs import BloscCodec, Crc32cCodec, GzipCodec

    return {
        "gzip": [GzipCodec(level=9)],
        "blosc": [BloscCodec(cname="lz4", shuffle="bitshuffle"), Crc32cCodec()],
    }.get(name)


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_lossless_relayout_keeps_the_id(zarr_lib, tmp_path, layout):
    tree = _rich_tree()
    options = dict(LAYOUTS[layout])
    if isinstance(options.get("compressors"), str):
        options["compressors"] = _compressors(options["compressors"])
    base = logical_id(_write(zarr_lib, tmp_path / "base.zarr", tree))
    assert base == _memory_id(tree)
    variant = logical_id(_write(zarr_lib, tmp_path / "variant.zarr", tree, layout=options))
    assert variant == base


def _change(tree, path, fn):
    tree = copy.deepcopy(tree)
    *parents, name = path
    node = tree
    for parent in parents:
        node = node["children"][parent]
    fn(node, name)
    return tree


def _set_value(index, value):
    def apply(node, name):
        values, dims, attributes = node["children"][name]
        values = values.copy()
        values[index] = value
        node["children"][name] = (values, dims, attributes)

    return apply


def _replace(leaf):
    def apply(node, name):
        node["children"][name] = leaf

    return apply


SENSITIVITY = {
    "one-ulp": (
        ["f32"],
        _set_value((1, 3), np.nextafter(np.float32(1.5), np.float32(2))),
    ),
    "nan-payload-bit": (["f32"], _set_value((1, 0), _f32(0x7FC00003))),
    "float-signed-zero": (["f64"], _set_value(1, 0.0)),
    "complex-signed-zero": (["c128"], _set_value(0, complex(0.0, 3.0))),
    "flipped-bool": (["flags"], _set_value((0, 1), True)),
    "string-trailing-nul": (["s_nul"], _set_value(0, "a")),
    "attribute-value": (["f32"], lambda n, k: n["children"][k][2].update(units="mJy")),
    "attribute-int-vs-float": (["f32"], lambda n, k: n["children"][k][2].update(scale=1)),
    "dimension-name": (
        ["f32"],
        lambda n, k: n["children"].__setitem__(k, (n["children"][k][0], ["row", "freq"], {})),
    ),
    "group-attribute": (["sub"], lambda n, k: n["children"][k]["attributes"].update(k="w")),
    "rename-array": (["u8"], lambda n, k: n["children"].__setitem__("u8x", n["children"].pop(k))),
    "add-empty-group": (["new"], lambda n, k: n["children"].__setitem__(k, {})),
    "reshape": (
        ["flags"],
        lambda n, k: n["children"].__setitem__(k, (n["children"][k][0].reshape(3, 2), None, {})),
    ),
    "int32-vs-int64": (
        ["i32"],
        lambda n, k: n["children"].__setitem__(k, (n["children"][k][0].astype(np.int64), None, {})),
    ),
}


@pytest.mark.parametrize("change", sorted(SENSITIVITY))
def test_content_changes_change_the_id(change):
    tree = _rich_tree()
    path, fn = SENSITIVITY[change]
    assert _memory_id(_change(tree, path, fn)) != _memory_id(tree)


def test_dimension_names_are_only_hashed_when_named():
    base = {"children": {"x": _leaf(np.arange(3, dtype=np.int8))}}
    unnamed = {"children": {"x": _leaf(np.arange(3, dtype=np.int8), [None])}}
    assert _memory_id(base) == _memory_id(unnamed)


def test_attribute_key_order_is_not_content(zarr_lib, tmp_path):
    tree = _rich_tree()
    root = _write(zarr_lib, tmp_path / "t.zarr", tree)
    base = logical_id(root)
    document = _meta(root / "f32")
    document["attributes"] = dict(reversed(list(document["attributes"].items())))
    _put_meta(root / "f32", document)
    assert logical_id(root) == base


def test_sensitivity_holds_through_a_store(zarr_lib, tmp_path):
    tree = _rich_tree()
    changed = _change(tree, *SENSITIVITY["one-ulp"])
    first = logical_id(_write(zarr_lib, tmp_path / "a.zarr", tree))
    second = logical_id(_write(zarr_lib, tmp_path / "b.zarr", changed))
    assert first != second


def test_zarr_eliding_signed_zero_chunks_is_detected_not_hidden(zarr_lib, tmp_path):
    """zarr's default drops a chunk 'equal' to fill_value, and for complex
    signed zeros that equality is not bitwise: the stored values change."""
    tree = {"children": {"c": _leaf(np.array([complex(-0.0, -0.0)] * 4, np.complex64))}}
    root = _write(
        zarr_lib,
        tmp_path / "elided.zarr",
        tree,
        layout={"write_empty_chunks": False, "chunks": lambda shape: (2,)},
    )
    assert logical_id(root) != _memory_id(tree)


@pytest.mark.parametrize("workers", [1, 8])
def test_workers_do_not_change_the_id(zarr_lib, tmp_path, workers):
    tree = {"children": {"y": _leaf(np.arange(3 * BLOCK_ELEMENTS + 5, dtype=np.float32))}}
    root = _write(zarr_lib, tmp_path / "w.zarr", tree, layout={"chunks": lambda s: (100000,)})
    assert logical_id(root, workers=workers) == _memory_id(tree)


def test_real_xarray_ms_export_survives_xarray_relayout(zarr_lib, tmp_path):
    pytest.importorskip("xarray_ms")
    import xarray as xr
    from msfactory import make_ms

    source = tmp_path / "source.ms"
    make_ms(source, nant=3, ntime=4, nspw=1, nfield=1, nscan_per_field=1, add_weight_spectrum=True)
    exported = tmp_path / "export.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_datatree(source, engine="xarray-ms:msv2", auto_corrs=True) as tree:
            tree.to_zarr(exported, mode="w", compute=True, consolidated=True)
    base = logical_id(exported)

    def relayout(ds):
        ds = ds.chunk({name: 1 for name, size in ds.sizes.items() if size > 0})
        for variable in ds.variables.values():
            for key in ("chunks", "preferred_chunks", "compressors", "filters", "shards"):
                variable.encoding.pop(key, None)
        return ds

    rechunked = tmp_path / "rechunked.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_datatree(exported, engine="zarr") as tree:
            tree.map_over_datasets(relayout).to_zarr(
                rechunked, mode="w", compute=True, consolidated=True
            )
    assert logical_id(rechunked) == base
    # One changed value in a real export changes the ID.
    weight = next(exported.glob("*/WEIGHT"))
    array = zarr_lib.open_array(zarr_lib.storage.LocalStore(str(weight)), mode="r+")
    values = array[...]
    values.flat[0] = np.float32(2.5) if values.flat[0] != 2.5 else np.float32(3.5)
    array[...] = values
    zarr_lib.consolidate_metadata(str(exported))
    assert logical_id(exported) != base


def test_msutils_own_msv4_writer_output_passes_the_reading_policy(zarr_lib, tmp_path):
    """to_msv4 (xradio) writes Zarr v3 with fixed-width strings and nested
    consolidated metadata; all of it must be inside the reading policy."""
    pytest.importorskip("xradio")
    from msfactory import make_ms

    from msutils import to_msv4

    source = tmp_path / "source.ms"
    make_ms(source, nant=3, ntime=4, nspw=1, nfield=1, nscan_per_field=1)
    state = tmp_path / "state.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_msv4(str(source), str(state))
    assert _meta(state)["zarr_format"] == 3
    assert "consolidated_metadata" in _meta(state)
    identity = logical_id(state)
    assert identity == logical_id(state, workers=1)


# --------------------------------------------------------------------------
# Reading policy


def _simple(zarr, root, **layout):
    tree = {
        "children": {
            "a": _leaf(np.arange(6, dtype=np.float64).reshape(2, 3), ["x", "y"]),
            "g": {"children": {"s": _leaf(np.array(["p", "q"], dtype=object))}},
        }
    }
    return _write(zarr, root, tree, layout=layout)


def _refused(root, code, **kwargs):
    with pytest.raises(LogicalIdRefusal) as exc:
        logical_id(root, **kwargs)
    assert exc.value.code == code, exc.value
    return exc.value


def test_symlinked_chunk_file_is_refused(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    chunk = next((root / "a" / "c").rglob("0"))
    if chunk.is_dir():
        chunk = next(p for p in chunk.rglob("*") if p.is_file())
    external = tmp_path / "external"
    external.write_bytes(chunk.read_bytes())
    chunk.unlink()
    chunk.symlink_to(external)
    _refused(root, "zarr-symlink")


def test_symlinked_group_and_root_are_refused(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    alias = tmp_path / "alias.zarr"
    alias.symlink_to(root, target_is_directory=True)
    _refused(alias, "zarr-symlink")
    shutil.move(root / "g", tmp_path / "g")
    (root / "g").symlink_to(tmp_path / "g", target_is_directory=True)
    _refused(root, "zarr-symlink")


def test_fifo_is_refused(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    os.mkfifo(root / "a" / "c" / "pipe")
    _refused(root, "zarr-special-file")


@pytest.mark.parametrize(
    "entry", ["notes.txt", ".hidden", "c/9/0", "c/0/5", "c/0/1/7", "c.0.0", "sub/zarr.json"]
)
def test_stray_and_out_of_grid_entries_are_refused(zarr_lib, tmp_path, entry):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    path = root / "a" / entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    _refused(root, "zarr-extra-entry")


def test_implicit_group_directory_is_refused(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    (root / "implicit").mkdir()
    _refused(root, "zarr-extra-entry")


def test_zarr_v2_and_mixed_trees_are_refused(zarr_lib, tmp_path):
    v2 = tmp_path / "v2.zarr"
    group = zarr_lib.open_group(str(v2), mode="w", zarr_format=2)
    group.create_array("a", shape=(2,), dtype="int32")
    refusal = _refused(v2, "zarr-format")
    assert "zarr_format=3" in refusal.reason
    mixed = _simple(zarr_lib, tmp_path / "mixed.zarr")
    (mixed / "g" / ".zattrs").write_text("{}")
    _refused(mixed, "zarr-format")
    three = _simple(zarr_lib, tmp_path / "three.zarr")
    document = _meta(three)
    document["zarr_format"] = 99
    _put_meta(three, document)
    _refused(three, "zarr-format")


def test_root_array_and_missing_root_are_refused(zarr_lib, tmp_path):
    root = tmp_path / "array.zarr"
    zarr_lib.create_array(
        store=zarr_lib.storage.LocalStore(str(root)), shape=(2,), dtype="int8", zarr_format=3
    )
    _refused(root, "zarr-root")
    _refused(tmp_path / "missing.zarr", "zarr-root")


def _edit_array(key, value):
    def edit(document):
        if value is _DELETE:
            document.pop(key)
        else:
            document[key] = value

    return edit


_DELETE = object()
METADATA_EDITS = {
    "pickle-codec": (
        _edit_array(
            "codecs",
            [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "numcodecs.pickle"},
            ],
        ),
        "zarr-codec",
    ),
    "quantize-codec": (
        _edit_array(
            "codecs",
            [
                {"name": "numcodecs.quantize", "configuration": {"digits": 2}},
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
        ),
        "zarr-codec",
    ),
    "bytes-without-endian": (_edit_array("codecs", [{"name": "bytes"}]), "zarr-codec"),
    "raw-bits-dtype": (_edit_array("data_type", "r64"), "zarr-dtype"),
    "datetime-dtype": (
        _edit_array("data_type", {"name": "numpy.datetime64", "configuration": {}}),
        "zarr-dtype",
    ),
    "storage-transformers": (
        _edit_array("storage_transformers", [{"name": "x", "configuration": {}}]),
        "zarr-metadata",
    ),
    "must-understand-extension": (
        _edit_array("extension", {"must_understand": True}),
        "zarr-metadata",
    ),
    "unknown-key": (_edit_array("extension", 1), "zarr-metadata"),
    "huge-shape": (_edit_array("shape", [2**40, 2**40]), "zarr-metadata"),
    "missing-codecs": (_edit_array("codecs", _DELETE), "zarr-metadata"),
    "lone-surrogate-attribute": (_edit_array("attributes", {"k": "\ud800"}), "zarr-metadata"),
    "irregular-grid": (
        _edit_array("chunk_grid", {"name": "rectilinear", "configuration": {}}),
        "zarr-metadata",
    ),
}


@pytest.mark.parametrize("edit", sorted(METADATA_EDITS))
def test_metadata_outside_the_policy_is_refused(zarr_lib, tmp_path, edit):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    fn, code = METADATA_EDITS[edit]
    document = _meta(root / "a")
    fn(document)
    _put_meta(root / "a", document)
    _refused(root, code)


def test_extension_that_need_not_be_understood_is_ignored(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    base = logical_id(root)
    document = _meta(root / "a")
    document["extension"] = {"must_understand": False, "anything": [1, 2]}
    _put_meta(root / "a", document)
    assert logical_id(root) == base


def test_duplicate_json_keys_are_refused(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    text = (root / "g" / "zarr.json").read_text()
    (root / "g" / "zarr.json").write_text(
        text.replace('"attributes"', '"attributes": {}, "attributes"', 1)
    )
    _refused(root, "zarr-metadata")


def test_reserved_names_are_refused(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    shutil.copytree(root / "g", root / "__reserved")
    _refused(root, "zarr-name")


def test_metadata_and_chunk_ceilings(zarr_lib, tmp_path, monkeypatch):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    monkeypatch.setattr(logical, "_MAX_ZARR_METADATA_BYTES", 10)
    _refused(root, "zarr-metadata")
    monkeypatch.undo()
    monkeypatch.setattr(logical, "_MAX_CHUNK_BYTES", 8)
    _refused(root, "zarr-chunk-size")


def test_stale_consolidated_metadata_is_refused(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr", consolidated=True)
    base = logical_id(root)
    assert base == logical_id(_simple(zarr_lib, tmp_path / "plain.zarr"))
    document = _meta(root / "a")
    document["attributes"] = {"edited": True}
    _put_meta(root / "a", document)
    _refused(root, "zarr-consolidated-stale")
    root = _simple(zarr_lib, tmp_path / "missing.zarr", consolidated=True)
    zarr_lib.create_array(
        store=zarr_lib.storage.LocalStore(str(root)), name="late", shape=(1,), dtype="int8"
    )
    _refused(root, "zarr-consolidated-stale")


def test_corrupt_chunk_is_unreadable(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    chunk = next(path for path in (root / "a" / "c").rglob("*") if path.is_file())
    chunk.write_bytes(chunk.read_bytes()[:5])
    _refused(root, "zarr-unreadable")


def test_metadata_changed_during_read_is_refused(zarr_lib, tmp_path, monkeypatch):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    original = logical._slabs

    def slabs_then_edit(*args):
        yield from original(*args)
        with open(root / "a" / "zarr.json", "a") as stream:
            stream.write(" ")

    monkeypatch.setattr(logical, "_slabs", slabs_then_edit)
    _refused(root, "zarr-changed-during-read")


def test_workers_must_be_positive(zarr_lib, tmp_path):
    root = _simple(zarr_lib, tmp_path / "t.zarr")
    with pytest.raises(ValueError):
        logical_id(root, workers=0)
