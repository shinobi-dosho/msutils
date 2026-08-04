# msutils -- design conventions

Guidance for coding agents (and humans) working in this repository. Read this
before changing the readers, the `MSInfo` model, or anything that touches
casacore tables. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, testing and
the contribution workflow; this file states the conventions the code holds to
and why.

## What this is

`msutils` is a Python library and CLI of **everyday Measurement Set operations** for radio-astronomy pipelines — inspect, subset, average, manage columns and flags. Explicitly *not* calibration or imaging. Public functions are exposed on the package root (`import msutils; msutils.msinfo(ms)`) and mirrored by a click CLI (`msutils` console script → `msutils.cli:cli`).

The headline entry point is **`msinfo`**, which returns a typed `MSInfo`. The old `summary()` is a deprecated adapter over it (`FutureWarning`); do not build anything new on `summary`.

## Commands

```bash
pip install -e ".[all]"   # everything; build backend is hatchling
pip install pytest ruff

pytest                    # full suite; needs no simulator (see Testing below)
pytest tests/test_msinfo.py::test_scan_duration_includes_final_integration
ruff check .              # lint gate: syntax + undefined names only (E9, F63, F7, F82)
```

`src/` layout; the package is `src/msutils/`. Packaging is entirely in `pyproject.toml` — no `setup.py`/`requirements.txt`/`MANIFEST.in`. Releases publish to PyPI on GitHub release creation (`.github/workflows/publish.yml`); bump `version` in `pyproject.toml`.

## Architecture

```
src/msutils/
  info/            msinfo + the MSInfo model          <- the core abstraction
    _model.py        dataclasses, Registry, formatting helpers
    _msv2.py         casacore/TaQL reader (default)
    _msv4.py         MSv4-schema reader (zarr / xradio / xarray-ms engines)
    _render.py       listobs-style text report
  _ms.py           column ops: addcol, delcol, renamecol, copycol, sumcols, addnoise
  subset.py        subset() and average()
  flags.py         flag version backup/restore
  flagstats.py     TaQL flag statistics  (+ _flagrender.py, _flagplot.py)
  diagnostics.py   du(), check(), taql()
  convert.py       MSv2 -> MSv4 via xradio
  _tables.py       open_table / query context managers
  _compat.py       the deprecated summary()
  cli.py           click CLI
```

### Non-negotiable patterns

- **Aggregate in TaQL, not in Python.** This is the single most important rule here. The 2.x `summary()` ran one `table.query()` per field and another per scan — O(fields × scans) full scans. `_msv2.py` does one `GROUPBY` pass and folds the result in Python. When adding metadata, extend the aggregate in `_msv2._AGGREGATES`; do not add another pass.
- **TaQL re-reads a column per expression.** Each extra `UVW`/`FLAG` term in a `SELECT` costs another pass over bulk data (0.17 s → 0.64 s for one `SUMSQR(UVW)` on 595k rows). This is why bulk-column aggregates live at `level="data"` and index-column ones at `level="full"`.
- **Bind tables as `$1`, never interpolate paths.** TaQL rejects bare paths containing `.` or `::`, so `FROM ./obs.ms` is a parse error. Use `_tables.query(cmd, [tab])`.
- **Always close tables.** casacore holds locks; a leaked handle makes later operations block. Everything goes through `_tables.open_table` / `query` / `closing_table`, which close on exit. (A per-SPW handle leak in the since-removed `weights.write_toms` was a real 2.x bug.)
- **`ack=False`.** `open_table` suppresses casacore's "Successful readonly open" banner so stdout stays clean for piped JSON.
- **Chunked read/write by SPW.** Bulk column writes iterate `set(tab.getcol('DATA_DESC_ID'))`, then rows in `rowchunk` blocks via `getcol/putcol(col, row0, nr)`. MSs do not fit in memory.
- **`addcol(clone='DATA')`** is the standard way to create a column matching an existing one; `sumcols` and `copycol` rely on it, and it is idempotent (returns `'exists'`).
- **`DATA_DESC_ID` is not `SPECTRAL_WINDOW_ID`.** Always map through `DATA_DESCRIPTION`. Conflating them is a quiet source of wrong results, and `MSInfo.data_descriptions` exists to make the mapping explicit.

### The MSInfo model

`info/_model.py` is the contract between readers and consumers. Conventions to preserve:

- Collections are `Registry` objects, addressable by **id or name** (`info.fields["DEEP_2"]`), where integer keys are ids, *not* positions — subset MSs keep non-contiguous ids.
- **Derived values are `@property`, listed in `_derived`**, never stored fields. `to_dict()` includes them. This keeps the JSON a faithful record of what was read and stops values drifting.
- Times are **MJD seconds** everywhere in the model. MSv4 stores unix seconds; `_msv4._mjd_seconds` converts on the way in.
- Bump `SCHEMA_VERSION` on any breaking JSON change.

### Readers and engines

`msinfo(path, engine=...)` picks a reader. MSv2 defaults to `casacore` (TaQL). `_msv4.py` reads the MSv4 *schema* from three sources — `zarr` (plain xarray, no xradio), `xradio`, and `xarray-ms` (an MSv2 viewed through the MSv4 schema, no conversion) — all folding into the same `MSInfo`.

**Keep casacore/TaQL the MSv2 default.** It is several times faster for metadata, needs nothing beyond the base install, and opens MSs that xarray-ms rejects (e.g. an empty `FEED`). A malformed MS is exactly the one a diagnostic tool must be able to open. xradio is needed only to *write* MSv4.

## Dependencies

Base install is `numpy` + `python-casacore` + `click`, and it covers `msinfo`, all column ops, `subset`, flags, `flagstats` and diagnostics. Extras: `plots` (matplotlib), `average` (codex-africanus), `msv4` (xarray, zarr — reading), `xarray-ms`, `convert` (xradio — writing). `tests/test_import.py` asserts in a subprocess that importing `msutils` pulls in none of them; keep optional imports inside functions.

`python-casacore` is a regular dependency (pip wheels bundle casacore), so `pip install .` is self-contained. Python ≥ 3.11.

## Testing

`tests/msfactory.py` builds structurally realistic MSs with python-casacore alone — several fields with distinct intents, several scans, several SPWs, a real `DATA_DESCRIPTION` mapping and a populated `FEED`. This replaced `simms`, whose absence used to make the MS-backed tests silently skip, including in CI.

- The synthetic MS is built to exercise the cases 2.x got wrong. When fixing a metadata bug, make sure the fixture can actually distinguish right from wrong — `patterned_ms` exists because the default fixture flags whole rows, which cannot tell a correct per-correlation breakdown from the broken one.
- `tests/test_regressions.py` pins the six audited bugs; each test failed before its fix.
- Tests for optional stacks `importorskip`. CI runs the suite twice: bare (proving the base install is self-sufficient) and with `[all]`.
- Do not assert on the contents of a column created by `addcol` without `init_with` — it is genuinely uninitialised, and comparisons pass or fail by luck.
