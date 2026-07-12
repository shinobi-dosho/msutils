# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`msutils` is a Python library of CASA Measurement Set (MS) manipulation tools used in radio-astronomy pipelines. The core column operations are exposed **directly on the package**: `import msutils; msutils.summary(msname)`, `msutils.addcol(...)`, etc. There is also a click **CLI** (`msutils` console script → `msutils.cli:cli`) with subcommands `summary`/`addcol`/`copycol`/`sumcols`/`addnoise`/`flagstats`.

**Two deprecated aliases** (both emit `FutureWarning`, kept for one release):
- The old **capitalised `MSUtils` package** — `src/MSUtils/__init__.py` warns, then swaps `sys.modules["MSUtils"]` for the real `msutils` package.
- The old **`msutils.msutils` submodule** — the functions used to live there; `src/msutils/msutils.py` now re-exports them from `_ms` and warns. `src/msutils/__init__.py` also has a `__getattr__` so lazy `msutils.msutils` attribute access triggers the (deprecated) import.

Use the flat `msutils.<func>` form in all new code.

## Commands

```bash
pip install .            # install (editable: pip install -e .); build backend is hatchling
pip install pytest ruff  # dev tooling (or: pip install with the [dependency-groups] dev set)

pytest                   # run the test suite (config in pyproject.toml [tool.pytest.ini_options])
pytest tests/test_import.py::test_flat_api          # run a single test

ruff check .             # lint; gate mirrors the old flake8 hard-fail set (see pyproject [tool.ruff.lint])
```

The package uses a **`src/` layout**: the real package is `src/msutils/`, plus the `src/MSUtils/` alias shim; both are shipped (`[tool.hatch.build.targets.wheel] packages`). The distribution/PyPI name is `msutils`. Packaging is configured entirely in `pyproject.toml` (hatchling) — there is no `setup.py`/`setup.cfg`/`requirements.txt`/`MANIFEST.in`. Internal code imports the core as `from . import _ms` (never the deprecated `msutils.msutils` shim), and opens tables via `_tables.open_table` (a context manager that guarantees `.close()`).

Tests: `tests/test_import.py` is import/API smoke checks (no deps). `tests/test_ms_ops.py` exercises the core column ops end-to-end against a **synthetic MS** built by the `ms` fixture (`conftest.py`), which uses `simms` (7-antenna kat-7, 3 times × 4 chans). `simms` is a heavy git-installed dependency in the `test` dependency group; when it's absent the MS-backed tests **skip** rather than fail (so a bare `pytest` still passes). CI installs `simms` as a non-fatal step. The ruff gate only enforces syntax + undefined-name rules (`E9`, `F63`, `F7`, `F82`).

Releases publish to PyPI automatically on GitHub release creation (`.github/workflows/publish.yml`). Bump `version` in `pyproject.toml` for a release.

## Critical dependency detail

All modules now use the **modern `casacore` namespace** (`from casacore.tables import table`; `flag_stats` additionally uses `daskms`). The legacy `pyrap` alias is gone. `python-casacore` is a regular `pyproject.toml` dependency (pip wheels bundle the casacore libs), so `pip install .` is self-contained.

**Dependencies are split into extras** (`[project.optional-dependencies]`): the base install is only `numpy` + `python-casacore` (enough for the core `_ms` column ops). Heavier stacks are opt-in — `msutils[flagstats]` pulls `dask-ms`+`matplotlib` (for `flag_stats`), `msutils[plots]` pulls `matplotlib`+`scipy` (for `ClassESW`), and `msutils[all]` gets everything. So importing `flag_stats`/`ClassESW` without the matching extra fails by design.

## Architecture

Modules under `src/msutils/`, unified by a shared table helper (`_tables.open_table`) and logger (`_log.create_logger`, one handler per named logger):

- **`_ms.py`** — the core (base-install only, needs just numpy+casacore), re-exported at the package root. Type-hinted column-level MS operations: `summary` (dumps MS metadata to a JSON-serializable dict), `addcol`, `sumcols`, `copycol`, `addnoise`, `compute_vis_noise`, `verify_antpos`. Defines `STOKES_TYPES` (correlation-code → label map) reused by other modules and lists the public API in `__all__`. (The old `prep()` — which shelled out to `addbitflagcol`/`flag-ms.py` and used `distutils` — was removed.)

- **`flag_stats.py`** — flag statistics computed **out-of-core with dask-ms + dask.array**, results plotted to a **matplotlib PNG** (2×2 bar-chart summary; `Agg` backend). The public entry points are `save_statistics` (→ JSON) and `plot_statistics` (→ PNG via `plotfile=`, + JSON via `outfile=`). Under the hood, per-axis functions (`antenna_flags_field`, `scan_flags_field`, `source_flags_field`, `correlation_flags_field`) use `xds_from_ms` grouped by a column, then `da.blockwise` + `da.reduction` with the module-level `_get_flags`/`_get_ant_flags`/`_chunk`/`_combine`/`_aggregate` helpers to accumulate `[flagged_sum, total_count]` pairs.

- **`ClassESW.py`** — `MSNoise` class: estimates per-channel visibility noise/weights from an SEFD-vs-frequency profile (fits a polynomial or spline), then writes `WEIGHT`/`WEIGHT_SPECTRUM` back to the MS. `MEERKAT_SEFD` is a built-in reference profile. Depends on `msutils.summary` and `msutils.addcol`.

(`imp_plotter.py` — matplotlib gain/cal-table plotters — was removed in 2.x; gain-table plotting is superseded by `ragavi-gains`, and gain tables aren't MSs.)

### Recurring patterns to preserve

- **Chunked read/write by SPW.** Bulk column operations iterate over `set(tab.getcol('DATA_DESC_ID'))` (spectral windows), then loop rows in `rowchunk`-sized blocks via `getcol/putcol(col, row0, nr)`. Follow this when adding column-writing functions — MS files are too large to load whole.
- **Always close tables.** casacore holds locks; every opened `table(...)` / subtable must be `.close()`d or subsequent operations hang (see the explicit note in `compute_vis_noise`).
- **`addcol(clone='DATA')`** is the standard way to create a new column matching an existing column's shape/type; `sumcols`/`copycol`/`MSNoise.write_toms` all rely on it and it is idempotent (returns `'exists'` if the column is already there).
