# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`msutils` is a Python library of CASA Measurement Set (MS) manipulation tools used in radio-astronomy pipelines. It is a pure library — there is no CLI entry point. Functions are imported and called (`from msutils import msutils`).

The import package was renamed from `MSUtils` → `msutils`. `MSUtils` still works as a **deprecated alias** (`src/MSUtils/__init__.py` emits a `FutureWarning` — shown by default, unlike `DeprecationWarning` — then swaps `sys.modules["MSUtils"]` for the real `msutils` package so submodule access transparently resolves). Use `msutils` in all new code.

## Commands

```bash
pip install .            # install (editable: pip install -e .); build backend is hatchling
pip install pytest ruff  # dev tooling (or: pip install with the [dependency-groups] dev set)

pytest                   # run the test suite (config in pyproject.toml [tool.pytest.ini_options])
pytest tests/test_import.py::test_import_msutils   # run a single test

ruff check .             # lint; gate mirrors the old flake8 hard-fail set (see pyproject [tool.ruff.lint])
```

The package uses a **`src/` layout**: the real package is `src/msutils/`, plus the `src/MSUtils/` alias shim; both are shipped (`[tool.hatch.build.targets.wheel] packages`). The distribution/PyPI name is `msutils`. Packaging is configured entirely in `pyproject.toml` (hatchling) — there is no `setup.py`/`setup.cfg`/`requirements.txt`/`MANIFEST.in`. Internal imports use relative form (`from . import msutils`) so the real package never routes through the deprecated alias.

Tests are thin: `tests/` holds import/smoke checks only, because almost every function reads/writes casacore tables. When adding real functionality, verify against a real `.ms`. The ruff gate only enforces syntax + undefined-name rules (`E9`, `F63`, `F7`, `F82`).

Releases publish to PyPI automatically on GitHub release creation (`.github/workflows/publish.yml`). Bump `version` in `pyproject.toml` for a release.

## Critical dependency detail

The core module `msutils.py` and `imp_plotter.py`/`ClassESW.py` import casacore via the **legacy `pyrap` namespace** (`from pyrap.tables import table`), while `flag_stats.py` uses the **modern `casacore` / `daskms`** namespace. Both refer to the same underlying `python-casacore` install (`pyrap` is an alias). `python-casacore` is a regular `pyproject.toml` dependency (pip wheels bundle the casacore libs), so `pip install .` is self-contained.

**Dependencies are split into extras** (`[project.optional-dependencies]`): the base install is only `numpy` + `python-casacore` (enough for `msutils.msutils`'s column ops). Heavier stacks are opt-in — `msutils[flagstats]` pulls `dask-ms`+`bokeh` (for `flag_stats`), `msutils[plots]` pulls `matplotlib`+`scipy` (for `ClassESW`/`imp_plotter`), and `msutils[all]` gets everything. So importing `flag_stats`/`ClassESW`/`imp_plotter` without the matching extra fails by design.

## Architecture

Four largely independent modules under `src/msutils/`, unified only by the shared casacore-table access pattern and a shared logger (`_log.create_logger`, one handler per named logger):

- **`msutils.py`** — the core (base-install only, needs just numpy+casacore). Column-level MS operations: `summary` (dumps MS metadata to a JSON-serializable dict), `addcol`, `sumcols`, `copycol`, `addnoise`, `compute_vis_noise`, `verify_antpos`. Defines `STOKES_TYPES` (correlation-code → label map) reused by other modules. (The old `prep()` — which shelled out to `addbitflagcol`/`flag-ms.py` and used `distutils` — was removed.)

- **`flag_stats.py`** — flag statistics computed **out-of-core with dask-ms + dask.array**, results plotted to interactive Bokeh HTML. The public entry points are `save_statistics` (→ JSON) and `plot_statistics` (→ HTML). Under the hood, per-axis functions (`antenna_flags_field`, `scan_flags_field`, `source_flags_field`, `correlation_flags_field`) use `xds_from_ms` grouped by a column, then `da.blockwise` + `da.reduction` with the module-level `_get_flags`/`_get_ant_flags`/`_chunk`/`_combine`/`_aggregate` helpers to accumulate `[flagged_sum, total_count]` pairs.

- **`ClassESW.py`** — `MSNoise` class: estimates per-channel visibility noise/weights from an SEFD-vs-frequency profile (fits a polynomial or spline), then writes `WEIGHT`/`WEIGHT_SPECTRUM` back to the MS. `MEERKAT_SEFD` is a built-in reference profile. Depends on `msutils.summary` and `msutils.addcol`.

- **`imp_plotter.py`** — matplotlib plotters for calibration/gain tables (not MS data). `gain_plotter(caltable, typ, ...)` dispatches on `typ` ∈ {`'delay'`, `'gain'`, `'bandpass'`} to `plot_delay_table` / `plot_gain_table` / `plot_bandpass_table`, reading `CPARAM`/`FPARAM` solution columns. Forces the `Agg` backend at import.

### Recurring patterns to preserve

- **Chunked read/write by SPW.** Bulk column operations iterate over `set(tab.getcol('DATA_DESC_ID'))` (spectral windows), then loop rows in `rowchunk`-sized blocks via `getcol/putcol(col, row0, nr)`. Follow this when adding column-writing functions — MS files are too large to load whole.
- **Always close tables.** casacore holds locks; every opened `table(...)` / subtable must be `.close()`d or subsequent operations hang (see the explicit note in `compute_vis_noise`).
- **`addcol(clone='DATA')`** is the standard way to create a new column matching an existing column's shape/type; `sumcols`/`copycol`/`MSNoise.write_toms` all rely on it and it is idempotent (returns `'exists'` if the column is already there).
