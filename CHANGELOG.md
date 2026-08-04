# Changelog

## 3.0.0

A rewrite of the metadata layer, a set of new everyday tools, and fixes for
the bugs found in a full audit of 2.x.

### `msinfo` replaces `summary`

`msutils.msinfo()` returns a typed, JSON-serialisable `MSInfo` whose
collections are addressable by name as well as by id
(`info.fields["PKS1934-638"]`, `info.spws[0].chan_width`,
`info.antennas["m003"].latitude`). It has a `render()` method producing a
`listobs`-style report, a versioned `to_dict()`, and three cost levels
(`meta` / `full` / `data`) so inspecting a large MS need not read it.

`summary()` still returns the old dict shape but emits a `FutureWarning`.

**Values `summary()` reported incorrectly, now fixed** (they change for
`summary()` too, since it is now built on `msinfo`):

- **Observing intents** were the raw `STATE::OBS_MODE` column presented as
  per-field data, which only lined up by coincidence. Now resolved per field
  and per scan through `STATE_ID`.
- **Scan durations** were `max(TIME) - min(TIME)`, understating every scan by
  one integration and reporting `0` for single-timestamp scans. The final
  integration is now included. `FIELD.PERIOD` follows.
- **`MAXBL`** used only the *u* and *v* components, making it a projected uv
  distance under a baseline-length name. `msinfo` reports `max_baseline` (the
  physical antenna separation) and `max_uv_distance` separately.
- **`EXPOSURE`** was row 0's value presented as an MS-wide property. It is
  now the shortest integration time present; the legacy dict holds one scalar,
  so `msinfo` exposes the full set as `integration_times`.
- **Multiple polarization setups** were ignored; only `POLARIZATION` row 0 was
  read.

**Newly available**: `DATA_DESC_ID` → (SPW, polarization) mapping, channel
widths (previously omitted from the hard-coded column list entirely),
telescope/observer/project, UTC time ranges, per-field and per-scan row counts
and time ranges, sexagesimal phase centres, antenna geodetic positions and
diameters, baseline counts, column inventory with shapes and sizes, and MS
size on disk.

**Performance**: `summary()` ran one full table scan per field and another per
scan, so its cost grew as O(fields × scans) — 0.55 s for a 595k-row MS with 6
scans, 1.53 s for the same MS with 60. `msinfo(level="full")` does one TaQL
`GROUPBY` in 0.37 s, flat in the number of scans; `level="meta"` is ~10 ms and
independent of MS size.

### MSv4

`msinfo` reads MSv4 processing sets into the same `MSInfo` as MSv2, so the
same code works on both. Reading needs only `msutils[msv4]` (xarray + zarr) —
xradio is required only to *write* MSv4, under `msutils[convert]`, which
backs the new `msutils convert` / `msutils.convert.to_msv4()`.

An MSv2 can also be read *through* the MSv4 schema without converting it, via
`msinfo(path, engine="xarray-ms")` and the `msutils[xarray-ms]` extra. The
default MSv2 engine remains casacore/TaQL: it is several times faster for
metadata, needs no extra dependencies, and opens Measurement Sets that
stricter readers reject — which matters for a tool whose job includes
diagnosing broken MSs.

### Fixed

- `addnoise()` with per-channel noise crashed on any MS needing more than one
  row chunk (`ValueError: non-broadcastable output operand`): the noise array
  was reshaped inside the row loop, gaining an axis per chunk.
- `addcol()` raised `UnboundLocalError: data_desc` when called without
  `clone`/`shape`, and its documented `data_desc_type` parameter was never
  read — the code branched on `valuetype == 'scalar'` instead, so scalar
  columns could not be created at all. Both fixed; invalid combinations now
  raise a clear `ValueError`.
- `flagstats` (was `flag_stats`) reported the **same flag count for every
  correlation** — its reduction kernel wrote the total into every slot — and
  inflated per-scan and per-field `sum`/`counts` by the number of groups.
- `flagstats` computed a field selection for the antenna axis and then never
  used it, so `fields=` was silently ignored there.

### Added

- `delcol()` / `renamecol()` — removing `CORRECTED_DATA` after calibration is
  a routine chore with no previous tool. Required MSv2 columns are protected
  unless `force=True`.
- `subset()` — write a selected copy by field/SPW/scan/antenna/TaQL. Keeps the
  original ids rather than renumbering them as CASA `split` does.
- `average()` — time and channel averaging, delegating the binning to
  `codex-africanus`. Reconciles `FLAG`/`FLAG_ROW` inconsistencies that
  africanus otherwise rejects.
- `flag_backup()` / `flag_restore()` / `flag_versions()` / `flag_delete()` —
  a `flagmanager` equivalent, storing versions in `<ms>.flagversions/`.
- `du()` — on-disk size by storage manager and subtable.
- `check()` — MSv2 conformance: required columns and subtables, and whether
  `FIELD_ID`/`DATA_DESC_ID`/`ANTENNA1`/`ANTENNA2` reference rows that exist.
- `taql()` — run a TaQL query and get the columns back as arrays.
- CLI commands for all of the above, plus `msutils info`.

### Changed

- `flagstats` is rewritten on TaQL and **no longer depends on `dask-ms`**. One
  `GNTRUES(FLAG)` aggregation per axis replaces a pass per antenna, and adds
  per-channel and per-baseline breakdowns the old code could not produce. Its
  JSON layout changed accordingly.
- The base install now covers `msinfo`, columns, `subset`, flags, `flagstats`
  and diagnostics on `numpy` + `python-casacore` + `click` alone. Extras are
  `plots`, `average`, `msv4`, `xarray-ms`, `convert`.
- Python ≥ 3.11 (was ≥ 3.8).
- casacore's "Successful readonly open of..." banner is suppressed, so stdout
  stays clean for piped JSON.
- `casacore.measures` is imported lazily, cutting package import time.

### Removed

- The 1.x aliases deprecated in 2.0, as announced: the `MSUtils` package, the
  `msutils.msutils` submodule, `msutils.flag_stats` and `msutils.ClassESW`.
  Use `msutils.<function>` and `msutils.flagstats`.
- `msutils.weights` (`MSNoise`, `MEERKAT_SEFD`). Estimating weights by fitting
  an SEFD profile is data processing, not an everyday MS operation, and it sat
  outside what this package is for. It also carried two of the audited bugs
  (`estimate_weights(plot_stats=False)` raised `UnboundLocalError`, and
  `write_toms()` leaked a casacore table handle per spectral window). Pin
  `msutils<3` if you depend on it. Dropping it also removed the `scipy`
  dependency from the `plots` extra.
- `flagstats.wgs84_to_ecef()`, an internal helper that had leaked into the
  public API and interpreted its documented degrees as radians. Use
  `msutils.info.geodetic_to_itrf()` / `itrf_to_geodetic()`.

### Testing

The suite builds its own Measurement Sets with `python-casacore`
(`tests/msfactory.py`), replacing the heavy git-installed `simms` dependency.
Previously the MS-backed tests skipped whenever `simms` was absent — including
in CI — so most of the suite was not actually running. It now runs everywhere:
208 tests with all extras installed, 160 on a bare install.

## 2.0.0

- Renamed the package to lowercase `msutils`; flattened the public API onto
  the package root; renamed `flag_stats` → `flagstats` and `ClassESW` →
  `weights`; added the click CLI; moved to the modern `casacore` namespace and
  a `src/` layout; removed `imp_plotter` and `prep()`.
