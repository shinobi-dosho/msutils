# msutils -- design conventions

Guidance for coding agents (and humans) working in this repository. Read this
before changing the readers, the `MSInfo` model, or anything that touches
casacore tables. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, testing and
the contribution workflow; this file states the conventions the code holds to
and why.

Organisation-wide conventions live in
[`shinobi-dosho/.github`](https://github.com/shinobi-dosho/.github/blob/main/AGENTS.md) — this file states what is
specific to `msutils` and wins where the two disagree.

## What this is

`msutils` is a Python library and CLI of **everyday Measurement Set operations** for radio-astronomy pipelines — inspect, subset, average, manage columns and flags — plus, since `gainutils`, the same kind of everyday operations on the *other* thing a pipeline produces: **calibration gain tables**. It does not *solve* for calibration and does not image; it manipulates what the solvers write. Public functions are exposed on the package root (`import msutils; msutils.msinfo(ms)`) and mirrored by a click CLI (`msutils` console script → `msutils.cli:cli`).

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
  gains/           gainutils: operations on calibration solutions
    _model.py        GainTable / GainBlock          <- the core abstraction
    _casa.py         CASA caltable reader + writer
    _quartical.py    QuartiCal zarr store reader + writer (`gains` extra)
    _io.py           format detection, read_gains / write_gains
    _stats.py        the shared flag-aware reduction (median / mean)
    fluxscale.py     flux-density bootstrap
    normalise.py     divide a scale out, leaving shape
    smooth.py        filter in time/frequency, complex then renormalised
    cli.py           click CLI (`gainutils` console script)
```

### gainutils

A second console script rather than a subcommand group, because these act on
calibration solutions rather than on Measurement Sets. The design rule is the
same one `MSInfo` follows: **one model, many formats**.

- **`GainBlock` is dense over `(time, freq, antenna, correlation)`**, one per
  (field, spw, direction). That is QuartiCal's own axis order, so its arrays
  need no transpose; a CASA caltable's rows are folded into it on read and
  unfolded on write via the `rows` map each block keeps.
- **Operations are `GainTable -> GainTable`** (`GainTable.map`), and never
  touch IO or a format. `fluxscale`, `normalise` and `smooth` are each
  arithmetic plus a report and nothing else.
- **A phase is never filtered as a number.** It is defined modulo 2*pi, so a
  running mean spikes at every wrap; `smooth` filters the complex gain for
  direction and the amplitude separately for magnitude, because the complex
  filter alone shrinks the amplitude wherever the phase rotates. Anything
  new that averages solutions inherits both halves of that.
- **A window may be physical.** `smooth` takes `"120s"` / `"8MHz"` as well
  as a sample count, converts against the block's own axes, and records what
  it resolved to -- the same reasoning that makes a physical solution
  interval survive being replayed on differently-averaged data.
- **Anything that reduces goes through `_stats.amplitude_statistic`.** Both
  operations ask "what is this antenna's amplitude", so the flag handling,
  the outlier rejection and the median/mean choice are written once. An
  operation that reduces differently is a sign the shared function needs an
  argument, not a copy.
- **Reductions start from `GainBlock.masked()`.** A median or mean over the
  raw array folds in flagged solutions and produces a plausible wrong number
  rather than an error. A (time, antenna) slot the source never had reads
  back *flagged*, not zero, for the same reason.
- **Antennas are matched by name, never by position.** Two tables can order
  antennas differently, and a positional match between them is a silent
  error. Correlations fall back to positional matching when a format records
  no labels (a caltable has no `CORR_TYPE`), and the result records which was
  used.
- **A block is a field's whole solution, never a storage chunk.** QuartiCal
  splits one field across datasets by time (normally per scan), so those are
  concatenated on read and split back on write via `GainBlock.chunks`. Read
  them as they sit on disk and a two-hour smoothing window silently becomes
  "whatever one scan held".
- **A parameterised term's `params` are the authoritative array**, not its
  gains: QuartiCal rebuilds the gains from them on load
  (`interpolate.py`'s `data_field = "params" if parameterized else
  "gains"`), so both are carried and both are written, always together.
  `amplitude` and `phase` are parameterised despite sounding elementary.
- **What an operation may do with a parameterisation is read off the
  parameter *names*** (`_params.py`), not a table of type names, so a new
  term type classifies itself. Three answers, and the distinction matters:
  a scaling maps onto amplitude parameters exactly; it is **meaningless**
  on phase-like ones, which describe a gain of unit modulus (verified on
  real stores: |g| = 1 to machine precision); and smoothing works on any of
  them but can only rebuild the gains for the unambiguous cases, since a
  delay's reconstruction needs the solver's reference frequency and a wrong
  one is invisible in the array.
- **Nothing is written in place.** Every operation takes an output path and
  refuses an existing one: a half-rewritten caltable is unrecoverable, and a
  pipeline that caches on declared outputs cannot see an in-place edit.
- **A measurement is a value, not a log line.** `fluxscale` returns a
  dataclass with a `to_dict()`/`save()`; CASA's own task writes its result to
  a logger that pipelines routinely discard, which is how a flux scale ends
  up recorded nowhere.

Where a statistic is a real choice, it is a knob with the trade documented:
`statistic` is `median` (robust) or `mean` (reproduces CASA to 0.1%), and on
real MeerKAT gains the two differ by 0.5% — more than either one's error
bar. `normalise`'s `scope` is the same kind of choice with more at stake:
`antenna` (CASA's `solnorm`) does **not** preserve relative antenna
amplitudes and `block` does, and picking the wrong one moves a flux scale
into a term the caller did not expect.

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

## Attribution: commit trailers yes, PR trailers no

A commit made with an assistant's help says so in a trailer on the
**commit message**, naming the assistant and the model behind it:

```
Assisted-by: <LLM> <MODEL>
```

— e.g. `Assisted-by: Claude Opus 5`, `Assisted-by: Codex GPT-5`. One
line, last in the message, after any `Co-authored-by:` for real people.

Agents default to a `Co-authored-by:` trailer with an address; replace
it. Both halves of that default are wrong here. `Co-authored-by:` claims
more than happened — these tools assist, and the person who ran them
owns the change and answers for it. The address is what makes the claim
bite: GitHub renders a trailer as co-authorship only when one is
present, so `Co-authored-by: Claude <noreply@anthropic.com>` attaches a
vendor to the authorship of work this organisation owns. `Assisted-by:`
with no address records the same fact as plain text and attaches nobody.

This reverses the rule as it stood until August 2026, which argued that
co-authorship was the honest description of tools doing real work. The
consideration that changed it is ownership, not accuracy of credit:
these are paid services operating on our IP, and the history should not
carry anything a vendor could read as a stake in it.

**Pull request descriptions carry no trailer at all** — no
`Assisted-by:`, no `Co-authored-by:`, no "Generated with", no tool
badge. A PR body is review material: it exists to tell a reviewer what
changed and why, and what to check. Provenance already lives on every
commit the PR contains, where it is attached to the specific change
rather than repeated once per PR, so a trailer in the description is
duplication in the one place that has no room for it. Agents default to adding one; delete it.

Neither form is a substitute for the message itself. A commit that
explains a decision badly does not improve by naming the model that
helped make it — see the existing history for the standard: what
changed, what it deviates from and why, and what a reviewer should not
assume held still.

## Reviewing changes: check the tree, not just the diff

A claim that something "doesn't exist" or "is unused" should be verified against
the actual tree before acting on it — a symbol absent from the diff is usually
present in the repo.
