# Contributing to msutils

Thanks for your interest in contributing! **msutils** is a library and CLI of
everyday Measurement Set operations for radio-astronomy pipelines. The most
valuable contributions are bug reports against real MSs, focused fixes, tests,
documentation, and feedback on the design.

## Scope and philosophy

msutils does **everyday MS operations**: inspect, subset, average, manage
columns and flags. It does not calibrate, image, or otherwise process
visibilities — those belong in the tools built for them
([CASA](https://casa.nrao.edu), [QuartiCal](https://github.com/ratt-ru/QuartiCal),
[WSClean](https://gitlab.com/aroffringa/wsclean), and friends). A feature that
would make msutils a processing package is out of scope, however useful; the
2.x `weights` module was removed in 3.0 for exactly that reason.

Two rules follow from that, and they shape most review comments here:

**Aggregate in TaQL, not in Python.** MSs do not fit in memory and the
interesting ones have hundreds of scans. Pushing work into casacore's C++
layer is the difference between seconds and minutes — the 2.x `summary()` ran
one table scan per field and another per scan, and replacing it with a single
`GROUPBY` made it flat in the number of scans. If you are adding metadata,
extend the existing aggregate rather than adding a pass.

**The base install stays small.** `numpy` + `python-casacore` + `click` covers
`msinfo`, every column operation, `subset`, flag versions, `flagstats` and the
diagnostics. Anything heavier is an extra, imported inside the function that
needs it, and `tests/test_import.py` enforces that in a subprocess.

See **[`AGENTS.md`](AGENTS.md)** for the full set of conventions — read it
before changing the readers, the `MSInfo` model, or anything that touches
casacore tables.

## Ways to contribute

- **Report bugs** via [issues](https://github.com/SpheMakh/msutils/issues).
  A bug report that names the telescope and the MS's shape (fields, SPWs,
  scans, whether `FEED`/`STATE`/`SOURCE` are populated) is worth several that
  don't — most metadata bugs are really "this MS is shaped in a way the code
  did not expect".
- **Fix a bug**, with a regression test that fails without the fix.
  `tests/test_regressions.py` collects the ones found in the 2.x audit and is
  the right home for more.
- **Improve documentation** under `docs/`, or the docstrings that feed the API
  reference.
- **Sizable change** — open an issue to discuss it first. That is especially
  true for anything that adds a dependency or widens the scope above.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SpheMakh/msutils.git
cd msutils
uv sync --all-extras --group dev
uv run pytest
uv run ruff check .

# enable the repo's pre-commit hook (once per clone)
git config core.hooksPath .githooks
```

`pip install -e ".[all]"` works too if you'd rather not use uv; CI runs
`uv sync --locked`, so a dependency change means committing the updated
`uv.lock`.

### The pre-commit hook

Enabling it is the only setup step that is not uv's job — git will not let a
repository turn on an executable hook by itself, which is why the `git config`
above is manual; skip it and you simply get no hook.

`.githooks/pre-commit` is a tracked shell script (no `pre-commit` framework, no
separate pinned tool universe). When a commit touches Python it runs
`ruff check` and `ruff format --check` through `uv run`, so it uses this
project's own pinned ruff and agrees with CI's lint job by construction. A
format failure is fixed with `uv run ruff format <file> && git add <file>`; the
hook never rewrites files behind your back. Any other commit skips it in
milliseconds, and `git commit --no-verify` bypasses it when you genuinely need
to.

## Testing

```bash
uv run pytest -q
```

The suite builds its own Measurement Sets with python-casacore
(`tests/msfactory.py`), so it needs no simulator and nothing skips for want of
test data. Tests for the optional extras `importorskip`; CI runs the suite
twice, once on a bare install and once with `[all]`, so the base install is
proven self-sufficient rather than assumed to be.

Two things worth knowing before you write a test here:

- **Make sure the fixture can tell right from wrong.** The default synthetic MS
  flags whole rows, which cannot distinguish a correct per-correlation flag
  breakdown from the 2.x bug that reported the same total for every
  correlation. `patterned_ms` exists for that reason — flag one correlation,
  one channel and one antenna, so each axis has a different known answer.
- **Do not assert on the contents of an uninitialised column.** `addcol`
  without `init_with` leaves cells genuinely unfilled, and comparisons against
  them pass or fail by luck.

## Documentation

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

## Commits

Commit messages explain *why*, not just what — the reasoning is the part that
is expensive to reconstruct later. Where a change fixes something subtle, say
what the old behaviour was and how it failed. See `git log` for the house
style.
