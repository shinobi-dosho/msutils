<!--
Thanks for contributing. Nothing below is mandatory -- delete what does not
apply. A one-line fix does not need a filled-in form; a change to the readers
or the MSInfo model probably does.
-->

## What and why

<!--
What changes, and the reasoning behind it. The *why* is the part that is
expensive to reconstruct in a year's time -- if this fixes something subtle,
say what the old behaviour was and how it failed.
-->

## How it was verified

<!--
"Tests pass" is the floor, not the answer. What did you actually run it
against -- the synthetic MS, a real one, which telescope, how big?
-->

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
- [ ] Ran against a real Measurement Set (say which, roughly, below)

## Checklist

- [ ] A regression test that **fails without this change** — for a bug fix,
      this is the part that matters most. `tests/test_regressions.py` is the
      right home if it is one of the known failure modes.
- [ ] The fixture can tell right from wrong. If you are testing a per-axis
      result, check the synthetic MS actually varies along that axis —
      `patterned_ms` exists because the default one flags whole rows and
      cannot distinguish a correct per-correlation breakdown from a broken
      one.
- [ ] No new metadata pass over the main table. If you are adding something to
      `msinfo`, extend the existing TaQL aggregate rather than adding a query
      — see `AGENTS.md`.
- [ ] No new base dependency. Anything heavier than numpy + python-casacore +
      click belongs in an extra, imported inside the function that needs it.
- [ ] `uv.lock` updated if dependencies changed (`uv lock`), since CI runs
      `--locked`.
- [ ] Docs updated if behaviour changed — `docs/`, the docstrings that feed
      the API reference, and `CHANGELOG.md`.

## Related issues

<!-- Closes #123 -->
