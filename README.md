# msutils

Tools for manipulating CASA Measurement Sets (MS), for radio-astronomy pipelines.

[![Python package](https://github.com/sphemakh/msutils/actions/workflows/installation.yml/badge.svg)](https://github.com/sphemakh/msutils/actions/workflows/installation.yml)
[![PyPI](https://img.shields.io/pypi/v/msutils.svg)](https://pypi.org/project/msutils/)

## Features

- Summarise an MS — fields, spectral windows, antennas, scans, correlations — to a dict / JSON.
- Add, copy, and sum/subtract visibility columns, chunked by spectral window.
- Add Gaussian noise to a column (from a stddev, or computed from an SEFD).
- Verify / fix antenna Y-position sign conventions.
- Flag statistics per antenna / scan / field / correlation, with a matplotlib summary plot (`flagstats` extra).
- Per-channel visibility weight estimation from an SEFD profile (`plots` extra).
- A `msutils` command-line tool wrapping the above.

## Install

Requires Python >= 3.8. `python-casacore` is pulled in automatically (its wheels bundle the casacore libraries).

```bash
pip install msutils                 # core column operations + CLI
pip install "msutils[flagstats]"    # + flag statistics & plots (dask-ms, matplotlib)
pip install "msutils[plots]"        # + weight estimation (matplotlib, scipy)
pip install "msutils[all]"          # everything
```

## Python API

The core operations are exposed directly on the package:

```python
import msutils

# Metadata summary (returns a JSON-serialisable dict)
info = msutils.summary("obs.ms")

# Column operations
msutils.addcol("obs.ms", "MODEL_DATA", clone="DATA")
msutils.copycol("obs.ms", "DATA", "CORRECTED_DATA")
msutils.sumcols("obs.ms", cols=["DATA", "MODEL_DATA"], outcol="CORRECTED_DATA")

# Add noise (computed from an SEFD, or a fixed stddev)
msutils.addnoise("obs.ms", column="MODEL_DATA", sefd=551)
```

Flag statistics and weight estimation live in optional submodules:

```python
from msutils import flagstats                 # needs msutils[flagstats]
flagstats.plot_statistics("obs.ms", plotfile="flags.png", outfile="flags.json")

from msutils.weights import MSNoise           # needs msutils[plots]
```

## Command line

```bash
msutils summary  obs.ms --json summary.json
msutils addcol   obs.ms MODEL_DATA --clone DATA
msutils copycol  obs.ms DATA CORRECTED_DATA
msutils sumcols  obs.ms DATA MODEL_DATA --out CORRECTED_DATA
msutils addnoise obs.ms --column MODEL_DATA --sefd 551
msutils flagstats obs.ms --plot flags.png --json flags.json   # needs msutils[flagstats]
```

Run `msutils --help` or `msutils <command> --help` for all options.

## Migrating from 1.x / `MSUtils`

- The import package is now lowercase **`msutils`** (the capitalised `MSUtils` still works, with a deprecation warning).
- The core functions are on the package root — use `msutils.summary(...)` instead of `from msutils import msutils`.
- Modules renamed: `flag_stats` → `flagstats`, `ClassESW` → `weights` (old names still importable, deprecated).
- `imp_plotter` (gain/calibration-table plotting) was removed — use [ragavi](https://github.com/ratt-ru/ragavi) (`ragavi-gains`) instead.
- The defunct `prep()` and its external-binary dependencies were removed.

## License

GNU GPL v2 or later. See [LICENSE](LICENSE).
