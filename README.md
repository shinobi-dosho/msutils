# msutils

Everyday Measurement Set operations for radio-astronomy pipelines — inspect,
subset, average, manage columns and flags. No calibration, no imaging.

[![Python package](https://github.com/sphemakh/msutils/actions/workflows/installation.yml/badge.svg)](https://github.com/sphemakh/msutils/actions/workflows/installation.yml)
[![PyPI](https://img.shields.io/pypi/v/msutils.svg)](https://pypi.org/project/msutils/)

```python
import msutils

info = msutils.msinfo("obs.ms")
print(info.render())

info.fields["PKS1934-638"].scan_numbers   # [1, 2]
info.spws[0].chan_width[0]                # 10000000.0
info.antennas["m003"].latitude            # -30.71053
```

```
$ msutils info obs.ms
Measurement Set: /data/obs.ms
  format      : MSv2    size: 271.4 MiB    rows: 595,200
  telescope   : MEERKAT      observer: sphe          project: TEST-2024-01
  observed    : 2024-01-01T00:00:00.000 -> 2024-01-01T05:12:44.000  (5h12m44s)
  totals      : 32 antennas, 496 baselines, 3 fields, 60 scans, 1 SPWs, 8 channels
  integration : 8s
  max baseline: 4,701.2 m

Fields (3):
  ID  Name         RA            Dec           Frame  Intents             Scans    Rows
  --  -----------  ------------  ------------  -----  ------------------  -----  ------
   0  PKS1934-638  01h20m12.845  -29d47m37.70  J2000  CALIBRATE_BANDPASS     20  198,400
   1  J0217+0144   02h51m53.240  -32d39m30.94  J2000  CALIBRATE_PHASE        20  198,400
   2  DEEP_2       04h23m33.635  -35d31m24.18  J2000  OBSERVE_TARGET         20  198,400
...
```

## Install

Requires Python ≥ 3.11. The base install is just `numpy` + `python-casacore`
+ `click`, and covers everything except averaging, plots and MSv4.

```bash
pip install msutils                    # msinfo, columns, subset, flags, flagstats, diagnostics
pip install "msutils[plots]"           # + PNG summary plots (matplotlib)
pip install "msutils[average]"         # + time/channel averaging (codex-africanus)
pip install "msutils[msv4]"            # + read MSv4 processing sets (xarray, zarr)
pip install "msutils[xarray-ms]"       # + read an MSv2 through the MSv4 schema
pip install "msutils[convert]"         # + write MSv2 -> MSv4 (xradio)
pip install "msutils[all]"             # everything
```

## `msinfo`

`msinfo` returns a typed, JSON-serialisable `MSInfo` whose collections are
addressable **by name as well as by id** — because real work says "the bandpass
calibrator", not "row 1".

```python
info = msutils.msinfo("obs.ms")

info.observation.telescope           # 'MEERKAT'
info.fields["DEEP_2"].intents        # ['OBSERVE_TARGET#ON_SOURCE']
info.fields["DEEP_2"].ra_hms         # '04h23m33.635'
info.scans[7].duration               # 24.0  (includes the final integration)
info.spws["SPW0"].centre_freq        # 1415000000.0
info.data_descriptions[1].spw_id     # DDID -> SPW, explicitly
info.columns["DATA"].shape           # [4, 4]

info.to_dict()                       # stable, versioned JSON
info.render(verbose=True)            # the listobs-style report above
```

Three cost levels, so inspecting a 100 GB MS doesn't have to read 100 GB:

| `level` | Reads | Gives you |
|---|---|---|
| `"meta"` | subtables only | fields, SPWs, antennas, columns, max baseline |
| `"full"` *(default)* | + one pass over the index columns | scans, time ranges, row counts |
| `"data"` | + UVW and FLAG | uv coverage, flag statistics |

`"meta"` is independent of MS size (~10 ms); `"full"` is one TaQL `GROUPBY`,
flat in the number of scans.

### MSv4 and engines

`msinfo` reads MSv2 tables and MSv4 processing sets into the same `MSInfo`:

```python
msutils.msinfo("obs.ms")                          # MSv2, via casacore/TaQL
msutils.msinfo("obs.zarr")                        # MSv4, via xarray + zarr
msutils.msinfo("obs.ms", engine="xarray-ms")      # MSv2 through the MSv4 schema
```

The default MSv2 engine is casacore/TaQL: it needs nothing beyond the base
install, it is several times faster for metadata, and it will open a
Measurement Set that stricter readers reject — which matters, because a
malformed MS is exactly the one you need to inspect. Pass
`engine="xarray-ms"` when you want the n-dimensional MSv4 view of an MS you
already have, with no conversion step.

Converting for real needs `msutils[convert]`:

```bash
msutils convert obs.ms obs.zarr
```

## Everything else

```python
# columns
msutils.addcol("obs.ms", "MODEL_DATA", clone="DATA")
msutils.copycol("obs.ms", "DATA", "CORRECTED_DATA")
msutils.sumcols("obs.ms", cols=["DATA", "MODEL_DATA"], outcol="CORRECTED_DATA")
msutils.delcol("obs.ms", "CORRECTED_DATA")          # reclaim the disk
msutils.renamecol("obs.ms", "MODEL_DATA", "OLD_MODEL")
msutils.addnoise("obs.ms", column="MODEL_DATA", sefd=551)

# datasets
msutils.subset("obs.ms", "target.ms", fields=["DEEP_2"], spws=[0])
msutils.average("obs.ms", "avg.ms", time_bin=8.0, chan_bin=4)   # [average]

# flags
stats = msutils.flagstats("obs.ms")
stats.by_correlation["XY"].percent
stats.by_channel[0]                                 # per-channel, per SPW
msutils.flag_backup("obs.ms", name="pre-rfi")
msutils.flag_restore("obs.ms", "pre-rfi")

# diagnostics
msutils.du("obs.ms")                                # where the bytes went
msutils.check("obs.ms")                             # MSv2 conformance
msutils.taql("SELECT DISTINCT FIELD_ID FROM $1", "obs.ms")
```

`subset` keeps the original field and SPW ids rather than renumbering them
the way CASA `split` does, so ids in a subset still match the parent MS.

## Command line

```bash
msutils info      obs.ms [-v] [--level meta|full|data] [--json out.json]
msutils flagstats obs.ms [--plot flags.png] [--json flags.json] [--field DEEP_2]
msutils subset    obs.ms target.ms --field DEEP_2 --spw 0
msutils average   obs.ms avg.ms --time-bin 8 --chan-bin 4
msutils delcol    obs.ms CORRECTED_DATA MODEL_DATA
msutils renamecol obs.ms MODEL_DATA OLD_MODEL
msutils addcol    obs.ms MODEL_DATA --clone DATA
msutils copycol   obs.ms DATA CORRECTED_DATA
msutils sumcols   obs.ms DATA MODEL_DATA --out CORRECTED_DATA
msutils addnoise  obs.ms --column MODEL_DATA --sefd 551
msutils flags     backup|restore|list|delete obs.ms [NAME]
msutils du        obs.ms
msutils check     obs.ms
msutils taql      'SELECT DISTINCT FIELD_ID FROM $1' --ms obs.ms
msutils convert   obs.ms obs.zarr
```

`msutils <command> --help` for the full options.

## Migrating from 2.x

`summary()` still works and returns the same dict shape, but is deprecated in
favour of `msinfo()` and now emits a `FutureWarning`. Several of its values
have been **corrected** — see [CHANGELOG.md](CHANGELOG.md) for the full list
and for the removed 1.x aliases.

`msutils.weights` (`MSNoise`, SEFD-profile weight estimation) was removed in
3.0: it is data processing rather than an everyday MS operation. Pin
`msutils<3` if you still need it.

```python
info = msutils.summary("obs.ms")     # deprecated
info = msutils.msinfo("obs.ms")      # use this
```

## License

GNU GPL v2 or later. See [LICENSE](LICENSE).
