"""Flag statistics, computed with TaQL.

One ``GNTRUES(FLAG)`` aggregation per axis returns a
``(group, channel, correlation)`` count array straight out of casacore's C++
layer, in a single streaming pass over the data. Three such queries -- grouped
by field, by scan, and by antenna pair -- cover every axis reported here,
because the channel and correlation breakdowns fall out of any one of them.

This replaces a dask-ms implementation that ran one full pass over the MS *per
antenna*, and whose per-correlation numbers were wrong: its reduction kernel
wrote the total flag count into every correlation slot, so XX, XY, YX and YY
always reported identical values. The same kernel inflated the per-scan and
per-field ``sum``/``counts`` by the number of groups. The dependency is gone;
only the optional matplotlib plot remains behind the ``plots`` extra.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as _field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy

from ._log import create_logger
from ._tables import open_table, query
from .info import msinfo
from .info._model import Registry, _Record

__all__ = [
    "flagstats",
    "FlagStats",
    "FlagCount",
    "save_statistics",
    "plot_statistics",
]

LOGGER = create_logger(__name__)

#: Axes that :func:`flagstats` can break the counts down by.
AXES = ("field", "scan", "antenna", "baseline", "spw", "correlation", "channel")


@dataclass
class FlagCount(_Record):
    """Flagged and total visibility counts for one bin of some axis."""

    id: Any
    name: str = ""
    flagged: int = 0
    total: int = 0

    _derived = ("fraction", "percent")

    @property
    def fraction(self) -> float:
        """Flagged fraction in [0, 1]; 0.0 for an empty bin."""
        return float(self.flagged) / float(self.total) if self.total else 0.0

    @property
    def percent(self) -> float:
        return self.fraction * 100.0

    def __iadd__(self, other: "FlagCount") -> "FlagCount":
        self.flagged += other.flagged
        self.total += other.total
        return self


@dataclass
class FlagStats(_Record):
    """Flag counts broken down along each axis."""

    path: str = ""
    total: FlagCount = _field(default_factory=lambda: FlagCount(id="all", name="all"))
    by_field: Registry = _field(default_factory=lambda: Registry([]))
    by_scan: Registry = _field(default_factory=lambda: Registry([]))
    by_antenna: Registry = _field(default_factory=lambda: Registry([]))
    by_baseline: Registry = _field(default_factory=lambda: Registry([]))
    by_spw: Registry = _field(default_factory=lambda: Registry([]))
    by_correlation: Registry = _field(default_factory=lambda: Registry([]))
    by_channel: Dict[int, List[FlagCount]] = _field(default_factory=dict)

    def render(self) -> str:
        """A plain-text report of the flag percentages."""
        from ._flagrender import render_flagstats
        return render_flagstats(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str, indent: Optional[int] = 2) -> str:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(self.to_json(indent=indent))
        LOGGER.info("Wrote flag statistics to %s", path)
        return path

    def __str__(self) -> str:
        return self.render()


def flagstats(msname: str,
              fields: Optional[Sequence] = None,
              spws: Optional[Sequence] = None,
              antennas: Optional[Sequence] = None,
              scans: Optional[Sequence] = None,
              outfile: Optional[str] = None) -> FlagStats:
    """Compute flag statistics for ``msname``.

    Args:
        msname: Path to the Measurement Set.
        fields: Field ids or names to restrict to. ``None`` means all.
        spws: Spectral window ids or names to restrict to.
        antennas: Antenna ids or names to restrict to. A baseline is included
            if *either* antenna is selected.
        scans: Scan numbers to restrict to.
        outfile: If given, also write the JSON dump there.

    Returns:
        A :class:`FlagStats`.
    """
    info = msinfo(msname, level="meta")
    field_ids = _resolve(fields, info.fields, "field")
    spw_ids = _resolve(spws, info.spws, "spectral window")
    antenna_ids = _resolve(antennas, info.antennas, "antenna")
    scan_ids = [int(s) for s in scans] if scans else None

    # DATA_DESC_ID is what the main table keys off, so translate the SPW
    # selection through DATA_DESCRIPTION rather than assuming ddid == spw.
    ddids = [d.id for d in info.data_descriptions
             if spw_ids is None or d.spw_id in spw_ids]
    spw_of_ddid = {d.id: d.spw_id for d in info.data_descriptions}
    if not ddids:
        raise ValueError("no DATA_DESC_IDs match the requested spectral windows")

    stats = FlagStats(path=info.path)
    per_field: Dict[int, FlagCount] = {}
    per_scan: Dict[int, FlagCount] = {}
    per_baseline: Dict[Tuple[int, int], FlagCount] = {}
    per_antenna: Dict[int, FlagCount] = {}
    per_spw: Dict[int, FlagCount] = {}
    per_corr: Dict[int, FlagCount] = {}
    per_channel: Dict[int, List[FlagCount]] = {}

    with open_table(msname) as tab:
        for ddid in ddids:
            spw_id = spw_of_ddid[ddid]
            where = _where(ddid, field_ids, antenna_ids, scan_ids)

            # One pass per axis. Each returns (ngroup, nchan, ncorr) flag
            # counts plus the row count per group, from which the totals for
            # every cell follow.
            fields_seen = _axis_counts(tab, "FIELD_ID", where)
            scans_seen = _axis_counts(tab, "SCAN_NUMBER", where)
            baselines = _axis_counts(tab, "ANTENNA1, ANTENNA2", where,
                                     keys=("ANTENNA1", "ANTENNA2"))

            if fields_seen is None:
                LOGGER.info("No rows for DATA_DESC_ID %d with this selection", ddid)
                continue

            keys, flagged, nrows = fields_seen
            nchan, ncorr = flagged.shape[1], flagged.shape[2]
            corr_labels = _corr_labels(info, ddid)

            for i, key in enumerate(keys):
                _accumulate(per_field, int(key[0]), info.fields, flagged[i],
                            nrows[i] * nchan * ncorr)

            if scans_seen is not None:
                skeys, sflagged, snrows = scans_seen
                for i, key in enumerate(skeys):
                    number = int(key[0])
                    _bump(per_scan, number, str(number), sflagged[i].sum(),
                          snrows[i] * nchan * ncorr)

            if baselines is not None:
                bkeys, bflagged, bnrows = baselines
                for i, key in enumerate(bkeys):
                    p, q = int(key[0]), int(key[1])
                    total = int(bnrows[i]) * nchan * ncorr
                    count = int(bflagged[i].sum())
                    _bump(per_baseline, (p, q),
                          "{0}-{1}".format(_name(info.antennas, p),
                                           _name(info.antennas, q)),
                          count, total)
                    # A baseline contributes to both of its antennas.
                    for antenna in (p, q):
                        _accumulate(per_antenna, antenna, info.antennas,
                                    count, total)

            # Channel and correlation breakdowns are free: sum the per-field
            # array over the group axis, then over the other data axis.
            spw_flagged = flagged.sum(axis=0)               # (nchan, ncorr)
            spw_total = int(nrows.sum())
            _bump(per_spw, spw_id, _name(info.spws, spw_id),
                  int(spw_flagged.sum()), spw_total * nchan * ncorr)
            for c in range(ncorr):
                _bump(per_corr, c, corr_labels[c] if c < len(corr_labels) else str(c),
                      int(spw_flagged[:, c].sum()), spw_total * nchan)
            per_channel[spw_id] = [
                FlagCount(id=ch, name=str(ch),
                          flagged=int(spw_flagged[ch].sum()),
                          total=spw_total * ncorr)
                for ch in range(nchan)]

    stats.by_field = Registry(_sorted(per_field))
    stats.by_scan = Registry(_sorted(per_scan))
    stats.by_antenna = Registry(_sorted(per_antenna))
    stats.by_baseline = Registry(list(per_baseline.values()))
    stats.by_spw = Registry(_sorted(per_spw))
    stats.by_correlation = Registry(_sorted(per_corr))
    stats.by_channel = per_channel
    stats.total = FlagCount(id="all", name="all",
                            flagged=sum(c.flagged for c in stats.by_spw),
                            total=sum(c.total for c in stats.by_spw))

    LOGGER.info("%s: %.2f%% of %d visibilities flagged",
                msname, stats.total.percent, stats.total.total)
    if outfile:
        stats.save(outfile)
    return stats


# --------------------------------------------------------------------------
# TaQL

def _axis_counts(tab, group_by: str, where: str,
                 keys: Sequence[str] = ()) -> Optional[Tuple]:
    """``GROUPBY group_by`` -> (key tuples, (n, nchan, ncorr) counts, row counts).

    ``GNTRUES`` gives the per-element flag count for each group, which is what
    makes the channel and correlation axes correct rather than a repeat of the
    group total.
    """
    key_names = list(keys) or [group_by]
    command = ("SELECT {0}, GNTRUES(FLAG) AS NFLAG, GCOUNT() AS NROW "
               "FROM $1 {1} GROUPBY {0}".format(group_by, where))
    LOGGER.debug("flag stats: %s", command)
    with query(command, [tab]) as res:
        if res.nrows() == 0:
            return None
        flagged = numpy.asarray(res.getcol("NFLAG"))
        nrows = numpy.asarray(res.getcol("NROW"))
        key_columns = [numpy.asarray(res.getcol(name)) for name in key_names]
        return list(zip(*key_columns)), flagged, nrows


def _where(ddid: int, field_ids, antenna_ids, scan_ids) -> str:
    """Build the WHERE clause for one DATA_DESC_ID plus the user's selection."""
    clauses = ["DATA_DESC_ID=={0:d}".format(ddid)]
    if field_ids:
        clauses.append("FIELD_ID IN [{0}]".format(",".join(map(str, field_ids))))
    if scan_ids:
        clauses.append("SCAN_NUMBER IN [{0}]".format(",".join(map(str, scan_ids))))
    if antenna_ids:
        listed = ",".join(map(str, antenna_ids))
        clauses.append("(ANTENNA1 IN [{0}] OR ANTENNA2 IN [{0}])".format(listed))
    return "WHERE " + " AND ".join(clauses)


# --------------------------------------------------------------------------
# helpers

def _resolve(selection, registry, label: str) -> Optional[List[int]]:
    """Map a mixed list of ids and names to ids, via the MSInfo registry."""
    if not selection:
        return None
    resolved = []
    for item in selection:
        if isinstance(item, str) and not item.lstrip("-").isdigit():
            try:
                resolved.append(registry[item].id)
            except KeyError:
                raise ValueError("unknown {0} {1!r}; have {2}".format(
                    label, item, registry.names)) from None
        else:
            ident = int(item)
            if ident not in registry.ids:
                raise ValueError("unknown {0} id {1}; have {2}".format(
                    label, ident, registry.ids))
            resolved.append(ident)
    return resolved


def _name(registry, ident: int) -> str:
    try:
        return registry[ident].name or str(ident)
    except KeyError:
        return str(ident)


def _bump(store: Dict, key, name: str, flagged, total) -> None:
    entry = store.get(key)
    if entry is None:
        store[key] = FlagCount(id=key, name=name, flagged=int(flagged),
                               total=int(total))
    else:
        entry.flagged += int(flagged)
        entry.total += int(total)


def _accumulate(store: Dict, ident: int, registry, flagged, total) -> None:
    _bump(store, ident, _name(registry, ident),
          int(numpy.sum(flagged)), int(total))


def _sorted(store: Dict) -> List[FlagCount]:
    return [store[k] for k in sorted(store)]


def _corr_labels(info, ddid: int) -> List[str]:
    """Correlation labels for the polarization setup behind a DATA_DESC_ID."""
    try:
        pol_id = info.data_descriptions[ddid].pol_id
        return info.polarizations[pol_id].corr_labels
    except KeyError:                           # pragma: no cover - malformed MS
        return []


# --------------------------------------------------------------------------
# public entry points kept from 2.x

def save_statistics(msname: str, antennas=None, fields=None,
                    outfile: Optional[str] = None, **kwargs) -> FlagStats:
    """Compute flag statistics and write them to JSON.

    Returns a :class:`FlagStats` rather than the old nested dict. The JSON
    layout changed with it: the previous one carried per-correlation values
    that were all identical, and per-scan/per-field sums inflated by the
    number of groups.
    """
    return flagstats(msname, fields=fields, antennas=antennas,
                     outfile=outfile or "default-flag-statistics.json",
                     **kwargs)


def plot_statistics(msname: str, antennas=None, fields=None,
                    plotfile: Optional[str] = None,
                    outfile: Optional[str] = None, **kwargs) -> FlagStats:
    """Compute flag statistics, write a PNG summary and optionally JSON.

    Needs the ``plots`` extra for matplotlib.
    """
    stats = flagstats(msname, fields=fields, antennas=antennas,
                      outfile=outfile, **kwargs)
    from ._flagplot import plot_flagstats
    plot_flagstats(stats, plotfile or "default-flagging-summary-plots.png")
    return stats
