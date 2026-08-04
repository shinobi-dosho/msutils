"""Disk usage, conformance checking, and raw TaQL access.

Three small tools that round out the "what is actually in this MS" story that
:func:`msutils.msinfo` starts:

* :func:`du` -- where the bytes went, per storage manager and per column.
* :func:`check` -- does this MS satisfy the MSv2 standard?
* :func:`taql` -- run an arbitrary TaQL query and get the columns back.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field as _field
from typing import Any, Dict, List, Optional, Sequence

import numpy

from ._log import create_logger
from ._tables import open_table
from ._tables import query as _query
from .info import msinfo
from .info._render import format_bytes

__all__ = ["du", "check", "taql", "DiskUsage", "StorageGroup", "CheckReport"]

LOGGER = create_logger(__name__)


@dataclass
class StorageGroup:
    """One casacore storage manager and the columns it holds.

    Bytes are attributed per *storage manager*, not per column: a
    ``StandardStMan`` typically holds dozens of scalar columns in shared
    files, so there is no per-column figure to report. Bulk columns such as
    DATA usually get a tiled manager to themselves, which is where the space
    goes anyway.
    """

    name: str
    type: str
    columns: List[str] = _field(default_factory=list)
    nbytes: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "columns": self.columns,
                "nbytes": self.nbytes}


@dataclass
class DiskUsage:
    """On-disk size of an MS, broken down by storage manager and subtable."""

    path: str
    total: int = 0
    main_table: int = 0
    storage: List[StorageGroup] = _field(default_factory=list)
    subtables: Dict[str, int] = _field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"path": self.path, "total": self.total,
                "main_table": self.main_table,
                "storage": [s.to_dict() for s in self.storage],
                "subtables": self.subtables}

    def render(self) -> str:
        lines = ["Disk usage: {0}".format(self.path),
                 "  total: {0}".format(format_bytes(self.total))]
        lines.append("")
        lines.append("Main table by storage manager:")
        if not self.storage:
            lines.append("  (none)")
        for group in sorted(self.storage, key=lambda g: g.nbytes, reverse=True):
            share = (100.0 * group.nbytes / self.total) if self.total else 0.0
            lines.append("  {0:>10}  {1:5.1f}%  {2:<18} {3}".format(
                format_bytes(group.nbytes), share, group.type,
                ", ".join(group.columns[:6])
                + (", ..." if len(group.columns) > 6 else "")))
        if self.subtables:
            lines.append("")
            lines.append("Subtables:")
            for name, size in sorted(self.subtables.items(),
                                     key=lambda kv: kv[1], reverse=True):
                lines.append("  {0:>10}  {1}".format(format_bytes(size), name))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def du(msname: str) -> DiskUsage:
    """Report where an MS's bytes are, by storage manager and subtable."""
    path = os.path.abspath(msname)
    usage = DiskUsage(path=path, total=_tree_size(path))

    with open_table(msname) as tab:
        try:
            dminfo = tab.getdminfo()
        except RuntimeError:                   # pragma: no cover - exotic tables
            dminfo = {}

    for name, spec in dminfo.items():
        seqnr = spec.get("SEQNR")
        usage.storage.append(StorageGroup(
            name=str(name), type=str(spec.get("TYPE", "")),
            columns=sorted(spec.get("COLUMNS", [])),
            nbytes=_manager_bytes(path, seqnr)))

    usage.main_table = sum(g.nbytes for g in usage.storage)
    for entry in sorted(os.listdir(path)):
        child = os.path.join(path, entry)
        if os.path.isdir(child):
            usage.subtables[entry] = _tree_size(child)
    return usage


@dataclass
class CheckReport:
    """Result of :func:`check`."""

    path: str
    ok: bool = True
    errors: List[str] = _field(default_factory=list)
    warnings: List[str] = _field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "ok": self.ok, "errors": self.errors,
                "warnings": self.warnings}

    def render(self) -> str:
        lines = ["Check: {0}".format(self.path)]
        for message in self.errors:
            lines.append("  ERROR   {0}".format(message))
        for message in self.warnings:
            lines.append("  WARNING {0}".format(message))
        lines.append("  {0}".format(
            "OK -- conforms to the MSv2 standard" if self.ok
            else "{0} error(s), {1} warning(s)".format(len(self.errors),
                                                       len(self.warnings))))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


#: Subtables the MSv2 standard requires.
REQUIRED_SUBTABLES = ("ANTENNA", "DATA_DESCRIPTION", "FEED", "FIELD",
                      "OBSERVATION", "POLARIZATION", "SPECTRAL_WINDOW")


def check(msname: str) -> CheckReport:
    """Validate an MS against the MSv2 standard.

    Checks the required main-table columns and subtables are present, that
    ``DATA_DESC_ID`` values resolve through DATA_DESCRIPTION to real SPW and
    polarization rows, and that ``FIELD_ID``/``ANTENNA`` references are in
    range. Structural problems are errors; things that merely tend to cause
    trouble downstream are warnings.
    """
    from casacore.tables import required_ms_desc

    report = CheckReport(path=os.path.abspath(msname))
    required_columns = sorted(k for k in required_ms_desc()
                              if not k.startswith("_"))

    with open_table(msname) as tab:
        present = set(tab.colnames())
        for column in required_columns:
            if column not in present:
                report.errors.append(
                    "main table is missing required column {0}".format(column))
        if "DATA" not in present and "FLOAT_DATA" not in present:
            report.warnings.append(
                "no DATA or FLOAT_DATA column: nothing to calibrate or image")

        keywords = set(tab.keywordnames())
        for subtable in REQUIRED_SUBTABLES:
            if subtable not in keywords:
                report.errors.append("missing required subtable {0}".format(subtable))

        if tab.nrows() == 0:
            report.warnings.append("main table has no rows")

    info = msinfo(msname, level="meta")

    # Referential integrity: every DATA_DESC_ID must resolve.
    spw_ids = set(info.spws.ids)
    pol_ids = set(info.polarizations.ids)
    for dd in info.data_descriptions:
        if dd.spw_id not in spw_ids:
            report.errors.append(
                "DATA_DESC_ID {0} points at SPECTRAL_WINDOW_ID {1}, which does "
                "not exist".format(dd.id, dd.spw_id))
        if dd.pol_id not in pol_ids:
            report.errors.append(
                "DATA_DESC_ID {0} points at POLARIZATION_ID {1}, which does "
                "not exist".format(dd.id, dd.pol_id))

    for spw in info.spws:
        if len(spw.chan_freq) != spw.num_chan:
            report.errors.append(
                "SPW {0}: NUM_CHAN is {1} but CHAN_FREQ has {2} entries".format(
                    spw.id, spw.num_chan, len(spw.chan_freq)))

    if info.nrows:
        _check_references(msname, info, report)

    report.ok = not report.errors
    return report


def _check_references(msname: str, info, report: CheckReport) -> None:
    """Verify the main table's index columns point at rows that exist."""
    with open_table(msname) as tab:
        checks = (("FIELD_ID", set(info.fields.ids), "FIELD"),
                  ("DATA_DESC_ID", set(info.data_descriptions.ids),
                   "DATA_DESCRIPTION"),
                  ("ANTENNA1", set(info.antennas.ids), "ANTENNA"),
                  ("ANTENNA2", set(info.antennas.ids), "ANTENNA"))
        for column, valid, subtable in checks:
            if column not in tab.colnames():
                continue
            with _query("SELECT DISTINCT {0} FROM $1".format(column), [tab]) as res:
                if res.nrows() == 0:
                    continue
                used = set(int(v) for v in res.getcol(column))
            dangling = sorted(used - valid)
            if dangling:
                report.errors.append(
                    "{0} references {1} row(s) that do not exist in {2}: "
                    "{3}".format(column, len(dangling), subtable,
                                 dangling[:10]))


def taql(command: str, msname: Optional[str] = None,
         columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Run a TaQL command and return the result columns as numpy arrays.

    Args:
        command: TaQL command. Refer to ``msname`` as ``$1``.
        msname: MS bound to ``$1``. Optional if the command names its own
            tables.
        columns: Columns to extract. Defaults to everything the result has.

    Returns:
        Column name -> array. Columns whose cells vary in shape are returned
        as lists rather than arrays.

        >>> taql("SELECT DISTINCT FIELD_ID FROM $1", "obs.ms")
        {'FIELD_ID': array([0, 1, 2], dtype=int32)}
    """
    if msname:
        with open_table(msname) as tab:
            return _run(command, [tab], columns)
    return _run(command, [], columns)


def _run(command: str, tables, columns) -> Dict[str, Any]:
    with _query(command, tables) as res:
        wanted = list(columns) if columns else list(res.colnames())
        out = {}
        for name in wanted:
            try:
                out[name] = numpy.asarray(res.getcol(name))
            except RuntimeError:
                # Ragged cells: getcol refuses, getvarcol returns a dict.
                out[name] = list(res.getvarcol(name).values())
        return out


def _manager_bytes(path: str, seqnr) -> int:
    """Total size of the files backing storage manager ``seqnr``.

    A manager owns ``table.f<seqnr>`` plus suffixed companions -- the
    ``table.f0i`` index of a StandardStMan, the ``table.f1_TSM0`` tiles of a
    TiledShapeStMan. Matching must not let ``table.f10`` count towards
    manager 1, so the character after the sequence number has to be a
    non-digit (or nothing at all).
    """
    if seqnr is None:
        return 0
    prefix = "table.f{0}".format(seqnr)
    total = 0
    for candidate in glob.glob(os.path.join(path, prefix + "*")):
        suffix = os.path.basename(candidate)[len(prefix):]
        if suffix[:1].isdigit():
            continue                           # e.g. table.f10 for seqnr 1
        if os.path.isfile(candidate):
            total += os.path.getsize(candidate)
    return total


def _tree_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total
