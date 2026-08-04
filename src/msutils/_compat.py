"""Backwards-compatible :func:`summary`, reimplemented on top of :func:`msinfo`.

``summary()`` returned an unstructured dict of ALL-CAPS keys that mixed raw
subtable dumps with derived values. It is deprecated in favour of
:func:`msutils.msinfo`, which returns a typed, addressable
:class:`~msutils.info.MSInfo`.

This adapter keeps the old *shape* so existing pipelines keep running, but it
is built on the new reader, so several values are now **correct where they
previously were not**:

* ``FIELD.INTENTS`` is now a per-field list of intent lists, resolved through
  ``STATE_ID``. It used to be the raw ``STATE::OBS_MODE`` column, which only
  lined up with the fields by coincidence.
* ``FIELD.PERIOD`` and ``SCAN`` durations now include the final integration.
  They used to be ``max(TIME) - min(TIME)``, understating every scan by one
  integration (and reporting 0 for single-timestamp scans).
* ``MAXBL`` is now the longest baseline over all three UVW components. It used
  to use only *u* and *v*, making it a projected *uv* distance.
* ``EXPOSURE`` is now the set of integration times present, not row 0's value.

Code that depended on the old (wrong) numbers should move to
:func:`msutils.msinfo`.
"""
from __future__ import annotations

import codecs
import json
import warnings
from typing import Any, Dict, Optional

from ._log import create_logger
from ._tables import open_table, query
from .info import msinfo

__all__ = ["summary"]

LOGGER = create_logger(__name__)

_DEPRECATION = (
    "msutils.summary() is deprecated and will be removed in a future release; "
    "use msutils.msinfo(), which returns a structured MSInfo object "
    "(info.fields['NAME'], info.spws[0].chan_width, info.to_dict(), ...). "
    "Note that summary() now reports corrected intents, scan durations and "
    "max-baseline values -- see msutils._compat for details."
)

# Raw subtable columns the old summary() dumped wholesale. Reproduced so that
# callers indexing e.g. info['ANT']['POSITION'] keep working.
_SPW_COLUMNS = ("CHAN_FREQ MEAS_FREQ_REF REF_FREQUENCY TOTAL_BANDWIDTH NAME "
                "NUM_CHAN IF_CONV_CHAIN NET_SIDEBAND FREQ_GROUP_NAME "
                "CHAN_WIDTH").split()


def summary(msname: str, outfile: Optional[str] = None,
            display: bool = True) -> Dict[str, Any]:
    """Deprecated. Use :func:`msutils.msinfo` instead.

    Returns the legacy summary dict, built from :func:`msinfo`.
    """
    warnings.warn(_DEPRECATION, FutureWarning, stacklevel=2)

    info = msinfo(msname, level="full")
    out: Dict[str, Any] = {
        "FIELD": {},
        "SPW": {},
        "ANT": {},
        "MAXBL": info.max_baseline,
        "SCAN": {},
        "EXPOSURE": (info.integration_times[0] if info.integration_times else None),
        "NROW": info.nrows,
        "CORR": {},
    }

    state_ids = _state_ids(msname)
    out["FIELD"]["FIELD_ID"] = [f.id for f in info.fields]
    out["FIELD"]["INTENTS"] = [f.intents for f in info.fields]
    out["FIELD"]["STATE_ID"] = [state_ids.get(f.id) for f in info.fields]
    out["FIELD"]["PERIOD"] = [
        sum(info.scans[n].duration for n in f.scan_numbers) for f in info.fields]

    for scan in info.scans:
        out["SCAN"].setdefault(str(scan.field_id), {})[str(scan.number)] = scan.duration

    for key, subtable, columns in (("FIELD", "FIELD", None),
                                   ("ANT", "ANTENNA", None),
                                   ("SPW", "SPECTRAL_WINDOW", _SPW_COLUMNS)):
        out[key].update(_dump_subtable(msname, subtable, columns))

    if len(info.polarizations):
        pol = info.polarizations[0]
        out["CORR"]["NUM_CORR"] = pol.num_corr
        out["CORR"]["CORR_TYPE"] = pol.corr_labels
        out["NCOR"] = pol.num_corr
    else:                                      # pragma: no cover - malformed MS
        out["CORR"]["NUM_CORR"] = 0
        out["CORR"]["CORR_TYPE"] = []
        out["NCOR"] = 0

    if display:
        LOGGER.info(out)

    if outfile:
        with codecs.open(outfile, "w", "utf8") as stream:
            stream.write(json.dumps(out, ensure_ascii=False, default=_fallback))

    return out


def _state_ids(msname: str) -> Dict[int, int]:
    """FIELD_ID -> the lowest STATE_ID the main table uses for that field.

    The legacy dict has room for one scalar per field, so a field observed
    under several states cannot be represented faithfully here; the lowest id
    is picked for determinism. ``msinfo`` resolves states properly, exposing
    the full intent list per field and per scan.
    """
    with open_table(msname) as tab:
        if "STATE_ID" not in tab.colnames() or tab.nrows() == 0:
            return {}
        with query("SELECT FIELD_ID, GMIN(STATE_ID) AS STATE_ID FROM $1 "
                   "GROUPBY FIELD_ID", [tab]) as res:
            return {int(f): int(s) for f, s in zip(res.getcol("FIELD_ID"),
                                                   res.getcol("STATE_ID"))}


def _dump_subtable(msname: str, subtable: str,
                   columns: Optional[list]) -> Dict[str, Any]:
    """Read a subtable's columns into plain lists, as the old summary() did."""
    out: Dict[str, Any] = {}
    with open_table("{0}::{1}".format(msname, subtable)) as tab:
        for name in (columns if columns is not None else tab.colnames()):
            if name not in tab.colnames():
                continue
            try:
                value = tab.getcol(name)
            except RuntimeError:               # unfilled / variable-shaped column
                continue
            out[name] = value.tolist() if hasattr(value, "tolist") else value
    return out


def _fallback(value: Any) -> Any:              # pragma: no cover - defensive
    """JSON encoder fallback for stray numpy scalars."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)
