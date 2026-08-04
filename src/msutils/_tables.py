"""Helpers for safely opening and querying casacore tables.

Two rules the rest of the package depends on:

* **Everything closes.** casacore holds file locks on open tables; a forgotten
  ``close()`` makes later operations block. Every helper here is a context
  manager that closes on the way out, exception or not.
* **Everything is quiet.** casacore prints ``Successful readonly open of ...``
  to stdout for every table it opens unless told otherwise. That noise
  corrupts piped JSON and buries CLI output, so ``ack=False`` is the default.

:func:`query` is the workhorse for metadata: pushing aggregation into TaQL
keeps it in casacore's C++ layer and costs one streaming pass over the MS,
instead of one Python-level pass per field/scan/antenna.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from casacore.tables import table, taql

__all__ = ["closing_table", "open_table", "query", "subtable_names", "table_exists"]


@contextmanager
def open_table(name: str, readonly: bool = True, ack: bool = False,
               **kwargs) -> Iterator[table]:
    """Open a casacore table by name, closing it on exit.

    Args:
        name: Table (or subtable, e.g. ``"ms::ANTENNA"``) path.
        readonly: Open read-only (the casacore default) unless ``False``.
        ack: Let casacore print its "Successful ... open" banner. Off by
            default so stdout stays clean for JSON and CLI output.
    """
    tab = table(name, readonly=readonly, ack=ack, **kwargs)
    try:
        yield tab
    finally:
        tab.close()


@contextmanager
def closing_table(tab: table) -> Iterator[table]:
    """Close an already-opened table (e.g. a ``table.query(...)`` result) on exit."""
    try:
        yield tab
    finally:
        tab.close()


@contextmanager
def query(command: str, tables: Sequence[table] | None = None,
          **kwargs: Any) -> Iterator[table]:
    """Run a TaQL command, yielding the result table and closing it on exit.

    Bind input tables positionally as ``$1``, ``$2``, ... rather than
    interpolating paths into the command string: TaQL rejects bare paths
    containing ``.`` or ``::`` (so ``FROM ./obs.ms`` is a parse error), and
    binding sidesteps quoting entirely.

        >>> with open_table(ms) as tab, \\
        ...         query("SELECT GCOUNT() AS N FROM $1", [tab]) as res:
        ...     nrows = int(res.getcol("N")[0])
    """
    res = taql(command, tables=list(tables) if tables else [], **kwargs)
    try:
        yield res
    finally:
        res.close()


def table_exists(name: str) -> bool:
    """True if ``name`` can be opened as a casacore table."""
    try:
        with open_table(name):
            return True
    except RuntimeError:
        return False


def subtable_names(tab: table) -> list[str]:
    """Names of the subtables attached to ``tab`` as table keywords."""
    return sorted(k for k in tab.keywordnames()
                  if isinstance(tab.getkeyword(k), str)
                  and tab.getkeyword(k).startswith("Table:"))
