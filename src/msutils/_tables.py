"""Helpers for safely opening casacore tables.

casacore holds file locks on open tables; forgetting to ``close()`` one makes
subsequent operations hang. These context managers guarantee the table is
closed even if the body raises.
"""
from contextlib import contextmanager
from typing import Iterator

from casacore.tables import table


@contextmanager
def open_table(name: str, readonly: bool = True, **kwargs) -> Iterator[table]:
    """Open a casacore table by name, closing it on exit.

    Args:
        name: Table (or subtable, e.g. ``"ms::ANTENNA"``) path.
        readonly: Open read-only (the casacore default) unless ``False``.
    """
    tab = table(name, readonly=readonly, **kwargs)
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
