"""Save and restore flag versions -- a ``flagmanager`` equivalent.

Flagging is destructive and iterative: you flag, look at the result, and often
want the previous state back. Versions are stored beside the MS in
``<ms>.flagversions/flags.<name>``, each a small casacore table holding just
``FLAG`` and ``FLAG_ROW`` for every row of the main table, plus a
``FLAG_VERSION_LIST`` file recording names and descriptions.

The directory layout follows CASA's ``flagmanager`` so the two sit side by
side, but the version tables are written by casacore here and are only
guaranteed to be readable by msutils.
"""

from __future__ import annotations

import datetime
import os
import shutil
from dataclasses import dataclass

from ._log import create_logger
from ._tables import open_table, query

__all__ = [
    "FlagVersion",
    "flag_backup",
    "flag_delete",
    "flag_restore",
    "flag_versions",
]

LOGGER = create_logger(__name__)

#: Columns carried in a flag version.
_FLAG_COLUMNS = ("FLAG", "FLAG_ROW")

_LIST_FILE = "FLAG_VERSION_LIST"


@dataclass
class FlagVersion:
    """One saved flag version."""

    name: str
    description: str = ""
    path: str = ""
    nrows: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "nrows": self.nrows,
        }


def _versions_dir(msname: str) -> str:
    return os.path.abspath(msname).rstrip(os.sep) + ".flagversions"


def _version_path(msname: str, name: str) -> str:
    return os.path.join(_versions_dir(msname), "flags." + name)


def flag_backup(
    msname: str, name: str | None = None, description: str = "", overwrite: bool = False
) -> FlagVersion:
    """Save the current FLAG/FLAG_ROW columns as a named version.

    Args:
        msname: MS to snapshot.
        name: Version name. Defaults to a UTC timestamp.
        description: Free-text note stored alongside the version.
        overwrite: Replace an existing version of the same name.

    Returns:
        The :class:`FlagVersion` written.
    """
    # `None` means "pick a default"; an explicit empty string is a mistake and
    # is reported as one rather than silently becoming a timestamp.
    if name is None:
        name = datetime.datetime.now(datetime.UTC).strftime("backup_%Y%m%d-%H%M%S")
    _validate_name(name)

    target = _version_path(msname, name)
    if os.path.exists(target):
        if not overwrite:
            raise FileExistsError(
                f"flag version {name!r} already exists at {target}; pass overwrite=True "
                "to replace it"
            )
        shutil.rmtree(target)
    os.makedirs(_versions_dir(msname), exist_ok=True)

    with open_table(msname) as tab:
        nrows = tab.nrows()
        with query("SELECT {} FROM $1".format(", ".join(_FLAG_COLUMNS)), [tab]) as selection:
            copied = selection.copy(target, deep=True)
            copied.close()

    version = FlagVersion(name=name, description=description, path=target, nrows=nrows)
    _append_to_list(msname, version)
    LOGGER.info("Saved flag version %r (%d rows) to %s", name, nrows, target)
    return version


def flag_restore(msname: str, name: str, rowchunk: int = 100000) -> FlagVersion:
    """Restore FLAG/FLAG_ROW from a saved version.

    Args:
        msname: MS to restore into.
        name: Version name, as passed to :func:`flag_backup`.
        rowchunk: Rows per write.
    """
    source = _version_path(msname, name)
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"no flag version {name!r} for {msname}; have {[v.name for v in flag_versions(msname)]}"
        )

    with open_table(msname, readonly=False) as tab, open_table(source) as saved:
        if saved.nrows() != tab.nrows():
            raise ValueError(
                f"flag version {name!r} has {saved.nrows()} rows but {msname} has {tab.nrows()}; the MS has "
                "changed since the backup"
            )

        nrows = tab.nrows()
        for row0 in range(0, nrows, rowchunk):
            nr = min(rowchunk, nrows - row0)
            for column in _FLAG_COLUMNS:
                tab.putcol(column, saved.getcol(column, row0, nr), row0, nr)
            LOGGER.info("Restored %s (rows %d to %d)", name, row0, row0 + nr - 1)

    LOGGER.info("Restored flag version %r into %s", name, msname)
    return FlagVersion(name=name, path=source, nrows=nrows)


def flag_versions(msname: str) -> list[FlagVersion]:
    """List the saved flag versions for an MS, newest last."""
    root = _versions_dir(msname)
    if not os.path.isdir(root):
        return []

    descriptions = _read_list(msname)
    versions = []
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("flags."):
            continue
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        name = entry[len("flags.") :]
        nrows = 0
        try:
            with open_table(path) as tab:
                nrows = tab.nrows()
        except RuntimeError:  # not a readable table
            LOGGER.warning("Skipping unreadable flag version at %s", path)
            continue
        versions.append(
            FlagVersion(name=name, description=descriptions.get(name, ""), path=path, nrows=nrows)
        )
    return versions


def flag_delete(msname: str, name: str) -> None:
    """Delete a saved flag version."""
    target = _version_path(msname, name)
    if not os.path.exists(target):
        raise FileNotFoundError(f"no flag version {name!r} for {msname}")
    shutil.rmtree(target)
    _rewrite_list(msname, [v for v in flag_versions(msname) if v.name != name])
    LOGGER.info("Deleted flag version %r", name)


def _validate_name(name: str) -> None:
    if not name or os.sep in name or name.startswith("."):
        raise ValueError(
            f"invalid flag version name {name!r}: must be non-empty, contain no "
            "path separator, and not start with '.'"
        )


def _append_to_list(msname: str, version: FlagVersion) -> None:
    existing = {v.name: v.description for v in flag_versions(msname)}
    existing[version.name] = version.description
    _rewrite_list(msname, [FlagVersion(name=n, description=d) for n, d in sorted(existing.items())])


def _rewrite_list(msname: str, versions: list[FlagVersion]) -> None:
    path = os.path.join(_versions_dir(msname), _LIST_FILE)
    try:
        with open(path, "w", encoding="utf-8") as stream:
            stream.writelines(f"{version.name} : {version.description}\n" for version in versions)
    except OSError as exc:  # pragma: no cover - permissions
        LOGGER.warning("Could not update %s: %s", path, exc)


def _read_list(msname: str) -> dict:
    path = os.path.join(_versions_dir(msname), _LIST_FILE)
    out = {}
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if " : " in line:
                    name, _, description = line.partition(" : ")
                    out[name.strip()] = description.strip()
    except OSError:
        pass
    return out
