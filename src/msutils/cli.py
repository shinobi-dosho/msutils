"""Command-line interface for msutils (``msutils`` console script)."""

import json

import click

from . import _ms
from .info import msinfo

try:
    from importlib.metadata import version as _version

    __version__ = _version("msutils")
except Exception:  # not installed / metadata missing
    __version__ = "unknown"


@click.group()
@click.version_option(__version__, prog_name="msutils")
def cli():
    """Everyday Measurement Set operations."""


@cli.command()
@click.argument("ms")
@click.option(
    "-v", "--verbose", is_flag=True, help="Also list scans, antennas and columns in full."
)
@click.option(
    "--level",
    type=click.Choice(["meta", "full", "data"]),
    default="full",
    show_default=True,
    help="meta: subtables only (no main-table scan); full: one pass "
    "over the index columns; data: also read UVW and FLAG.",
)
@click.option(
    "--json",
    "outfile",
    type=click.Path(),
    default=None,
    help="Write the structured metadata to this JSON file.",
)
@click.option(
    "--json-stdout", is_flag=True, help="Print JSON to stdout instead of the text report."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["MSv2", "MSv4"]),
    default=None,
    help="Force the format instead of auto-detecting.",
)
@click.option(
    "--engine",
    type=click.Choice(["casacore", "zarr", "xradio", "xarray-ms"]),
    default=None,
    help="How to read the data. MSv2 defaults to casacore (TaQL); "
    "'xarray-ms' reads an MSv2 through the MSv4 schema.",
)
def info(ms, verbose, level, outfile, json_stdout, fmt, engine):
    """Report MS metadata: fields, SPWs, scans, antennas, columns.

    Works on both MSv2 tables and MSv4 processing sets.
    """
    result = msinfo(ms, level=level, outfile=outfile, format=fmt, engine=engine)
    click.echo(result.to_json() if json_stdout else result.render(verbose=verbose))


@cli.command()
@click.argument("ms")
@click.option(
    "--json",
    "outfile",
    type=click.Path(),
    default=None,
    help="Also write the summary to this JSON file.",
)
@click.option("--quiet", is_flag=True, help="Do not print the summary.")
def summary(ms, outfile, quiet):
    """Deprecated alias for `info` (legacy dict layout)."""
    click.echo("warning: 'msutils summary' is deprecated; use 'msutils info'.", err=True)
    _ms.summary(ms, outfile=outfile, display=not quiet)


@cli.command()
@click.argument("ms")
@click.argument("colname")
@click.option(
    "--clone", default="DATA", show_default=True, help="Existing column to clone shape/type from."
)
@click.option(
    "--valuetype", default=None, help="Column value type, e.g. 'complex', 'float', 'scalar'."
)
@click.option(
    "--init-with",
    "init_with",
    type=float,
    default=None,
    help="Initialise the new column with this (float) value.",
)
def addcol(ms, colname, clone, valuetype, init_with):
    """Add column COLNAME to MS."""
    result = _ms.addcol(ms, colname, clone=clone, valuetype=valuetype, init_with=init_with)
    click.echo(result)


@cli.command()
@click.argument("ms")
@click.argument("fromcol")
@click.argument("tocol")
def copycol(ms, fromcol, tocol):
    """Copy column FROMCOL to TOCOL in MS (creating TOCOL if needed)."""
    _ms.copycol(ms, fromcol, tocol)


@cli.command()
@click.argument("ms")
@click.argument("cols", nargs=-1, required=True)
@click.option("--out", "outcol", required=True, help="Output column.")
@click.option(
    "--subtract",
    is_flag=True,
    help="Subtract the second column from the first (needs exactly 2 cols).",
)
def sumcols(ms, cols, outcol, subtract):
    """Sum COLS (2 or more) into --out, or subtract with --subtract."""
    if subtract:
        if len(cols) != 2:
            raise click.UsageError("--subtract requires exactly two columns.")
        _ms.sumcols(ms, col1=cols[0], col2=cols[1], outcol=outcol, subtract=True)
    else:
        _ms.sumcols(ms, cols=list(cols), outcol=outcol)


@cli.command()
@click.argument("ms")
@click.option(
    "--column",
    default="MODEL_DATA",
    show_default=True,
    help="Column to write noisy visibilities into.",
)
@click.option(
    "--sefd",
    type=float,
    default=551.0,
    show_default=True,
    help="SEFD (Jy) used to compute per-visibility noise.",
)
@click.option("--noise", type=float, default=0.0, help="Noise stddev; if 0, computed from --sefd.")
@click.option(
    "--add-to",
    "add_to",
    default=None,
    help="Add the noise to this column's data instead of pure noise.",
)
def addnoise(ms, column, sefd, noise, add_to):
    """Add Gaussian noise to MS."""
    _ms.addnoise(ms, column=column, sefd=sefd, noise=noise, addToCol=add_to)


@cli.command()
@click.argument("ms")
@click.option(
    "--plot",
    "plotfile",
    type=click.Path(),
    default=None,
    help="Write a PNG summary plot (needs the 'plots' extra).",
)
@click.option(
    "--json", "outfile", type=click.Path(), default=None, help="Output JSON statistics file."
)
@click.option("--field", "fields", multiple=True, help="Field id/name to include (repeatable).")
@click.option(
    "--antenna", "antennas", multiple=True, help="Antenna id/name to include (repeatable)."
)
@click.option(
    "--spw", "spws", multiple=True, help="Spectral window id/name to include (repeatable)."
)
@click.option(
    "--scan", "scans", multiple=True, type=int, help="Scan number to include (repeatable)."
)
@click.option("--quiet", is_flag=True, help="Do not print the report.")
def flagstats(ms, plotfile, outfile, fields, antennas, spws, scans, quiet):
    """Flag statistics per field, scan, antenna, baseline, SPW, corr, channel."""
    # Import the function from the submodule directly: the package root
    # re-exports `flagstats` as a *function*, which shadows the module of the
    # same name, so `from . import flagstats` would not give us the module.
    from .flagstats import flagstats as compute

    stats = compute(
        ms,
        fields=list(fields) or None,
        antennas=list(antennas) or None,
        spws=list(spws) or None,
        scans=list(scans) or None,
        outfile=outfile,
    )
    if plotfile:
        try:
            from ._flagplot import plot_flagstats
        except ImportError as exc:
            raise click.ClickException(
                "plotting needs matplotlib. Install with: pip install 'msutils[plots]'"
            ) from exc
        plot_flagstats(stats, plotfile)
    if not quiet:
        click.echo(stats.render())


@cli.command()
@click.argument("ms")
@click.argument("colnames", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Allow removing columns the MSv2 standard requires.")
def delcol(ms, colnames, force):
    """Remove COLNAMES from MS (e.g. CORRECTED_DATA to reclaim disk)."""
    removed = _ms.delcol(ms, *colnames, force=force)
    click.echo("removed: {}".format(", ".join(removed) if removed else "(nothing)"))


@cli.command()
@click.argument("ms")
@click.argument("fromcol")
@click.argument("tocol")
def renamecol(ms, fromcol, tocol):
    """Rename column FROMCOL to TOCOL in MS."""
    _ms.renamecol(ms, fromcol, tocol)
    click.echo(f"{fromcol} -> {tocol}")


def _selection_options(command):
    """Shared --field/--spw/--scan/--antenna options."""
    for option in reversed(
        [
            click.option(
                "--field", "fields", multiple=True, help="Field id/name to keep (repeatable)."
            ),
            click.option(
                "--spw", "spws", multiple=True, help="Spectral window id/name to keep (repeatable)."
            ),
            click.option(
                "--scan", "scans", multiple=True, type=int, help="Scan number to keep (repeatable)."
            ),
            click.option(
                "--antenna", "antennas", multiple=True, help="Antenna id/name to keep (repeatable)."
            ),
        ]
    ):
        command = option(command)
    return command


@cli.command()
@click.argument("ms")
@click.argument("outms")
@_selection_options
@click.option(
    "--taql",
    "taql_where",
    default=None,
    help="Extra TaQL predicate, ANDed with the other selections.",
)
@click.option("--overwrite", is_flag=True, help="Replace OUTMS if it exists.")
def subset(ms, outms, fields, spws, scans, antennas, taql_where, overwrite):
    """Write the selected rows of MS to a new MS at OUTMS.

    Field and SPW ids are preserved, not renumbered as CASA split does.
    """
    from .subset import subset as _subset

    _subset(
        ms,
        outms,
        fields=list(fields) or None,
        spws=list(spws) or None,
        scans=list(scans) or None,
        antennas=list(antennas) or None,
        taql=taql_where,
        overwrite=overwrite,
    )
    click.echo(outms)


@cli.command()
@click.argument("ms")
@click.argument("outms")
@click.option(
    "--time-bin", type=float, default=1.0, show_default=True, help="Averaging interval in seconds."
)
@click.option(
    "--chan-bin", type=int, default=1, show_default=True, help="Channels per output channel."
)
@_selection_options
@click.option(
    "--datacolumn", default="DATA", show_default=True, help="Visibility column to average."
)
@click.option("--overwrite", is_flag=True, help="Replace OUTMS if it exists.")
def average(ms, outms, time_bin, chan_bin, fields, spws, scans, antennas, datacolumn, overwrite):
    """Time- and channel-average MS into OUTMS (needs the 'average' extra)."""
    from .subset import average as _average

    try:
        _average(
            ms,
            outms,
            time_bin=time_bin,
            chan_bin=chan_bin,
            fields=list(fields) or None,
            spws=list(spws) or None,
            scans=list(scans) or None,
            antennas=list(antennas) or None,
            datacolumn=datacolumn,
            overwrite=overwrite,
        )
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(outms)


@cli.group()
def flags():
    """Save and restore flag versions."""


@flags.command("backup")
@click.argument("ms")
@click.option("--name", default=None, help="Version name (default: timestamp).")
@click.option("--description", default="", help="Note stored with the version.")
@click.option("--overwrite", is_flag=True, help="Replace a version of this name.")
def flags_backup(ms, name, description, overwrite):
    """Save the current flags as a named version."""
    from .flags import flag_backup

    version = flag_backup(ms, name=name, description=description, overwrite=overwrite)
    click.echo(f"{version.name} ({version.nrows} rows)")


@flags.command("restore")
@click.argument("ms")
@click.argument("name")
def flags_restore(ms, name):
    """Restore flags from a saved version."""
    from .flags import flag_restore

    flag_restore(ms, name)
    click.echo(f"restored {name}")


@flags.command("list")
@click.argument("ms")
def flags_list(ms):
    """List saved flag versions."""
    from .flags import flag_versions

    versions = flag_versions(ms)
    if not versions:
        click.echo("(no saved flag versions)")
        return
    for version in versions:
        click.echo(
            f"{version.name:<24} {version.nrows:>10} rows  "
            f"{version.created_utc or '-':<20} {version.description}"
        )


@flags.command("delete")
@click.argument("ms")
@click.argument("name")
def flags_delete(ms, name):
    """Delete a saved flag version."""
    from .flags import flag_delete

    flag_delete(ms, name)
    click.echo(f"deleted {name}")


@cli.command()
@click.argument("ms")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead.")
def du(ms, as_json):
    """Report where an MS's bytes are, by storage manager and subtable."""
    from .diagnostics import du as _du

    usage = _du(ms)
    click.echo(json.dumps(usage.to_dict(), indent=2) if as_json else usage.render())


@cli.command()
@click.argument("ms")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead.")
def check(ms, as_json):
    """Validate MS against the MSv2 standard. Exits non-zero on errors."""
    from .diagnostics import check as _check

    report = _check(ms)
    click.echo(json.dumps(report.to_dict(), indent=2) if as_json else report.render())
    if not report.ok:
        raise SystemExit(1)


@cli.command()
@click.argument("command")
@click.option("--ms", default=None, help="MS bound to $1 in the command.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead.")
def taql(command, ms, as_json):
    """Run a TaQL COMMAND and print the result columns.

    Refer to the MS given by --ms as $1, e.g.

        msutils taql 'SELECT DISTINCT FIELD_ID FROM $1' --ms obs.ms
    """
    from .diagnostics import taql as _taql

    result = _taql(command, msname=ms)
    if as_json:
        click.echo(
            json.dumps(
                {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in result.items()},
                indent=2,
                default=str,
            )
        )
        return
    if not result:
        click.echo("(no columns)")
        return
    for name, values in result.items():
        click.echo(f"{name}: {values}")


@cli.command()
@click.argument("ms")
@click.argument("outpath")
@click.option(
    "--partition",
    "partitions",
    multiple=True,
    type=click.Choice(["FIELD_ID", "SCAN_NUMBER", "STATE_ID", "ANTENNA1"]),
    help="Extra column to partition on (repeatable).",
)
@click.option(
    "--no-pointing", is_flag=True, help="Skip the POINTING subtable (often large and unused)."
)
@click.option("--overwrite", is_flag=True, help="Replace OUTPATH if it exists.")
def convert(ms, outpath, partitions, no_pointing, overwrite):
    """Convert MSv2 at MS to an MSv4 processing set at OUTPATH.

    Needs the 'convert' extra (xradio). Reading an MSv4 set afterwards needs
    only 'msv4'.
    """
    from .convert import to_msv4

    try:
        info = to_msv4(
            ms,
            outpath,
            partition_scheme=list(partitions) or None,
            with_pointing=not no_pointing,
            overwrite=overwrite,
        )
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(info.render())


if __name__ == "__main__":
    cli()
