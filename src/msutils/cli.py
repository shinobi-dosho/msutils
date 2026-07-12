"""Command-line interface for msutils (``msutils`` console script)."""
import click

from . import _ms

try:
    from importlib.metadata import version as _version
    __version__ = _version("msutils")
except Exception:  # not installed / metadata missing
    __version__ = "unknown"


@click.group()
@click.version_option(__version__, prog_name="msutils")
def cli():
    """CASA Measurement Set manipulation utilities."""


@cli.command()
@click.argument("ms")
@click.option("--json", "outfile", type=click.Path(), default=None,
              help="Also write the summary to this JSON file.")
@click.option("--quiet", is_flag=True, help="Do not print the summary.")
def summary(ms, outfile, quiet):
    """Summarise MS (fields, SPWs, antennas, scans, correlations)."""
    _ms.summary(ms, outfile=outfile, display=not quiet)


@cli.command()
@click.argument("ms")
@click.argument("colname")
@click.option("--clone", default="DATA", show_default=True,
              help="Existing column to clone shape/type from.")
@click.option("--valuetype", default=None,
              help="Column value type, e.g. 'complex', 'float', 'scalar'.")
@click.option("--init-with", "init_with", type=float, default=None,
              help="Initialise the new column with this (float) value.")
def addcol(ms, colname, clone, valuetype, init_with):
    """Add column COLNAME to MS."""
    result = _ms.addcol(ms, colname, clone=clone, valuetype=valuetype,
                        init_with=init_with)
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
@click.option("--subtract", is_flag=True,
              help="Subtract the second column from the first (needs exactly 2 cols).")
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
@click.option("--column", default="MODEL_DATA", show_default=True,
              help="Column to write noisy visibilities into.")
@click.option("--sefd", type=float, default=551.0, show_default=True,
              help="SEFD (Jy) used to compute per-visibility noise.")
@click.option("--noise", type=float, default=0.0,
              help="Noise stddev; if 0, computed from --sefd.")
@click.option("--add-to", "add_to", default=None,
              help="Add the noise to this column's data instead of pure noise.")
def addnoise(ms, column, sefd, noise, add_to):
    """Add Gaussian noise to MS."""
    _ms.addnoise(ms, column=column, sefd=sefd, noise=noise, addToCol=add_to)


@cli.command()
@click.argument("ms")
@click.option("--html", "htmlfile", type=click.Path(), default=None,
              help="Output HTML plot file.")
@click.option("--json", "outfile", type=click.Path(), default=None,
              help="Output JSON statistics file.")
@click.option("--field", "fields", multiple=True,
              help="Field id/name to include (repeatable).")
@click.option("--antenna", "antennas", multiple=True,
              help="Antenna id/name to include (repeatable).")
def flagstats(ms, htmlfile, outfile, fields, antennas):
    """Compute + plot flag statistics (requires the 'flagstats' extra)."""
    try:
        from . import flag_stats
    except ImportError:
        raise click.ClickException(
            "flagstats needs extra dependencies. Install with: "
            "pip install 'msutils[flagstats]'")
    flag_stats.plot_statistics(
        ms,
        antennas=list(antennas) or None,
        fields=list(fields) or None,
        htmlfile=htmlfile,
        outfile=outfile,
    )


if __name__ == "__main__":
    cli()
