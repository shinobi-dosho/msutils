"""Command-line interface for gain operations (``gainutils`` console script).

Its own script rather than a subcommand group under ``msutils``, because
these act on calibration solutions rather than on Measurement Sets, and
because a cab wrapping ``gainutils fluxscale`` should not have to know that
the two share a package.
"""

from __future__ import annotations

import click

from .fluxscale import fluxscale as _fluxscale
from .normalise import normalise as _normalise

try:
    from importlib.metadata import version as _version

    __version__ = _version("msutils")
except Exception:  # not installed / metadata missing
    __version__ = "unknown"


@click.group()
@click.version_option(__version__, prog_name="gainutils")
def cli():
    """Operations on calibration gain tables."""


@cli.command()
@click.argument("transfer")
@click.option(
    "--reference",
    default=None,
    help="Gain table/store of the calibrator with a known flux. Defaults to TRANSFER, "
    "i.e. both fields in one table.",
)
@click.option("--transfer-field", required=True, help="Field id or name whose flux is unknown.")
@click.option("--reference-field", required=True, help="Field id or name to measure against.")
@click.option(
    "--reference-flux",
    type=float,
    default=1.0,
    show_default=True,
    help="Flux density (Jy) the reference was solved against. 1.0 whenever its model carried "
    "its true flux, which leaves its gains instrumental.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Write the rescaled transfer gains here. Never written in place; must not exist.",
)
@click.option(
    "--json",
    "json_out",
    default=None,
    type=click.Path(),
    help="Write the measurement to this JSON file. The number exists nowhere else once "
    "the process exits, so a pipeline should always ask for it.",
)
@click.option(
    "--gain-threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="Drop gain amplitudes deviating from the median by more than this fraction. 0 disables.",
)
@click.option(
    "--statistic",
    type=click.Choice(["median", "mean"]),
    default="median",
    show_default=True,
    help="How to collapse each antenna's amplitudes over time/frequency. "
    "'median' resists a bad scan; 'mean' is what reproduces CASA. On real data "
    "they differ by ~0.5%, which is larger than either one's formal error.",
)
@click.option("--term", default=None, help="Term to read from a QuartiCal store holding several.")
def fluxscale(
    transfer,
    reference,
    transfer_field,
    reference_field,
    reference_flux,
    output,
    json_out,
    gain_threshold,
    statistic,
    term,
):
    """Bootstrap TRANSFER's flux density from a reference calibrator's gains.

    Both calibrators must have been solved the same way, the reference
    against a model carrying its true flux and the transfer against an
    assumed one (conventionally 1 Jy). What the two solutions disagree
    about is then the sky, and the ratio of their median gain amplitudes
    squared is the transfer calibrator's flux density.
    """
    result = _fluxscale(
        transfer,
        reference,
        transfer_field=transfer_field,
        reference_field=reference_field,
        reference_flux=reference_flux,
        output=output,
        gain_threshold=gain_threshold,
        statistic=statistic,
        term=term,
        json_out=json_out,
    )
    click.echo(result.render())


if __name__ == "__main__":  # pragma: no cover
    cli()


@cli.command()
@click.argument("gains")
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Write the normalised gains here. Never written in place; must not exist.",
)
@click.option(
    "--statistic",
    type=click.Choice(["median", "mean"]),
    default="median",
    show_default=True,
    help="Average to divide out. 'median' resists a bad scan; 'mean' is CASA's solnorm.",
)
@click.option(
    "--axis",
    type=click.Choice(["all", "time", "freq"]),
    default="all",
    show_default=True,
    help="Axes the factor is averaged over. 'freq' is bandpass normalisation "
    "(unit amplitude per spectrum, time variation kept); 'all' takes out one level.",
)
@click.option(
    "--scope",
    type=click.Choice(["antenna", "correlation", "block"]),
    default="antenna",
    show_default=True,
    help="How widely one factor applies. 'antenna' matches CASA and does NOT preserve "
    "relative antenna amplitudes -- they move into whatever term is applied alongside. "
    "'block' preserves them and removes only the overall level, which is what moving a "
    "flux scale between terms needs.",
)
@click.option(
    "--json",
    "json_out",
    default=None,
    type=click.Path(),
    help="Write the factors taken out to this JSON file. That scale has left the table; "
    "a pipeline that cannot find it later cannot put it back.",
)
@click.option("--term", default=None, help="Term to read from a QuartiCal store holding several.")
def normalise(gains, output, statistic, axis, scope, json_out, term):
    """Divide a scale out of GAINS, leaving their shape behind.

    Where the overall amplitude sits among a chain's terms is an artefact of
    how the solver was run -- CASA's sequential chain leaves a bandpass at
    unit amplitude because it is solved last, QuartiCal's joint solve makes
    no such promise. This puts the scale where you say it goes.
    """
    result = _normalise(
        gains,
        output=output,
        statistic=statistic,
        axis=axis,
        scope=scope,
        term=term,
        json_out=json_out,
    )
    click.echo(result.render())
