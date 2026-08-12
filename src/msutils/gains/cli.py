"""Command-line interface for gain operations (``gainutils`` console script).

Its own script rather than a subcommand group under ``msutils``, because
these act on calibration solutions rather than on Measurement Sets, and
because a cab wrapping ``gainutils fluxscale`` should not have to know that
the two share a package.
"""

from __future__ import annotations

import click

from .fluxscale import fluxscale as _fluxscale

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
