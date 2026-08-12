"""Bootstrap a flux-density scale from one calibrator's gains onto another's.

The physics is one assumption: **the antenna gains are the same, on
average, for both calibrators**. They are observed minutes apart through
the same instrument, so whatever differs between their solutions is the
sky, not the receiver.

That is exploited by solving the two deliberately differently. The
reference is solved against a model with its true flux density, so its
gains come out purely instrumental. The transfer calibrator's flux is
unknown, so it is solved against an assumed model -- conventionally a 1 Jy
point source -- and the solver, having nowhere else to put the discrepancy,
absorbs it into the gain amplitudes::

    |g_transfer| = |g_instrumental| * sqrt(S_true / S_assumed)

So the flux falls out of a ratio, per antenna and correlation, of the
median gain amplitude::

    S = (median|g_transfer| / median|g_reference|)^2 * S_reference

and the *scatter* of that ratio across antennas is the uncertainty. The
transfer gains are then divided by ``sqrt(S)`` so they too become
instrumental, which is what makes them applicable to a target.

## What this does that CASA's `fluxscale` does not

* **Two tables, not one.** CASA compares fields *within* a single caltable,
  which forces a caller to solve the second calibrator by appending into a
  copy of the first one's table. Here the reference and the transfer are
  separate inputs (the same path twice is still allowed, for a table that
  does hold both).
* **The number is a value, not a log line.** `fluxscale()` returns a
  :class:`FluxscaleResult`; the CLI can write it as JSON. CASA writes the
  fit to a `listfile` if asked and to its logger otherwise, and a pipeline
  that discards the logger discards the only record of the measurement.
* **The output holds the transfer field alone.** CASA's fluxtable carries
  the reference calibrator's solutions through as well, so applying it
  without naming a field corrects with a blend of two calibrators.
* **Antennas are matched by name**, not by position, so two tables that
  disagree on antenna ordering produce an error rather than a wrong number.

It is deliberately *not* a re-implementation of everything CASA's task
does: there is no spectral fit across spectral windows yet (each window is
reported on its own), and no incremental mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from msutils._log import create_logger

from ._io import read_gains, write_gains
from ._model import GainBlock, GainTable

__all__ = ["STATISTICS", "FluxDensity", "FluxscaleResult", "fluxscale"]

LOGGER = create_logger(__name__)

#: Bumped on any breaking change to the JSON report.
SCHEMA_VERSION = 1


@dataclass
class FluxDensity:
    """One spectral window's bootstrapped flux density.

    Attributes:
        flux_jy: The measured flux density of the transfer field.
        error_jy: Standard error of the mean over the (antenna,
            correlation) samples -- the spread of the per-antenna answers,
            which is what makes a bad antenna visible instead of merely
            wrong. **Not comparable to CASA's**, which propagates per-solution
            errors and comes out several times smaller on the same data; this
            one asks "how much do the antennas disagree", which is the
            question a bootstrap can actually fail at.
        snr: `flux_jy / error_jy`.
        n_samples: How many (antenna, correlation) pairs contributed. For a
            healthy bootstrap this is every antenna times every
            correlation; a smaller number means solutions were flagged.
        per_antenna: Antenna label -> its own flux estimate, averaged over
            correlations. Kept because the failure mode this measurement
            has is one antenna disagreeing, and a single number cannot show
            that.
    """

    spw_id: int | None
    frequency_hz: float
    flux_jy: float
    error_jy: float
    snr: float
    n_samples: int
    per_antenna: dict[str, float] = field(default_factory=dict)

    def render(self, field_name: str) -> str:
        """The measurement as CASA's own `listfile` phrases it.

        Deliberately identical in shape, so a bootstrapped flux from this
        tool and one from CASA can be compared by eye or by diff.
        """
        return (
            f"# Flux density for {field_name} in SpW={self.spw_id} "
            f"(freq={self.frequency_hz:.6g} Hz) is: {self.flux_jy:.6g} "
            f"+/- {self.error_jy:.6g} (SNR = {self.snr:.6g}, N = {self.n_samples})"
        )


@dataclass
class FluxscaleResult:
    """Everything one `fluxscale` run measured and did."""

    transfer_field: str
    reference_field: str
    reference_flux_jy: float
    fluxes: list[FluxDensity]
    transfer_path: str = ""
    reference_path: str = ""
    term: str = ""
    output_path: str = ""
    correlation_matching: str = "label"
    antennas_used: tuple[str, ...] = ()
    statistic: str = "median"
    schema_version: int = SCHEMA_VERSION

    @property
    def flux_jy(self) -> float:
        """The single-window flux, for the common one-window case."""
        if len(self.fluxes) != 1:
            raise ValueError(
                f"this result covers {len(self.fluxes)} spectral windows -- "
                "read `fluxes` rather than asking for one number"
            )
        return self.fluxes[0].flux_jy

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "transfer_field": self.transfer_field,
            "reference_field": self.reference_field,
            "reference_flux_jy": self.reference_flux_jy,
            "transfer_path": self.transfer_path,
            "reference_path": self.reference_path,
            "term": self.term,
            "output_path": self.output_path,
            "correlation_matching": self.correlation_matching,
            "statistic": self.statistic,
            "antennas_used": list(self.antennas_used),
            "fluxes": [asdict(f) for f in self.fluxes],
        }

    def render(self) -> str:
        """The report as text, one line per spectral window."""
        return "\n".join(f.render(self.transfer_field) for f in self.fluxes)

    def save(self, path: str | Path) -> str:
        """Write the report as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return str(path)


#: How a block's gain amplitudes are collapsed onto one value per (antenna,
#: correlation). The choice is worth a knob rather than a constant: measured
#: against a real MeerKAT bootstrap (10 antennas, 3 reference scans, 16
#: transfer scans) the two differ by **0.5%** -- 5.969 Jy by median against
#: 5.943 Jy by mean, where CASA's own task reported 5.938 Jy. The median is
#: the default because one bad scan should not move a flux scale; `mean` is
#: the one to use when reproducing CASA, whose behaviour it matches to
#: within 0.1% despite its documentation saying "median gain norm".
STATISTICS = ("median", "mean")


def _average_amplitude(block: GainBlock, threshold: float, statistic: str) -> np.ma.MaskedArray:
    """|g| collapsed over time and frequency, per (antenna, correlation).

    Time and frequency both: a `G` solve has one channel and many times, a
    `B` solve the reverse, and the statistic that makes both work is one
    taken over whatever axes the term actually resolves.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r} (known: {list(STATISTICS)})")
    amplitude = np.ma.abs(block.masked())
    if threshold > 0:
        # CASA's `gainthreshold`: drop solutions deviating from the median by
        # more than this fraction, so one bad interval cannot drag the average
        # of a short scan. Always measured from the median, whatever the
        # statistic -- an outlier test against a mean the outlier is already in
        # is a weaker test.
        median = np.ma.median(amplitude, axis=(0, 1))
        deviation = np.ma.abs(amplitude - median) / np.ma.masked_equal(median, 0)
        amplitude = np.ma.masked_where(deviation > threshold, amplitude)
    if statistic == "mean":
        return amplitude.mean(axis=(0, 1))
    return np.ma.median(amplitude, axis=(0, 1))


def _align(reference: GainBlock, transfer: GainBlock) -> tuple[list[int], list[int], list[int], list[int], str]:
    """Index maps putting the two blocks' antennas and correlations in step.

    Antennas match by label and nothing else: a positional match between two
    tables that happen to order antennas differently is the kind of error
    that produces a believable number. Correlations match by label when both
    sides carry real ones (a CASA caltable records no ``CORR_TYPE``, so it
    cannot), and fall back to position when the counts agree -- recorded in
    the result either way.
    """
    shared = [a for a in transfer.antennas if a in set(reference.antennas)]
    if not shared:
        raise ValueError(
            f"the two tables share no antennas: reference has {list(reference.antennas)}, "
            f"transfer has {list(transfer.antennas)}"
        )
    ref_ants = [reference.antennas.index(a) for a in shared]
    trn_ants = [transfer.antennas.index(a) for a in shared]

    if set(reference.corrs) == set(transfer.corrs):
        matching = "label"
        shared_corrs = [c for c in transfer.corrs if c in set(reference.corrs)]
        ref_corrs = [reference.corrs.index(c) for c in shared_corrs]
        trn_corrs = [transfer.corrs.index(c) for c in shared_corrs]
    elif len(reference.corrs) == len(transfer.corrs):
        matching = "position"
        ref_corrs = trn_corrs = list(range(len(transfer.corrs)))
    else:
        raise ValueError(
            f"cannot match correlations: reference has {list(reference.corrs)}, "
            f"transfer has {list(transfer.corrs)}"
        )
    return ref_ants, trn_ants, ref_corrs, trn_corrs, matching


def _measure(
    reference: GainBlock,
    transfer: GainBlock,
    *,
    reference_flux: float,
    threshold: float,
    statistic: str,
) -> tuple[FluxDensity, str, tuple[str, ...]]:
    """One spectral window's flux density, from the two blocks' gains."""
    ref_ants, trn_ants, ref_corrs, trn_corrs, matching = _align(reference, transfer)
    ref_median = _average_amplitude(reference, threshold, statistic)[np.ix_(ref_ants, ref_corrs)]
    trn_median = _average_amplitude(transfer, threshold, statistic)[np.ix_(trn_ants, trn_corrs)]

    ratio = trn_median / np.ma.masked_equal(ref_median, 0)
    samples = np.ma.filled(ratio.astype(float) ** 2 * reference_flux, np.nan)
    valid = np.isfinite(samples)
    if not valid.any():
        raise ValueError(
            "every antenna/correlation is flagged in one table or the other -- there is nothing to compare"
        )
    values = samples[valid]
    flux = float(values.mean())
    # Standard error of the mean. ddof=1 because the antennas are a sample of
    # the array's behaviour, not the whole of it; with one antenna there is no
    # spread to quote and the error is undefined rather than zero.
    error = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
    antennas = [reference.antennas[i] for i in ref_ants]
    per_antenna = {
        antennas[i]: float(np.nanmean(samples[i]))
        for i in range(len(antennas))
        if np.isfinite(samples[i]).any()
    }
    return (
        FluxDensity(
            spw_id=transfer.spw_id,
            frequency_hz=float(np.nanmean(transfer.freqs)),
            flux_jy=flux,
            error_jy=error,
            snr=float(flux / error) if error and np.isfinite(error) else float("nan"),
            n_samples=int(values.size),
            per_antenna=per_antenna,
        ),
        matching,
        tuple(antennas),
    )


def fluxscale(
    transfer: str | Path | GainTable,
    reference: str | Path | GainTable | None = None,
    *,
    transfer_field: int | str,
    reference_field: int | str,
    reference_flux: float = 1.0,
    output: str | Path | None = None,
    gain_threshold: float = 0.0,
    statistic: str = "median",
    term: str | None = None,
    json_out: str | Path | None = None,
) -> FluxscaleResult:
    """Measure a calibrator's flux density from another's gains, and rescale.

    Args:
        transfer: Gains of the calibrator whose flux is unknown -- a path
            (CASA caltable or QuartiCal store) or an already-read
            :class:`~msutils.gains.GainTable`.
        reference: Gains of the calibrator with a known flux. Defaults to
            `transfer`, i.e. both fields in one table, which is the
            arrangement CASA's own task requires.
        transfer_field: Field id or name to measure.
        reference_field: Field id or name to measure against.
        reference_flux: Flux density (Jy) the reference was solved against.
            1.0 is right whenever its model carried its true flux, which is
            the normal case -- its gains are then already instrumental.
        output: Write the rescaled transfer gains here (never in place).
        gain_threshold: Drop gain amplitudes deviating from the median by
            more than this fraction before averaging. 0 disables.
        statistic: How each antenna's amplitudes are collapsed over time and
            frequency -- ``"median"`` (default, robust) or ``"mean"`` (what
            reproduces CASA). See `STATISTICS`: on real data the two differ
            by about half a percent, which is larger than either one's
            formal error, so it is a choice worth making deliberately.
        term: Which term to read, for QuartiCal stores holding several.
        json_out: Write the report to this JSON file.

    Returns:
        A :class:`FluxscaleResult`: one flux density per spectral window,
        plus what was matched and what was written.

    Raises:
        ValueError: If the fields, spectral windows, antennas or
            correlations cannot be matched -- every one of which would
            otherwise yield a plausible wrong number.
    """
    transfer_gains = transfer if isinstance(transfer, GainTable) else read_gains(transfer, term=term)
    if reference is None:
        reference_gains = transfer_gains
    elif isinstance(reference, GainTable):
        reference_gains = reference
    else:
        reference_gains = read_gains(reference, term=term)

    transfer_id = transfer_gains.resolve_field(transfer_field)
    reference_id = reference_gains.resolve_field(reference_field)
    if transfer_gains.path == reference_gains.path and transfer_id == reference_id:
        raise ValueError(
            f"the reference and transfer field are both {transfer_id} in the same table -- "
            "a calibrator cannot bootstrap its own flux scale"
        )

    transfer_blocks = {b.spw_id: b for b in transfer_gains.select(field=transfer_id)}
    reference_blocks = {b.spw_id: b for b in reference_gains.select(field=reference_id)}
    shared_spws = sorted(set(transfer_blocks) & set(reference_blocks), key=lambda s: (s is None, s))
    if not shared_spws:
        raise ValueError(
            f"the two fields share no spectral window: transfer has {sorted(transfer_blocks)}, "
            f"reference has {sorted(reference_blocks)}"
        )

    fluxes: list[FluxDensity] = []
    scales: dict[int | None, float] = {}
    matching = "label"
    antennas: tuple[str, ...] = ()
    for spw_id in shared_spws:
        measurement, matching, antennas = _measure(
            reference_blocks[spw_id],
            transfer_blocks[spw_id],
            reference_flux=reference_flux,
            threshold=gain_threshold,
            statistic=statistic,
        )
        fluxes.append(measurement)
        scales[spw_id] = measurement.flux_jy
        LOGGER.info(measurement.render(transfer_gains.field_names.get(transfer_id, str(transfer_id))))

    result = FluxscaleResult(
        transfer_field=transfer_gains.field_names.get(transfer_id, str(transfer_id)),
        reference_field=reference_gains.field_names.get(reference_id, str(reference_id)),
        reference_flux_jy=reference_flux,
        fluxes=fluxes,
        transfer_path=transfer_gains.path,
        reference_path=reference_gains.path,
        term=transfer_gains.term,
        correlation_matching=matching,
        antennas_used=antennas,
        statistic=statistic,
    )

    if output is not None:
        # Dividing by sqrt(S) is what the measurement is *for*: it leaves the
        # transfer calibrator's gains instrumental, like the reference's, so
        # they can be applied to a target without carrying its flux with them.
        rescaled = transfer_gains.select(field=transfer_id).map(
            lambda block: block.with_gains(block.gains / np.sqrt(scales[block.spw_id]))
        )
        result.output_path = write_gains(rescaled, output, template=transfer_gains.path)

    if json_out is not None:
        result.save(json_out)
    return result
