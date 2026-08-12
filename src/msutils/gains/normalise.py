"""Divide a scale out of a set of gains, leaving their shape behind.

A calibration chain solves several terms against the same data, and where
the overall amplitude scale ends up among them is not physics -- it is an
artefact of how the solver was run. CASA's sequential chain leaves a
bandpass with a mean amplitude of essentially exactly 1 because `B` is
solved *after* `G` and is therefore a residual; QuartiCal solves the chain
jointly and makes no such promise, so the flat part of the amplitude can sit
in either term. Both are self-consistent as long as every term is applied
together, and both are wrong the moment a caller takes one term out of the
set -- transfers a bandpass to another observation, say, or seeds a chain
with a subset of another chain's products.

Normalising makes that explicit: the scale goes where you say it goes,
rather than where the solver happened to leave it.

## The three choices, and why each is a choice

**What to average** (`statistic`) -- `median` or `mean`. Shared with
`fluxscale` (see `msutils.gains._stats`); the median resists a bad scan and
is the default, the mean is what CASA's `solnorm` computes.

**Over which axes** (`axis`) -- `freq` normalises each solution's spectrum
to unit amplitude and keeps its time variation, which is bandpass
normalisation; `time` does the reverse; `all` (the default) takes out one
number per solution set, leaving both structures intact.

**Over what scope** (`scope`) -- and this is the one with teeth:

* `antenna` (default) gives every (antenna, correlation) its own factor.
  This is CASA `solnorm`'s behaviour, and it does **not** preserve the
  relative amplitudes between antennas: those move wholesale into whatever
  term is applied alongside. That is exactly right for separating a
  bandpass's shape from a gain's scale, and exactly wrong if you meant to
  take out only an overall scale.
* `block` uses one factor for the whole block, preserving every relative
  amplitude -- between antennas, between correlations -- and removing only
  the overall level. This is the one to use when moving a flux scale
  between terms.
* `correlation` sits between them: one factor per correlation, so relative
  antenna amplitudes survive but the two polarisations are levelled
  independently.

Phases are never touched. A normalisation divides by a positive real
number, so what changes is amplitude alone; nothing here re-references
phase to an antenna, which is a different operation with a different name.

For a **parameterised** term the parameters are what QuartiCal reads back,
so they are what gets divided -- and only when they are amplitudes. A
delay or phase parameterisation describes a gain of unit modulus, where
normalising is not unsupported but meaningless (see
`msutils.gains._params`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from msutils._log import create_logger

from ._io import read_gains, write_gains
from ._model import GainBlock, GainTable
from ._params import scale_block
from ._stats import amplitude_statistic, check_statistic

__all__ = ["AXIS_CHOICES", "SCOPES", "NormaliseResult", "NormalisedBlock", "normalise"]

LOGGER = create_logger(__name__)

#: Bumped on any breaking change to the JSON report.
SCHEMA_VERSION = 1

#: Which axes the normalisation factor is computed over.
AXIS_CHOICES = ("all", "time", "freq")

#: How widely one factor applies. See the module docstring -- `antenna` does
#: not preserve relative antenna amplitudes and `block` does, which is the
#: difference between "separate shape from scale" and "take out a scale".
SCOPES = ("antenna", "correlation", "block")

#: Axis -> the `AXES` positions it reduces over, before scope is applied.
_AXIS_INDICES = {"all": (0, 1), "time": (0,), "freq": (1,)}

#: Scope -> the extra `AXES` positions it reduces over (antenna, correlation).
_SCOPE_INDICES = {"antenna": (), "correlation": (2,), "block": (2, 3)}

#: Gain types a normalisation is meaningless for: a delay is a slope in
#: frequency, not an amplitude, so dividing it by its own average is
#: arithmetic without a meaning. Matched case-insensitively as a substring,
#: which covers QuartiCal's `delay_and_offset` and CASA's `K Jones` alike.
_NOT_AMPLITUDES = ("delay", "tec", "k jones", "kcross")


@dataclass
class NormalisedBlock:
    """What one block's normalisation took out.

    The factors are the point of the record, not a diagnostic: they are the
    scale that *left* this table, and applying these gains without knowing
    it is how a flux scale goes missing. `factor_summary` is the median of
    them, for the common case of wanting one number.
    """

    field_id: int | None
    spw_id: int | None
    direction: int
    factor_summary: float
    factors: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class NormaliseResult:
    """Everything one `normalise` run did."""

    statistic: str
    axis: str
    scope: str
    blocks: list[NormalisedBlock]
    input_path: str = ""
    output_path: str = ""
    term: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "statistic": self.statistic,
            "axis": self.axis,
            "scope": self.scope,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "term": self.term,
            "blocks": [asdict(b) for b in self.blocks],
        }

    def render(self) -> str:
        """The report as text, one line per block."""
        return "\n".join(
            f"# Normalised field={b.field_id} spw={b.spw_id} direction={b.direction} "
            f"by {self.statistic} over {self.axis} per {self.scope}: factor {b.factor_summary:.6g}"
            for b in self.blocks
        )

    def save(self, path: str | Path) -> str:
        """Write the report as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return str(path)


def _check_normalisable(gains: GainTable) -> None:
    kind = f"{gains.gain_type} {gains.term}".lower()
    if any(name in kind for name in _NOT_AMPLITUDES):
        raise ValueError(
            f"{gains.path or 'these gains'} hold {gains.gain_type or gains.term!r}, which is not an "
            "amplitude: a delay is a slope in frequency, so dividing it by its own average means "
            "nothing. Normalisation applies to gain-like terms (G, B, and their QuartiCal equivalents)."
        )


def _factors(block: GainBlock, *, statistic: str, axis: str, scope: str) -> np.ndarray:
    """The positive real factor to divide each solution by, broadcastable
    against the block."""
    reduce_axes = tuple(sorted(set(_AXIS_INDICES[axis]) | set(_SCOPE_INDICES[scope])))
    factor = amplitude_statistic(block, statistic=statistic, axis=reduce_axes, keepdims=True)
    # A factor of zero or one that is entirely flagged means "nothing to
    # measure here"; dividing by 1 leaves those solutions exactly as they
    # were, which is the only answer that cannot invent values.
    return np.ma.filled(np.ma.masked_equal(factor, 0), 1.0).astype(float)


def normalise(
    gains: str | Path | GainTable,
    *,
    output: str | Path | None = None,
    statistic: str = "median",
    axis: str = "all",
    scope: str = "antenna",
    term: str | None = None,
    json_out: str | Path | None = None,
) -> NormaliseResult:
    """Normalise gain amplitudes, dividing out a scale.

    Args:
        gains: Table/store to read, or an already-read
            :class:`~msutils.gains.GainTable`.
        output: Write the normalised gains here (never in place).
        statistic: ``"median"`` (default, robust) or ``"mean"`` (CASA's
            `solnorm`).
        axis: ``"all"`` (default), ``"time"`` or ``"freq"`` -- which axes the
            factor is averaged over. ``"freq"`` is bandpass normalisation.
        scope: ``"antenna"`` (default, per antenna and correlation, CASA's
            behaviour), ``"correlation"`` or ``"block"``. Only ``"block"``
            preserves the relative amplitudes between antennas -- read the
            module docstring before choosing, because the difference is a
            science decision rather than a formatting one.
        term: Which term to read, for QuartiCal stores holding several.
        json_out: Write the report to this JSON file.

    Returns:
        A :class:`NormaliseResult` recording the factors taken out -- which
        is the scale that left the table, and therefore something a caller
        has to be able to find later.

    Raises:
        ValueError: On an unknown statistic/axis/scope, or on a delay-like
            term, where the operation has no meaning.
    """
    check_statistic(statistic)
    if axis not in AXIS_CHOICES:
        raise ValueError(f"unknown axis {axis!r} (known: {list(AXIS_CHOICES)})")
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} (known: {list(SCOPES)})")

    table = gains if isinstance(gains, GainTable) else read_gains(gains, term=term)
    _check_normalisable(table)

    records: list[NormalisedBlock] = []

    def _normalise_block(block: GainBlock) -> GainBlock:
        factor = _factors(block, statistic=statistic, axis=axis, scope=scope)
        # Per (antenna, correlation), for the record: whatever the scope, the
        # useful thing to write down is what each antenna's amplitude was
        # divided by.
        broadcast = np.broadcast_to(factor, block.shape)
        per_antenna = {
            label: [float(broadcast[..., i, c].flat[0]) for c in range(len(block.corrs))]
            for i, label in enumerate(block.antennas)
        }
        records.append(
            NormalisedBlock(
                field_id=block.field_id,
                spw_id=block.spw_id,
                direction=block.direction,
                factor_summary=float(np.median(factor)),
                factors=per_antenna,
            )
        )
        return scale_block(block, factor, what="normalise")

    normalised = table.map(_normalise_block)
    result = NormaliseResult(
        statistic=statistic,
        axis=axis,
        scope=scope,
        blocks=records,
        input_path=table.path,
        term=table.term,
    )
    for line in result.render().splitlines():
        LOGGER.info(line)

    if output is not None:
        result.output_path = write_gains(normalised, output, template=table.path)
    if json_out is not None:
        result.save(json_out)
    return result
