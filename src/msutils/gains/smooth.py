"""Smooth gain solutions in time and/or frequency.

Solutions from short scans or weak calibrators are noisy, and that noise is
transferred wholesale onto the target when they are applied. Smoothing
trades resolution for signal -- but only if it is done on the right
quantity, and the obvious way is wrong.

## Why the complex gain, and why renormalise

**Phase cannot be filtered directly.** It is defined modulo 2*pi, so a
running mean over a solution set that crosses +/-pi averages values that are
adjacent on the circle and far apart as numbers, and produces a spike at
every wrap. Unwrapping first only moves the problem: an unwrap decision on
noisy solutions is itself unreliable, and one wrong branch propagates
through everything after it.

Filtering the **complex** gain has no such failure: adjacent phases are
adjacent vectors whatever their numerical values, so the wrap simply is not
represented.

**But vector averaging shrinks the amplitude.** Averaging unit vectors
pointing in different directions gives something shorter than unit length --
the faster the phase varies, the more amplitude a complex filter destroys.
Left uncorrected, smoothing a rapidly-rotating gain would quietly rescale
the flux of everything it is applied to.

So this smooths the complex gain for its *phase* and the amplitude
separately for its *magnitude*, and puts them back together::

    g_smooth = (conv(g) / |conv(g)|) * conv(|g|)

which is what "smooth the complex gain and renormalise" means in practice:
the direction comes from the vector average, and the length is restored from
a filter that never had a wrap to trip over.

## Flags, and gaps

Every convolution here is **normalised**: flagged samples contribute
nothing, and each output is divided by the weight that actually reached it.
That handles the edges of the axis and holes in the middle identically, with
no special cases and no zeros leaking in from beyond the data.

By default the flags survive the operation -- a flagged solution stays
flagged, however many neighbours it now has. `fill=True` unflags anything
whose window contained real data, which is the general form of CASA's
`fillgaps`: use it when the flagged solutions are gaps to be bridged rather
than data to be avoided.

## Parameterised terms

A CASA `K` table stores delays as real parameters, not phasors. There the
complex/renormalise dance would be meaningless, so `param_type == "float"`
smooths the values directly -- same normalised convolution, no phase, no
renormalisation.

A QuartiCal parameterised term is the same idea with more structure: the
solution is a set of named parameters (`delay_XX`, `phase_offset_YY`,
`amplitude_XX`) on their own frequency grid, and QuartiCal reads *those*
back rather than the gains. So they are what gets filtered -- each on its
own terms: a parameter named as a phase goes through the complex plane for
the same wrap reason as above, and everything else is filtered directly.

The gains array is then rebuilt from the smoothed parameters where that is
unambiguous (an amplitude *is* the gain; a phase is its argument) and the
write is **refused** where it is not. A delay's reconstruction needs the
reference frequency the solver used, and getting it wrong is invisible in
the array while adding a per-antenna constant phase to everything the term
corrects.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from msutils._log import create_logger

from ._io import read_gains, write_gains
from ._model import GainBlock, GainTable
from ._params import classify_parameters, rebuild_gains

__all__ = ["KERNELS", "SmoothResult", "SmoothedBlock", "smooth"]

LOGGER = create_logger(__name__)

#: Bumped on any breaking change to the JSON report.
SCHEMA_VERSION = 1

#: Kernel shapes. No `median`: a running median of a complex quantity is not
#: defined (there is no ordering on the plane), and taking it of amplitude
#: and phase separately reintroduces exactly the wrap problem the complex
#: filter exists to avoid. Reject outliers before smoothing instead.
KERNELS = ("boxcar", "gaussian")

#: Physical window units -> the factor converting them to the axis's own
#: units (seconds for time, Hz for frequency).
_TIME_UNITS = {"s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0}
_FREQ_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}

_WINDOW_PATTERN = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")


@dataclass
class SmoothedBlock:
    """What one block's smoothing actually did.

    `time_samples`/`freq_samples` are the windows *in samples*, which is the
    only form that says what happened: a physical window means different
    numbers of solutions on differently-averaged data, and that is the point
    of writing it in physical units but the reason for recording it in
    samples.
    """

    field_id: int | None
    spw_id: int | None
    direction: int
    time_samples: int
    freq_samples: int
    filled: int = 0


@dataclass
class SmoothResult:
    """Everything one `smooth` run did."""

    kernel: str
    time_window: str
    freq_window: str
    fill: bool
    blocks: list[SmoothedBlock]
    input_path: str = ""
    output_path: str = ""
    term: str = ""
    param_type: str = "complex"
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "kernel": self.kernel,
            "time_window": self.time_window,
            "freq_window": self.freq_window,
            "fill": self.fill,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "term": self.term,
            "param_type": self.param_type,
            "blocks": [asdict(b) for b in self.blocks],
        }

    def render(self) -> str:
        return "\n".join(
            f"# Smoothed field={b.field_id} spw={b.spw_id} direction={b.direction} "
            f"with a {self.kernel} kernel over {b.time_samples} time x {b.freq_samples} freq samples"
            + (f", filling {b.filled} flagged solutions" if self.fill else "")
            for b in self.blocks
        )

    def save(self, path: str | Path) -> str:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return str(path)


def _spacing(axis: np.ndarray) -> float:
    """The median step of a coordinate axis, or NaN if it has fewer than two
    points (a single solution has no spacing, and inventing one would turn a
    physical window into a silent no-op)."""
    if axis.size < 2:
        return float("nan")
    return float(np.median(np.diff(np.sort(axis))))


def _window_samples(
    window: int | str | None, coordinate: np.ndarray, *, units: dict, what: str
) -> int:
    """A window as a number of samples.

    A bare integer counts solutions; a string with a unit (``"120s"``,
    ``"8MHz"``) is physical and is converted against this block's own axis --
    the same distinction, and for the same reason, as a solution interval:
    the physical form means the same thing on differently-averaged data.
    """
    if window in (None, "", 0):
        return 1
    if isinstance(window, (int, np.integer)) and not isinstance(window, bool):
        return max(int(window), 1)
    match = _WINDOW_PATTERN.match(str(window))
    if not match:
        raise ValueError(
            f"cannot read {what} window {window!r} -- expected a sample count or e.g. '120s' / '8MHz'"
        )
    value, unit = float(match.group(1)), match.group(2).lower()
    if not unit:
        return max(round(value), 1)
    if unit not in units:
        raise ValueError(f"unknown {what} unit in {window!r} (known: {sorted(units)})")
    spacing = _spacing(coordinate)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(
            f"cannot convert the physical {what} window {window!r} to samples: this block's {what} axis "
            "has no usable spacing (fewer than two points, or unknown frequencies). Give a sample count instead."
        )
    return max(round(value * units[unit] / spacing), 1)


def _kernel(size: int, shape: str) -> np.ndarray:
    """A normalised 1-D kernel of odd length covering `size` samples."""
    if shape not in KERNELS:
        raise ValueError(f"unknown kernel {shape!r} (known: {list(KERNELS)})")
    size = max(int(size), 1)
    if size == 1:
        return np.ones(1)
    length = size if size % 2 else size + 1  # odd, so the kernel is centred
    if shape == "boxcar":
        return np.ones(length) / length
    # `size` is the FWHM, which is the width a user means by "smooth over
    # this much"; sigma follows from it.
    sigma = size / 2.3548200450309493
    x = np.arange(length) - length // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve along one axis, keeping the shape (`mode="same"`)."""
    if kernel.size == 1:
        return values
    moved = np.moveaxis(values, axis, -1)
    shape = moved.shape
    flat = moved.reshape(-1, shape[-1])
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = np.convolve(flat[i], kernel, mode="same")
    return np.moveaxis(out.reshape(shape), -1, axis)


def _normalised_convolve(
    values: np.ndarray, valid: np.ndarray, kernels: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve `values` over the given axes, ignoring invalid samples.

    Returns the smoothed values and the weight that reached each one. The
    division by weight is what makes a partial window -- at the edge of the
    axis, or beside a flagged solution -- come out as the average of what was
    actually there, rather than of what was there plus zeros.
    """
    numerator = values * valid
    denominator = valid.astype(float)
    for axis, kernel in kernels.items():
        numerator = _convolve(numerator, kernel, axis)
        denominator = _convolve(denominator, kernel, axis)
    weight = denominator
    smoothed = np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > 0)
    return smoothed, weight


def _smooth_parameters(block: GainBlock, kernels: dict, kernel: str, *, fill: bool) -> GainBlock:
    """Filter a parameterised term's parameters, then rebuild its gains.

    Each parameter is filtered on its own terms: one named as a phase goes
    through the complex plane (the wrap argument again, one level down), and
    everything else -- a delay, a TEC, an amplitude -- is a plain number and
    is filtered directly.
    """
    kinds = classify_parameters(block.param_names)
    # The parameters carry their own frequency axis: a delay is one number
    # across a band whose gains are per channel, so the frequency kernel is
    # sized against the parameter grid rather than the gains'.
    param_kernels = dict(kernels)
    if 1 in param_kernels and block.param_freqs is not None and block.param_freqs.size == 1:
        param_kernels.pop(1)

    values = block.masked_params()
    valid = ~np.ma.getmaskarray(values)
    smoothed = np.array(values.filled(0.0), dtype=float)
    for index, kind in enumerate(kinds):
        column = smoothed[..., index : index + 1]
        column_valid = valid[..., index : index + 1]
        if kind == "phase":
            vector, weight = _normalised_convolve(np.exp(1j * column), column_valid, param_kernels)
            filtered = np.angle(vector)
        else:
            filtered, weight = _normalised_convolve(column, column_valid, param_kernels)
        smoothed[..., index : index + 1] = np.where(weight > 0, filtered, column)

    rebuilt = rebuild_gains(smoothed, block.param_names, n_corr=len(block.corrs))
    if rebuilt is None:
        raise ValueError(
            f"this term's parameters {list(block.param_names)} can be smoothed, but its gains cannot be "
            "rebuilt from them: that needs the reference frequency the solver measured against, and a wrong "
            "one is invisible in the array while adding a per-antenna constant phase to everything the term "
            "corrects. Smoothing is supported for amplitude- and phase-parameterised terms."
        )
    flags = block.flags
    if fill:
        flags = np.zeros_like(block.flags)
    return replace(block, params=smoothed, gains=rebuilt, flags=flags)


def smooth(
    gains: str | Path | GainTable,
    *,
    output: str | Path | None = None,
    time_window: int | str | None = None,
    freq_window: int | str | None = None,
    kernel: str = "boxcar",
    fill: bool = False,
    term: str | None = None,
    json_out: str | Path | None = None,
) -> SmoothResult:
    """Smooth gains in time and/or frequency.

    The complex gain is filtered for its phase and the amplitude separately
    for its magnitude, then recombined -- see the module docstring for why
    neither alone is correct.

    Args:
        gains: Table/store to read, or an already-read
            :class:`~msutils.gains.GainTable`.
        output: Write the smoothed gains here (never in place).
        time_window: Window along time: a sample count, or a physical
            interval (``"120s"``, ``"5min"``). None leaves time alone.
        freq_window: Window along frequency: a sample count, or a physical
            width (``"8MHz"``). None leaves frequency alone.
        kernel: ``"boxcar"`` (default) or ``"gaussian"``, whose window is
            read as a FWHM.
        fill: Unflag solutions whose window contained real data. Off by
            default: a flagged solution stays flagged, which is the safe
            reading when the flags mean "bad" rather than "missing".
        term: Which term to read, for QuartiCal stores holding several.
        json_out: Write the report to this JSON file.

    Returns:
        A :class:`SmoothResult`, recording the windows **in samples** per
        block -- what a physical window actually resolved to on this data.

    Raises:
        ValueError: If neither window is given (there would be nothing to
            do), or if a window cannot be read or converted.
    """
    if time_window in (None, "", 0) and freq_window in (None, "", 0):
        raise ValueError(
            "smooth needs a time_window, a freq_window, or both -- with neither there is nothing to smooth"
        )
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r} (known: {list(KERNELS)})")

    table = gains if isinstance(gains, GainTable) else read_gains(gains, term=term)
    is_parameter = table.param_type == "float"
    records: list[SmoothedBlock] = []

    def _smooth_block(block: GainBlock) -> GainBlock:
        time_samples = _window_samples(time_window, block.times, units=_TIME_UNITS, what="time")
        freq_samples = _window_samples(
            freq_window, block.freqs, units=_FREQ_UNITS, what="frequency"
        )
        kernels = {}
        if time_samples > 1:
            kernels[0] = _kernel(time_samples, kernel)
        if freq_samples > 1:
            kernels[1] = _kernel(freq_samples, kernel)

        valid = ~block.flags
        if not kernels:
            records.append(
                SmoothedBlock(
                    block.field_id, block.spw_id, block.direction, time_samples, freq_samples
                )
            )
            return block

        if block.parameterised:
            result_block = _smooth_parameters(block, kernels, kernel, fill=fill)
            records.append(
                SmoothedBlock(
                    block.field_id,
                    block.spw_id,
                    block.direction,
                    time_samples,
                    freq_samples,
                    int((block.flags & ~result_block.flags).sum()),
                )
            )
            return result_block

        if is_parameter:
            # A delay is a number, not a phasor: filter it and stop. The
            # complex path below would treat its magnitude as a gain
            # amplitude, which it is not.
            smoothed, weight = _normalised_convolve(block.gains.real.astype(float), valid, kernels)
            result = smoothed.astype(complex)
        else:
            vector, weight = _normalised_convolve(block.gains, valid, kernels)
            amplitude, _ = _normalised_convolve(np.abs(block.gains), valid, kernels)
            length = np.abs(vector)
            # Where the vector average cancelled completely there is no
            # direction to keep, so the original phase is the only honest one
            # to fall back on.
            direction = np.divide(vector, length, out=np.zeros_like(vector), where=length > 0)
            fallback = np.divide(
                block.gains,
                np.abs(block.gains),
                out=np.zeros_like(vector),
                where=np.abs(block.gains) > 0,
            )
            direction = np.where(length > 0, direction, fallback)
            result = direction * amplitude

        # Untouched wherever nothing valid reached: no window, no answer.
        result = np.where(weight > 0, result, block.gains)
        flags = block.flags & ~(weight > 0) if fill else block.flags
        filled = int((block.flags & ~flags).sum())
        records.append(
            SmoothedBlock(
                block.field_id, block.spw_id, block.direction, time_samples, freq_samples, filled
            )
        )
        return block.with_gains(result, flags=flags)

    smoothed_table = table.map(_smooth_block)
    result = SmoothResult(
        kernel=kernel,
        time_window=str(time_window) if time_window not in (None, "", 0) else "",
        freq_window=str(freq_window) if freq_window not in (None, "", 0) else "",
        fill=fill,
        blocks=records,
        input_path=table.path,
        term=table.term,
        param_type=table.param_type,
    )
    for line in result.render().splitlines():
        LOGGER.info(line)

    if output is not None:
        result.output_path = write_gains(smoothed_table, output, template=table.path)
    if json_out is not None:
        result.save(json_out)
    return result
