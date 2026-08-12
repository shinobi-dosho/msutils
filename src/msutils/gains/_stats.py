"""How gain amplitudes are collapsed, shared by every operation that reduces.

Both operations here answer a question of the form "what is this antenna's
amplitude, over time or over the band" -- a flux bootstrap compares two of
them, a normalisation divides by one -- so the reduction, the outlier
rejection and the flag handling live in one place rather than being written
twice with two sets of edge cases.

The one real choice is `mean` against `median`, and it is not cosmetic.
Measured on a real MeerKAT bootstrap (10 antennas, 3 reference scans, 16
transfer scans) the two differ by **0.5%** -- 5.969 Jy by median against
5.943 Jy by mean, where CASA's own task reported 5.938 Jy for the same
table. That is larger than either one's error bar, so which was used is
recorded in every result rather than assumed.

`median` is the default throughout: one bad scan should not move a flux
scale or a bandpass normalisation. `mean` is the one that reproduces CASA
(to 0.08% on that table), despite CASA's documentation describing a
"median gain norm".
"""

from __future__ import annotations

import numpy as np

from ._model import GainBlock

__all__ = ["STATISTICS", "amplitude_statistic", "check_statistic"]

#: The reductions an operation may use. See the module docstring for what
#: separates them in practice.
STATISTICS = ("median", "mean")


def check_statistic(statistic: str) -> str:
    """Validate a statistic name, returning it."""
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r} (known: {list(STATISTICS)})")
    return statistic


def amplitude_statistic(
    block: GainBlock,
    *,
    statistic: str = "median",
    axis: tuple[int, ...] = (0, 1),
    threshold: float = 0.0,
    keepdims: bool = False,
) -> np.ma.MaskedArray:
    """|g| collapsed over `axis`, flags respected.

    Args:
        block: The block to reduce.
        statistic: ``"median"`` or ``"mean"``.
        axis: Axes to reduce over, in `AXES` order -- ``(0, 1)`` (time and
            frequency) collapses a `G` solve's times and a `B` solve's
            channels alike, which is what makes one statistic serve both.
        threshold: Drop amplitudes deviating fractionally from the median by
            more than this before reducing; 0 disables. Always measured from
            the *median*, whatever `statistic` is -- an outlier test against
            a mean the outlier is already in is a weaker test.
        keepdims: Keep the reduced axes as length 1, for broadcasting the
            result back against the block.

    Returns:
        A masked array; entries where everything was flagged stay masked
        rather than becoming zero, which is what lets a caller decide what
        "no solutions here" should mean for it.
    """
    check_statistic(statistic)
    amplitude = np.ma.abs(block.masked())
    if threshold > 0:
        median = np.ma.median(amplitude, axis=axis, keepdims=True)
        deviation = np.ma.abs(amplitude - median) / np.ma.masked_equal(median, 0)
        amplitude = np.ma.masked_where(deviation > threshold, amplitude)
    if statistic == "mean":
        return amplitude.mean(axis=axis, keepdims=keepdims)
    # np.ma.median does not take `keepdims` on every numpy it must run under,
    # so the axes are restored by hand.
    reduced = np.ma.median(amplitude, axis=axis)
    if keepdims:
        shape = list(block.shape)
        for ax in axis:
            shape[ax] = 1
        reduced = reduced.reshape(shape)
    return reduced
