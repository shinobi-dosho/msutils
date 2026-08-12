"""The format-neutral gain model every ``gainutils`` operation works on.

A calibration solution is the same object whichever solver wrote it: a
complex gain per (time, frequency, antenna, correlation), with flags, for
one field/spectral-window/direction. The two formats we read disagree only
about how that is laid out on disk --

* a **CASA caltable** is row-major: one row per (time, antenna), each row
  holding a ``(channel, correlation)`` array, with ``FIELD_ID`` and
  ``SPECTRAL_WINDOW_ID`` as columns;
* a **QuartiCal store** is already n-dimensional: ``gains`` with axes
  ``(gain_time, gain_freq, antenna, direction, correlation)``, one dataset
  per (field, data-description), with the field and ddid as attributes.

-- so this module defines the shape they both fold into, and every
operation (``fluxscale`` now; ``smooth``, ``normalise`` and whatever else
later) is written against it alone. Adding a format is a reader plus a
writer; adding an operation is a function `GainTable -> GainTable`. Neither
has to know about the other, which is the entire point of having a model.

The axis order is **(time, freq, antenna, correlation)**: QuartiCal's own,
so its arrays need no transpose, and CASA's rows are folded into it on
read. `direction` is not an axis but a block key -- QuartiCal's
direction-dependent stores become several blocks, which keeps every
operation four-dimensional instead of sometimes-five.

Conventions inherited from the rest of msutils: times are **MJD seconds**,
frequencies are **Hz**, collections are addressable by id *or* name, and
ids are ids rather than positions (a subset table keeps non-contiguous
ones).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["AXES", "GainBlock", "GainTable"]

#: The dimensions of a block's `gains`/`flags` arrays, in order.
AXES = ("time", "freq", "antenna", "correlation")


@dataclass
class GainBlock:
    """One field/spw/direction's solutions, dense over `AXES`.

    Attributes:
        gains: Complex gains, shape ``(n_time, n_freq, n_antenna, n_corr)``.
            Parameterised terms (a delay, say) are stored as the complex
            gain they represent when the format records one, and as real
            values cast to complex when it does not -- see `param_type` on
            the owning :class:`GainTable` before doing arithmetic that
            assumes a phasor.
        flags: Boolean, same shape; True means "do not use". A solution
            missing from the source entirely (a CASA row that does not
            exist for one antenna at one time) is flagged rather than
            silently zero, so a mean over the block cannot quietly include
            it.
        times: MJD seconds, shape ``(n_time,)``.
        freqs: Hz, shape ``(n_freq,)``.
        antennas: Antenna **labels**, one per antenna axis position --
            names where the format records them (QuartiCal stores them;
            a CASA table has them in its ANTENNA subtable), else the id as
            a string. Labels rather than ids because the one thing an
            operation comparing two tables must not do is match antennas by
            position: two tables can disagree on ordering, and the result is
            a plausible wrong answer rather than an error.
        corrs: Correlation labels (``"XX"``, ``"RL"``, ...), one per
            correlation axis position.
        field_id: Field this block belongs to, or None if the format does
            not say.
        spw_id: Spectral window (CASA) / data description (QuartiCal).
        direction: Direction index; 0 for every direction-independent
            solution, which is all a CASA caltable can hold.
        rows: For a CASA-sourced block, the source row number of each
            (time, antenna) pair, ``-1`` where the source had no row. This
            is what lets a writer put values back exactly where they came
            from instead of reconstructing an ordering; None for formats
            that are not row-based.
    """

    gains: np.ndarray
    flags: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    antennas: tuple[str, ...]
    corrs: tuple[str, ...]
    field_id: int | None = None
    spw_id: int | None = None
    direction: int = 0
    rows: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.gains = np.asarray(self.gains)
        self.flags = np.asarray(self.flags, dtype=bool)
        self.times = np.asarray(self.times, dtype=float)
        self.freqs = np.asarray(self.freqs, dtype=float)
        self.antennas = tuple(str(a) for a in self.antennas)
        self.corrs = tuple(str(c) for c in self.corrs)
        expected = (self.times.size, self.freqs.size, len(self.antennas), len(self.corrs))
        if self.gains.shape != expected:
            raise ValueError(
                f"gains have shape {self.gains.shape}, but the axes say {expected} "
                f"({' x '.join(AXES)})"
            )
        if self.flags.shape != self.gains.shape:
            raise ValueError(f"flags {self.flags.shape} do not match gains {self.gains.shape}")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.gains.shape  # type: ignore[return-value]

    @property
    def key(self) -> tuple[int | None, int | None, int]:
        """`(field_id, spw_id, direction)` -- what makes a block unique."""
        return (self.field_id, self.spw_id, self.direction)

    def masked(self) -> np.ma.MaskedArray:
        """The gains as a masked array, flags applied.

        Every reduction in every operation should start here. A median or a
        mean taken over the raw array silently folds in flagged solutions,
        which is the failure mode that produces a plausible wrong number
        rather than an error.
        """
        return np.ma.masked_array(self.gains, mask=self.flags)

    def with_gains(self, gains: np.ndarray, flags: np.ndarray | None = None) -> GainBlock:
        """A copy carrying new values -- the only way an operation should
        produce output, since it keeps every axis and the row map intact."""
        return replace(self, gains=np.asarray(gains), flags=self.flags if flags is None else flags)


@dataclass
class GainTable:
    """A set of :class:`GainBlock` s plus what the format said about them.

    Attributes:
        blocks: One per (field, spw, direction) present in the source.
        term: The solver's own name for the term (``"G"``, ``"B"``, ``"K"``
            for QuartiCal; the ``VisCal`` keyword for CASA).
        gain_type: What kind of solution it is, in the writing solver's
            vocabulary (``"complex"``, ``"delay_and_offset"``, ``"G
            Jones"``, ...). Recorded rather than interpreted: an operation
            that cares (smoothing a delay is not smoothing a phasor) must
            check it explicitly.
        param_type: ``"complex"`` or ``"float"`` -- whether the source
            stored complex gains or real parameters. A CASA ``K`` table is
            ``float``.
        format: ``"casa"`` or ``"quartical"``.
        path: Where it was read from.
        antenna_names: Antenna names indexed by antenna **id**, as the
            source recorded them (empty when it records none).
        field_names: Field id -> name, for the ids that appear.
    """

    blocks: list[GainBlock]
    term: str = ""
    gain_type: str = ""
    param_type: str = "complex"
    format: str = ""
    path: str = ""
    antenna_names: tuple[str, ...] = ()
    field_names: dict[int, str] = field(default_factory=dict)

    def __iter__(self) -> Iterator[GainBlock]:
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def fields(self) -> tuple[int, ...]:
        """Field ids present, in first-seen order."""
        return tuple(dict.fromkeys(b.field_id for b in self.blocks if b.field_id is not None))

    @property
    def spws(self) -> tuple[int, ...]:
        """Spectral windows present, in first-seen order."""
        return tuple(dict.fromkeys(b.spw_id for b in self.blocks if b.spw_id is not None))

    def resolve_field(self, field_ref: int | str) -> int:
        """A field id or name -> the id, checked against what is here.

        Names, not just ids, because a configuration written before the
        observation was read cannot know the ids -- and because an id that
        is *absent* is the mistake worth catching loudly (an empty
        selection otherwise yields an empty statistic rather than an
        error).
        """
        if isinstance(field_ref, (int, np.integer)) or str(field_ref).lstrip("-").isdigit():
            wanted = int(field_ref)
            if wanted not in self.fields:
                raise ValueError(
                    f"{self.path or 'table'} has no field id {wanted} "
                    f"(present: {list(self.fields)}, named {list(self.field_names.values())})"
                )
            return wanted
        matches = [fid for fid, name in self.field_names.items() if name == str(field_ref)]
        if not matches:
            raise ValueError(
                f"{self.path or 'table'} has no field named {field_ref!r} "
                f"(present: {sorted(self.field_names.values())})"
            )
        return matches[0]

    def select(
        self,
        *,
        field: int | str | None = None,
        spw: int | Sequence[int] | None = None,
        direction: int | None = None,
    ) -> GainTable:
        """The sub-table matching a field/spw/direction selection."""
        field_id = None if field is None else self.resolve_field(field)
        spws = None if spw is None else {int(s) for s in (spw if isinstance(spw, Sequence) else [spw])}
        blocks = [
            b
            for b in self.blocks
            if (field_id is None or b.field_id == field_id)
            and (spws is None or b.spw_id in spws)
            and (direction is None or b.direction == direction)
        ]
        return replace(self, blocks=blocks)

    def map(self, fn: Callable[[GainBlock], GainBlock]) -> GainTable:
        """A copy with `fn` applied to every block.

        The shape every operation takes: `fluxscale` rescales, a `normalise`
        divides, a `smooth` filters, and none of them touch IO or the
        table-level metadata.
        """
        return replace(self, blocks=[fn(b) for b in self.blocks])
