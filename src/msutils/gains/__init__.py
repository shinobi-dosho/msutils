"""``gainutils``: operations on calibration gain solutions.

A companion to the MS operations in the rest of msutils, for the *other*
thing a calibration pipeline produces. The design is deliberately narrow:

* one format-neutral model (:class:`GainTable` of :class:`GainBlock` s,
  dense over ``(time, freq, antenna, correlation)``) that a CASA caltable
  and a QuartiCal gain store both fold into;
* readers and writers behind :func:`read_gains` / :func:`write_gains`, so an
  operation never touches a format;
* operations as functions of a `GainTable`, so a new one is arithmetic and
  a report -- not IO, not a CLI, not a format.

:func:`fluxscale` is the first. `smooth` and `normalise` are the obvious
next two, and the shape they should take is already visible here:
`GainTable.map` over blocks, `GainBlock.masked` for anything that reduces,
`with_gains` to produce the result, and never a write in place.

Usage::

    import msutils.gains as gains

    result = gains.fluxscale(
        "secondary.G0", "primary.G0",
        transfer_field="J0825-5010", reference_field="J0408-6545",
        output="secondary.G0.fluxscaled",
    )
    print(result.flux_jy)

The CLI mirrors it (``gainutils fluxscale ...``).
"""

from ._io import detect_format, read_gains, write_gains
from ._model import AXES, GainBlock, GainTable
from ._quartical import quartical_terms
from .fluxscale import FluxDensity, FluxscaleResult, fluxscale

__all__ = [
    "AXES",
    "FluxDensity",
    "FluxscaleResult",
    "GainBlock",
    "GainTable",
    "detect_format",
    "fluxscale",
    "quartical_terms",
    "read_gains",
    "write_gains",
]
