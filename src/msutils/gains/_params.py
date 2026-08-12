"""What a parameterised term's parameters are, read off their own names.

QuartiCal stores most terms twice: the `gains` array, and the `params` it
was solved for -- ``amplitude_XX``, ``phase_offset_YY``, ``delay_XX``,
``tec_YY``. On load it reads the **params** for those terms and rebuilds
the gains from them, so an operation that changes gains alone is silently
discarded (`quartical/interpolation/interpolate.py`: ``data_field =
"params" if parameterized else "gains"``).

So an operation on a parameterised term has to act on the parameters, and
whether it *can* depends on what they are -- which QuartiCal labels
explicitly. Reading the labels rather than a table of type names is what
makes this survive a new term type: ``anything_amplitude_XX`` classifies
itself.

Three questions, and this module answers exactly those:

* is a parameter an **amplitude**? Then a scaling (a flux bootstrap, a
  normalisation) maps onto it exactly: multiply.
* is it a **phase**? Then it is a wrapped quantity: filtering it as a
  number spikes at every turn, so smoothing goes through the complex plane
  (`smooth`'s own argument, one level down).
* can the **gains** be rebuilt from it? Only for the parameterisations
  whose reconstruction is unambiguous. A delay's is not: it needs the
  reference frequency the solver used, and guessing that produces gains
  that look right and are wrong by a per-antenna constant phase.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PARAMETER_KINDS",
    "classify_parameters",
    "rebuild_gains",
    "scale_block",
    "scaling_is_expressible",
]

#: Substrings QuartiCal uses in its parameter names, mapped to what the
#: parameter is. Matched case-insensitively; anything unmatched is "other",
#: which is a real answer rather than a failure -- it says "a number we can
#: filter but not interpret".
PARAMETER_KINDS = {"amplitude": "amplitude", "phase": "phase", "delay": "delay", "tec": "tec"}


def classify_parameters(names) -> tuple[str, ...]:
    """Each parameter's kind, by name (``"amplitude"``, ``"phase"``, ...)."""
    kinds = []
    for name in names:
        lowered = str(name).lower()
        kinds.append(
            next((kind for hint, kind in PARAMETER_KINDS.items() if hint in lowered), "other")
        )
    return tuple(kinds)


def scaling_is_expressible(names) -> bool:
    """Whether multiplying a gain's amplitude maps onto these parameters.

    True only when **every** parameter is an amplitude. A phase-like
    parameterisation (a delay, a phase offset, a TEC) describes a gain of
    unit modulus -- verified on real stores, where |g| = 1 to machine
    precision -- so there is no amplitude in it to scale. That is not a
    missing feature: the operation has no meaning there.
    """
    kinds = classify_parameters(names)
    return bool(kinds) and all(kind == "amplitude" for kind in kinds)


def rebuild_gains(params: np.ndarray, names, *, n_corr: int) -> np.ndarray | None:
    """The gains a parameterisation implies, or None if it cannot be said.

    Handles the two unambiguous cases -- an amplitude is the gain, a phase
    is its argument -- and declines everything else. A delay is the one
    that matters: reconstructing it needs the reference frequency the
    solver measured it against, and a wrong choice there is invisible in
    the array and shows up as a per-antenna constant phase in the corrected
    data (the same trap `docs/APPLYGAINS_WORKER_DESIGN.md` documents in
    caracal for transferring a K between bands).

    Args:
        params: ``(time, freq, antenna, parameter)``.
        names: The parameter names, one per parameter.
        n_corr: Correlations the gains have.

    Returns:
        Gains shaped ``(time, freq, antenna, n_corr)``, or None when the
        parameterisation is not one this can invert.
    """
    kinds = classify_parameters(names)
    if len(kinds) != n_corr or len(set(kinds)) != 1:
        return None
    if kinds[0] == "amplitude":
        return params.astype(complex)
    if kinds[0] == "phase":
        return np.exp(1j * params.astype(float))
    return None


def scale_block(block, factor, *, what: str):
    """Divide a block's amplitudes by `factor`, parameters included.

    The one operation `fluxscale` and `normalise` share: both measure an
    amplitude and divide it out, and both have to do it to whichever array
    is authoritative for the term.

    Args:
        block: The block to scale.
        factor: A positive real array broadcastable against the gains.
        what: The caller's name, for the error message.

    Returns:
        A new block.

    Raises:
        ValueError: If the term is parameterised by something other than
            amplitudes -- a phase-like parameterisation describes a gain of
            unit modulus, so there is no amplitude in it to scale and the
            operation has no meaning -- or if its parameters do not share
            the gains' grid, where the factor cannot be applied unambiguously.
    """
    if not block.parameterised:
        return block.with_gains(block.gains / factor)
    if not scaling_is_expressible(block.param_names):
        kinds = sorted(set(classify_parameters(block.param_names)))
        raise ValueError(
            f"{what}: this term is parameterised by {kinds} ({list(block.param_names)}), which describes a "
            "gain of unit modulus -- there is no amplitude in it to scale. Scaling applies to amplitude-like "
            "terms; a phase, delay or TEC term is not one."
        )
    if block.params.shape != block.gains.shape:
        raise ValueError(
            f"{what}: this term's parameters {block.params.shape} do not share the gains' grid "
            f"{block.gains.shape}, so one factor cannot be applied to both. Scaling is supported where the "
            "parameters are the amplitudes themselves."
        )
    params = block.params / factor
    rebuilt = rebuild_gains(params, block.param_names, n_corr=len(block.corrs))
    return block.with_params(params, gains=rebuilt if rebuilt is not None else block.gains / factor)
