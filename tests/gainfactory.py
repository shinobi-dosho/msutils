"""Build small CASA calibration tables with python-casacore alone.

Same principle as ``msfactory``: no solver, no simulation stack, nothing to
skip on. A caltable is a plain casacore table with a handful of index
columns, a ``CPARAM``/``FPARAM`` array column and three subtables, so it can
be written directly -- and written with *known* gains, which is what lets a
flux-scale bootstrap be tested against an exact answer rather than against
whatever a solver happened to produce.

The tables here deliberately carry two fields: a reference calibrator whose
gains are purely instrumental, and a transfer calibrator whose gains are the
same instrumental values scaled by ``sqrt(flux)`` -- exactly the relationship
a solve against a 1 Jy model produces for a source of ``flux`` Jy.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
from casacore.tables import makearrcoldesc, makescacoldesc, maketabdesc, table

#: MJD seconds for 2024-01-01T00:00:00 UTC, matching ``msfactory``.
T0 = 60310.0 * 86400.0


def _subtable(path: str, name: str, coldescs, rows: int) -> str:
    sub_path = os.path.join(path, name)
    sub = table(sub_path, maketabdesc(coldescs), nrow=rows, ack=False)
    return sub_path, sub


def make_caltable(
    path,
    *,
    nant=4,
    ntime=3,
    nchan=1,
    ncorr=2,
    nspw=1,
    flux=5.0,
    reference_field=0,
    transfer_field=1,
    jitter=0.02,
    seed=0,
    viscal="G Jones",
    param_col="CPARAM",
):
    """Write a two-field CASA caltable at ``path`` and return ``(path, gains)``.

    Args:
        nant: Antennas; each gets one row per (time, field, spw).
        ntime: Timestamps per field.
        nchan: Channels in the solution (1 for a `G`-like table).
        ncorr: Correlations.
        nspw: Spectral windows, each 10 MHz wide and 100 MHz apart.
        flux: The transfer field's true flux density in Jy -- its gains are
            written as the reference's times ``sqrt(flux)``.
        jitter: Fractional random scatter added per (time, antenna), so the
            medians have something to average over and the error bar is not
            identically zero.
        seed: RNG seed; the gains are deterministic given one.
        param_col: ``"CPARAM"`` for a complex table or ``"FPARAM"`` for a
            real-parameter one -- a `K` (delay) table is the latter, and it
            takes a different path through every operation that has to know
            whether it is holding a phasor.

    Returns:
        ``(path, instrumental)`` where `instrumental` is the
        ``(nant, ncorr)`` array of reference-field gain amplitudes.
    """
    path = str(path)
    if os.path.exists(path):
        shutil.rmtree(path)

    rng = np.random.default_rng(seed)
    instrumental = 0.7 + 0.2 * rng.random((nant, ncorr))

    fields = [reference_field, transfer_field]
    rows = len(fields) * nspw * ntime * nant
    coldescs = [
        makescacoldesc("TIME", 0.0),
        makescacoldesc("FIELD_ID", 0),
        makescacoldesc("SPECTRAL_WINDOW_ID", 0),
        makescacoldesc("ANTENNA1", 0),
        makescacoldesc("ANTENNA2", -1),
        makearrcoldesc(param_col, 0 + 0j, shape=[nchan, ncorr], valuetype="complex")
        if param_col == "CPARAM"
        else makearrcoldesc(param_col, 0.0, shape=[nchan, ncorr], valuetype="float"),
        makearrcoldesc("FLAG", False, shape=[nchan, ncorr], valuetype="boolean"),
        makearrcoldesc("SNR", 0.0, shape=[nchan, ncorr], valuetype="float"),
    ]
    tab = table(path, maketabdesc(coldescs), nrow=rows, ack=False)

    times, field_ids, spw_ids, ant_ids = [], [], [], []
    params = np.zeros((rows, nchan, ncorr), dtype=complex)
    row = 0
    for field_id in fields:
        scale = 1.0 if field_id == reference_field else np.sqrt(flux)
        for spw in range(nspw):
            for t in range(ntime):
                for ant in range(nant):
                    noise = 1.0 + jitter * (rng.random((ncorr,)) - 0.5)
                    amplitude = instrumental[ant] * scale * noise
                    # A phase that varies with antenna and time, so nothing
                    # here accidentally tests a real-valued special case.
                    phase = 0.1 * ant + 0.01 * t
                    params[row, :, :] = amplitude * np.exp(1j * phase)
                    times.append(T0 + t * 60.0 + (0 if field_id == reference_field else 3600.0))
                    field_ids.append(field_id)
                    spw_ids.append(spw)
                    ant_ids.append(ant)
                    row += 1

    tab.putcol("TIME", np.array(times))
    tab.putcol("FIELD_ID", np.array(field_ids))
    tab.putcol("SPECTRAL_WINDOW_ID", np.array(spw_ids))
    tab.putcol("ANTENNA1", np.array(ant_ids))
    tab.putcol("ANTENNA2", np.full(rows, -1))
    tab.putcol(param_col, params if param_col == "CPARAM" else params.real)
    tab.putcol("FLAG", np.zeros((rows, nchan, ncorr), dtype=bool))
    tab.putcol("SNR", np.full((rows, nchan, ncorr), 100.0))
    tab.putkeyword("VisCal", viscal)
    tab.putkeyword("ParType", "Complex" if param_col == "CPARAM" else "Float")

    ant_path, ant_tab = _subtable(path, "ANTENNA", [makescacoldesc("NAME", "")], nant)
    ant_tab.putcol("NAME", np.array([f"m{i:03d}" for i in range(nant)]))
    ant_tab.close()
    tab.putkeyword("ANTENNA", f"Table: {ant_path}")

    n_fields = max(fields) + 1
    field_path, field_tab = _subtable(path, "FIELD", [makescacoldesc("NAME", "")], n_fields)
    # Built as a list and converted once: assigning into a numpy string array
    # truncates to the dtype's width, so seeding it with "FIELD0" would store
    # "GAINCA".
    names = [f"FIELD{i}" for i in range(n_fields)]
    names[reference_field] = "REFCAL"
    names[transfer_field] = "GAINCAL"
    field_tab.putcol("NAME", np.array(names))
    field_tab.close()
    tab.putkeyword("FIELD", f"Table: {field_path}")

    spw_path, spw_tab = _subtable(
        path,
        "SPECTRAL_WINDOW",
        [makearrcoldesc("CHAN_FREQ", 0.0, shape=[max(nchan, 4)], valuetype="double")],
        nspw,
    )
    width = 10e6 / max(nchan, 4)
    spw_tab.putcol(
        "CHAN_FREQ",
        np.array([[1.4e9 + s * 100e6 + c * width for c in range(max(nchan, 4))] for s in range(nspw)]),
    )
    spw_tab.close()
    tab.putkeyword("SPECTRAL_WINDOW", f"Table: {spw_path}")

    tab.close()
    return path, instrumental


def flag_solution(path, *, field_id, antenna, corr=None):
    """Flag every solution of one antenna on one field, in place."""
    with table(path, readonly=False, ack=False) as tab:
        fields = tab.getcol("FIELD_ID")
        ants = tab.getcol("ANTENNA1")
        flags = tab.getcol("FLAG")
        rows = np.flatnonzero((fields == field_id) & (ants == antenna))
        if corr is None:
            flags[rows] = True
        else:
            flags[rows, :, corr] = True
        tab.putcol("FLAG", flags)
    return rows


def corrupt_solution(path, *, field_id, antenna, factor):
    """Multiply one antenna's gains on one field by ``factor``, in place."""
    with table(path, readonly=False, ack=False) as tab:
        fields = tab.getcol("FIELD_ID")
        ants = tab.getcol("ANTENNA1")
        column = "CPARAM" if "CPARAM" in tab.colnames() else "FPARAM"
        params = tab.getcol(column)
        rows = np.flatnonzero((fields == field_id) & (ants == antenna))
        params[rows] *= factor
        tab.putcol(column, params)
    return rows
