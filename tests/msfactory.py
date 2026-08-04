"""Build small, structurally realistic Measurement Sets with python-casacore alone.

This replaces the old ``simms`` test dependency. ``simms`` is a heavy
git-installed package that pulled in a whole simulation stack just to lay down
a few hundred rows, and when it was missing the MS-backed tests silently
skipped -- which meant CI was mostly green on nothing.

``casacore.tables.default_ms`` already knows the MSv2 schema, so we only have
to fill it. The MSs produced here are deliberately *awkward*: several fields,
several SPWs, several scans per field, multiple observing intents and a
non-trivial ``DATA_DESC_ID`` -> ``(SPW, POL)`` mapping. Those are exactly the
cases the old ``summary()`` got wrong, so the fixture has to exercise them.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
from casacore.tables import default_ms, maketabdesc, makearrcoldesc, table

# MJD seconds for 2024-01-01T00:00:00 UTC (MJD 60310); arbitrary but fixed so
# that rendered dates are stable across runs.
T0 = 60310.0 * 86400.0

DEFAULT_INTENTS = [
    "CALIBRATE_BANDPASS#ON_SOURCE,CALIBRATE_FLUX#ON_SOURCE",
    "CALIBRATE_PHASE#ON_SOURCE",
    "OBSERVE_TARGET#ON_SOURCE",
]


def make_ms(path, nant=4, ntime=3, nchan=4, nspw=2, nfield=3, nscan_per_field=2,
            dt=8.0, corr_type=(9, 10, 11, 12), telescope="MEERKAT",
            flag_every=3, add_weight_spectrum=False):
    """Write a synthetic MSv2 at ``path`` and return the path.

    Args:
        nant: Number of antennas (baselines are all cross-correlations).
        ntime: Timestamps per scan.
        nchan: Channels per spectral window.
        nspw: Spectral windows (each gets its own DATA_DESCRIPTION row).
        nfield: Fields; field i is assigned intent i (cycling).
        nscan_per_field: Scans per field.
        dt: Integration time in seconds.
        corr_type: Correlation type codes (9-12 = XX,XY,YX,YY).
        flag_every: Flag every Nth row wholesale; ``None`` leaves data unflagged.
        add_weight_spectrum: Also create and fill WEIGHT_SPECTRUM.
    """
    path = str(path)
    if os.path.exists(path):
        shutil.rmtree(path)

    ncorr = len(corr_type)
    coldescs = [makearrcoldesc("DATA", 0 + 0j, shape=[nchan, ncorr], valuetype="complex")]
    if add_weight_spectrum:
        coldescs.append(makearrcoldesc("WEIGHT_SPECTRUM", 0.0, shape=[nchan, ncorr],
                                       valuetype="float"))
    default_ms(path, maketabdesc(coldescs))

    _write_antenna(path, nant)
    _write_feed(path, nant, nspw, corr_type)
    _write_spw(path, nspw, nchan)
    _write_polarization(path, corr_type)
    _write_data_description(path, nspw)
    _write_field(path, nfield)
    _write_state(path, nfield)

    baselines = [(i, j) for i in range(nant) for j in range(i + 1, nant)]
    tstart, tend = _write_main(path, baselines, nspw, nchan, ncorr, nfield,
                               nscan_per_field, ntime, dt, flag_every,
                               add_weight_spectrum)
    # Written last so TIME_RANGE actually matches the data, inter-scan gaps
    # and all.
    _write_observation(path, telescope, tstart, tend)
    return path


def _write_antenna(path, nant):
    with table(path + "::ANTENNA", readonly=False, ack=False) as tab:
        tab.addrows(nant)
        # Roughly MeerKAT-like ITRF positions, spread over a ~1 km footprint.
        pos = np.array([[5109360.0 + i * 137.0, 2006852.0 + i * 91.0, -3238948.0 - i * 73.0]
                        for i in range(nant)])
        tab.putcol("NAME", ["m%03d" % i for i in range(nant)])
        tab.putcol("STATION", ["m%03d" % i for i in range(nant)])
        tab.putcol("POSITION", pos)
        tab.putcol("DISH_DIAMETER", np.full(nant, 13.5))
        tab.putcol("MOUNT", ["ALT-AZ"] * nant)
        tab.putcol("TYPE", ["GROUND-BASED"] * nant)
        tab.putcol("FLAG_ROW", np.zeros(nant, bool))


def _write_feed(path, nant, nspw, corr_type):
    """One FEED row per (antenna, SPW).

    FEED is required by MSv2 and easy to forget, since casacore itself will
    happily read an MS without it -- but stricter readers (xarray-ms, for one)
    refuse an MS whose ANTENNA1/2 values are not covered by FEED::ANTENNA_ID.
    """
    # Correlation codes 9-12 are linear (X/Y), 5-8 circular (R/L).
    receptors = ["X", "Y"] if corr_type[0] >= 9 else ["R", "L"]
    nrec = len(receptors)

    with table(path + "::FEED", readonly=False, ack=False) as tab:
        tab.addrows(nant * nspw)
        row = 0
        for spw in range(nspw):
            for antenna in range(nant):
                tab.putcell("ANTENNA_ID", row, antenna)
                tab.putcell("FEED_ID", row, 0)
                tab.putcell("SPECTRAL_WINDOW_ID", row, spw)
                tab.putcell("TIME", row, T0)
                tab.putcell("INTERVAL", row, 1e30)      # valid for all time
                tab.putcell("NUM_RECEPTORS", row, nrec)
                tab.putcell("BEAM_ID", row, -1)
                tab.putcell("BEAM_OFFSET", row, np.zeros((nrec, 2)))
                tab.putcell("POLARIZATION_TYPE", row, receptors)
                tab.putcell("POL_RESPONSE", row, np.eye(nrec, dtype=np.complex64))
                tab.putcell("POSITION", row, np.zeros(3))
                tab.putcell("RECEPTOR_ANGLE", row, np.zeros(nrec))
                row += 1


def _write_spw(path, nspw, nchan):
    with table(path + "::SPECTRAL_WINDOW", readonly=False, ack=False) as tab:
        tab.addrows(nspw)
        for spw in range(nspw):
            f0, df = 1.4e9 + spw * 100e6, 10e6
            tab.putcell("NUM_CHAN", spw, nchan)
            tab.putcell("CHAN_FREQ", spw, f0 + df * np.arange(nchan))
            tab.putcell("CHAN_WIDTH", spw, np.full(nchan, df))
            tab.putcell("EFFECTIVE_BW", spw, np.full(nchan, df))
            tab.putcell("RESOLUTION", spw, np.full(nchan, df))
            tab.putcell("REF_FREQUENCY", spw, f0)
            tab.putcell("TOTAL_BANDWIDTH", spw, df * nchan)
            tab.putcell("NAME", spw, "SPW%d" % spw)
            tab.putcell("MEAS_FREQ_REF", spw, 5)  # TOPO
            tab.putcell("FREQ_GROUP_NAME", spw, "Group1")


def _write_polarization(path, corr_type):
    ncorr = len(corr_type)
    with table(path + "::POLARIZATION", readonly=False, ack=False) as tab:
        tab.addrows(1)
        tab.putcell("NUM_CORR", 0, ncorr)
        tab.putcell("CORR_TYPE", 0, np.array(corr_type))
        # feed indices per correlation, shape (2, ncorr)
        products = np.array([[c // 2 % 2, c % 2] for c in range(ncorr)]).T
        tab.putcell("CORR_PRODUCT", 0, products)


def _write_data_description(path, nspw):
    with table(path + "::DATA_DESCRIPTION", readonly=False, ack=False) as tab:
        tab.addrows(nspw)
        tab.putcol("SPECTRAL_WINDOW_ID", np.arange(nspw))
        tab.putcol("POLARIZATION_ID", np.zeros(nspw, int))
        tab.putcol("FLAG_ROW", np.zeros(nspw, bool))


def _write_field(path, nfield):
    names = ["PKS1934-638", "J0217+0144", "DEEP_2"]
    with table(path + "::FIELD", readonly=False, ack=False) as tab:
        tab.addrows(nfield)
        for f in range(nfield):
            # (ra, dec) in radians, shape (1, 2) -- one polynomial term
            direction = np.array([[0.35 + f * 0.4, -0.52 - f * 0.05]])
            tab.putcell("NAME", f, names[f] if f < len(names) else "FIELD%d" % f)
            tab.putcell("CODE", f, "")
            tab.putcell("PHASE_DIR", f, direction)
            tab.putcell("DELAY_DIR", f, direction)
            tab.putcell("REFERENCE_DIR", f, direction)
            tab.putcell("SOURCE_ID", f, f)
            tab.putcell("TIME", f, T0)
            tab.putcell("NUM_POLY", f, 0)


def _write_observation(path, telescope, tstart, tend):
    with table(path + "::OBSERVATION", readonly=False, ack=False) as tab:
        tab.addrows(1)
        tab.putcell("TELESCOPE_NAME", 0, telescope)
        tab.putcell("OBSERVER", 0, "msutils-tests")
        tab.putcell("PROJECT", 0, "TEST-2024-01")
        tab.putcell("TIME_RANGE", 0, np.array([tstart, tend]))


def _write_state(path, nfield):
    """One STATE row per field, each with a distinct observing intent."""
    with table(path + "::STATE", readonly=False, ack=False) as tab:
        tab.addrows(nfield)
        tab.putcol("OBS_MODE", [DEFAULT_INTENTS[f % len(DEFAULT_INTENTS)]
                                for f in range(nfield)])
        tab.putcol("SIG", np.ones(nfield, bool))
        tab.putcol("REF", np.zeros(nfield, bool))
        tab.putcol("FLAG_ROW", np.zeros(nfield, bool))


def _write_main(path, baselines, nspw, nchan, ncorr, nfield, nscan_per_field,
                ntime, dt, flag_every, add_weight_spectrum):
    time, a1, a2, ddid, field, scan, state = [], [], [], [], [], [], []
    scan_no = 0
    stamp = T0
    for f in range(nfield):
        for _ in range(nscan_per_field):
            scan_no += 1
            for _t in range(ntime):
                for dd in range(nspw):
                    for (p, q) in baselines:
                        time.append(stamp)
                        a1.append(p)
                        a2.append(q)
                        ddid.append(dd)
                        field.append(f)
                        scan.append(scan_no)
                        # STATE_ID indexes the STATE table, one row per field
                        state.append(f)
                stamp += dt
            # a gap between scans, so scan boundaries are unambiguous
            stamp += 5 * dt

    nrow = len(time)
    rng = np.random.default_rng(42)
    with table(path, readonly=False, ack=False) as tab:
        tab.addrows(nrow)
        tab.putcol("TIME", np.asarray(time, float))
        tab.putcol("TIME_CENTROID", np.asarray(time, float))
        tab.putcol("ANTENNA1", np.asarray(a1, np.int32))
        tab.putcol("ANTENNA2", np.asarray(a2, np.int32))
        tab.putcol("DATA_DESC_ID", np.asarray(ddid, np.int32))
        tab.putcol("FIELD_ID", np.asarray(field, np.int32))
        tab.putcol("SCAN_NUMBER", np.asarray(scan, np.int32))
        tab.putcol("STATE_ID", np.asarray(state, np.int32))
        tab.putcol("EXPOSURE", np.full(nrow, dt))
        tab.putcol("INTERVAL", np.full(nrow, dt))
        for col in ("ARRAY_ID", "OBSERVATION_ID", "PROCESSOR_ID", "FEED1", "FEED2"):
            tab.putcol(col, np.zeros(nrow, np.int32))
        tab.putcol("UVW", rng.normal(0.0, 800.0, (nrow, 3)))
        tab.putcol("DATA", (rng.normal(size=(nrow, nchan, ncorr))
                            + 1j * rng.normal(size=(nrow, nchan, ncorr))).astype(np.complex64))
        flag = np.zeros((nrow, nchan, ncorr), bool)
        if flag_every:
            flag[::flag_every] = True
        tab.putcol("FLAG", flag)
        tab.putcol("FLAG_ROW", np.zeros(nrow, bool))
        tab.putcol("SIGMA", np.ones((nrow, ncorr), np.float32))
        tab.putcol("WEIGHT", np.ones((nrow, ncorr), np.float32))
        if add_weight_spectrum:
            tab.putcol("WEIGHT_SPECTRUM", np.ones((nrow, nchan, ncorr), np.float32))

    return time[0], time[-1] + dt
