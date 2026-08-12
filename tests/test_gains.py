"""The gain model, its readers, and the flux-scale bootstrap.

The point of building the test tables with *known* gains (``gainfactory``)
is that a flux bootstrap has an exact answer: write a transfer calibrator
whose gains are the reference's times ``sqrt(S)`` and the measurement must
return ``S``. Every other property here -- flags, antenna matching, what the
output table contains -- is a way for that number to be wrong while still
looking plausible, which is why each gets its own test rather than being
folded into a "it works" case.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from gainfactory import corrupt_solution, flag_solution, make_caltable

import msutils.gains as gains
from msutils.gains.cli import cli


@pytest.fixture
def caltable(tmp_path):
    """A two-field caltable whose transfer calibrator is exactly 5 Jy."""
    path, instrumental = make_caltable(tmp_path / "cal.G0", flux=5.0)
    return path, instrumental


def test_reader_folds_rows_into_blocks_per_field_and_spw(tmp_path):
    path, _ = make_caltable(tmp_path / "two-spw.G0", nspw=2, nant=4, ntime=3)
    table = gains.read_gains(path)

    assert table.format == "casa"
    assert table.term == "G Jones"
    assert len(table) == 4  # 2 fields x 2 spws
    assert table.fields == (0, 1)
    assert table.spws == (0, 1)
    block = table.select(field="GAINCAL", spw=1).blocks[0]
    assert block.shape == (3, 1, 4, 2)  # time, freq, antenna, correlation
    # antennas come back as names, which is what makes cross-table matching safe
    assert block.antennas == ("m000", "m001", "m002", "m003")
    # a single-channel solve is labelled with its window's centre frequency
    assert block.freqs[0] == pytest.approx(1.5e9 + 10e6 / 4 * 1.5)


def test_a_missing_solution_reads_back_flagged_not_zero(tmp_path):
    """Folding rows into a dense array invents (time, antenna) slots that the
    table never had. If those read as unflagged zeros, every mean over the
    block is quietly wrong."""
    path, _ = make_caltable(tmp_path / "holes.G0")
    from casacore.tables import table as casa_table

    with casa_table(path, readonly=False, ack=False) as tab:
        tab.removerows([0])  # drop one (time, antenna) solution

    block = gains.read_gains(path).select(field=0).blocks[0]
    assert block.flags.sum() > 0
    assert block.masked().count() == block.gains.size - block.flags.sum()


def test_the_bootstrap_recovers_the_flux_it_was_given(caltable):
    path, _ = caltable
    result = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL")

    # Within its own error bar, which is the claim that matters: the fixture
    # jitters the gains, so an answer that landed exactly on 5.0 would mean the
    # scatter was being ignored rather than measured.
    measured = result.fluxes[0]
    assert abs(measured.flux_jy - 5.0) < 3 * measured.error_jy
    assert result.flux_jy == pytest.approx(5.0, rel=2e-2)
    assert measured.n_samples == 8  # 4 antennas x 2 correlations
    assert result.fluxes[0].snr > 50
    assert set(result.fluxes[0].per_antenna) == {"m000", "m001", "m002", "m003"}


def test_the_report_reads_like_casas_own(caltable):
    path, _ = caltable
    result = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL")
    line = result.render()
    assert line.startswith("# Flux density for GAINCAL in SpW=0")
    assert "SNR = " in line and "N = 8" in line


def test_fields_resolve_by_name_or_id_and_a_missing_one_raises(caltable):
    path, _ = caltable
    by_id = gains.fluxscale(path, transfer_field=1, reference_field=0)
    by_name = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL")
    assert by_id.flux_jy == pytest.approx(by_name.flux_jy)

    with pytest.raises(ValueError, match="no field named 'NOPE'"):
        gains.fluxscale(path, transfer_field="NOPE", reference_field="REFCAL")


def test_a_calibrator_cannot_bootstrap_off_itself(caltable):
    path, _ = caltable
    with pytest.raises(ValueError, match="cannot bootstrap its own flux scale"):
        gains.fluxscale(path, transfer_field="GAINCAL", reference_field="GAINCAL")


def test_flagged_solutions_are_excluded_from_the_measurement(tmp_path):
    """A flagged solution that still counts is the failure mode with no
    symptom: the answer stays finite and is simply wrong."""
    path, _ = make_caltable(tmp_path / "flagged.G0", flux=5.0)
    corrupt_solution(path, field_id=1, antenna=2, factor=100.0)
    flag_solution(path, field_id=1, antenna=2)

    result = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL")
    assert result.flux_jy == pytest.approx(5.0, rel=2e-2)
    assert result.fluxes[0].n_samples == 6  # the flagged antenna dropped out
    assert "m002" not in result.fluxes[0].per_antenna


def test_a_deviant_antenna_shows_up_per_antenna_and_in_the_error(tmp_path):
    path, _ = make_caltable(tmp_path / "deviant.G0", flux=5.0)
    corrupt_solution(path, field_id=1, antenna=1, factor=2.0)

    result = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL")
    per_antenna = result.fluxes[0].per_antenna
    assert per_antenna["m001"] == pytest.approx(20.0, rel=1e-2)  # 4x in amplitude
    # and the mean is dragged with a wide error rather than silently accepted
    assert result.fluxes[0].snr < 10


def test_the_gain_threshold_drops_outlying_amplitudes(tmp_path):
    path, _ = make_caltable(tmp_path / "outlier.G0", flux=5.0, ntime=5)
    from casacore.tables import table as casa_table

    with casa_table(path, readonly=False, ack=False) as tab:
        fields, ants = tab.getcol("FIELD_ID"), tab.getcol("ANTENNA1")
        params = tab.getcol("CPARAM")
        row = int(np.flatnonzero((fields == 1) & (ants == 0))[0])
        params[row] *= 50.0
        tab.putcol("CPARAM", params)

    loose = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL")
    tight = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL", gain_threshold=0.5)
    # the median already resists one outlier in five times; the threshold makes
    # that explicit rather than incidental
    assert tight.flux_jy == pytest.approx(5.0, rel=2e-2)
    assert tight.fluxes[0].per_antenna["m000"] == pytest.approx(5.0, rel=2e-2)
    assert loose.fluxes[0].n_samples == tight.fluxes[0].n_samples


def test_antennas_are_matched_by_name_not_position(tmp_path):
    """Two tables ordering antennas differently must not be compared
    positionally -- that yields a number rather than an error."""
    reference, _ = make_caltable(tmp_path / "ref.G0", nant=4, flux=1.0)
    transfer, _ = make_caltable(tmp_path / "trn.G0", nant=3, flux=5.0)
    from casacore.tables import table as casa_table

    with casa_table(f"{transfer}::ANTENNA", readonly=False, ack=False) as tab:
        tab.putcol("NAME", np.array(["m003", "m002", "m001"]))

    result = gains.fluxscale(
        transfer, reference, transfer_field="GAINCAL", reference_field="REFCAL"
    )
    assert set(result.antennas_used) == {"m001", "m002", "m003"}


def test_two_separate_tables_need_no_appended_copy(tmp_path):
    """The reason this exists at all: CASA compares fields inside one table,
    so a caller must solve the second calibrator by appending into a copy of
    the first one's. Two paths, one call, no copy."""
    reference, _ = make_caltable(tmp_path / "primary.G0", flux=1.0, seed=3)
    transfer, _ = make_caltable(tmp_path / "secondary.G0", flux=7.5, seed=3)

    result = gains.fluxscale(
        transfer, reference, transfer_field="GAINCAL", reference_field="REFCAL"
    )
    assert result.flux_jy == pytest.approx(7.5, rel=1e-2)


def test_the_output_holds_the_transfer_field_alone_rescaled(caltable, tmp_path):
    path, instrumental = caltable
    out = tmp_path / "rescaled.G0"
    result = gains.fluxscale(
        path, transfer_field="GAINCAL", reference_field="REFCAL", output=out
    )
    assert result.output_path == str(out)

    written = gains.read_gains(out)
    assert written.fields == (1,)  # the reference calibrator is NOT carried through
    block = written.blocks[0]
    # the rescaled gains are instrumental again: what the reference's are
    amplitude = np.ma.abs(block.masked()).mean(axis=(0, 1))
    assert np.allclose(amplitude, instrumental, rtol=0.05)

    # and the input is untouched
    original = gains.read_gains(path).select(field=1).blocks[0]
    assert np.ma.abs(original.masked()).mean() > np.ma.abs(block.masked()).mean()


def test_writing_refuses_an_existing_destination(caltable, tmp_path):
    path, _ = caltable
    out = tmp_path / "already-there"
    out.mkdir()
    with pytest.raises(FileExistsError):
        gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL", output=out)


def test_json_report_carries_the_measurement_and_its_provenance(caltable, tmp_path):
    path, _ = caltable
    report = tmp_path / "flux.json"
    gains.fluxscale(
        path, transfer_field="GAINCAL", reference_field="REFCAL", json_out=report
    )
    written = json.loads(report.read_text())
    assert written["schema_version"] == 1
    assert written["transfer_field"] == "GAINCAL"
    assert written["reference_field"] == "REFCAL"
    assert written["fluxes"][0]["flux_jy"] == pytest.approx(5.0, rel=2e-2)
    assert written["transfer_path"] == str(path)


def test_cli_prints_the_measurement_and_writes_json(caltable, tmp_path):
    path, _ = caltable
    report = tmp_path / "cli.json"
    result = CliRunner().invoke(
        cli,
        [
            "fluxscale",
            str(path),
            "--transfer-field",
            "GAINCAL",
            "--reference-field",
            "REFCAL",
            "--json",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "# Flux density for GAINCAL" in result.output
    assert json.loads(report.read_text())["fluxes"][0]["flux_jy"] == pytest.approx(5.0, rel=2e-2)


def test_detect_format_names_what_is_there(caltable, tmp_path):
    path, _ = caltable
    assert gains.detect_format(path) == "casa"
    with pytest.raises(FileNotFoundError):
        gains.detect_format(tmp_path / "nothing")
    empty = tmp_path / "neither"
    empty.mkdir()
    with pytest.raises(ValueError, match="neither a CASA"):
        gains.detect_format(empty)


def test_the_statistic_is_a_recorded_choice(caltable):
    """median and mean answer the same question and can disagree: on real
    MeerKAT gains they differ by 0.5%, which is larger than either one's
    error bar. So which was used has to travel with the number."""
    path, _ = caltable
    median = gains.fluxscale(path, transfer_field="GAINCAL", reference_field="REFCAL")
    mean = gains.fluxscale(
        path, transfer_field="GAINCAL", reference_field="REFCAL", statistic="mean"
    )
    assert median.statistic == "median"
    assert mean.statistic == "mean"
    # on gains jittered symmetrically the two agree; the point is that the
    # choice is recorded, not that it never matters
    assert mean.flux_jy == pytest.approx(median.flux_jy, rel=1e-2)

    with pytest.raises(ValueError, match="unknown statistic"):
        gains.fluxscale(
            path, transfer_field="GAINCAL", reference_field="REFCAL", statistic="mode"
        )


# ---- normalise --------------------------------------------------------


def _amplitudes(path, field=1):
    """Per (antenna, correlation) mean amplitude of one field's block."""
    block = gains.read_gains(path).select(field=field).blocks[0]
    return np.ma.abs(block.masked()).mean(axis=(0, 1))


def test_normalise_leaves_the_chosen_statistic_at_unity(caltable, tmp_path):
    path, _ = caltable
    out = tmp_path / "normalised.G0"
    result = gains.normalise(path, output=out, statistic="mean")

    block = gains.read_gains(out).select(field=1).blocks[0]
    amplitude = np.ma.abs(block.masked()).mean(axis=(0, 1))
    assert np.allclose(amplitude, 1.0)
    assert result.statistic == "mean"
    assert result.output_path == str(out)


def test_median_and_mean_normalisation_differ_on_skewed_gains(tmp_path):
    """The two answer different questions, and a skewed set is where that
    shows: one bad interval drags a mean and leaves a median alone."""
    path, _ = make_caltable(tmp_path / "skew.G0", ntime=5, flux=1.0)
    from casacore.tables import table as casa_table

    with casa_table(path, readonly=False, ack=False) as tab:
        params = tab.getcol("CPARAM")
        rows = np.flatnonzero((tab.getcol("FIELD_ID") == 1) & (tab.getcol("ANTENNA1") == 0))
        params[rows[0]] *= 10.0  # one interval, one antenna
        tab.putcol("CPARAM", params)

    by_median = gains.normalise(path, output=tmp_path / "med.G0", statistic="median")
    by_mean = gains.normalise(path, output=tmp_path / "avg.G0", statistic="mean")
    median_factor = by_median.blocks[1].factors["m000"][0]
    mean_factor = by_mean.blocks[1].factors["m000"][0]
    assert mean_factor > median_factor * 1.5


def test_scope_block_preserves_relative_antenna_amplitudes(caltable, tmp_path):
    """The scope choice is a science decision: per-antenna normalisation moves
    the relative antenna amplitudes into whatever term is applied alongside,
    and only `block` leaves them where they were."""
    path, _ = caltable
    before = _amplitudes(path)

    # `mean` on both, because the assertion below measures the mean: a median
    # normalisation leaves the mean near 1 rather than at it, and testing one
    # statistic through the other only measures their difference.
    per_antenna = gains.normalise(
        path, output=tmp_path / "per-ant.G0", scope="antenna", statistic="mean"
    )
    whole_block = gains.normalise(
        path, output=tmp_path / "per-block.G0", scope="block", statistic="mean"
    )
    assert per_antenna.scope == "antenna" and whole_block.scope == "block"

    flattened = _amplitudes(tmp_path / "per-ant.G0")
    kept = _amplitudes(tmp_path / "per-block.G0")
    # per antenna: every antenna ends at 1, so the spread is gone
    assert np.allclose(flattened, 1.0, atol=1e-6)
    # per block: the ratios between antennas survive exactly
    assert np.allclose(kept / kept.mean(), before / before.mean(), rtol=1e-6)


def test_scope_correlation_levels_the_polarisations_separately(caltable, tmp_path):
    path, _ = caltable
    result = gains.normalise(
        path, output=tmp_path / "per-corr.G0", scope="correlation", statistic="mean"
    )
    after = _amplitudes(tmp_path / "per-corr.G0")
    # each correlation is levelled on its own...
    assert np.allclose(after.mean(axis=0), 1.0, rtol=1e-6)
    # ...while the antennas within one still differ, because they share a
    # single factor rather than getting one each
    assert after[:, 0].std() > 0
    assert result.blocks[1].factors["m000"] == result.blocks[1].factors["m001"]
    # and the two polarisations were levelled independently
    assert result.blocks[1].factors["m000"][0] != result.blocks[1].factors["m000"][1]


def test_axis_freq_normalises_the_spectrum_and_keeps_the_time_variation(tmp_path):
    path, _ = make_caltable(tmp_path / "bandpass.B0", nchan=4, ntime=3, viscal="B Jones")
    out = tmp_path / "normalised.B0"
    gains.normalise(path, output=out, axis="freq", statistic="mean")

    block = gains.read_gains(out).select(field=1).blocks[0]
    amplitude = np.ma.abs(block.masked())
    # every (time, antenna, correlation) spectrum now averages to 1
    assert np.allclose(amplitude.mean(axis=1), 1.0)
    # and the solutions still differ from one time to the next
    assert amplitude.std(axis=0).max() > 0


def test_phases_are_untouched(caltable, tmp_path):
    path, _ = caltable
    out = tmp_path / "phase-safe.G0"
    gains.normalise(path, output=out)
    before = gains.read_gains(path).select(field=1).blocks[0]
    after = gains.read_gains(out).select(field=1).blocks[0]
    assert np.allclose(np.angle(before.gains), np.angle(after.gains))


def test_flagged_solutions_do_not_enter_the_factor(tmp_path):
    path, _ = make_caltable(tmp_path / "flagged-norm.G0", flux=1.0)
    corrupt_solution(path, field_id=1, antenna=0, factor=1000.0)
    flag_solution(path, field_id=1, antenna=0)

    result = gains.normalise(path, scope="block", statistic="mean")
    # the corrupted antenna is flagged, so the block factor stays near the
    # honest amplitudes rather than being dragged by a factor of 1000
    assert result.blocks[1].factor_summary < 2.0


def test_a_delay_table_is_refused(tmp_path):
    """Normalising a delay is arithmetic without a meaning -- it is a slope in
    frequency, not an amplitude."""
    path, _ = make_caltable(tmp_path / "delay-norm.K0", viscal="K Jones", param_col="FPARAM")
    with pytest.raises(ValueError, match=r"which is not an"):
        gains.normalise(path)


def test_the_report_records_the_factors_that_left(caltable, tmp_path):
    path, _ = caltable
    report = tmp_path / "norm.json"
    result = gains.normalise(path, json_out=report, scope="block")
    written = json.loads(report.read_text())
    assert written["schema_version"] == 1
    assert written["scope"] == "block"
    assert written["statistic"] == "median"
    # field 1 was solved with sqrt(5) x the reference's gains, so that is what
    # a block normalisation should have taken out
    transfer = next(b for b in written["blocks"] if b["field_id"] == 1)
    reference = next(b for b in written["blocks"] if b["field_id"] == 0)
    assert transfer["factor_summary"] / reference["factor_summary"] == pytest.approx(np.sqrt(5.0), rel=0.05)
    assert result.blocks[0].factors["m000"]


def test_normalise_never_writes_in_place(caltable, tmp_path):
    path, _ = caltable
    before = _amplitudes(path)
    gains.normalise(path, output=tmp_path / "elsewhere.G0")
    assert np.allclose(_amplitudes(path), before)


def test_cli_normalise_writes_gains_and_a_report(caltable, tmp_path):
    path, _ = caltable
    out, report = tmp_path / "cli-norm.G0", tmp_path / "cli-norm.json"
    result = CliRunner().invoke(
        cli,
        ["normalise", str(path), "-o", str(out), "--scope", "block", "--json", str(report)],
    )
    assert result.exit_code == 0, result.output
    assert "# Normalised field=" in result.output
    assert json.loads(report.read_text())["scope"] == "block"
    assert gains.read_gains(out)


def test_unknown_axis_and_scope_are_refused(caltable):
    path, _ = caltable
    with pytest.raises(ValueError, match="unknown axis"):
        gains.normalise(path, axis="channel")
    with pytest.raises(ValueError, match="unknown scope"):
        gains.normalise(path, scope="spw")


# ---- smooth -----------------------------------------------------------


def _write_gains(path, values, *, field_id=1):
    """Overwrite one field's CPARAM with `values`, shaped (time, ant, corr)."""
    from casacore.tables import table as casa_table

    with casa_table(path, readonly=False, ack=False) as tab:
        fields, ants, times = tab.getcol("FIELD_ID"), tab.getcol("ANTENNA1"), tab.getcol("TIME")
        column = "CPARAM" if "CPARAM" in tab.colnames() else "FPARAM"
        params = tab.getcol(column)
        order = sorted(set(times[fields == field_id].tolist()))
        for row in np.flatnonzero(fields == field_id):
            ti = order.index(times[row])
            params[row, 0, :] = values[ti, ants[row], :].real if column == "FPARAM" else values[ti, ants[row], :]
        tab.putcol(column, params)


def test_smoothing_averages_noise_down(tmp_path):
    path, _ = make_caltable(tmp_path / "noisy.G0", nant=2, ntime=21, jitter=0.4, flux=1.0)
    out = tmp_path / "smoothed.G0"
    gains.smooth(path, output=out, time_window=5)

    before = gains.read_gains(path).select(field=1).blocks[0]
    after = gains.read_gains(out).select(field=1).blocks[0]
    assert np.abs(after.gains).std() < np.abs(before.gains).std()


def test_a_phase_wrap_does_not_spike(tmp_path):
    """The reason this filters the complex gain: a running mean of the phase
    itself averages values adjacent on the circle and far apart as numbers,
    putting a spike at every +/-pi crossing."""
    path, _ = make_caltable(tmp_path / "wrap.G0", nant=1, ntime=41, jitter=0.0, flux=1.0)
    # a phase ramp of 0.4 rad per solution: it crosses +/-pi twice
    phase = 0.4 * np.arange(41)
    values = np.exp(1j * phase)[:, None, None] * np.ones((41, 1, 2))
    _write_gains(path, values)

    out = tmp_path / "wrap-smoothed.G0"
    gains.smooth(path, output=out, time_window=5)
    after = gains.read_gains(out).select(field=1).blocks[0]

    # the smoothed phase still steps by 0.4 rad everywhere except the wraps,
    # which is what "no spike" means once the values are re-wrapped
    step = np.diff(np.angle(after.gains[:, 0, 0, 0]))
    step = (step + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(step[2:-2], 0.4, atol=1e-6)

    # and for contrast: smoothing the phase as a number would have produced
    # excursions of order pi at the wraps
    naive = np.convolve(np.angle(np.exp(1j * phase)), np.ones(5) / 5, mode="same")
    naive_step = np.diff(naive)
    assert np.abs(naive_step - 0.4).max() > 0.4  # off by more than the ramp itself


def test_renormalising_keeps_the_amplitude_a_vector_average_would_shrink(tmp_path):
    """Vector averaging a rotating gain shortens it -- by ~13% for a 0.8
    rad/sample ramp over 5 samples. Uncorrected, that would rescale the flux
    of everything the solutions are applied to."""
    path, _ = make_caltable(tmp_path / "rotate.G0", nant=1, ntime=31, jitter=0.0, flux=1.0)
    amplitude = 2.0
    values = amplitude * np.exp(1j * 0.8 * np.arange(31))[:, None, None] * np.ones((31, 1, 2))
    _write_gains(path, values)

    out = tmp_path / "rotate-smoothed.G0"
    gains.smooth(path, output=out, time_window=5)
    after = gains.read_gains(out).select(field=1).blocks[0]
    assert np.allclose(np.abs(after.gains[2:-2]), amplitude, rtol=1e-6)

    # what the same filter does without the renormalisation
    shrunk = np.abs(np.convolve(values[:, 0, 0], np.ones(5) / 5, mode="same"))
    assert shrunk[5] < 0.9 * amplitude


def test_physical_windows_convert_against_the_axis(tmp_path):
    """A window in seconds means the same thing on differently-averaged data;
    the report says what it resolved to here."""
    path, _ = make_caltable(tmp_path / "physical.G0", ntime=9)  # 60 s apart
    result = gains.smooth(path, time_window="180s")
    assert result.blocks[0].time_samples == 3

    with pytest.raises(ValueError, match="unknown time unit"):
        gains.smooth(path, time_window="180parsec")
    with pytest.raises(ValueError, match="cannot read time window"):
        gains.smooth(path, time_window="soon")


def test_flagged_solutions_neither_contribute_nor_are_revived(tmp_path):
    path, _ = make_caltable(tmp_path / "gappy.G0", nant=2, ntime=9, jitter=0.0, flux=1.0)
    corrupt_solution(path, field_id=1, antenna=0, factor=100.0)
    flag_solution(path, field_id=1, antenna=0)
    out = tmp_path / "gappy-smoothed.G0"
    gains.smooth(path, output=out, time_window=5)

    after = gains.read_gains(out).select(field=1).blocks[0]
    # antenna 1 is untouched by antenna 0's absurd values (they are flagged,
    # and smoothing is per antenna anyway), and antenna 0 stays flagged
    assert after.flags[:, :, 0, :].all()
    assert not after.flags[:, :, 1, :].any()


def test_fill_revives_solutions_a_window_could_reach(tmp_path):
    path, _ = make_caltable(tmp_path / "fill.G0", nant=2, ntime=9, jitter=0.0, flux=1.0)
    flag_solution(path, field_id=1, antenna=0)

    kept = gains.smooth(path, output=tmp_path / "kept.G0", time_window=5)
    filled = gains.smooth(path, output=tmp_path / "filled.G0", time_window=5, fill=True)
    assert kept.blocks[1].filled == 0
    # an antenna flagged at every time has no neighbours of its own to borrow
    # from, so even fill cannot revive it -- smoothing does not cross antennas
    assert filled.blocks[1].filled == 0

    # but a hole in time is bridged
    path2, _ = make_caltable(tmp_path / "hole.G0", nant=2, ntime=9, jitter=0.0, flux=1.0)
    from casacore.tables import table as casa_table

    with casa_table(path2, readonly=False, ack=False) as tab:
        fields, times = tab.getcol("FIELD_ID"), tab.getcol("TIME")
        flags = tab.getcol("FLAG")
        middle = sorted(set(times[fields == 1].tolist()))[4]
        flags[np.flatnonzero((fields == 1) & (times == middle))] = True
        tab.putcol("FLAG", flags)

    bridged = gains.smooth(path2, output=tmp_path / "bridged.G0", time_window=5, fill=True)
    assert bridged.blocks[1].filled == 4  # 2 antennas x 2 correlations


def test_a_delay_table_is_smoothed_as_a_parameter(tmp_path):
    """A delay is a number, not a phasor: the complex/renormalise path would
    treat its magnitude as a gain amplitude, which it is not."""
    path, _ = make_caltable(
        tmp_path / "delay.K0", nant=1, ntime=9, jitter=0.0, viscal="K Jones", param_col="FPARAM"
    )
    values = np.zeros((9, 1, 2), dtype=complex)
    values[:, 0, :] = np.array([1, 2, 3, 4, 100, 6, 7, 8, 9])[:, None]  # ns, with a spike
    _write_gains(path, values)

    out = tmp_path / "delay-smoothed.K0"
    result = gains.smooth(path, output=out, time_window=3)
    assert result.param_type == "float"
    after = gains.read_gains(out).select(field=1).blocks[0]
    smoothed = after.gains[:, 0, 0, 0].real
    # the spike is averaged with its neighbours as a NUMBER: (4 + 100 + 6)/3
    assert smoothed[4] == pytest.approx((4 + 100 + 6) / 3)
    # and nothing acquired a phase on the way through
    assert np.allclose(after.gains.imag, 0.0)


def test_a_gaussian_kernel_weights_the_centre_more(tmp_path):
    path, _ = make_caltable(tmp_path / "kernel.G0", nant=1, ntime=21, jitter=0.0, flux=1.0)
    values = np.ones((21, 1, 2), dtype=complex)
    values[10] = 11.0  # one spike
    _write_gains(path, values)

    box = gains.smooth(path, output=tmp_path / "box.G0", time_window=5, kernel="boxcar")
    gauss = gains.smooth(path, output=tmp_path / "gauss.G0", time_window=5, kernel="gaussian")
    assert box.kernel == "boxcar" and gauss.kernel == "gaussian"
    box_peak = np.abs(gains.read_gains(tmp_path / "box.G0").select(field=1).blocks[0].gains[10, 0, 0, 0])
    gauss_peak = np.abs(gains.read_gains(tmp_path / "gauss.G0").select(field=1).blocks[0].gains[10, 0, 0, 0])
    assert gauss_peak > box_peak  # the gaussian keeps more of the centre sample


def test_smoothing_with_no_window_is_refused(caltable):
    path, _ = caltable
    with pytest.raises(ValueError, match="nothing to smooth"):
        gains.smooth(path)
    with pytest.raises(ValueError, match="unknown kernel"):
        gains.smooth(path, time_window=3, kernel="sinc")


def test_cli_smooth_writes_gains_and_a_report(caltable, tmp_path):
    path, _ = caltable
    out, report = tmp_path / "cli-smooth.G0", tmp_path / "cli-smooth.json"
    result = CliRunner().invoke(
        cli,
        ["smooth", str(path), "-o", str(out), "--time-window", "180s", "--json", str(report)],
    )
    assert result.exit_code == 0, result.output
    assert "# Smoothed field=" in result.output
    written = json.loads(report.read_text())
    assert written["time_window"] == "180s"
    assert written["blocks"][0]["time_samples"] == 3
    assert gains.read_gains(out)


# ---- QuartiCal stores -------------------------------------------------

xr = pytest.importorskip("xarray", reason="xarray/zarr not installed (msutils[gains])")


def test_a_quartical_store_reads_into_the_same_model(tmp_path):
    from gainfactory import make_quartical_store

    path = make_quartical_store(tmp_path / "solve.qc", nant=4, ntime=3, nchan=2, ndir=2)
    table = gains.read_gains(path)

    assert table.format == "quartical"
    assert table.term == "G" and table.gain_type == "complex"
    # a direction is a block, not an axis, so every operation stays 4-D
    assert len(table) == 2
    assert [b.direction for b in table.blocks] == [0, 1]
    block = table.blocks[0]
    assert block.shape == (3, 2, 4, 2)
    assert block.antennas == ("m000", "m001", "m002", "m003")
    assert block.corrs == ("XX", "YY")


def test_a_store_holding_several_terms_refuses_to_guess(tmp_path):
    from gainfactory import make_quartical_store

    path = make_quartical_store(tmp_path / "chain.qc", term="G")
    make_quartical_store(tmp_path / "k.qc", term="K", gain_type="delay_and_offset")
    shutil.copytree(tmp_path / "k.qc" / "K", Path(path) / "K")

    assert set(gains.quartical_terms(path)) == {"G", "K"}
    with pytest.raises(ValueError, match="name the one you mean"):
        gains.read_gains(path)
    assert gains.read_gains(path, term="K").gain_type == "delay_and_offset"


def test_writing_a_store_round_trips_through_an_operation(tmp_path):
    from gainfactory import make_quartical_store

    path = make_quartical_store(tmp_path / "solve.qc", flux=4.0)
    out = tmp_path / "normalised.qc"
    gains.normalise(path, output=out, scope="block", statistic="mean")

    written = gains.read_gains(out)
    assert written.format == "quartical"
    amplitude = np.ma.abs(written.blocks[0].masked()).mean()
    assert amplitude == pytest.approx(1.0)
    # the source is untouched, and everything the model does not represent
    # survived the copy
    assert np.ma.abs(gains.read_gains(path).blocks[0].masked()).mean() > 1.5
    assert (Path(out) / "G" / "G_0" / "gain_flags").exists()
    assert (Path(out) / "G" / "G_0" / ".zattrs").exists()


def test_flags_collapse_onto_the_solution_when_written(tmp_path):
    """QuartiCal flags a solution, not a correlation of one. Writing has to
    collapse the model's correlation axis, and it does so with `any` -- a
    solution one of whose correlations is unusable is not one to apply."""
    from gainfactory import make_quartical_store

    path = make_quartical_store(tmp_path / "flagme.qc", ntime=5)
    table = gains.read_gains(path)
    block = table.blocks[0]
    flags = block.flags.copy()
    flags[1, :, 0, 0] = True  # one correlation only
    table.blocks[0] = block.with_gains(block.gains, flags=flags)

    out = tmp_path / "flagged.qc"
    gains.write_gains(table, out)
    written = gains.read_gains(out).blocks[0]
    assert written.flags[1, :, 0, :].all()  # both correlations, after the collapse
    assert not written.flags[2, :, 0, :].any()


def test_a_parameterised_terms_parameters_come_through_and_go_back(tmp_path):
    """QuartiCal rebuilds such a term's gains from `params` when it loads
    them, so the parameters are the authoritative array: an operation that
    changed gains alone would read back unchanged."""
    from gainfactory import make_quartical_store

    path = make_quartical_store(
        tmp_path / "amp.qc",
        gain_type="amplitude",
        param_names=("amplitude_XX", "amplitude_YY"),
        flux=9.0,
    )
    block = gains.read_gains(path).blocks[0]
    assert block.parameterised
    assert block.param_names == ("amplitude_XX", "amplitude_YY")
    # an amplitude term's parameters ARE its gains
    assert np.allclose(block.params, block.gains.real)

    out = tmp_path / "amp-normalised.qc"
    gains.normalise(path, output=out, scope="block", statistic="mean")
    written = gains.read_gains(out).blocks[0]
    # both arrays moved together, and they still agree
    assert np.ma.abs(written.masked()).mean() == pytest.approx(1.0)
    assert np.allclose(written.params, written.gains.real)


def test_scaling_a_phase_like_term_is_refused_as_meaningless(tmp_path):
    """A delay or phase parameterisation describes a gain of unit modulus --
    verified on real stores, |g| = 1 to machine precision -- so there is no
    amplitude in it to normalise or bootstrap."""
    from gainfactory import make_quartical_store

    # A `phase` term, deliberately: its *type* name says nothing about
    # amplitudes, so the refusal has to come from reading the parameters
    # rather than from matching the type against a list.
    path = make_quartical_store(
        tmp_path / "phase.qc",
        term="P",
        gain_type="phase",
        param_names=("phase_XX", "phase_YY"),
    )
    with pytest.raises(ValueError, match="no amplitude in it to scale"):
        gains.normalise(path, term="P", output=tmp_path / "nope.qc")

    # a delay is refused too, by the coarser check that also covers a CASA
    # K table, which has no parameters to read
    delay = make_quartical_store(
        tmp_path / "delay.qc",
        term="K",
        gain_type="delay_and_offset",
        param_names=("phase_offset_XX", "delay_XX", "phase_offset_YY", "delay_YY"),
    )
    with pytest.raises(ValueError, match=r"not an\s+amplitude|no amplitude in it to scale"):
        gains.normalise(delay, term="K", output=tmp_path / "nope-delay.qc")


def test_smoothing_a_parameterised_term_filters_its_parameters(tmp_path):
    from gainfactory import make_quartical_store

    path = make_quartical_store(
        tmp_path / "amp-smooth.qc",
        gain_type="amplitude",
        param_names=("amplitude_XX", "amplitude_YY"),
        ntime=9,
        nchan=1,
    )
    out = tmp_path / "amp-smoothed.qc"
    gains.smooth(path, output=out, time_window=3)
    before = gains.read_gains(path).blocks[0]
    after = gains.read_gains(out).blocks[0]
    assert after.params.std() < before.params.std()
    # the two arrays still agree, which is what makes the store readable by
    # QuartiCal (params) and by everything else (gains)
    assert np.allclose(after.params, after.gains.real)


def test_smoothing_a_delay_refuses_to_invent_a_reference_frequency(tmp_path):
    """The parameters could be filtered; the gains cannot be rebuilt from
    them without knowing what the solver referenced the delay to."""
    from gainfactory import make_quartical_store

    path = make_quartical_store(
        tmp_path / "delay-smooth.qc",
        term="K",
        gain_type="delay_and_offset",
        param_names=("phase_offset_XX", "delay_XX", "phase_offset_YY", "delay_YY"),
        ntime=9,
    )
    with pytest.raises(ValueError, match="reference frequency"):
        gains.smooth(path, term="K", output=tmp_path / "nope.qc", time_window=3)


def test_blocks_from_another_store_are_refused(tmp_path):
    from gainfactory import make_quartical_store

    path = make_quartical_store(tmp_path / "one.qc", fields=(0,))
    other = make_quartical_store(tmp_path / "two.qc", fields=(7,))
    table = gains.read_gains(other)
    table.path = path  # pretend it came from the first store
    with pytest.raises(ValueError, match=r"did not\s+come from this store"):
        gains.write_gains(table, tmp_path / "mismatch.qc")


def test_writing_refuses_an_existing_store(tmp_path):
    from gainfactory import make_quartical_store

    path = make_quartical_store(tmp_path / "solve.qc")
    out = tmp_path / "taken.qc"
    out.mkdir()
    with pytest.raises(FileExistsError):
        gains.write_gains(gains.read_gains(path), out)
