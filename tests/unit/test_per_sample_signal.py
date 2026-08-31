"""Per-sample view of one annotated signal.

Annotation grades on the mean spectrum, so it cannot say in how many samples a
signal is actually there. These read what detection already recorded per sample
instead of re-measuring, which is what keeps "present here" from disagreeing
with the peak list.
"""
import numpy as np
import pytest

from processing.per_sample_signal import (
    alignment_shifts, line_positions, sample_count, signal_across_samples,
    signal_in_sample)

MHZ = 600.0
J = 7.0
D = J / MHZ          # one coupling, in ppm


def _line(ppm, present, drift=None, area=1.0, n=4):
    """One detected line: same apex in every sample unless drift says otherwise."""
    return {
        'ppm': ppm,
        'ppm_per_sample': [ppm + (drift[i] if drift else 0.0) for i in range(n)],
        'aligned_per_sample': list(present),
        'area_per_sample': [area] * n,
    }


def _doublet(centre=1.47, present=(True, True, True, True), drift=None, n=4):
    return [_line(centre - D / 2, present, drift, n=n),
            _line(centre + D / 2, present, drift, n=n)]


class TestPresence:
    def test_every_line_detected_is_present(self):
        r = signal_in_sample(_doublet(), 0)
        assert r['present'] is True

    def test_one_missing_leg_is_absent(self):
        """A doublet with one leg is not a doublet in that sample."""
        lines = [_line(1.4642, (True,) * 4), _line(1.4758, (False, True, True, True))]
        assert signal_in_sample(lines, 0)['present'] is False
        assert signal_in_sample(lines, 1)['present'] is True

    def test_absent_sample_reports_no_j(self):
        """An undetected line's recorded apex is noise; a J from it is fiction."""
        lines = [_line(1.4642, (False,) * 4), _line(1.4758, (False,) * 4)]
        r = signal_in_sample(lines, 0)
        assert r['present'] is False
        assert r['j_obs_hz'] is None
        assert r['delta_j_hz'] is None

    def test_missing_presence_vector_is_not_absence(self):
        """Detection that never recorded presence must not be read as absent."""
        lines = [{'ppm': 1.4642, 'ppm_per_sample': [1.4642] * 4},
                 {'ppm': 1.4758, 'ppm_per_sample': [1.4758] * 4}]
        assert signal_in_sample(lines, 0)['present'] is True


class TestMeasurement:
    def test_j_is_the_line_spacing_in_that_sample(self):
        r = signal_in_sample(_doublet(), 0, db_j_hz=J, field_mhz=MHZ)
        assert r['j_obs_hz'] == pytest.approx(J, abs=1e-6)
        assert r['delta_j_hz'] == pytest.approx(0.0, abs=1e-6)

    def test_drift_moves_the_centre_but_not_the_coupling(self):
        """The whole pattern shifts together, so J survives a 20 Hz drift."""
        shift = 20.0 / MHZ
        lines = _doublet(drift=[0.0, shift, 0.0, 0.0])
        r = signal_in_sample(lines, 1, db_ppm=1.47, db_j_hz=J, field_mhz=MHZ)
        assert r['j_obs_hz'] == pytest.approx(J, abs=1e-6)
        assert r['ppm_diff'] == pytest.approx(shift, abs=1e-9)

    def test_singlet_has_no_j(self):
        r = signal_in_sample([_line(3.03, (True,) * 4)], 0, db_j_hz=None)
        assert r['j_obs_hz'] is None
        assert r['n_lines'] == 1

    def test_area_sums_the_lines(self):
        r = signal_in_sample(_doublet(), 0)
        assert r['area'] == pytest.approx(2.0)

    def test_no_peaks_measures_nothing(self):
        r = signal_in_sample([], 0)
        assert r['present'] is False and r['area'] is None

    def test_sample_index_out_of_range(self):
        assert signal_in_sample(_doublet(), 99)['present'] is False


class TestAcrossSamples:
    def test_counts_only_the_included_samples(self):
        """Blanks live outside `included`, the way detection already treats them."""
        lines = _doublet(present=(True, True, False, True))
        out = signal_across_samples(lines, included=[1, 2, 3])
        assert out['n_included'] == 3
        assert out['n_present'] == 2
        assert [r['sample_idx'] for r in out['per_sample']] == [1, 2, 3]

    def test_all_samples_when_nothing_excluded(self):
        out = signal_across_samples(_doublet())
        assert out['n_included'] == 4 and out['n_present'] == 4

    def test_spread_reports_the_drift(self):
        shift = 0.03
        out = signal_across_samples(_doublet(drift=[0.0, shift, 0.0, 0.0]))
        assert out['ppm_spread'] == pytest.approx(shift, abs=1e-9)

    def test_spread_needs_two_present_samples(self):
        out = signal_across_samples(_doublet(present=(True, False, False, False)))
        assert out['ppm_spread'] is None

    def test_sample_count_survives_peaks_without_vectors(self):
        assert sample_count([{'ppm': 1.0}]) == 0
        assert sample_count(_doublet()) == 4


class TestAlignment:
    def test_shift_is_the_drift_of_the_pattern_centre(self):
        shift = 15.0 / MHZ
        lines = _doublet(drift=[0.0, shift, 0.0, 0.0])
        sh = alignment_shifts(lines)
        # Reference is the mean centre, so the shifts sum to zero.
        assert sum(sh.values()) == pytest.approx(0.0, abs=1e-12)
        assert sh[1] - sh[0] == pytest.approx(shift, abs=1e-9)

    def test_subtracting_the_shift_lines_the_centres_up(self):
        lines = _doublet(drift=[0.0, 0.01, -0.004, 0.007])
        sh = alignment_shifts(lines)
        aligned = [signal_in_sample(lines, i)['ppm_centre'] - sh[i] for i in sh]
        assert max(aligned) - min(aligned) == pytest.approx(0.0, abs=1e-9)

    def test_absent_samples_get_no_shift(self):
        sh = alignment_shifts(_doublet(present=(True, False, True, True)))
        assert set(sh) == {0, 2, 3}

    def test_explicit_reference_pins_the_axis(self):
        lines = _doublet(centre=1.47, drift=[0.0, 0.01, 0.0, 0.0])
        sh = alignment_shifts(lines, reference=1.47)
        assert sh[0] == pytest.approx(0.0, abs=1e-9)
        assert sh[1] == pytest.approx(0.01, abs=1e-9)

    def test_nothing_present_yields_no_shifts(self):
        assert alignment_shifts(_doublet(present=(False,) * 4)) == {}


def test_line_positions_falls_back_to_the_master_apex():
    """A peak without a per-sample vector still has one position to offer."""
    assert line_positions([{'ppm': 2.5}], 0) == [2.5]


class TestCollapsedLines:
    """The per-sample apex search scans ±0.02 ppm — 12 Hz at 600 MHz — while the
    median multiplet line spacing is 7.1 Hz, so the taller leg wins BOTH searches
    and both lines are reported at one position. 64% of (multiplet, sample) pairs
    on the reference set. Nothing may be measured from such a reading.
    """

    def _collapsed_doublet(self, n=4):
        """Sample 1 has both legs folded onto the left one."""
        left, right = 1.47 - D / 2, 1.47 + D / 2
        return [
            {'ppm': left, 'aligned_per_sample': [True] * n,
             'ppm_per_sample': [left, left, left, left],
             'area_per_sample': [1.0] * n},
            {'ppm': right, 'aligned_per_sample': [True] * n,
             'ppm_per_sample': [right, left, right, right],
             'area_per_sample': [1.0] * n},
        ]

    def test_collapsed_sample_reports_no_j(self):
        r = signal_in_sample(self._collapsed_doublet(), 1, db_j_hz=J, field_mhz=MHZ)
        assert r['collapsed'] is True
        assert r['j_obs_hz'] is None
        assert r['delta_j_hz'] is None

    def test_collapsed_sample_reports_no_centre(self):
        """The positions describe one leg, so their mean is off by half a J."""
        r = signal_in_sample(self._collapsed_doublet(), 1, db_ppm=1.47)
        assert r['ppm_centre'] is None
        assert r['ppm_diff'] is None

    def test_intact_sample_is_unaffected(self):
        r = signal_in_sample(self._collapsed_doublet(), 0, db_j_hz=J, field_mhz=MHZ)
        assert r['collapsed'] is False
        assert r['j_obs_hz'] == pytest.approx(J, abs=1e-6)

    def test_presence_still_reflects_detection(self):
        """Collapsing is a measurement failure, not an absence."""
        assert signal_in_sample(self._collapsed_doublet(), 1)['present'] is True

    def test_area_survives_the_collapse(self):
        """Areas come from integration, not from the apex search."""
        assert signal_in_sample(self._collapsed_doublet(), 1)['area'] == pytest.approx(2.0)

    def test_singlet_is_never_collapsed(self):
        r = signal_in_sample([_line(3.03, (True,) * 4)], 0)
        assert r['collapsed'] is False
        assert r['ppm_centre'] == pytest.approx(3.03)

    def test_across_samples_counts_the_collapses(self):
        out = signal_across_samples(self._collapsed_doublet())
        assert out['n_collapsed'] == 1
        assert out['n_present'] == 4

    def test_alignment_skips_a_collapsed_sample(self):
        """A shift from a folded pattern would move the sample by half a J."""
        assert 1 not in alignment_shifts(self._collapsed_doublet())

    def test_threshold_scales_with_the_multiplet(self):
        """What counts as collapsed for a 3 Hz doublet is not what counts for 12."""
        wide = 12.0 / MHZ
        lines = [
            {'ppm': 1.47, 'aligned_per_sample': [True] * 2,
             'ppm_per_sample': [1.47, 1.47]},
            {'ppm': 1.47 + wide, 'aligned_per_sample': [True] * 2,
             'ppm_per_sample': [1.47 + wide, 1.47 + wide * 0.7]},
        ]
        # 70% of the master spacing is not a collapse.
        assert signal_in_sample(lines, 1)['collapsed'] is False


class TestRelativeSD:
    def test_cv_of_a_measured_quantity(self):
        from processing.per_sample_signal import relative_sd_percent
        # J readings around 7 Hz, sd 0.1 → ~1.4%
        assert relative_sd_percent([6.9, 7.0, 7.1]) == pytest.approx(
            np.std([6.9, 7.0, 7.1], ddof=1) / 7.0 * 100.0)

    def test_identical_readings_have_no_spread(self):
        from processing.per_sample_signal import relative_sd_percent
        assert relative_sd_percent([7.0, 7.0, 7.0]) == pytest.approx(0.0)

    def test_one_reading_has_no_cv(self):
        from processing.per_sample_signal import relative_sd_percent
        assert relative_sd_percent([7.0]) is None
        assert relative_sd_percent([]) is None

    def test_a_mean_at_zero_has_no_cv(self):
        """Why the CV is taken on the measured value and not on the deviation:
        a deviation is centred on zero, so its ratio is meaningless."""
        from processing.per_sample_signal import relative_sd_percent
        assert relative_sd_percent([-0.2, 0.0, 0.2]) is None

    def test_nones_and_nans_are_skipped(self):
        from processing.per_sample_signal import relative_sd_percent
        assert relative_sd_percent([7.0, None, float('nan'), 7.2]) is not None


class TestPercentDeviation:
    def test_distance_from_the_reference(self):
        from processing.per_sample_signal import percent_deviation
        assert percent_deviation(7.14, 7.0) == pytest.approx(2.0)
        assert percent_deviation(6.86, 7.0) == pytest.approx(-2.0)

    def test_a_sample_on_the_mean_is_zero(self):
        from processing.per_sample_signal import percent_deviation
        assert percent_deviation(7.0, 7.0) == pytest.approx(0.0)

    def test_sign_survives_a_negative_reference(self):
        """The magnitude divides, so a value above its reference reads positive
        whichever side of zero the reference sits."""
        from processing.per_sample_signal import percent_deviation
        assert percent_deviation(-0.9, -1.0) == pytest.approx(10.0)

    def test_nothing_to_compare_against(self):
        from processing.per_sample_signal import percent_deviation
        assert percent_deviation(7.0, None) is None
        assert percent_deviation(None, 7.0) is None
        assert percent_deviation(7.0, 0.0) is None


class TestGroupMeans:
    def _rows(self):
        return [{'group': 'FL', 'j_obs': 7.0}, {'group': 'FL', 'j_obs': 7.2},
                {'group': 'HC', 'j_obs': 6.0}, {'group': 'HC', 'j_obs': None}]

    def test_means_are_taken_within_each_group(self):
        from processing.per_sample_signal import group_means
        m = group_means(self._rows(), 'j_obs')
        assert m['FL'] == pytest.approx(7.1)
        assert m['HC'] == pytest.approx(6.0)

    def test_missing_values_do_not_enter_the_mean(self):
        from processing.per_sample_signal import group_means
        assert group_means([{'group': 'A', 'j_obs': None}], 'j_obs') == {}

    def test_non_numeric_is_skipped(self):
        from processing.per_sample_signal import group_means
        rows = [{'group': 'A', 'j_obs': 'x'}, {'group': 'A', 'j_obs': 7.0}]
        assert group_means(rows, 'j_obs')['A'] == pytest.approx(7.0)


class TestAbsentSamplesReportNothing:
    """An undetected line keeps its reference position, so a centre built from
    it is the study value handed back as if it had been measured here — the
    same number in every absent sample, which reads as data and is not.
    """

    def test_no_centre_where_the_signal_was_not_detected(self):
        lines = _doublet(present=(False, True, True, True))
        assert signal_in_sample(lines, 0)['ppm_centre'] is None
        assert signal_in_sample(lines, 1)['ppm_centre'] is not None

    def test_no_ppm_diff_either(self):
        lines = _doublet(present=(False, True, True, True))
        assert signal_in_sample(lines, 0, db_ppm=1.47)['ppm_diff'] is None

    def test_area_is_still_reported(self):
        """Integration does not depend on finding an apex."""
        lines = _doublet(present=(False, True, True, True))
        assert signal_in_sample(lines, 0)['area'] == pytest.approx(2.0)

    def test_absent_samples_do_not_all_report_the_same_position(self):
        """The regression this guards: every absent row showed one identical
        Δppm, because it was the master centre repeated."""
        lines = _doublet(present=(False, False, True, True))
        centres = [signal_in_sample(lines, i)['ppm_centre'] for i in (0, 1)]
        assert centres == [None, None]
