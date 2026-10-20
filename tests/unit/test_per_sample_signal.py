# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Per-sample view of one annotated signal, and the consensus it adds up to.

Where a signal's lines sit in a sample comes from the multiplet fit, which
assigns one apex per line. This module reads that assignment, so the tests
build the assignment the way the fit hands it over.
"""
import numpy as np
import pytest

from processing.per_sample_signal import (
    alignment_shifts, consensus, sample_count, signal_across_samples,
    signal_in_sample)

MHZ = 600.0
J = 7.0
D = J / MHZ          # one coupling, in ppm


def _line(ppm, area=1.0, n=4):
    """One detected line, carrying only what this module still reads."""
    return {'ppm': ppm,
            'ppm_per_sample': [ppm] * n,
            'area_per_sample': [area] * n}


def _doublet(centre=1.47, n=4):
    return [_line(centre - D / 2, n=n), _line(centre + D / 2, n=n)]


def _fit(ppms, ranks, *, residual=0.0, alts=0):
    """What multiplet_line_fit hands back for one sample."""
    return dict(ppms=list(ppms), ranks=list(ranks), residual_hz=residual,
                drift_ppm=0.0, n_alternatives=alts, fitted=True,
                n_candidates=len(ppms))


class TestPresence:
    def test_every_line_assigned_is_present(self):
        f = _fit([1.47 - D / 2, 1.47 + D / 2], [0, 1])
        r = signal_in_sample(_doublet(), 0, fit=f, field_mhz=MHZ)
        assert r['present'] is True
        assert r['partial'] is False

    def test_one_leg_only_is_not_a_doublet_here(self):
        f = _fit([1.47 - D / 2], [0])
        r = signal_in_sample(_doublet(), 0, fit=f, field_mhz=MHZ)
        assert r['present'] is False

    def test_no_fit_is_no_reading(self):
        """The master positions belong to the study, not to this sample."""
        r = signal_in_sample(_doublet(), 0, db_ppm=1.47, db_j_hz=J,
                             field_mhz=MHZ)
        assert r['present'] is False
        assert r['ppm_centre'] is None and r['j_obs_hz'] is None
        assert r['ppm_diff'] is None and r['delta_j_hz'] is None

    def test_a_fit_that_failed_measures_nothing(self):
        f = _fit([1.47, 1.48], [0, 1])
        f['fitted'] = False
        r = signal_in_sample(_doublet(), 0, fit=f, field_mhz=MHZ)
        assert r['n_found'] == 0 and r['j_obs_hz'] is None

    def test_area_does_not_depend_on_the_fit(self):
        """Integration runs over the master peak bounds; no apex enters it."""
        r = signal_in_sample(_doublet(), 0, field_mhz=MHZ)
        assert r['area'] == pytest.approx(2.0)


class TestMeasurement:
    def test_adjacent_lines_span_one_coupling(self):
        f = _fit([1.47 - D / 2, 1.47 + D / 2], [0, 1])
        r = signal_in_sample(_doublet(), 0, fit=f, db_j_hz=J, field_mhz=MHZ)
        assert r['j_obs_hz'] == pytest.approx(J, abs=1e-6)
        assert r['delta_j_hz'] == pytest.approx(0.0, abs=1e-6)

    def test_outer_lines_of_a_triplet_span_two(self):
        """Lines 0 and 2 sit 2J apart; the index span is what divides."""
        lines = [_line(1.47 - D), _line(1.47), _line(1.47 + D)]
        f = _fit([1.47 - D, 1.47 + D], [0, 2])
        r = signal_in_sample(lines, 0, fit=f, db_j_hz=J, field_mhz=MHZ,
                             n_lines=3)
        assert r['j_obs_hz'] == pytest.approx(J, abs=1e-6)
        assert r['partial'] is True and r['present'] is False

    def test_the_centre_is_the_mean_of_the_assigned_lines(self):
        f = _fit([1.47 - D / 2, 1.47 + D / 2], [0, 1])
        r = signal_in_sample(_doublet(), 0, fit=f, db_ppm=1.46, field_mhz=MHZ)
        assert r['ppm_centre'] == pytest.approx(1.47)
        assert r['ppm_diff'] == pytest.approx(0.01)

    def test_a_single_assigned_line_has_no_coupling(self):
        f = _fit([1.47], [0])
        r = signal_in_sample(_doublet(), 0, fit=f, field_mhz=MHZ)
        assert r['j_obs_hz'] is None

    def test_alternatives_travel_with_the_reading(self):
        f = _fit([1.47 - D / 2, 1.47 + D / 2], [0, 1], alts=3)
        r = signal_in_sample(_doublet(), 0, fit=f, field_mhz=MHZ)
        assert r['n_alternatives'] == 3


class TestAcrossSamples:
    def _fits(self, js):
        """One fit per sample, each doublet carrying the J it was given."""
        out = {}
        for i, j_hz in enumerate(js):
            d = j_hz / MHZ
            out[i] = _fit([1.47 - d / 2, 1.47 + d / 2], [0, 1])
        return out

    def test_present_count_is_over_the_included_samples(self):
        fits = self._fits([7.0, 7.1])
        out = signal_across_samples(_doublet(), fits=fits, included=[0, 1, 2],
                                    field_mhz=MHZ)
        assert out['n_included'] == 3
        assert out['n_present'] == 2

    def test_excluded_samples_are_never_read(self):
        out = signal_across_samples(_doublet(), fits=self._fits([7.0] * 4),
                                    included=[1, 2], field_mhz=MHZ)
        assert [r['sample_idx'] for r in out['per_sample']] == [1, 2]


class TestConsensus:
    """The signal's own δ and J: the median of what the samples measured, and
    the distance to the database taken AFTER that median."""

    def _rows(self, js, ppms=None):
        ppms = ppms or [1.47] * len(js)
        return [{'sample_idx': i, 'present': True, 'ppm_centre': p,
                 'j_obs_hz': j} for i, (j, p) in enumerate(zip(js, ppms))]

    def test_the_median_j_is_reported(self):
        c = consensus(self._rows([6.9, 7.0, 7.4]), db_j_hz=J)
        assert c['observed_j_hz'] == pytest.approx(7.0)
        assert c['j_n'] == 3

    def test_noise_around_the_database_value_cancels(self):
        """Averaging |ΔJ| instead would report ~0.5 Hz of disagreement for a
        signal that agrees exactly — an absolute value cannot cancel."""
        c = consensus(self._rows([6.5, 7.0, 7.5]), db_j_hz=J)
        assert c['delta_j_hz'] == pytest.approx(0.0, abs=1e-9)
        assert float(np.mean([abs(x - J) for x in (6.5, 7.0, 7.5)])) > 0.3

    def test_one_bad_reading_does_not_drag_the_consensus(self):
        c = consensus(self._rows([7.0, 7.0, 7.0, 7.0, 25.0]), db_j_hz=J)
        assert c['observed_j_hz'] == pytest.approx(7.0)

    def test_the_scatter_is_reported_with_the_value(self):
        c = consensus(self._rows([6.9, 7.0, 7.1]), db_j_hz=J)
        assert c['j_sd'] == pytest.approx(np.std([6.9, 7.0, 7.1], ddof=1))

    def test_one_sample_has_no_scatter_but_still_a_value(self):
        c = consensus(self._rows([7.2]), db_j_hz=J)
        assert c['observed_j_hz'] == pytest.approx(7.2)
        assert c['j_sd'] is None and c['j_n'] == 1

    def test_nothing_measured_is_reported_as_nothing(self):
        c = consensus([], db_ppm=1.47, db_j_hz=J)
        assert c['observed_j_hz'] is None and c['delta_j_hz'] is None
        assert c['ppm_centre'] is None and c['j_n'] == 0

    def test_delta_ppm_follows_the_same_rule(self):
        c = consensus(self._rows([7.0] * 3, ppms=[1.46, 1.47, 1.48]),
                      db_ppm=1.47)
        assert c['ppm_centre'] == pytest.approx(1.47)
        assert c['ppm_diff'] == pytest.approx(0.0, abs=1e-9)

    def test_samples_without_a_j_still_contribute_their_position(self):
        rows = [{'sample_idx': 0, 'ppm_centre': 1.47, 'j_obs_hz': 7.0},
                {'sample_idx': 1, 'ppm_centre': 1.48, 'j_obs_hz': None}]
        c = consensus(rows, db_ppm=1.47, db_j_hz=J)
        assert c['ppm_n'] == 2 and c['j_n'] == 1


class TestAlignmentShifts:
    def _rows(self, centres):
        return [{'sample_idx': i, 'present': True, 'ppm_centre': c}
                for i, c in enumerate(centres)]

    def test_each_sample_moves_by_its_own_offset(self):
        sh = alignment_shifts(self._rows([1.46, 1.47, 1.48]), reference=1.47)
        assert sh == {0: pytest.approx(-0.01), 1: pytest.approx(0.0),
                      2: pytest.approx(0.01)}

    def test_the_group_centres_itself_by_default(self):
        sh = alignment_shifts(self._rows([1.46, 1.48]))
        assert sum(sh.values()) == pytest.approx(0.0)

    def test_a_sample_without_a_reading_is_not_shifted(self):
        rows = self._rows([1.47, 1.48])
        rows[0]['present'] = False
        assert 0 not in alignment_shifts(rows)

    def test_nothing_present_shifts_nothing(self):
        assert alignment_shifts([]) == {}


class TestSampleCount:
    def test_counts_from_the_recorded_vector(self):
        assert sample_count(_doublet(n=6)) == 6

    def test_no_vector_is_no_samples(self):
        assert sample_count([{'ppm': 1.47}]) == 0


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


class TestJFromTheDatabaseIsNotEvidence:
    """When the pattern searched for was laid out from the database J, the
    spacing found among its own lines is that J returned. Measured against the
    reference study: perturbing the database J by 4 Hz moved the fitted J by
    3.99 Hz on that path and by 0.00 Hz when the pattern came from detected
    peaks. Aggregating the first as a measurement would confirm the database
    against itself.
    """

    def _rows(self):
        return [{'present': True, 'ppm_centre': 1.470, 'j_obs_hz': 7.0},
                {'present': True, 'ppm_centre': 1.472, 'j_obs_hz': 7.2}]

    def test_the_j_is_withheld(self):
        from processing.per_sample_signal import consensus
        c = consensus(self._rows(), db_j_hz=7.0, j_from_db=True)
        assert c['observed_j_hz'] is None
        assert c['delta_j_hz'] is None
        assert c['j_n'] == 0
        assert c['j_from_db'] is True

    def test_the_positions_are_still_a_measurement(self):
        """Where the lines sit is the sample's own answer, whatever led there."""
        from processing.per_sample_signal import consensus
        c = consensus(self._rows(), db_ppm=1.47, j_from_db=True)
        assert c['ppm_centre'] == pytest.approx(1.471)
        assert c['ppm_n'] == 2

    def test_a_pattern_from_detected_peaks_keeps_its_j(self):
        from processing.per_sample_signal import consensus
        c = consensus(self._rows(), db_j_hz=7.0, j_from_db=False)
        assert c['observed_j_hz'] == pytest.approx(7.1)
        assert c['delta_j_hz'] == pytest.approx(0.1)
        assert c['j_from_db'] is False

    def test_the_flag_travels_through_the_summary(self):
        from processing.per_sample_signal import signal_across_samples
        lines = [{'ppm': 1.47, 'ppm_per_sample': [1.47],
                  'aligned_per_sample': [True], 'area_per_sample': [1.0]}]
        out = signal_across_samples(lines, j_from_db=True, db_j_hz=7.0)
        assert out['j_from_db'] is True
        assert out['observed_j_hz'] is None
