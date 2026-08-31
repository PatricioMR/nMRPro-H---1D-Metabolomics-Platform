"""Fitting a multiplet against the apices ONE sample really has.

The defect this replaces: each line was located by its own apex search, so the
taller leg of a doublet won both searches and the pattern was recorded twice at
one position. The assignment here is injective, which makes that unsayable.
"""
import numpy as np
import pytest

from processing.multiplet_line_fit import (
    candidate_apices, fit_across_samples, fit_lines, fit_window, observed_j_hz)

MHZ = 600.0
J = 7.0
D = J / MHZ
GOLD, BAD = 0.20, 0.50


def _axis(lo=1.40, hi=1.55, step=0.0002):
    """A descending ppm axis, the convention every spectrum here uses."""
    return np.arange(hi, lo, -step)


def _spectrum(ppm, peaks, width=0.0015):
    """Lorentzian lines at ``peaks`` = [(ppm, height), ...]."""
    y = np.zeros_like(ppm)
    for p, h in peaks:
        y = y + h * (width ** 2) / ((ppm - p) ** 2 + width ** 2)
    return y


class TestCandidateApices:
    def test_finds_both_legs_of_a_doublet(self):
        ppm = _axis()
        y = _spectrum(ppm, [(1.47 - D / 2, 1.0), (1.47 + D / 2, 1.0)])
        got = [p for p, _ in candidate_apices(ppm, y, 1.44, 1.50)]
        assert len(got) == 2
        assert got[0] == pytest.approx(1.47 - D / 2, abs=3e-4)
        assert got[1] == pytest.approx(1.47 + D / 2, abs=3e-4)

    def test_returned_in_ascending_ppm(self):
        ppm = _axis()
        y = _spectrum(ppm, [(1.46, 1.0), (1.48, 1.0), (1.50, 1.0)])
        got = [p for p, _ in candidate_apices(ppm, y, 1.44, 1.52)]
        assert got == sorted(got)

    def test_noise_is_not_an_apex(self):
        """A sample without this analyte must give an empty list, not noise."""
        ppm = _axis()
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.01, ppm.size)
        assert candidate_apices(ppm, y, 1.44, 1.50, noise_floor=0.05) == []

    def test_a_window_outside_the_axis_finds_nothing(self):
        ppm = _axis()
        y = _spectrum(ppm, [(1.47, 1.0)])
        assert candidate_apices(ppm, y, 9.0, 9.5) == []


class TestInjectiveAssignment:
    """The whole point: one apex cannot be two lines."""

    def test_two_lines_never_share_an_apex(self):
        cands = [(1.47, 10.0), (1.47 + D, 1.0)]
        fit = fit_lines([1.47, 1.47 + D], cands, field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD)
        assert len(set(fit['ppms'])) == len(fit['ppms']) == 2

    def test_the_taller_leg_cannot_win_both(self):
        """The old per-peak search returned the same position twice here."""
        ppm = _axis()
        y = _spectrum(ppm, [(1.47 - D / 2, 10.0), (1.47 + D / 2, 1.0)])
        cands = candidate_apices(ppm, y, 1.44, 1.50)
        fit = fit_lines([1.47 - D / 2, 1.47 + D / 2], cands, field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD)
        assert fit['fitted'] is True
        assert fit['ppms'][0] != fit['ppms'][1]
        j = observed_j_hz(fit['ranks'], fit['ppms'], MHZ)
        assert j == pytest.approx(J, abs=0.3)

    def test_one_apex_cannot_serve_a_doublet(self):
        """A single line in the window is one line, never a folded pair."""
        cands = [(1.47, 10.0)]
        fit = fit_lines([1.47 - D / 2, 1.47 + D / 2], cands, field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD)
        assert fit['ppms'] == [] and fit['fitted'] is False


class TestChoosingTheBestSpacing:
    def test_the_pair_matching_the_coupling_wins(self):
        """A foreign resonance sits between the legs; the fit must not take it
        just because it is closer to the expected centre."""
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(1.47 - D / 2, 1.0), (1.4702, 5.0), (1.47 + D / 2, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD)
        assert fit['ppms'] == pytest.approx([exp[0], exp[1]])

    def test_a_drifted_multiplet_is_not_penalised_for_drifting(self):
        """Translation is free: only the gaps between lines are judged."""
        exp = [1.47 - D / 2, 1.47 + D / 2]
        shift = 0.02
        cands = [(exp[0] + shift, 1.0), (exp[1] + shift, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is True
        assert fit['residual_hz'] == pytest.approx(0.0, abs=1e-6)
        assert fit['drift_ppm'] == pytest.approx(shift, abs=1e-9)

    def test_drift_beyond_the_tolerance_is_refused(self):
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(exp[0] + 0.4, 1.0), (exp[1] + 0.4, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is False and fit['ppms'] == []

    def test_a_window_with_no_such_pattern_does_not_fit(self):
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(1.46, 1.0), (1.50, 1.0)]      # 24 Hz apart, not 7
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is False

    def test_the_worst_pair_decides_not_the_average(self):
        """One leg on a foreign resonance must not be averaged away by the
        others agreeing."""
        exp = [1.47 - D, 1.47, 1.47 + D]
        cands = [(1.47 - D, 1.0), (1.47, 1.0), (1.47 + D + 0.004, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert len(fit['ppms']) == 2       # drops the displaced line
        assert fit['fitted'] is True


class TestPartialPatterns:
    def test_more_lines_win_when_the_spacing_holds(self):
        exp = [1.47 - D, 1.47, 1.47 + D]
        cands = [(p, 1.0) for p in exp]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['ranks'] == [0, 1, 2]

    def test_outer_lines_alone_still_fit_and_give_j(self):
        exp = [1.47 - D, 1.47, 1.47 + D]
        cands = [(1.47 - D, 1.0), (1.47 + D, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['ranks'] == [0, 2]
        assert observed_j_hz(fit['ranks'], fit['ppms'], MHZ) == pytest.approx(
            J, abs=1e-6)


class TestSinglets:
    def test_the_nearest_apex_is_the_reading(self):
        fit = fit_lines([3.20], [(3.19, 1.0), (3.205, 1.0)], field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['ppms'] == pytest.approx([3.205])
        assert fit['fitted'] is True

    def test_a_singlet_has_no_coupling(self):
        fit = fit_lines([3.20], [(3.20, 1.0)], field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert observed_j_hz(fit['ranks'], fit['ppms'], MHZ) is None

    def test_rival_apices_are_counted(self):
        fit = fit_lines([3.20], [(3.199, 1.0), (3.201, 1.0), (3.30, 1.0)],
                        field_mhz=MHZ, dj_bad_hz=BAD, dj_gold_hz=GOLD,
                        max_drift_ppm=0.01)
        assert fit['n_alternatives'] == 1

    def test_nothing_within_the_tolerance_is_no_reading(self):
        fit = fit_lines([3.20], [(3.40, 1.0)], field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is False


class TestUncertainty:
    def test_a_lone_good_fit_has_no_alternatives(self):
        exp = [1.47 - D / 2, 1.47 + D / 2]
        fit = fit_lines(exp, [(exp[0], 1.0), (exp[1], 1.0)], field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['n_alternatives'] == 0

    def test_a_second_pair_at_the_same_spacing_is_reported(self):
        """Two pairs 7 Hz apart in one window: the reading stands, but it was
        not the only one the window allowed."""
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(exp[0], 1.0), (exp[1], 1.0),
                 (exp[0] + 0.01, 1.0), (exp[1] + 0.01, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['n_alternatives'] >= 1


class TestWindow:
    def test_spans_the_pattern_plus_the_drift_allowed(self):
        lo, hi = fit_window([1.46, 1.48], 0.01)
        assert (lo, hi) == pytest.approx((1.45, 1.49))

    def test_no_lines_no_window(self):
        assert fit_window([], 0.01) is None


class TestAcrossSamples:
    def test_each_sample_is_fitted_on_its_own_trace(self):
        ppm = _axis()
        a = _spectrum(ppm, [(1.47 - D / 2, 1.0), (1.47 + D / 2, 1.0)])
        b = _spectrum(ppm, [(1.48 - D / 2, 1.0), (1.48 + D / 2, 1.0)])
        fits = fit_across_samples([1.47 - D / 2, 1.47 + D / 2], ppm,
                                  np.column_stack([a, b]), field_mhz=MHZ,
                                  dj_bad_hz=BAD, dj_gold_hz=GOLD,
                                  max_drift_ppm=0.05)
        assert float(np.mean(fits[0]['ppms'])) == pytest.approx(1.47, abs=5e-4)
        assert float(np.mean(fits[1]['ppms'])) == pytest.approx(1.48, abs=5e-4)

    def test_a_sample_without_the_analyte_reports_no_fit(self):
        ppm = _axis()
        a = _spectrum(ppm, [(1.47 - D / 2, 1.0), (1.47 + D / 2, 1.0)])
        b = np.zeros_like(ppm)
        fits = fit_across_samples([1.47 - D / 2, 1.47 + D / 2], ppm,
                                  np.column_stack([a, b]), field_mhz=MHZ,
                                  dj_bad_hz=BAD, dj_gold_hz=GOLD,
                                  max_drift_ppm=0.05, noise_floor=0.01)
        assert fits[0]['fitted'] is True
        assert fits[1]['fitted'] is False

    def test_excluded_samples_are_not_fitted(self):
        ppm = _axis()
        cols = np.column_stack([_spectrum(ppm, [(1.47, 1.0)])] * 3)
        fits = fit_across_samples([1.47], ppm, cols, included=[0, 2],
                                  field_mhz=MHZ, dj_bad_hz=BAD,
                                  dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert sorted(fits) == [0, 2]
