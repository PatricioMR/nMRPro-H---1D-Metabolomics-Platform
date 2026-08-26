"""Scoring one signal: the confirmation floor, the coverage gate, the MCR ΔJ."""
import pytest

from processing.annotation_scoring import (
    apply_mcr_dj, is_confirmed, score_signal)

PARAMS = {'w_dj': 0.57, 'w_ppm_ns': 0.10, 'w_cosine_ns': 0.14,
          'w_ppm_s': 0.27, 'w_cosine_s': 0.32,
          'bad_score_thresh': 0.50, 'good_score_thresh': 0.75,
          'dppm_gold': 0.05, 'dj_gold_hz': 0.20, 'dj_bad_hz': 0.50,
          'dj_neutral': 0.0}
GOOD = PARAMS['good_score_thresh']


def _mp(**kw):
    base = {'type': 'd', 'ppm_diff': 0.001, 'delta_j_hz': 0.1,
            '_n_found_lines': 2, '_n_exp_lines': 2, '_s_cosine_local': 0.8}
    base.update(kw)
    return base


class TestConfirmation:
    def test_three_agreements_confirm(self):
        assert is_confirmed(_mp(), dj_conf=1.0, params=PARAMS)

    def test_a_missing_line_does_not(self):
        assert not is_confirmed(_mp(_n_found_lines=1), dj_conf=1.0, params=PARAMS)

    def test_a_wrong_shift_does_not(self):
        assert not is_confirmed(_mp(ppm_diff=0.06), dj_conf=1.0, params=PARAMS)

    def test_a_wrong_coupling_does_not(self):
        assert not is_confirmed(_mp(delta_j_hz=0.9), dj_conf=1.0, params=PARAMS)

    def test_an_unmeasured_coupling_does_not(self):
        assert not is_confirmed(_mp(delta_j_hz=None), dj_conf=0.0, params=PARAMS)

    def test_a_doublet_is_the_smallest_multiplet_that_can_confirm(self):
        assert not is_confirmed(_mp(_n_found_lines=1, _n_exp_lines=1),
                                dj_conf=1.0, params=PARAMS)


class TestConfirmationFloor:
    def test_a_confirmed_signal_reaches_the_good_band(self):
        """Three independent agreements are an identification; a composite
        number must not rank it below a signal with less evidence."""
        mp = _mp(_s_cosine_local=0.0)
        score_signal(mp, s_ppm=0.2, s_dj=0.3, dj_conf=1.0, coverage=1.0,
                     params=PARAMS)
        assert mp['composite_score'] >= GOOD
        assert mp['_confirmed'] is True
        assert 'floored' in mp['_score_breakdown']

    def test_a_higher_score_is_not_pulled_down_to_the_floor(self):
        mp = _mp()
        score_signal(mp, s_ppm=1.0, s_dj=1.0, dj_conf=1.0, coverage=1.0,
                     params=PARAMS)
        assert mp['composite_score'] > GOOD
        assert 'floored' not in mp['_score_breakdown']

    def test_an_unconfirmed_signal_is_not_floored(self):
        mp = _mp(delta_j_hz=0.9)
        score_signal(mp, s_ppm=0.2, s_dj=0.0, dj_conf=1.0, coverage=1.0,
                     params=PARAMS)
        assert mp['composite_score'] < GOOD
        assert '_confirmed' not in mp


class TestCoverageGate:
    def test_an_analyte_that_matched_nothing_scores_zero(self):
        """Coverage of zero means not one database signal was found — there is
        no fingerprint to have recognised."""
        mp = _mp()
        score_signal(mp, s_ppm=1.0, s_dj=1.0, dj_conf=1.0, coverage=0.0,
                     params=PARAMS)
        assert mp['composite_score'] == 0.0
        assert mp['_score_gate'] == 0.0

    def test_the_equation_explains_the_zero(self):
        """The gate used to be applied by the caller, leaving the equation
        describing a number the caller had since overwritten."""
        mp = _mp()
        score_signal(mp, s_ppm=1.0, s_dj=1.0, dj_conf=1.0, coverage=0.0,
                     params=PARAMS)
        assert '0.000' in mp['_score_breakdown']
        assert 'no database signal' in mp['_score_breakdown']

    def test_partial_coverage_does_not_gate(self):
        mp = _mp()
        score_signal(mp, s_ppm=1.0, s_dj=1.0, dj_conf=1.0, coverage=0.2,
                     params=PARAMS)
        assert mp['composite_score'] > 0


class TestMcrOverride:
    def test_no_mcr_result_leaves_the_measurement_alone(self):
        mp = _mp()
        assert apply_mcr_dj(mp, params=PARAMS) is None
        assert mp['delta_j_hz'] == 0.1

    def test_mcr_replaces_the_coupling_measured_on_the_merged_spectrum(self):
        """Inside an overlap the snapped lines belong to more than one analyte;
        MCR measures J on the isolated component."""
        mp = _mp(delta_j_hz=0.9, mcr_delta_j_hz=0.05)
        s_dj, conf = apply_mcr_dj(mp, params=PARAMS)
        assert mp['delta_j_hz'] == 0.05
        assert mp['_dj_source'] == 'mcr'
        assert s_dj == 1.0 and conf == 1.0

    def test_position_is_untouched(self):
        """Deconvolution resolves shape and coupling, not chemical shift."""
        mp = _mp(ppm_diff=0.02, peak_ppm=3.5, mcr_delta_j_hz=0.05)
        apply_mcr_dj(mp, params=PARAMS)
        assert mp['ppm_diff'] == 0.02 and mp['peak_ppm'] == 3.5


class TestStoredTerms:
    def test_every_term_is_kept_for_display(self):
        mp = _mp()
        score_signal(mp, s_ppm=0.9, s_dj=0.8, dj_conf=1.0, coverage=0.5,
                     params=PARAMS)
        assert mp['_s_ppm'] == 0.9 and mp['_s_dj'] == 0.8
        assert mp['_s_found'] == 0.5
        assert mp['_score_breakdown']
