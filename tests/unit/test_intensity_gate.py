"""Lines all found, but the multiplet accounts for too little of their
height: a triplet claimed on a singlet of SNR 417 and two lines of noise."""
import pytest

from processing.annotation_quality import QUALITY_BAD, quality_verdict
from processing.annotation_scoring import (
    PATTERN_MULT, SCORING_DEFAULTS, TIER_CONFIRMED_MULTIPLET, TIER_SINGLET,
    TIER_UNCONFIRMED_MULTIPLET, candidate_tier, is_competitor,
    lines_contradict_multiplicity, lines_explained, pascal_agreement)


def test_a_singlet_with_two_noise_lines_is_not_a_triplet():
    ex = lines_explained([0, 1, 2], [5.0, 417.0, 5.0], 't')
    assert ex == pytest.approx(20.0 / 427.0, abs=0.01)
    assert lines_contradict_multiplicity('t', ex)
    assert candidate_tier('t', 0.1, 3, 3, ex) == TIER_UNCONFIRMED_MULTIPLET
    assert not is_competitor({'multiplicity': 't', 'delta_j_hz': 0.1,
                              'n_found': 3, 'n_expected': 3,
                              'lines_explained': ex})


def test_agreement_alone_could_not_tell_that_from_an_overlapped_leg():
    singlet = pascal_agreement([0, 1, 2], [5.0, 417.0, 5.0], 't')
    overlapped = pascal_agreement([0, 1], [100.0, 300.0], 'd')
    assert singlet == pytest.approx(overlapped, abs=0.01)
    assert lines_explained([0, 1], [100.0, 300.0], 'd') == pytest.approx(0.5)


def test_an_overlapped_leg_still_passes():
    ex = lines_explained([0, 1], [100.0, 300.0], 'd')
    assert not lines_contradict_multiplicity('d', ex)
    assert candidate_tier('d', 0.1, 2, 2, ex) == TIER_CONFIRMED_MULTIPLET
    ex = lines_explained([0, 1, 2], [50.0, 400.0, 50.0], 't')
    assert candidate_tier('t', 0.1, 3, 3, ex) == TIER_CONFIRMED_MULTIPLET


def test_a_clean_multiplet_explains_all_of_itself():
    assert lines_explained([0, 1], [200.0, 200.0], 'd') == pytest.approx(1.0)
    assert lines_explained([0, 1, 2], [100.0, 200.0, 100.0], 't') == pytest.approx(1.0)
    # the outer pair of a triplet, judged as the pair it is
    assert lines_explained([0, 2], [100.0, 100.0], 't') == pytest.approx(1.0)


def test_the_gate_is_for_first_order_letters_only():
    assert not lines_contradict_multiplicity('s', 0.1)
    assert not lines_contradict_multiplicity(PATTERN_MULT, 0.1)
    assert not lines_contradict_multiplicity('t', None)
    assert lines_explained([0, 1], [1.0, 1.0], 'm*') is None


def test_a_shape_with_one_apex_is_a_singlets_worth_of_evidence():
    assert candidate_tier(PATTERN_MULT, 0.0, 1, 1) == TIER_SINGLET
    assert candidate_tier(PATTERN_MULT, 0.0, 3, 3) == TIER_CONFIRMED_MULTIPLET


def test_the_verdict_names_the_contradiction():
    mp = {'type': 't', '_n_found_lines': 3, '_n_exp_lines': 3,
          'ppm_diff': 0.001, 'delta_j_hz': 0.1, 'composite_score': 0.9,
          'lines_explained': 0.05}
    q, why = quality_verdict(mp, params=SCORING_DEFAULTS)
    assert q == QUALITY_BAD and 'accounts for 5%' in why
    mp['lines_explained'] = 0.9
    q, why = quality_verdict(mp, params=SCORING_DEFAULTS)
    assert q == 'good'


class TestLinesCovary:
    def _fits(self, a, b):
        return {i: {'fitted': True, 'ranks': [0, 1], 'heights': [x, y]}
                for i, (x, y) in enumerate(zip(a, b))}

    def test_one_molecules_lines_rise_and_fall_together(self):
        from processing.annotation_scoring import lines_covary
        a = [100, 120, 90, 130, 110, 95, 125, 105]
        b = [200, 240, 180, 260, 220, 190, 250, 210]
        assert lines_covary(self._fits(a, b), noise=1.0) is True

    def test_two_molecules_at_the_right_spacing_do_not(self):
        from processing.annotation_scoring import lines_covary
        a = [100, 120, 90, 130, 110, 95, 125, 105]
        b = [210, 190, 250, 180, 230, 260, 200, 220]
        assert lines_covary(self._fits(a, b), noise=1.0) is False

    def test_one_overlapped_line_does_not_refuse_the_multiplet(self):
        from processing.annotation_scoring import lines_covary
        a = [100, 120, 90, 130, 110, 95, 125, 105]
        b = [200, 240, 180, 260, 220, 190, 250, 210]
        c = [210, 190, 250, 180, 230, 260, 200, 220]     # under a neighbour
        fits = {i: {'fitted': True, 'ranks': [0, 1, 2], 'heights': [x, y, z]}
                for i, (x, y, z) in enumerate(zip(a, b, c))}
        assert lines_covary(fits, noise=1.0) is True

    def test_faint_lines_are_not_judged(self):
        from processing.annotation_scoring import lines_covary
        a = [10, 12, 9, 13, 11, 9.5, 12.5, 10.5]
        b = [21, 19, 25, 18, 23, 26, 20, 22]
        assert lines_covary(self._fits(a, b), noise=1.0) is None   # SNR 10 < 30
        assert lines_covary(self._fits(a[:3], b[:3]), noise=0.01) is None

    def test_read_off_the_traces(self):
        import numpy as np
        from processing.annotation_scoring import lines_covary_on_trace
        from processing.pattern_shape import lines_model
        ppm = np.linspace(3.0, 3.2, 2000)
        rng = np.random.default_rng(0)
        cols = []
        for k in range(8):
            a = 100 * np.exp(rng.normal(0, 0.15))
            b = 100 * np.exp(rng.normal(0, 0.15))
            cols.append(a * lines_model(ppm, [(0.0, 1.0), (1e-3, 0.0)], 3.08, 2.6, 500.0)
                        + b * lines_model(ppm, [(0.0, 1.0), (1e-3, 0.0)], 3.10, 2.6, 500.0)
                        + rng.normal(0, 0.5, ppm.size))
        X = np.column_stack(cols)
        assert lines_covary_on_trace(ppm, X, [3.08, 3.10], half_ppm=0.003,
                                     noise=0.5) is False
        cols = []
        for k in range(8):
            a = 100 * np.exp(rng.normal(0, 0.15))
            cols.append(a * lines_model(ppm, [(0.0, 1.0), (1e-3, 0.0)], 3.08, 2.6, 500.0)
                        + a * lines_model(ppm, [(0.0, 1.0), (1e-3, 0.0)], 3.10, 2.6, 500.0)
                        + rng.normal(0, 0.5, ppm.size))
        X = np.column_stack(cols)
        assert lines_covary_on_trace(ppm, X, [3.08, 3.10], half_ppm=0.003,
                                     noise=0.5) is True

    def test_lines_that_do_not_covary_are_an_unverified_multiplet(self):
        from processing.annotation_scoring import (
            TIER_UNCONFIRMED_MULTIPLET, candidate_tier, is_competitor)
        assert candidate_tier('d', 0.1, 2, 2, 0.9, False) == TIER_UNCONFIRMED_MULTIPLET
        assert not is_competitor({'multiplicity': 'd', 'delta_j_hz': 0.1,
                                  'n_found': 2, 'n_expected': 2,
                                  'lines_covary': False})


class TestDriftCoherence:
    def test_the_spread_of_a_metabolites_own_shifts(self):
        from processing.annotation_scoring import drift_band, drift_spread
        cre = [{'peak_rank': 6, 'peak_ppm': 3.032, 'db_ppm': 3.024},
               {'peak_rank': 7, 'peak_ppm': 3.928, 'db_ppm': 3.920}]
        pcr = [{'peak_rank': 6, 'peak_ppm': 3.032, 'db_ppm': 3.029},
               {'peak_rank': 7, 'peak_ppm': 3.928, 'db_ppm': 3.936}]
        assert drift_spread(cre) == pytest.approx(0.0, abs=1e-9)
        assert drift_spread(pcr) == pytest.approx(0.0055, abs=1e-9)
        assert drift_band(drift_spread(cre)) < drift_band(drift_spread(pcr))
        assert drift_spread(cre[:1]) is None and drift_band(None) == 0

    def test_the_ranking_prefers_the_coherent_molecule(self):
        from processing.annotation_scoring import rank_candidates
        s = dict(multiplicity='s', ppm_diff=0.008, stocsy_partners=1,
                 quality=None, score=0.5)
        cre = dict(s, drift_spread_ppm=0.0001)
        pcr = dict(s, ppm_diff=0.003, drift_spread_ppm=0.0055)
        ordered, amb = rank_candidates([pcr, cre], dj_gold_hz=0.2)
        assert ordered[0] is cre and not amb


def test_two_singlets_tied_on_every_band_are_ambiguous_whatever_the_score():
    from processing.annotation_scoring import rank_candidates
    a = dict(multiplicity='s', ppm_diff=0.008, quality=None, score=0.47)
    b = dict(multiplicity='s', ppm_diff=0.006, quality=None, score=0.70)
    _o, amb = rank_candidates([a, b], dj_gold_hz=0.2)
    assert amb is True
    # a cosine that differs settles it
    b2 = dict(b, cosine=0.8)
    _o, amb = rank_candidates([a, b2], dj_gold_hz=0.2)
    assert amb is False


def test_delta_ppm_is_banded_from_the_studys_common_drift():
    from processing.annotation_scoring import (
        annotate_drift_spread, candidate_key, study_drift)
    lac = {'id': 'LAC', 'matched_peaks': [
        {'peak_rank': 1, 'peak_ppm': 1.335, 'db_ppm': 1.318, 'delta_j_hz': 0.1,
         '_n_found_lines': 2, '_n_exp_lines': 2, 'multiplicity': 'd',
         'annotation_quality': 'good'},
        # a candidate the annotation did not accept says nothing
        {'peak_rank': 9, 'peak_ppm': 4.20, 'db_ppm': 4.26, 'delta_j_hz': 0.1,
         '_n_found_lines': 2, '_n_exp_lines': 2, 'multiplicity': 'd',
         'annotation_quality': None}]}
    cre = {'id': 'CRE', 'matched_peaks': [
        {'peak_rank': 6, 'peak_ppm': 3.044, 'db_ppm': 3.024, 'multiplicity': 's'}]}
    pcr = {'id': 'PCR', 'matched_peaks': [
        {'peak_rank': 6, 'peak_ppm': 3.044, 'db_ppm': 3.029, 'multiplicity': 's'}]}
    assert study_drift([lac, cre, pcr]) == pytest.approx(0.017)
    annotate_drift_spread([lac, cre, pcr])
    a = cre['matched_peaks'][0]; b = pcr['matched_peaks'][0]
    assert a['ppm_off_drift'] == pytest.approx(0.003)
    assert b['ppm_off_drift'] == pytest.approx(0.002)
    ka = candidate_key(dict(a, ppm_diff=0.020), dj_gold_hz=0.2)
    kb = candidate_key(dict(b, ppm_diff=0.015), dj_gold_hz=0.2)
    assert ka.ppm == kb.ppm == 0
    # without a confirmed multiplet there is no common drift, and delta-ppm
    # is banded from zero as before
    annotate_drift_spread([cre, pcr])
    assert cre['matched_peaks'][0]['ppm_off_drift'] is None
