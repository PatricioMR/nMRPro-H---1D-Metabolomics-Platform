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
