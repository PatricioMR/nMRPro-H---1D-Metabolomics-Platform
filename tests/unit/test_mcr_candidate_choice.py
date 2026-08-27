"""MCR must judge the candidate the ranking elects.

_run_mcr_on_results picks one candidate per peak, runs the Pearson r against it
and writes mcr_confirmed — which then re-enters candidate_key as a tie-break.
It used to pick by composite_score, a Δppm-weighted blend that orders
candidates differently from the ranking, so the confirmation could describe a
compound the pipeline had not elected.
"""
import pytest

from gui.stages._annotation_search import _mcr_candidate
from processing.annotation_scoring import candidate_key, DJ_GOLD_HZ_DEFAULT


def _pick(cands):
    """The index _run_mcr_on_results elects, by the same key it uses."""
    return min(range(len(cands)),
               key=lambda i: candidate_key(_mcr_candidate(cands[i]),
                                           dj_gold_hz=DJ_GOLD_HZ_DEFAULT))


def test_mapping_reads_the_search_line_count_keys():
    """The search writes _n_found_lines/_n_exp_lines; candidate_key reads
    n_found/n_expected. A mismatch silently demotes every multiplet to the
    unverified tier."""
    c = _mcr_candidate({'type': 'd', '_n_found_lines': 2, '_n_exp_lines': 2,
                        'delta_j_hz': 0.1, 'ppm_diff': 0.001})
    assert c['n_found'] == 2
    assert c['n_expected'] == 2
    assert c['multiplicity'] == 'd'


def test_confirmed_multiplet_beats_higher_scoring_singlet():
    """A singlet with the better blended score must not be the one MCR judges:
    a confirmed multiplet outranks it on evidence tier."""
    cands = [
        {'type': 's', 'ppm_diff': 0.001, 'composite_score': 0.95},
        {'type': 'd', 'ppm_diff': 0.010, 'delta_j_hz': 0.05,
         '_n_found_lines': 2, '_n_exp_lines': 2, 'composite_score': 0.40},
    ]
    assert _pick(cands) == 1


def test_delta_j_outranks_delta_ppm():
    """Δppm already found the candidate; ΔJ is what separates them."""
    cands = [
        {'type': 'd', 'ppm_diff': 0.0005, 'delta_j_hz': 3.0,
         '_n_found_lines': 2, '_n_exp_lines': 2, 'composite_score': 0.90},
        {'type': 'd', 'ppm_diff': 0.0150, 'delta_j_hz': 0.05,
         '_n_found_lines': 2, '_n_exp_lines': 2, 'composite_score': 0.50},
    ]
    assert _pick(cands) == 1


def test_incomplete_multiplet_loses_to_complete_one():
    """A missing line means the pattern was never verified."""
    cands = [
        {'type': 't', 'ppm_diff': 0.001, 'delta_j_hz': 0.05,
         '_n_found_lines': 2, '_n_exp_lines': 3, 'composite_score': 0.99},
        {'type': 't', 'ppm_diff': 0.008, 'delta_j_hz': 0.10,
         '_n_found_lines': 3, '_n_exp_lines': 3, 'composite_score': 0.30},
    ]
    assert _pick(cands) == 1


def test_score_only_breaks_a_full_tie():
    """With every physical criterion equal, the blended score still orders."""
    base = {'type': 'd', 'ppm_diff': 0.002, 'delta_j_hz': 0.05,
            '_n_found_lines': 2, '_n_exp_lines': 2}
    cands = [dict(base, composite_score=0.30), dict(base, composite_score=0.80)]
    assert _pick(cands) == 1


def test_missing_cosine_and_quality_are_neutral():
    """Neither exists at MCR time; absent, they must not favour a candidate."""
    a = _mcr_candidate({'type': 'd', 'ppm_diff': 0.002, 'delta_j_hz': 0.05,
                        '_n_found_lines': 2, '_n_exp_lines': 2})
    assert a['cosine'] is None
    assert a['quality'] is None
    assert candidate_key(a, dj_gold_hz=DJ_GOLD_HZ_DEFAULT) == \
        candidate_key(dict(a), dj_gold_hz=DJ_GOLD_HZ_DEFAULT)
