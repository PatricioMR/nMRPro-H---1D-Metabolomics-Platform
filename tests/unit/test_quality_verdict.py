"""The automatic quality verdict for one signal.

Undecided (None) is the middle band, not a failure: the evidence neither
confirms nor rejects, and the analyst decides.
"""
import pytest

from processing.annotation_quality import (
    QUALITY_BAD, QUALITY_GOOD, QUALITY_UNDECIDED, quality_verdict)

PARAMS = {'dppm_gold': 0.05, 'dj_bad_hz': 0.50,
          'good_score_thresh': 0.75, 'bad_score_thresh': 0.50}


def _mp(**kw):
    base = {'type': 'd', 'ppm_diff': 0.001, 'delta_j_hz': 0.1,
            '_n_found_lines': 2, '_n_exp_lines': 2, 'composite_score': 0.9}
    base.update(kw)
    return base


def _verdict(**kw):
    return quality_verdict(_mp(**kw), params=PARAMS)[0]


class TestOrderOfEvidence:
    def test_a_complete_multiplet_is_good(self):
        assert _verdict() == QUALITY_GOOD

    def test_a_missing_line_is_bad_whatever_the_score(self):
        """Some lines absent means the coupling was never confirmed."""
        assert _verdict(_n_found_lines=1, composite_score=1.0) == QUALITY_BAD

    def test_a_complete_multiplet_outranks_a_low_score(self):
        """Δppm, ΔJ and the line count agreeing is an identification; a
        composite number cannot overrule three independent agreements."""
        assert _verdict(composite_score=0.1) == QUALITY_GOOD

    def test_a_bad_coupling_blocks_the_full_multiplet_gate(self):
        assert _verdict(delta_j_hz=0.9, composite_score=0.1) == QUALITY_BAD

    def test_a_database_without_j_still_allows_the_gate(self):
        assert _verdict(delta_j_hz=None, composite_score=0.1) == QUALITY_GOOD


class TestPpmGate:
    def test_a_high_score_at_the_wrong_shift_is_not_promoted(self):
        """A strong coupling term can carry the score while the signal sits at
        the wrong shift."""
        q, reason = quality_verdict(
            _mp(ppm_diff=0.06, _n_found_lines=1, composite_score=0.95),
            params=PARAMS)
        assert q == QUALITY_BAD          # incomplete lines settle it first

    def test_the_block_is_explained_to_the_analyst(self):
        q, reason = quality_verdict(
            _mp(ppm_diff=0.06, _n_found_lines=None, _n_exp_lines=None,
                composite_score=0.95), params=PARAMS)
        assert q is QUALITY_UNDECIDED
        assert 'Auto-good blocked' in reason


class TestScoreBands:
    def test_below_the_bad_threshold(self):
        assert _verdict(_n_found_lines=None, _n_exp_lines=None,
                        composite_score=0.2) == QUALITY_BAD

    def test_between_the_thresholds_is_undecided(self):
        assert _verdict(_n_found_lines=None, _n_exp_lines=None,
                        composite_score=0.6) is QUALITY_UNDECIDED

    def test_above_the_good_threshold(self):
        assert _verdict(_n_found_lines=None, _n_exp_lines=None,
                        composite_score=0.8) == QUALITY_GOOD


class TestSinglets:
    def test_a_singlet_is_never_decided_here(self):
        """Only the singlet rule can promote one, and only when it is alone in
        its window and uncontested."""
        for score in (0.0, 0.6, 1.0):
            assert quality_verdict(_mp(type='s', composite_score=score),
                                   params=PARAMS)[0] is QUALITY_UNDECIDED

    def test_an_unresolved_multiplet_counts_as_a_singlet(self):
        assert _verdict(type='m', composite_score=1.0) is QUALITY_UNDECIDED


class TestReasons:
    def test_an_incomplete_multiplet_says_how_many_lines(self):
        _, reason = quality_verdict(_mp(_n_found_lines=1, _n_exp_lines=3),
                                    params=PARAMS)
        assert '1 of 3' in reason

    def test_a_confirmed_multiplet_says_so(self):
        _, reason = quality_verdict(_mp(), params=PARAMS)
        assert 'all lines found' in reason
