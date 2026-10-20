# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Two scores, two questions.

signal_score  — can I trust THIS signal to quantify with?
analyte_score — is this analyte present at all?

They were one number. Conflating them penalised a perfectly good signal for
circumstances that had nothing to do with it: alanine's CH3 doublet is exactly
as trustworthy whether or not its CH quartet was found under the water region.
"""
import pytest

from processing.annotation_scoring import (
    ScoreBreakdown, analyte_score, coverage_fraction, signal_score)

PARAMS = {'w_dj': 0.57, 'w_ppm_ns': 0.10, 'w_cosine_ns': 0.14,
          'w_ppm_s': 0.27, 'w_cosine_s': 0.32,
          'bad_score_thresh': 0.50, 'good_score_thresh': 0.75,
          'w_coverage': 0.05}
GOOD = PARAMS['good_score_thresh']
BAD = PARAMS['bad_score_thresh']


def _sig(ppm=0.88, dj=1.0, cos=0.82, singlet=False):
    return signal_score(s_ppm=ppm, s_dj=dj, s_cosine=cos,
                        is_singlet=singlet, params=PARAMS)


class TestSignalScore:
    def test_good_multiplet_is_well_inside_good(self):
        assert _sig().value > GOOD

    def test_dj_dominates(self):
        # Same Δppm and shape, ΔJ collapses → must fall out of the Good band.
        assert _sig(dj=0.0).value < BAD

    def test_perfect_ppm_cannot_rescue_bad_dj(self):
        assert _sig(ppm=1.0, dj=0.0).value < _sig(ppm=0.5, dj=1.0).value

    def test_shape_moves_the_score_but_does_not_dominate(self):
        low, high = _sig(cos=0.1).value, _sig(cos=1.0).value
        assert low < high
        assert high - low < 0.57      # smaller than the ΔJ weight

    def test_unmeasured_dj_renormalises_instead_of_scoring_zero(self):
        # ΔJ excluded, not counted as 0 — an unmeasured J is not a bad J.
        assert _sig(dj=None).value > _sig(dj=0.0).value

    def test_unmeasured_shape_renormalises(self):
        assert _sig(cos=None).value > _sig(cos=0.0).value

    def test_lone_singlet_cannot_reach_good(self):
        # Even with a perfect Δppm and shape: one line is not identification.
        assert _sig(ppm=1.0, dj=None, cos=1.0, singlet=True).value <= BAD

    def test_breakdown_lists_every_term_used(self):
        b = _sig()
        assert set(b.terms) == {'Δppm', 'ΔJ', 'shape'}
        assert pytest.approx(sum(b.weights.values()), abs=1e-9) == 1.0

    def test_equation_reconstructs_the_value(self):
        b = _sig()
        assert b.equation().endswith(f'= {b.value:.3f}')
        assert 'ΔJ' in b.equation()

    def test_signal_score_ignores_the_other_signals(self):
        # There is no parameter for them — the guarantee is structural.
        assert 'coverage' not in _sig().terms


class TestAnalyteScore:
    def test_single_confirmed_signal_stays_good(self):
        """Alanine seen only through its CH3 doublet is still identified."""
        d = _sig().value
        assert analyte_score([d], coverage=1 / 3, params=PARAMS).value > GOOD

    def test_missing_signals_penalise_only_slightly(self):
        d = _sig().value
        full = analyte_score([d], coverage=1.0, params=PARAMS).value
        partial = analyte_score([d], coverage=1 / 3, params=PARAMS).value
        assert partial < full
        assert (full - partial) < 0.05        # a few percent, not a band

    def test_coverage_penalty_is_monotonic(self):
        d = _sig().value
        vals = [analyte_score([d], coverage=c, params=PARAMS).value
                for c in (0.2, 0.5, 0.8, 1.0)]
        assert vals == sorted(vals)

    def test_best_signal_dominates(self):
        strong = analyte_score([0.95, 0.30], coverage=1.0, params=PARAMS).value
        assert strong > 0.70

    def test_corroborating_signals_raise_it(self):
        one = analyte_score([0.90], coverage=1.0, params=PARAMS).value
        two = analyte_score([0.90, 0.90], coverage=1.0, params=PARAMS).value
        assert two >= one

    def test_a_weak_second_signal_never_drags_it_down(self):
        """A weak resonance means overlap, not a wrong assignment.

        Averaging the signals dropped L-lactic acid out of the Good band in the
        reference dataset — a metabolite nobody would question.
        """
        clean = analyte_score([0.90], coverage=1.0, params=PARAMS).value
        mixed = analyte_score([0.90, 0.10], coverage=1.0, params=PARAMS).value
        assert mixed >= clean

    def test_the_best_signal_is_the_floor(self):
        best = 0.82
        v = analyte_score([best, 0.05, 0.11], coverage=1.0, params=PARAMS).value
        assert v >= best

    def test_no_signal_scores_zero(self):
        b = analyte_score([], coverage=0.0, params=PARAMS)
        assert b.value == 0.0 and b.note == 'no signal'

    def test_breakdown_is_auditable(self):
        b = analyte_score([0.9, 0.5], coverage=0.5, params=PARAMS)
        assert set(b.terms) == {'best', 'others', 'coverage'}
        assert 'coverage ×' in b.note
        assert f'= {b.value:.3f}' in b.equation()
        assert b.equation().endswith(f'[{b.note}]')


class TestScoreBreakdown:
    def test_equation_without_terms_shows_the_value(self):
        assert ScoreBreakdown(value=0.42).equation() == '0.420 = 0.420'

    def test_note_is_appended(self):
        b = ScoreBreakdown(value=0.4, terms={'a': 1.0}, weights={'a': 0.4},
                           note='capped')
        assert b.equation().endswith('[capped]')


class TestConfirmationFloor:
    """Coverage must not undo an identification the confirmation rule made."""

    def test_floor_holds_against_the_coverage_penalty(self):
        b = analyte_score([GOOD], coverage=0.2, params=PARAMS, floor=GOOD)
        assert b.value == pytest.approx(GOOD)
        assert 'confirmed signal' in b.note

    def test_floor_does_not_inflate_a_higher_score(self):
        b = analyte_score([0.95], coverage=1.0, params=PARAMS, floor=GOOD)
        assert b.value > GOOD

    def test_no_floor_leaves_the_penalty_in_place(self):
        assert analyte_score([GOOD], coverage=0.2, params=PARAMS).value < GOOD


class TestOneScoreEverywhere:
    """The search, the rescue path and the results window share one formula.

    They used to hold four: a 45%-Δppm blend in the search (then multiplied by a
    database provenance factor), 0.4·Δppm + 0.6·multiplet_score in rescue, a
    ΔJ-led blend in the results window, and score·0.6 + coverage·0.4 on tree
    collapse. The same signal read 0.317 in one and 0.972 in another.
    """

    def test_same_inputs_give_the_same_number(self):
        a = signal_score(s_ppm=0.88, s_dj=1.0, s_cosine=0.82,
                         is_singlet=False, params=PARAMS)
        b = signal_score(s_ppm=0.88, s_dj=1.0, s_cosine=0.82,
                         is_singlet=False, params=PARAMS)
        assert a.value == b.value

    def test_better_dj_is_the_only_reason_two_passes_differ(self):
        """The search has a preliminary ΔJ; the window measures it properly.

        The difference between the two numbers must come from the input, not
        from the formula.
        """
        from processing.annotation_scoring import dj_score
        rough = signal_score(
            s_ppm=0.88, s_cosine=0.82, is_singlet=False, params=PARAMS,
            s_dj=dj_score(0.35, PARAMS.get('dj_gold_hz', 0.20),
                          PARAMS.get('dj_bad_hz', 0.50), 0.0))
        measured = signal_score(
            s_ppm=0.88, s_cosine=0.82, is_singlet=False, params=PARAMS,
            s_dj=dj_score(0.09, PARAMS.get('dj_gold_hz', 0.20),
                          PARAMS.get('dj_bad_hz', 0.50), 0.0))
        assert measured.value > rough.value
        assert rough.terms['Δppm'] == measured.terms['Δppm']
        assert rough.terms['shape'] == measured.terms['shape']
        assert rough.weights == measured.weights      # same formula

    def test_rescue_shape_only_renormalises(self):
        """Rescue never measures J — the ΔJ term is excluded, not scored zero."""
        no_j = signal_score(s_ppm=0.9, s_dj=None, s_cosine=0.8,
                            is_singlet=False, params=PARAMS)
        assert 'ΔJ' not in no_j.terms
        assert pytest.approx(sum(no_j.weights.values()), abs=1e-9) == 1.0
        assert no_j.value > signal_score(s_ppm=0.9, s_dj=0.0, s_cosine=0.8,
                                         is_singlet=False, params=PARAMS).value


class TestScoringDefaults:
    def test_defaults_cover_every_weight_signal_score_reads(self):
        from processing.annotation_scoring import SCORING_DEFAULTS
        for k in ('w_dj', 'w_ppm_ns', 'w_cosine_ns', 'w_ppm_s', 'w_cosine_s',
                  'bad_score_thresh', 'w_coverage', 'w_corroboration',
                  'dj_gold_hz', 'dj_bad_hz', 'dj_neutral'):
            assert k in SCORING_DEFAULTS, k

    def test_search_can_score_without_a_window(self):
        from processing.annotation_scoring import SCORING_DEFAULTS
        b = signal_score(s_ppm=0.9, s_dj=1.0, s_cosine=0.8,
                         is_singlet=False, params=SCORING_DEFAULTS)
        assert 0.0 <= b.value <= 1.0


class TestCoverageFraction:
    """How much of the analyte's fingerprint was actually found."""

    @staticmethod
    def _met(db, matched):
        return {'db_peaks': db, 'matched_peaks': matched}

    def test_all_signals_found(self):
        met = self._met([{'ppm_center': 1.0, 'type': 'd'},
                         {'ppm_center': 3.0, 'type': 's'}],
                        [{'db_ppm': 1.0, 'type': 'd', 'delta_j_hz': 0.1},
                         {'db_ppm': 3.0, 'type': 's'}])
        assert coverage_fraction(met, dj_bad_hz=0.5) == pytest.approx(1.0)

    def test_a_missing_signal_lowers_it(self):
        met = self._met([{'ppm_center': 1.0, 'type': 'd'},
                         {'ppm_center': 3.0, 'type': 's'}],
                        [{'db_ppm': 1.0, 'type': 'd', 'delta_j_hz': 0.1}])
        assert coverage_fraction(met, dj_bad_hz=0.5) == pytest.approx(0.5)

    def test_a_wrong_coupling_does_not_count_as_found(self):
        """Right place, wrong J — that is something else sitting there."""
        met = self._met([{'ppm_center': 1.0, 'type': 'd'}],
                        [{'db_ppm': 1.0, 'type': 'd', 'delta_j_hz': 0.9}])
        assert coverage_fraction(met, dj_bad_hz=0.5) == pytest.approx(0.0)

    def test_singlets_have_no_coupling_to_disagree_with(self):
        met = self._met([{'ppm_center': 3.0, 'type': 's'}],
                        [{'db_ppm': 3.0, 'type': 's'}])
        assert coverage_fraction(met, dj_bad_hz=0.5) == pytest.approx(1.0)

    def test_unmeasured_j_is_allowed_through(self):
        met = self._met([{'ppm_center': 1.0, 'type': 'd'}],
                        [{'db_ppm': 1.0, 'type': 'd', 'delta_j_hz': None}])
        assert coverage_fraction(met, dj_bad_hz=0.5) == pytest.approx(1.0)

    def test_unresolved_multiplets_count_on_neither_side(self):
        """'m' carries no J and no line pattern; its absence says nothing."""
        met = self._met([{'ppm_center': 1.0, 'type': 'd'},
                         {'ppm_center': 2.0, 'type': 'm'}],
                        [{'db_ppm': 1.0, 'type': 'd', 'delta_j_hz': 0.1}])
        assert coverage_fraction(met, dj_bad_hz=0.5) == pytest.approx(1.0)

    def test_no_db_signals_scores_zero(self):
        assert coverage_fraction({'db_peaks': [], 'matched_peaks': []},
                                 dj_bad_hz=0.5) == 0.0

    def test_faint_signals_are_no_longer_excused(self):
        """They used to be dropped from both sides, so a metabolite could look
        fully covered with most of its fingerprint missing."""
        met = self._met([{'ppm_center': 1.0, 'type': 'd'},
                         {'ppm_center': 3.0, 'type': 's', 'expected_snr': 0.01}],
                        [{'db_ppm': 1.0, 'type': 'd', 'delta_j_hz': 0.1}])
        assert coverage_fraction(met, dj_bad_hz=0.5) == pytest.approx(0.5)
