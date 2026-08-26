"""Lexicographic candidate ranking shared by annotation and quantification.

The cases below are the three contest regimes measured in a real 37-sample
mouse study (128 of 236 annotated peaks were contested): multiplet vs multiplet,
multiplet vs singlet, and singlet vs singlet.
"""
import pytest

from processing.annotation_scoring import (
    COSINE_TIE, PPM_BAND_PPM, TIER_CONFIRMED_MULTIPLET,
    TIER_UNCONFIRMED_MULTIPLET, TIER_SINGLET,
    candidate_key, candidate_tier, dj_band, ppm_band, rank_candidates)

GOLD = 0.20     # dj_gold_hz from _params


def _c(name, **kw):
    base = {'name': name, 'type': 'd', 'delta_j_hz': 0.0, 'ppm_diff': 0.0,
            'n_found': 2, 'n_expected': 2, 'cosine': 0.5}
    base.update(kw)
    return base


def _order(cands, band_ppm=PPM_BAND_PPM):
    ordered, amb = rank_candidates(cands, dj_gold_hz=GOLD, band_ppm=band_ppm)
    return [c['name'] for c in ordered], amb


class TestTier:
    def test_complete_multiplet_with_j_is_confirmed(self):
        assert candidate_tier('d', 0.1, 2, 2) == TIER_CONFIRMED_MULTIPLET

    def test_missing_line_is_unconfirmed(self):
        assert candidate_tier('t', 0.1, 2, 3) == TIER_UNCONFIRMED_MULTIPLET

    def test_unmeasured_j_is_unconfirmed(self):
        assert candidate_tier('d', None, 2, 2) == TIER_UNCONFIRMED_MULTIPLET

    def test_singlet_is_its_own_tier(self):
        assert candidate_tier('s', None) == TIER_SINGLET

    def test_confirmed_multiplet_beats_singlet(self):
        names, _ = _order([_c('sing', type='s', delta_j_hz=None, ppm_diff=0.0),
                           _c('mult', delta_j_hz=0.4, ppm_diff=0.018)])
        assert names == ['mult', 'sing']


class TestDjBand:
    def test_sub_gold_deltas_share_band_zero(self):
        # 0.046 and 0.096 Hz — both far below the 0.28-0.78 Hz between-sample
        # SD of a measured J. Ordering them would be ordering noise.
        assert dj_band(0.046, GOLD) == dj_band(0.096, GOLD) == 0

    def test_band_separates_above_gold(self):
        assert dj_band(0.15, GOLD) < dj_band(0.45, GOLD)

    def test_unmeasured_sorts_last(self):
        assert dj_band(None, GOLD) > dj_band(10.0, GOLD)

    def test_dj_decides_when_bands_differ(self):
        names, amb = _order([_c('far', delta_j_hz=0.45, ppm_diff=0.0),
                             _c('near', delta_j_hz=0.05, ppm_diff=0.019)])
        assert names == ['near', 'far']
        assert amb is False


class TestPpmBand:
    def test_differences_below_the_drift_share_a_band(self):
        # 0.001 vs 0.016 ppm = 9 Hz apart at 600 MHz; measured peak drift is
        # 0.030 ppm (18 Hz), so this difference is not evidence.
        assert ppm_band(0.001) == ppm_band(0.016) == 0

    def test_band_separates_a_real_difference(self):
        assert ppm_band(0.001) < ppm_band(0.043)

    def test_ppm_decides_only_after_dj_ties(self):
        names, _ = _order([_c('a', delta_j_hz=0.05, ppm_diff=0.045),
                           _c('b', delta_j_hz=0.05, ppm_diff=0.001)])
        assert names == ['b', 'a']

    def test_ppm_never_overrides_dj(self):
        # Perfect Δppm must not rescue a candidate two ΔJ bands behind —
        # this is the inversion the old weighted blend produced.
        names, _ = _order([_c('good_ppm_bad_j', delta_j_hz=0.45, ppm_diff=0.0),
                           _c('good_j_bad_ppm', delta_j_hz=0.05, ppm_diff=0.045)])
        assert names == ['good_j_bad_ppm', 'good_ppm_bad_j']


class TestCosineTieBreak:
    def test_cosine_breaks_a_full_tie(self):
        names, amb = _order([_c('low', cosine=0.50), _c('high', cosine=0.73)])
        assert names == ['high', 'low']
        assert amb is False

    def test_indistinguishable_cosine_is_ambiguous(self):
        _, amb = _order([_c('a', cosine=0.50),
                         _c('b', cosine=0.50 + COSINE_TIE / 2)])
        assert amb is True

    def test_missing_cosine_defaults_to_zero(self):
        names, _ = _order([_c('none', cosine=None), _c('some', cosine=0.4)])
        assert names == ['some', 'none']

    def test_ambiguity_needs_every_earlier_key_equal(self):
        # Same cosine but different ΔJ band → decided, not ambiguous.
        _, amb = _order([_c('a', delta_j_hz=0.05, cosine=0.5),
                         _c('b', delta_j_hz=0.45, cosine=0.5)])
        assert amb is False


class TestManualOverride:
    def test_manual_good_outranks_every_computed_criterion(self):
        names, _ = _order([
            _c('auto_best', delta_j_hz=0.0, ppm_diff=0.0, cosine=0.99),
            _c('user_pick', delta_j_hz=0.45, ppm_diff=0.045, cosine=0.1,
               manual_quality='good')])
        assert names == ['user_pick', 'auto_best']

    def test_manual_good_on_a_singlet_still_wins(self):
        names, _ = _order([
            _c('mult', delta_j_hz=0.0, ppm_diff=0.0),
            _c('user_sing', type='s', delta_j_hz=None, ppm_diff=0.04,
               manual_quality='good')])
        assert names == ['user_sing', 'mult']

    def test_two_manual_goods_fall_back_to_the_normal_order(self):
        names, _ = _order([
            _c('worse', delta_j_hz=0.45, manual_quality='good'),
            _c('better', delta_j_hz=0.05, manual_quality='good')])
        assert names == ['better', 'worse']


class TestSingletRegime:
    """35% of contested peaks in the reference study had only singlets."""

    def test_singlets_fall_to_ppm_then_cosine(self):
        s = dict(type='s', delta_j_hz=None, n_found=None, n_expected=None)
        names, _ = _order([_c('far', ppm_diff=0.045, cosine=0.9, **s),
                           _c('near', ppm_diff=0.002, cosine=0.2, **s)])
        assert names == ['near', 'far']

    def test_singlets_tied_on_ppm_use_cosine(self):
        s = dict(type='s', delta_j_hz=None, n_found=None, n_expected=None)
        names, amb = _order([_c('a', ppm_diff=0.002, cosine=0.20, **s),
                             _c('b', ppm_diff=0.010, cosine=0.71, **s)])
        assert names == ['b', 'a']
        assert amb is False


class TestRankCandidates:
    def test_single_candidate_is_never_ambiguous(self):
        ordered, amb = rank_candidates([_c('only')], dj_gold_hz=GOLD)
        assert len(ordered) == 1 and amb is False

    def test_empty_input(self):
        ordered, amb = rank_candidates([], dj_gold_hz=GOLD)
        assert ordered == [] and amb is False

    def test_bcaa_contest_from_the_reference_study(self):
        """Real contest at ~1.0 ppm: three metabolites on peaks 58/60.

        The old weighted blend (45% Δppm) elected 2-hydroxy-3-methylbutyric
        acid, whose Δppm is 0.043 — 44x worse than isoleucine's 0.001.
        """
        names, amb = _order([
            _c('2h3mb', delta_j_hz=0.046, ppm_diff=0.043, cosine=0.499),
            _c('isoleucine', delta_j_hz=0.096, ppm_diff=0.001, cosine=0.543),
            _c('valine', delta_j_hz=0.046, ppm_diff=0.016, cosine=0.730)])
        assert names[-1] == '2h3mb'          # worst Δppm band, loses
        assert amb is False


class TestAmbiguousPeakRanks:
    """The annotation-side consumer of the same rule."""

    @staticmethod
    def _met(name, matched):
        return {'id': name, 'name': name, 'matched_peaks': matched}

    @staticmethod
    def _mp(rank, **kw):
        d = {'peak_rank': rank, 'type': 'd', 'delta_j_hz': 0.05,
             'ppm_diff': 0.001, '_s_cosine': 0.5, 'composite_score': 0.9,
             'annotation_quality': None}
        d.update(kw)
        return d

    def test_indistinguishable_candidates_are_ambiguous(self):
        from gui.stages._annotation_helpers import ambiguous_peak_ranks
        mets = [self._met('A', [self._mp(1)]), self._met('B', [self._mp(1)])]
        assert ambiguous_peak_ranks(mets) == {1}

    def test_dj_band_difference_resolves(self):
        from gui.stages._annotation_helpers import ambiguous_peak_ranks
        mets = [self._met('A', [self._mp(1, delta_j_hz=0.05)]),
                self._met('B', [self._mp(1, delta_j_hz=0.45)])]
        assert ambiguous_peak_ranks(mets) == set()

    def test_manual_good_resolves(self):
        from gui.stages._annotation_helpers import ambiguous_peak_ranks
        mets = [self._met('A', [self._mp(1)]), self._met('B', [self._mp(1)])]
        assert ambiguous_peak_ranks(
            mets, manual_quality={('B', 1): 'good'}) == set()

    def test_bad_candidates_do_not_create_ambiguity(self):
        from gui.stages._annotation_helpers import ambiguous_peak_ranks
        mets = [self._met('A', [self._mp(1)]),
                self._met('B', [self._mp(1, annotation_quality='bad')])]
        assert ambiguous_peak_ranks(mets) == set()

    def test_score_still_resolves_when_nothing_else_does(self):
        from gui.stages._annotation_helpers import ambiguous_peak_ranks
        mets = [self._met('A', [self._mp(1, composite_score=0.90)]),
                self._met('B', [self._mp(1, composite_score=0.40)])]
        assert ambiguous_peak_ranks(mets) == set()


class TestQualityOutranksCosine:
    """Regression: cosine used to sit above annotation quality.

    On a real contest at 6.449 ppm, dopamine (annotation_quality='good') lost to
    gibberellic acid (no verdict) purely on fingerprint cosine — 0.679 vs 0.296.
    A candidate the annotation pipeline did not accept must not win a similarity
    tie-break against one it did.
    """

    def test_good_beats_higher_cosine(self):
        names, _ = _order([
            _c('unverified', type='s', delta_j_hz=None, ppm_diff=0.030,
               n_found=None, n_expected=None, cosine=0.679),
            _c('good', type='s', delta_j_hz=None, ppm_diff=0.026,
               n_found=None, n_expected=None, cosine=0.296, quality='good')])
        assert names == ['good', 'unverified']

    def test_cosine_still_decides_between_two_good_ones(self):
        names, _ = _order([
            _c('lower_cos', cosine=0.615, quality='good'),
            _c('higher_cos', cosine=0.874, quality='good')])
        assert names == ['higher_cos', 'lower_cos']

    def test_quality_never_outranks_dj(self):
        names, _ = _order([_c('good_far_j', delta_j_hz=0.45, quality='good'),
                           _c('near_j', delta_j_hz=0.05)])
        assert names == ['near_j', 'good_far_j']


class TestIsCompetitor:
    """An unverified multiplet did not identify a signal — it must not compete."""

    def test_complete_multiplet_competes(self):
        from processing.annotation_scoring import is_competitor
        assert is_competitor(_c('ok', delta_j_hz=0.1, n_found=2, n_expected=2))

    def test_missing_line_does_not_compete(self):
        from processing.annotation_scoring import is_competitor
        # DB says triplet, only 2 of 3 lines found: J cannot be measured from
        # lines that are absent, so the multiplicity was never verified.
        assert not is_competitor(_c('partial', type='t', delta_j_hz=0.1,
                                    n_found=2, n_expected=3))

    def test_unmeasured_j_does_not_compete(self):
        from processing.annotation_scoring import is_competitor
        assert not is_competitor(_c('no_j', delta_j_hz=None,
                                    n_found=2, n_expected=2))

    def test_singlet_always_competes(self):
        from processing.annotation_scoring import is_competitor
        assert is_competitor(_c('s', type='s', delta_j_hz=None,
                                n_found=None, n_expected=None))

    def test_unverified_cannot_take_a_peak_from_a_verified_one(self):
        names, _ = _order([
            _c('unverified', type='t', delta_j_hz=None, ppm_diff=0.0,
               n_found=1, n_expected=3, cosine=0.99),
            _c('verified', delta_j_hz=0.4, ppm_diff=0.045, cosine=0.1)])
        assert names == ['verified']          # the unverified one is dropped

    def test_peak_with_only_unverified_candidates_has_no_winner(self):
        ordered, amb = rank_candidates(
            [_c('a', type='t', delta_j_hz=None, n_found=1, n_expected=3),
             _c('b', type='q', delta_j_hz=None, n_found=2, n_expected=4)],
            dj_gold_hz=GOLD)
        assert ordered == [] and amb is False

    def test_unverified_does_not_create_ambiguity(self):
        _, amb = _order([_c('verified', delta_j_hz=0.1, cosine=0.5),
                         _c('unverified', delta_j_hz=None, n_found=1,
                            n_expected=2, cosine=0.5)])
        assert amb is False
