"""When the contested signal cannot tell two analytes apart, ask their partners.

Every criterion above this one judges the contested signal alone. In
BMRB_Metabolomics_v01, 1341 of 2205 signals (60.8%) have a rival within
0.005 ppm carrying the same multiplicity — so for the majority of contests
those criteria are comparing evidence that is identical for both claimants by
construction, and the winner is decided by a tie-break on a fingerprint blend.

The reference case: L-lactic acid and N-Acetyl-L-Alanine.

    L-lactic acid        1.318 d J 7.50 3H    4.101 q J 7.00 1H
    N-Acetyl-L-Alanine   1.317 d J 7.20 3H    4.109 q J 7.20 1H    1.999 s 3H

0.001 ppm and 0.30 Hz apart at the methyl doublet they both claim. On the
reference study N-acetyl-alanine took it and lactate — one of the most
abundant metabolites in any biological sample — was reported as having lost
the competition.

What separates them is the acetyl singlet at 1.999 that only one of them has,
and STOCSY is how that becomes evidence: resonances of one molecule co-vary
across samples because they are the same molecule at the same concentration.

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic 1H
    NMR data sets. Analytical Chemistry 77(5):1282-1289.
"""
import pytest

from processing.annotation_scoring import (
    STOCSY_BAND_CAP, STOCSY_MIN_R, candidate_key, corroborating_partners,
    corroboration_band, rank_candidates)


class TestCountingPartners:
    def test_a_correlated_partner_counts(self):
        assert corroborating_partners(5, [9], {(5, 9): 0.97}) == 1

    def test_an_uncorrelated_one_does_not(self):
        assert corroborating_partners(5, [9], {(5, 9): 0.10}) == 0

    def test_the_pair_key_is_order_free(self):
        """The matrix stores the upper triangle only; reading it the other way
        round would silently find nothing."""
        assert corroborating_partners(9, [5], {(5, 9): 0.97}) == 1

    def test_a_peak_does_not_corroborate_itself(self):
        assert corroborating_partners(5, [5, 5], {(5, 5): 1.0}) == 0

    def test_partners_with_no_entry_are_not_counted(self):
        """A pair absent from the matrix was never measured — one of the peaks
        had too few samples or no variance. Absent is not refuted, but it is
        not support either."""
        assert corroborating_partners(5, [9, 11], {(5, 9): 0.97}) == 1

    def test_no_matrix_means_no_corroboration_measured(self):
        assert corroborating_partners(5, [9], {}) == 0
        assert corroborating_partners(5, [9], None) == 0

    def test_a_candidate_with_no_peak_has_no_partners(self):
        assert corroborating_partners(None, [9], {(5, 9): 0.97}) == 0

    def test_it_counts_partners_rather_than_averaging_them(self):
        """One strong partner is real evidence that a weak second must not
        dilute. A mean would let an analyte with four uncorrelated database
        signals lose to one with none."""
        strong_and_weak = corroborating_partners(
            5, [9, 11], {(5, 9): 0.99, (5, 11): 0.05})
        assert strong_and_weak == 1

    def test_the_threshold_sits_between_chance_and_a_shared_molecule(self):
        assert 0.5 < STOCSY_MIN_R < 0.95


class TestTheBand:
    def test_more_partners_sort_better(self):
        assert corroboration_band(2) < corroboration_band(1)
        assert corroboration_band(1) < corroboration_band(0)

    def test_it_is_capped(self):
        """Past the cap an analyte with many database signals would outrank one
        with few for having a bigger molecule rather than better support."""
        assert corroboration_band(STOCSY_BAND_CAP) == corroboration_band(99)

    def test_nothing_measured_sorts_the_same_as_nothing_found(self):
        """An unmeasured correlation is not evidence against a candidate, so it
        must not be given a worse band than a measured refutation."""
        assert corroboration_band(None) == corroboration_band(0)

    def test_a_nonsense_value_does_not_become_support(self):
        assert corroboration_band('many') == corroboration_band(0)


class TestWhereItSitsInTheKey:
    def _cand(self, **kw):
        c = {'multiplicity': 'd', 'delta_j_hz': 0.1, 'ppm_diff': 0.001,
             'n_found': 2, 'n_expected': 2, 'quality': 'good', 'score': 0.9,
             'cosine': 0.5}
        c.update(kw)
        return c

    def test_it_outranks_the_cosine_and_the_score(self):
        """Both are fingerprint blends. This asks a sharper question — does
        this analyte's own other resonance track this peak."""
        corroborated = self._cand(stocsy_partners=2, cosine=0.1, score=0.1)
        polished = self._cand(stocsy_partners=0, cosine=0.99, score=0.99)
        k_c = candidate_key(corroborated, dj_gold_hz=0.5)
        k_p = candidate_key(polished, dj_gold_hz=0.5)
        assert k_c < k_p

    def test_it_does_not_outrank_the_direct_evidence(self):
        """A partner can co-vary for reasons that are not identity — two
        metabolites of one pathway track each other. Measured coupling on the
        contested signal itself is the stronger claim."""
        corroborated = self._cand(stocsy_partners=3, delta_j_hz=9.0)
        direct = self._cand(stocsy_partners=0, delta_j_hz=0.05)
        assert candidate_key(direct, dj_gold_hz=0.5) < candidate_key(
            corroborated, dj_gold_hz=0.5)

    def test_it_does_not_outrank_a_manual_call(self):
        corroborated = self._cand(stocsy_partners=3)
        marked = self._cand(stocsy_partners=0, manual_quality='good')
        assert candidate_key(marked, dj_gold_hz=0.5) < candidate_key(
            corroborated, dj_gold_hz=0.5)

    def test_a_candidate_that_never_measured_it_is_not_penalised_below_a_loss(self):
        assert (candidate_key(self._cand(), dj_gold_hz=0.5).corroboration
                == candidate_key(self._cand(stocsy_partners=0),
                                 dj_gold_hz=0.5).corroboration)


class TestTheLactateContest:
    """The two entries exactly as BMRB_Metabolomics_v01 holds them."""

    LACTATE = {'multiplicity': 'd', 'ppm_diff': 0.012, 'delta_j_hz': 0.30,
               'n_found': 2, 'n_expected': 2, 'quality': 'good',
               'cosine': 0.62, 'score': 0.88, 'intensity_agreement': 0.95}
    NAA = {'multiplicity': 'd', 'ppm_diff': 0.013, 'delta_j_hz': 0.00,
           'n_found': 2, 'n_expected': 2, 'quality': 'good',
           'cosine': 0.71, 'score': 0.91, 'intensity_agreement': 0.95}

    def test_without_corroboration_the_doublet_alone_decides_it(self):
        """Not a ranking bug: 0.001 ppm is 0.6 Hz at 600 MHz, well under the
        study's own drift, and the two J values differ by 0.30 Hz. The
        criteria have nothing to separate."""
        ordered, _ = rank_candidates(
            [dict(self.LACTATE, _id='lactate'), dict(self.NAA, _id='naa')],
            dj_gold_hz=0.5)
        assert ordered[0]['_id'] == 'naa'

    def test_the_acetyl_singlet_decides_it_when_it_is_absent(self):
        """N-acetyl-alanine's 1.999 singlet does not track the doublet, so its
        claim has no support; lactate's 4.10 quartet does."""
        ordered, _ = rank_candidates(
            [dict(self.LACTATE, _id='lactate', stocsy_partners=1),
             dict(self.NAA, _id='naa', stocsy_partners=0)],
            dj_gold_hz=0.5)
        assert ordered[0]['_id'] == 'lactate'

    def test_the_acetylated_compound_keeps_it_when_its_partners_are_there(self):
        """If the acetyl singlet and the quartet both track the doublet, the
        acetylated compound really is present and should win. The criterion
        reports what the data says, not a preference for lactate."""
        ordered, _ = rank_candidates(
            [dict(self.LACTATE, _id='lactate', stocsy_partners=1),
             dict(self.NAA, _id='naa', stocsy_partners=2)],
            dj_gold_hz=0.5)
        assert ordered[0]['_id'] == 'naa'


class TestTheWiring:
    """Measured and never reaching the decision is this project's recurring
    failure — per_sample_consensus was written, wired in, and a silent no-op
    for want of one whitelist key."""

    def test_the_search_writes_it_onto_the_signal(self):
        import inspect

        from gui.stages import _annotation_search as a
        src = inspect.getsource(a)
        assert "r['stocsy_partners'] = _stocsy_partner_count(" in src

    def test_the_whitelist_copies_it_to_matched_peaks(self):
        import inspect

        from gui.widgets import annotation_results_window as w
        assert "'stocsy_partners': r.get('stocsy_partners')" in \
            inspect.getsource(w)

    def test_quantification_ranks_on_the_same_number(self):
        """The winner here and the winner the Annotation window shows must be
        decided on the same evidence."""
        import inspect

        from gui.stages import _quant_adapter as a
        assert "'stocsy_partners': mp.get('stocsy_partners')" in \
            inspect.getsource(a)

    def test_the_coherence_score_and_the_criterion_agree_on_what_a_signal_is(self):
        """Two answers to "which peaks are this analyte's" would let a peak
        corroborate one and not the other."""
        import inspect

        from gui.stages import _annotation_helpers as h
        for fn in (h._stocsy_coherence, h._stocsy_partner_count):
            assert '_metabolite_matched_ranks' in inspect.getsource(fn)


def test_a_partner_matched_twice_does_not_count_twice():
    """Greedy matching consumes each observed peak once. Without it two close
    database peaks map to the same observed peak and the metabolite appears to
    have more independent evidence than it has."""
    from gui.stages._annotation_helpers import _metabolite_matched_ranks
    ranks = _metabolite_matched_ranks(
        'm', {'m': [(1.000, 1), (1.001, 1)]},
        [{'ppm': 1.0005, 'rank': 7}], 0.05)
    assert ranks == [7]


def test_a_contest_corroboration_settles_is_not_reported_as_ambiguous():
    """An ambiguous contest has no winner at all, so a criterion left out of
    the ambiguity check would leave both analytes with nothing — including the
    two this one exists to separate."""
    base = {'multiplicity': 'd', 'delta_j_hz': 0.1, 'ppm_diff': 0.001,
            'n_found': 2, 'n_expected': 2, 'quality': 'good',
            'cosine': 0.7, 'score': 0.9}
    ordered, ambiguous = rank_candidates(
        [dict(base, _id='a', stocsy_partners=2),
         dict(base, _id='b', stocsy_partners=0)], dj_gold_hz=0.5)
    assert not ambiguous
    assert ordered[0]['_id'] == 'a'


def test_a_contest_nothing_settles_is_still_ambiguous():
    """The criterion must not make ambiguity disappear — two claimants with the
    same support are still a tie the data does not resolve."""
    base = {'multiplicity': 'd', 'delta_j_hz': 0.1, 'ppm_diff': 0.001,
            'n_found': 2, 'n_expected': 2, 'quality': 'good',
            'cosine': 0.7, 'score': 0.9}
    _, ambiguous = rank_candidates(
        [dict(base, _id='a', stocsy_partners=1),
         dict(base, _id='b', stocsy_partners=1)], dj_gold_hz=0.5)
    assert ambiguous


def test_the_ambiguity_check_covers_every_banded_criterion():
    """Named from the key's own fields, so adding a criterion cannot silently
    leave it out the way corroboration was left out."""
    import inspect
    src = inspect.getsource(rank_candidates)
    k = candidate_key({}, dj_gold_hz=0.5)
    banded = [f for f in k._fields if f not in ('cosine', 'score')]
    for f in banded:
        assert f"'{f}'" in src, f


# ── Blanks manufacture correlation ───────────────────────────────────────────

class TestTheCorrelationIsMeasuredOnSamplesOnly:
    """A blank drags every peak to nothing at once, and that shared descent
    reads as co-variation between molecules with nothing to do with each other.

    Measured on 32 samples at 20% CV with five blanks appended: two independent
    analytes read r = +0.74 to +0.80, against |r| < 0.25 over the samples
    alone. At a 0.80 threshold that makes almost everything look corroborated.
    """

    def _peaks(self, seed=1):
        import numpy as np
        from processing.stocsy import compute_peak_correlation_matrix
        rng = np.random.default_rng(seed)
        a = np.concatenate([rng.normal(0, 2, 5), rng.normal(100, 20, 32)])
        b = np.concatenate([rng.normal(0, 2, 5), rng.normal(80, 16, 32)])
        return ([{'rank': 1, 'area_per_sample': a},
                 {'rank': 2, 'area_per_sample': b}],
                compute_peak_correlation_matrix)

    def test_blanks_make_independent_analytes_look_correlated(self):
        peaks, corr = self._peaks()
        r = corr(peaks)[(1, 2)]
        assert r > 0.6, r

    def test_excluding_them_gives_back_the_real_correlation(self):
        peaks, corr = self._peaks()
        r = corr(peaks, included=list(range(5, 37)))[(1, 2)]
        assert abs(r) < 0.4, r

    def test_the_inflated_value_would_have_crossed_the_threshold(self):
        """Which is the whole point: the criterion would have counted every
        pair as corroborating and separated nothing."""
        peaks, corr = self._peaks(seed=1)
        inflated = corr(peaks)[(1, 2)]
        honest = corr(peaks, included=list(range(5, 37)))[(1, 2)]
        assert honest < STOCSY_MIN_R
        assert inflated > honest + 0.4

    def test_a_peak_with_too_few_included_samples_is_dropped(self):
        peaks, corr = self._peaks()
        assert corr(peaks, included=[0, 1]) == {}

    def test_no_selection_still_uses_every_column(self):
        peaks, corr = self._peaks()
        assert (1, 2) in corr(peaks)


def test_detection_correlates_over_the_included_samples():
    """Same selection rule detection and quantification use, so a study's
    engines cannot disagree about which samples it contains."""
    import inspect

    from gui.stages import peak_detection_stage as pds
    src = inspect.getsource(pds.PeakDetectionStage.detect_peaks)
    assert 'included_sample_indices(self.app.samples_info' in src
    assert 'included=_incl_corr' in src


class TestTheMatrixIsComputedInOneGo:
    """The pair count is quadratic: a 1047-peak study is 547581 pairs, and one
    np.corrcoef call per pair took ~25 s with the detection overlay frozen on
    it. Vectorised it is 0.53 s.
    """

    def _pairwise_reference(self, peaks):
        """What the loop used to compute, kept here as the thing to match."""
        import numpy as np
        out = {}
        elig = [(p['rank'], np.asarray(p['area_per_sample'], float))
                for p in peaks]
        for i, (ri, ai) in enumerate(elig):
            if np.std(ai) == 0.0:
                continue
            for rj, aj in elig[i + 1:]:
                if np.std(aj) == 0.0:
                    continue
                r = float(np.corrcoef(ai, aj)[0, 1])
                if np.isfinite(r):
                    out[(min(ri, rj), max(ri, rj))] = r
        return out

    def _peaks(self, n=40, seed=3):
        import numpy as np
        rng = np.random.default_rng(seed)
        return [{'rank': i, 'area_per_sample': rng.normal(100, 20, 37)}
                for i in range(1, n + 1)]

    def test_it_gives_the_same_answer_as_the_pairwise_loop(self):
        import numpy as np
        from processing.stocsy import compute_peak_correlation_matrix
        peaks = self._peaks()
        got = compute_peak_correlation_matrix(peaks)
        want = self._pairwise_reference(peaks)
        assert set(got) == set(want)
        for k in want:
            assert got[k] == pytest.approx(want[k], abs=1e-12)

    def test_a_flat_peak_is_still_dropped(self):
        """No variance means no correlation, and numpy would return NaN."""
        import numpy as np
        from processing.stocsy import compute_peak_correlation_matrix
        peaks = self._peaks(n=3)
        peaks[1]['area_per_sample'] = np.full(37, 5.0)
        got = compute_peak_correlation_matrix(peaks)
        assert all(2 not in pair for pair in got)
        assert (1, 3) in got

    def test_fewer_than_two_usable_peaks_is_an_empty_matrix(self):
        from processing.stocsy import compute_peak_correlation_matrix
        assert compute_peak_correlation_matrix(self._peaks(n=1)) == {}
        assert compute_peak_correlation_matrix([]) == {}

    def test_the_sample_selection_still_applies(self):
        """Vectorising must not quietly bring the blanks back in."""
        import numpy as np
        from processing.stocsy import compute_peak_correlation_matrix
        peaks = self._peaks(n=2)
        for p in peaks:
            p['area_per_sample'] = np.concatenate(
                [np.zeros(5), p['area_per_sample'][5:]])
        blanks_in = compute_peak_correlation_matrix(peaks)[(1, 2)]
        blanks_out = compute_peak_correlation_matrix(
            peaks, included=list(range(5, 37)))[(1, 2)]
        assert blanks_in != pytest.approx(blanks_out)
