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


class TestPerSampleConsensus:
    """What the samples measured replaces what the mean spectrum did. The mean
    spectrum is a pointwise maximum across samples — a trace none of them is.
    """

    def _con(self, **kw):
        base = {'ppm_centre': 1.470, 'ppm_diff': 0.002, 'ppm_n': 5,
                'ppm_sd': 0.001, 'observed_j_hz': 7.05, 'delta_j_hz': 0.05,
                'j_n': 5, 'j_sd': 0.12}
        base.update(kw)
        return base

    def test_the_consensus_replaces_the_master_reading(self):
        from processing.annotation_scoring import apply_per_sample_consensus
        mp = _mp(delta_j_hz=0.9, ppm_diff=0.04,
                 per_sample_consensus=self._con())
        out = apply_per_sample_consensus(mp, params=PARAMS)
        assert mp['delta_j_hz'] == pytest.approx(0.05)
        assert mp['ppm_diff'] == pytest.approx(0.002)
        assert mp['observed_j_hz'] == pytest.approx(7.05)
        assert out is not None and out[1] == 1.0

    def test_it_says_where_the_numbers_came_from(self):
        """A transformation the analyst cannot trace is one they cannot check."""
        from processing.annotation_scoring import apply_per_sample_consensus
        mp = _mp(per_sample_consensus=self._con())
        apply_per_sample_consensus(mp, params=PARAMS)
        assert mp['_dj_source'] == 'per_sample'
        assert mp['_ppm_source'] == 'per_sample'

    def test_no_consensus_leaves_the_stored_value_alone(self):
        from processing.annotation_scoring import apply_per_sample_consensus
        mp = _mp(delta_j_hz=0.9, ppm_diff=0.04)
        assert apply_per_sample_consensus(mp, params=PARAMS) is None
        assert mp['delta_j_hz'] == pytest.approx(0.9)
        assert mp['ppm_diff'] == pytest.approx(0.04)

    def test_a_singlet_gets_its_position_without_a_coupling(self):
        from processing.annotation_scoring import apply_per_sample_consensus
        mp = _mp(type='s', per_sample_consensus=self._con(
            observed_j_hz=None, delta_j_hz=None, j_n=0, j_sd=None))
        assert apply_per_sample_consensus(mp, params=PARAMS) is None
        assert mp['ppm_diff'] == pytest.approx(0.002)

    def test_a_disagreeing_consensus_lowers_the_j_term(self):
        from processing.annotation_scoring import apply_per_sample_consensus
        mp = _mp(per_sample_consensus=self._con(delta_j_hz=0.8))
        s_dj, conf = apply_per_sample_consensus(mp, params=PARAMS)
        assert s_dj == 0.0 and conf == 1.0

    def test_a_confirmation_can_be_withdrawn_by_the_samples(self):
        """The master lines agreed; the samples did not. The verdict follows
        the samples."""
        from processing.annotation_scoring import apply_per_sample_consensus
        mp = _mp(delta_j_hz=0.05, per_sample_consensus=self._con(
            delta_j_hz=0.9, ppm_diff=0.002))
        assert is_confirmed(mp, dj_conf=1.0, params=PARAMS) is True
        apply_per_sample_consensus(mp, params=PARAMS)
        assert is_confirmed(mp, dj_conf=1.0, params=PARAMS) is False


def test_expected_lines_report_where_they_came_from():
    """A fit anchored on the database J recovers it by construction; the caller
    has to be able to tell that apart. The flag follows how many lines were
    actually confirmed, not how long the list is — a list kept full by
    extrapolation is measured, a list of two where only one was confirmed is
    not."""
    import inspect
    from gui.stages._annotation_search import _AnnotationSearchMixin as M
    src = inspect.getsource(M._expected_line_ppms)
    assert "n_measured = sum(1 for l in lines if l.get('measured'))" in src
    assert 'n_measured < 2' in src
    assert 'return pred, from_db' in src
    assert "], True" in src, 'the synthetic fallback must flag itself'


class TestSyncKeepsTheWholeComb:
    """A line that found no peak used to be dropped, truncating the list. A
    later reader seeing fewer lines than the multiplicity calls for rebuilt the
    pattern from the database J, which then reappeared in whatever was measured
    against it — 23% of the reference study's multiplets took that path.
    """

    def _lines(self, *ppms):
        return [{'ppm': p, 'rel_intensity': 1.0} for p in ppms]

    def test_an_unfound_line_is_kept_not_dropped(self):
        from processing.annotation_scoring import sync_predicted_lines
        out = sync_predicted_lines(self._lines(1.00, 1.01, 1.02),
                                   [1.001, 1.021], tolerance=0.005)
        assert len(out) == 3

    def test_the_kept_line_sits_where_the_found_ones_say(self):
        """Two confirmed positions fix the whole comb by their index, so the
        placed line carries the sample's spacing and not the database's."""
        from processing.annotation_scoring import sync_predicted_lines
        out = sync_predicted_lines(self._lines(1.00, 1.01, 1.02),
                                   [1.000, 1.020], tolerance=0.005)
        mid = [l for l in out if not l.get('measured')]
        assert len(mid) == 1
        assert mid[0]['ppm'] == pytest.approx(1.010, abs=1e-9)

    def test_each_line_records_whether_it_was_confirmed(self):
        from processing.annotation_scoring import sync_predicted_lines
        out = sync_predicted_lines(self._lines(1.00, 1.01), [1.001],
                                   tolerance=0.005)
        assert sum(1 for l in out if l.get('measured')) == 1
        assert sum(1 for l in out if not l.get('measured')) == 1

    def test_one_confirmed_line_cannot_fix_a_spacing(self):
        """With a single anchor there is nothing to extrapolate from, so the
        original positions stand — and that is the caller's cue that the
        pattern is the database's."""
        from processing.annotation_scoring import sync_predicted_lines
        out = sync_predicted_lines(self._lines(1.00, 1.01), [1.001],
                                   tolerance=0.005)
        kept = [l for l in out if not l.get('measured')]
        assert kept[0]['ppm'] == pytest.approx(1.01)

    def test_nothing_found_leaves_the_original_alone(self):
        from processing.annotation_scoring import sync_predicted_lines
        orig = self._lines(1.00, 1.01)
        assert sync_predicted_lines(orig, [], tolerance=0.005) == orig
