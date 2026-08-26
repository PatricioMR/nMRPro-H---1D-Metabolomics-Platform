"""A singlet's whole evidence is a position.

There is no J to validate, so the question is not "does it score well" but "is
there anything else this peak could be". Two kinds of ambiguity disqualify it:
another detected signal it might be, or another metabolite claiming it.
"""
import inspect

import pytest

from gui.widgets.annotation_results_window import AnnotationResultsWindow as ARW
from processing.annotation_quality import promote_singlets




def _mp(rank, ppm, ppm_diff, mult='s'):
    return {'peak_rank': rank, 'peak_ppm': ppm, 'db_ppm': ppm,
            'ppm_diff': ppm_diff, 'type': mult}


def _run(peaks, mets, manual=None):
    promote_singlets(mets, [{'ppm': p} for p in peaks],
                     params={'dppm_gold': 0.05}, manual_marks=manual)
    return mets


class TestSingletQuality:
    def test_alone_and_uncontested_is_good_and_anchor(self):
        mp = _mp(1, 3.20, 0.002)
        _run([3.20], [{'id': 'A', 'matched_peaks': [mp]}])
        assert mp['annotation_quality'] == 'good'
        assert mp['auto_anchor'] is True

    def test_a_rival_metabolite_keeps_it_average(self):
        a, b = _mp(1, 3.20, 0.002), _mp(1, 3.20, 0.004)
        _run([3.20], [{'id': 'A', 'matched_peaks': [a]},
                      {'id': 'B', 'matched_peaks': [b]}])
        assert a['annotation_quality'] is None
        assert b['annotation_quality'] is None
        assert not a.get('auto_anchor')

    def test_another_signal_in_the_window_keeps_it_average(self):
        """No rival metabolite, but two peaks it could be."""
        mp = _mp(1, 3.20, 0.002)
        _run([3.20, 3.23], [{'id': 'A', 'matched_peaks': [mp]}])
        assert mp['annotation_quality'] is None

    def test_a_signal_outside_the_window_does_not_count(self):
        mp = _mp(1, 3.20, 0.002)
        _run([3.20, 3.40], [{'id': 'A', 'matched_peaks': [mp]}])
        assert mp['annotation_quality'] == 'good'

    def test_ppm_beyond_the_gate_is_bad(self):
        """0.05 ppm is the floor for average — beyond it no DB singlet is here."""
        mp = _mp(1, 3.20, 0.06)
        _run([3.20], [{'id': 'A', 'matched_peaks': [mp]}])
        assert mp['annotation_quality'] == 'bad'

    def test_a_rejected_rival_is_not_a_rival(self):
        a = _mp(1, 3.20, 0.002)
        b = _mp(1, 3.20, 0.004)
        b['annotation_quality'] = 'bad'
        _run([3.20], [{'id': 'A', 'matched_peaks': [a]},
                      {'id': 'B', 'matched_peaks': [b]}])
        assert a['annotation_quality'] == 'good'

    def test_multiplets_are_left_alone(self):
        mp = _mp(1, 3.20, 0.002, mult='d')
        _run([3.20], [{'id': 'A', 'matched_peaks': [mp]}])
        assert 'annotation_quality' not in mp

    def test_a_faint_singlet_is_judged_like_any_other(self):
        """Faintness no longer excuses a signal from the rule."""
        mp = _mp(1, 3.20, 0.002)
        mp['expected_snr'] = 0.1
        _run([3.20], [{'id': 'A', 'matched_peaks': [mp]}])
        assert mp['annotation_quality'] == 'good'

    def test_reasons_are_recorded_for_the_analyst(self):
        mp = _mp(1, 3.20, 0.002)
        _run([3.20, 3.23], [{'id': 'A', 'matched_peaks': [mp]}])
        assert any('contested' in r for r in mp['_status_reasons'])


class TestOneRuleForSinglets:
    def test_the_overlapping_rules_are_gone(self):
        """Three rules promoted singlets with two different Δppm thresholds —
        0.05 ppm in one and 0.00083 ppm in the other two, sixty times apart."""
        assert not hasattr(ARW, '_promote_exclusive_singlets')
        assert not hasattr(ARW, '_promote_corroborated_singlets')

    def test_the_uncontested_rule_is_gone_too(self):
        """Once singlets left it, it promoted only multiplets that
        _auto_mark_good's full-multiplet gate had already promoted: measured on
        a real dataset, 47 promotions, 47 of them duplicates, 0 unique."""
        assert not hasattr(ARW, '_promote_uncontested_candidates')

    def test_threshold_comes_from_params(self):
        src = inspect.getsource(promote_singlets)
        body = src[src.index('"""', src.index('"""') + 3):]   # skip the docstring
        assert "params['dppm_gold']" in body
        assert 'dppm_gold_hz' not in body, 'the 0.00083 ppm threshold is back'


def test_multi_peak_demotion_rule_is_gone():
    """It demoted nothing: its exemption tested what the promotion guaranteed.

    _auto_mark_good promotes a multiplet when all expected lines are found and
    ΔJ < dj_bad_hz. The demotion then exempted a peak on exactly that condition
    (nf == ne, ΔJ ok) — and nf can never exceed ne, since measure_multiplet
    snaps at most n_expected lines. Measured on a real dataset: 50 candidates
    for demotion, 50 exempted, 0 demoted.

    The corroboration principle it stood for — an analyte with several DB
    signals should have more than one position confirmed — lives on in
    analyte_score's coverage term, at the analyte level where it belongs.
    """
    assert not hasattr(ARW, '_require_multi_peak_confirmation')


class TestDppmGateFollowsTheTolerance:
    """The Δppm gate uses the analyst's 'Tolerance (ppm)' from the panel.

    They were two independent inputs sharing a default of 0.05: tightening the
    tolerance to 0.03 left every quality gate still judging at 0.05, and
    widening it to 0.10 produced candidates the gate then rejected. Tolerance is
    where the analyst states how far a signal may drift.
    """

    def test_window_takes_dppm_gold_from_the_tolerance_entry(self):
        src = inspect.getsource(ARW.__init__)
        assert 'tolerance_var' in src
        assert "self._params['dppm_gold'] = _tol_hud" in src

    def test_there_is_no_second_input_for_the_same_number(self):
        """The Parameters dialogs no longer offer their own Δppm threshold.

        Measured on a real project: 1 of 540 matched peaks exceeded the
        tolerance, by 0.002 ppm, and that one was a multiplet-centre artefact.
        A separate acceptance threshold could not reject anything the search had
        not already excluded — the same number wearing two names.
        """
        import gui.app as app_mod
        from gui.widgets import annotation_results_window as arw
        for mod in (app_mod, arw):
            src = inspect.getsource(mod)
            assert 'Gold standard Δppm' not in src, f'{mod.__name__} still offers it'
            assert 'dppm_gold_hz' not in src, f'{mod.__name__} keeps the Hz twin'

    def test_search_passes_the_tolerance_as_the_gate(self):
        import gui.stages._annotation_search as srch
        src = inspect.getsource(srch)
        assert "_sc_params['dppm_gold'] = tolerance" in src
        assert "_resc_params['dppm_gold'] = relax_tol" in src

    def test_the_singlet_rule_reads_that_parameter(self):
        src = inspect.getsource(promote_singlets)
        body = src[src.index('"""', src.index('"""') + 3):]
        assert "params['dppm_gold']" in body


class TestManualMarksArePreserved:
    def test_a_manual_mark_is_never_overwritten(self):
        mp = _mp(1, 3.20, 0.002)
        _run([3.20, 3.23], [{'id': 'A', 'matched_peaks': [mp]}],
             manual={('A', 1): 'good'})
        assert 'annotation_quality' not in mp

    def test_promoted_count_is_returned(self):
        from processing.annotation_quality import promote_singlets
        mets = [{'id': 'A', 'matched_peaks': [_mp(1, 3.20, 0.002)]}]
        assert promote_singlets(mets, [{'ppm': 3.20}],
                                params={'dppm_gold': 0.05}) == 1
