"""The analyst's accept / reject from the deconvolution review, in the engine
and in the review's own model.

A decision rides in ``_deconv_overrides[met_id]`` beside the shared-window
settings, so a project keeps it with no new field:
  {'decision': 'reject'}  out of the election, the joint fit and the cascade
  {'decision': 'accept'}  quantified over what the engine refused, one dot
"""
import numpy as np
import pytest

from processing.deconv_review import (
    KIND_CANDIDATE, KIND_CASCADE, KIND_JOINT, KIND_REJECTED, KIND_SHARED,
    clean_decision, counts, decisions_changed, describe_decision, item_window,
    kind_of, region_curves, review_items, set_decision, unexplained_share)
from processing.quant_engine import quantify
from processing.quantification import (
    AMOUNT_CONTRADICTED, ANALYST_REJECTED, CASCADE_CROWDED, Candidate)
from tests.unit.test_quant_cascade import (
    FIELD, LW, PPM, SIGMA, _cand, _met, _peak, _signal, _study)


def _run(cands, peaks, X, **kw):
    return {r['hmdb_id']: r for r in quantify(
        candidates=cands, peaks_by_rank=peaks, ppm=PPM, intensities=X,
        samples_info=[], included=[0, 1, 2], groups={'G': [0, 1, 2]},
        dj_gold_hz=0.2, noise_sigma=SIGMA, linewidth_ppm=LW,
        deconvolve=False, field_mhz=FIELD, **kw)}


def _singlet_missing_its_second_signal():
    """The 3.0 ppm singlet the database lists is not in the spectrum."""
    X = _study([(2.0, 100.0, ((0.0, 1.0),))])
    db = [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
          {'ppm_center': 3.0, 'type': 's', 'protons': 3}]
    return [_cand('A', 1, 2.0, db)], {1: _peak(1, 2.0)}, X


# ── in the engine ────────────────────────────────────────────────────────

def test_a_rejected_analyte_is_out_and_its_row_says_so():
    cands, peaks, X = _singlet_missing_its_second_signal()
    X = _study([(2.0, 100.0, ((0.0, 1.0),)), (3.0, 100.0, ((0.0, 1.0),))])
    base = _run(cands, peaks, X)
    assert not base['A']['_not_quantified']
    rows = _run(cands, peaks, X, analyst_rejected=['A'])
    r = rows['A']
    assert r['_not_quantified'] and r['_rejection'] == ANALYST_REJECTED
    assert r['_analyst_decision'] == 'rejected'
    assert 'rejected by the analyst' in r['_status_reasons'][0]
    assert len([x for x in quantify(
        candidates=cands, peaks_by_rank=peaks, ppm=PPM, intensities=X,
        samples_info=[], included=[0, 1, 2], groups={'G': [0, 1, 2]},
        dj_gold_hz=0.2, noise_sigma=SIGMA, linewidth_ppm=LW, deconvolve=False,
        analyst_rejected=['A']) if x['hmdb_id'] == 'A']) == 1, 'one row, not two'


def test_a_rejected_analyte_gives_its_peak_to_the_next():
    X = _study([(2.0, 100.0, ((0.0, 1.0),)), (3.0, 100.0, ((0.0, 1.0),))])
    db_a = [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
            {'ppm_center': 3.0, 'type': 's', 'protons': 3}]
    first = _cand('F', 1, 2.0, db_a, ppm_diff=None) if False else \
        Candidate(met_id='F', peak_rank=1, line_ranks=frozenset({1}),
                  evidence={'multiplicity': 's', 'ppm_diff': 0.0005, 'protons': 3,
                            'db_ppm': 2.0, 'study_drift_ppm': 0.0, 'n_found': 1,
                            'n_expected': 1, 'peak_ppm': 2.0,
                            'predicted_ppms': [2.0], 'score': 0.9},
                  payload={'met': {'db_peaks': db_a}, 'name': 'F'})
    second = Candidate(met_id='S', peak_rank=1, line_ranks=frozenset({1}),
                       evidence={'multiplicity': 's', 'ppm_diff': 0.002, 'protons': 3,
                                 'db_ppm': 2.0, 'study_drift_ppm': 0.0, 'n_found': 1,
                                 'n_expected': 1, 'peak_ppm': 2.0,
                                 'predicted_ppms': [2.0], 'score': 0.5},
                       payload={'met': {'db_peaks': db_a}, 'name': 'S'})
    peaks = {1: _peak(1, 2.0)}
    both = _run([first, second], peaks, X)
    assert not both['F']['_not_quantified'] and both['S']['_not_quantified']
    rej = _run([first, second], peaks, X, analyst_rejected=['F'])
    assert rej['F']['_rejection'] == ANALYST_REJECTED
    assert not rej['S']['_not_quantified'], 'the peak went to the runner-up'


def test_a_rejected_analyte_is_taken_out_of_the_cascade_too():
    missed = _met('missed', 'Missed', [_signal(1.0, 's', 3)])
    X = _study([(1.0, 200.0, ((0.0, 1.0),))])
    cands = [_cand('seen', 2, 3.0, [{'ppm_center': 3.0, 'type': 's', 'protons': 1}],
                   protons=1)]
    kw = dict(cascade=True, metabolites=[missed])
    with_it = _run([], {}, X, **kw)
    assert with_it['missed']['_by_cascade']
    without = _run([], {}, X, analyst_rejected=['missed'], **kw)
    assert without['missed']['_rejection'] == ANALYST_REJECTED
    assert without['missed']['metabolite'] == 'Missed', 'named, not its id'


def test_an_accepted_analyte_is_quantified_over_a_contradiction_at_one_dot():
    cands, peaks, X = _singlet_missing_its_second_signal()
    refused = _run(cands, peaks, X)
    assert refused['A']['_rejection'] == AMOUNT_CONTRADICTED
    rows = _run(cands, peaks, X, analyst_accepted=['A'])
    r = rows['A']
    assert not r['_not_quantified'] and r['_confidence'] == 1
    assert r['_analyst_decision'] == 'accepted'
    assert r['_status_reasons'][0].startswith('Accepted by the analyst')


def test_an_accepted_cascade_candidate_gets_its_number(monkeypatch):
    import processing.quant_cascade as qc
    # a bar no identification clears: every cascade row is "no number"
    monkeypatch.setattr(qc, 'CASCADE_SCORE_MIN', 0.999)
    monkeypatch.setattr(qc, 'CASCADE_LATE_SCORE_MIN', 0.999)
    missed = _met('missed', 'Missed', [_signal(1.0, 's', 3)])
    X = _study([(1.0, 200.0, ((0.0, 1.0),))])
    kw = dict(cascade=True, metabolites=[missed])
    r = _run([], {}, X, **kw)['missed']
    assert r['_not_quantified'] and r['_rejection'] == CASCADE_CROWDED
    got = _run([], {}, X, analyst_accepted=['missed'], **kw)['missed']
    assert not got['_not_quantified'] and got['_analyst_decision'] == 'accepted'
    assert np.isfinite(got['conc_mm']) and got['_confidence'] >= 1
    assert got['_status_reasons'][0].startswith('Accepted by the analyst')


def test_a_decision_on_an_analyte_that_is_not_there_changes_nothing():
    cands, peaks, X = _singlet_missing_its_second_signal()
    a = _run(cands, peaks, X)
    b = _run(cands, peaks, X, analyst_accepted=['nobody'], analyst_rejected=['nobody2'])
    assert a['A']['_rejection'] == b['A']['_rejection']
    assert 'nobody2' in b and b['nobody2']['_rejection'] == ANALYST_REJECTED


def test_rejecting_wins_over_accepting():
    cands, peaks, X = _singlet_missing_its_second_signal()
    rows = _run(cands, peaks, X, analyst_accepted=['A'], analyst_rejected=['A'])
    assert rows['A']['_rejection'] == ANALYST_REJECTED


def test_the_decisions_reach_the_engine_from_the_overrides_the_project_keeps():
    import inspect
    from gui.stages._quant_adapter import quantify_project
    src = inspect.getsource(quantify_project)
    assert "'decision') == 'reject'" in src and "'decision') == 'accept'" in src


# ── in the review's model ────────────────────────────────────────────────

def _row(mid, **kw):
    base = {'hmdb_id': mid, 'metabolite': mid.upper(), '_confidence': 1,
            '_not_quantified': False, 'ppm': 1.0}
    base.update(kw)
    return base


def test_every_kind_of_deconvolution_is_recognised():
    assert kind_of(_row('a', _by_cascade=True)) == KIND_CASCADE
    assert kind_of(_row('a', _joint_fit={'group': ['a']})) == KIND_JOINT
    assert kind_of(_row('a', _deconv={'window': (1, 2)})) == KIND_SHARED
    assert kind_of(_row('a', _not_quantified=True, _rejection='fit_misplaced')) == KIND_CANDIDATE
    assert kind_of(_row('a', _not_quantified=True, _rejection='cascade_crowded')) == KIND_CANDIDATE
    assert kind_of(_row('a', _not_quantified=True, _rejection=ANALYST_REJECTED)) == KIND_REJECTED
    assert kind_of(_row('a')) is None, 'an integral is not a deconvolution'
    assert kind_of(_row('a', _not_quantified=True, _rejection='lost_competition')) is None
    assert kind_of(_row('a', _is_is=True, _by_cascade=True)) is None


def test_the_window_comes_from_the_fit_the_window_or_where_the_cascade_looked():
    assert item_window(_row('a', _fit_curve={'focus': [2.0, 1.0]})) == (1.0, 2.0)
    assert item_window(_row('a', _deconv={'window': (3.0, 2.5)})) == (2.5, 3.0)
    assert item_window(_row('a', _multiplet_window=[1.5, 1.6])) == (1.5, 1.6)
    lo, hi = item_window(_row('a', _cascade={'signals': [(1.0, 'x', 3, 5.0), (2.0, 'y', 3, 9.0)]}))
    assert (lo + hi) / 2 == pytest.approx(2.0)
    assert item_window(_row('a', ppm=0.0)) is None


def test_the_list_puts_what_needs_a_look_first():
    rows = [_row('hi', _confidence=3, _joint_fit={'group': []}, ppm=3.0,
                 _fit_curve={'focus': [3, 3]}),
            _row('lo', _confidence=1, _by_cascade=True, ppm=2.0,
                 _fit_curve={'focus': [2, 2]}),
            _row('no', _not_quantified=True, _rejection='cascade_crowded', ppm=4.0,
                 _confidence=0, _cascade={'signals': [(4.0, 'x', 3, 5.0)]})]
    assert [i['id'] for i in review_items(rows)] == ['no', 'lo', 'hi']
    assert counts(review_items(rows))[KIND_CANDIDATE] == 1


def test_decisions_are_set_cleared_and_never_alter_the_argument():
    ov = {'a': {'off': True}}
    new = set_decision(ov, 'a', 'reject')
    assert new == {'a': {'decision': 'reject', 'off': True}} and ov == {'a': {'off': True}}
    assert set_decision(new, 'a', None, off=False) == {}
    assert set_decision({}, 'b', 'accept', exclude=['z', 'y']) == {
        'b': {'decision': 'accept', 'exclude': ['y', 'z']}}
    assert clean_decision({'decision': 'maybe', 'x': 1}) == {}


def test_what_changed_since_the_last_run_is_what_gets_applied():
    applied = {'a': {'decision': 'reject'}, 'b': {'off': True}}
    pending = {'a': {'decision': 'reject'}, 'b': {}, 'c': {'decision': 'accept'}}
    assert decisions_changed(applied, pending) == ['b', 'c']
    assert decisions_changed(applied, applied) == []


def test_a_decision_is_said_in_words():
    assert describe_decision(None) == ''
    assert describe_decision({'decision': 'accept'}) == '✓ accepted'
    assert describe_decision({'decision': 'reject', 'off': True}) == '✗ rejected · window kept whole'
    assert describe_decision({'exclude': ['a', 'b']}) == '−2 from window'


def test_the_regions_curves_are_the_fit_records_of_the_rows_standing_there():
    ppm = np.linspace(2.0, 1.0, 1000)

    def rec(centre, amp, base=None):
        x = ppm[(ppm > centre - 0.1) & (ppm < centre + 0.1)]
        u = 1.0 / (1.0 + ((x - centre) / 0.01) ** 2)
        return {'ppm': x.tolist(), 'unit_by_group': {'G': u.tolist()},
                'amp_by_group': {'G': amp}, 'n_by_group': {'G': [3, 3]},
                'baseline_by_group': ({'G': [base] * x.size} if base else None),
                'focus': [centre, centre]}
    rows = [_row('a', _fit_curve=rec(1.5, 10.0, base=2.0)),
            _row('b', _fit_curve=rec(1.52, 4.0)),
            _row('far', _fit_curve=rec(1.9, 5.0)),
            _row('gone', _not_quantified=True, _fit_curve=rec(1.5, 99.0))]
    got = region_curves(rows, ppm, 1.4, 1.6, 'G')
    ids = [g[0] for g in got]
    assert ids == ['a', 'b', 'baseline']
    assert max(got[0][3]) == pytest.approx(10.0, rel=0.02)
    assert got[2][3][0] == pytest.approx(2.0)
    assert region_curves(rows, ppm, 1.4, 1.6, 'Nobody') == []


def test_what_the_fit_leaves_is_the_positive_remainder_over_the_trace():
    x = np.linspace(2.0, 1.0, 101)
    trace = np.where(np.abs(x - 1.5) < 0.1, 10.0, 0.0)
    assert unexplained_share(trace, trace, x) == pytest.approx(0.0)
    assert unexplained_share(trace, 0 * trace, x) == pytest.approx(1.0)
    assert unexplained_share(trace, 0.5 * trace, x) == pytest.approx(0.5)
    assert unexplained_share(trace, trace * 3, x) == pytest.approx(0.0), 'over-fitting is not left over'
    assert unexplained_share(0 * trace, trace, x) is None


def test_what_a_re_quantification_changed_is_said():
    from processing.deconv_review import change_summary, change_text
    def r(mid, conc, **kw):
        return {'hmdb_id': mid, 'metabolite': mid.upper(), 'conc_mm': conc,
                '_not_quantified': False, **kw}
    before = [r('a', 1.0), r('b', 2.0), r('c', 3.0), r('is', 1.0, _is_is=True)]
    after = [r('a', 1.05), r('b', 3.0), r('d', 4.0),
             r('c', 3.0, _not_quantified=True), r('is', 1.0, _is_is=True)]
    c = change_summary(before, after)
    assert c['entered'] == ['D'] and c['left'] == ['C']
    assert [m for m, _ in c['moved']] == ['B'] and c['moved'][0][1] == pytest.approx(1.5)
    assert (c['n_before'], c['n_after']) == (3, 3)
    text = change_text(c)
    assert '1 entered' in text and '1 left' in text and 'greedy' in text
    quiet = change_summary(before, before)
    assert 'nothing else changed' in change_text(quiet)
