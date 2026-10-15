"""The identity score: its pieces, and what it does to the rows when it
decides."""
import numpy as np
import pytest

from processing import identity_score as ids


def test_the_dilution_every_peak_shares_is_found():
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 10, (30, 1))
    dil = np.array([1.0, 1.3, 0.7, 1.1])
    M = base * dil[None, :] * np.exp(rng.normal(0, 0.02, (30, 4)))
    f = ids.pqn_dilution(M)
    assert f / f[0] == pytest.approx(dil, rel=0.03)


def test_one_molecule_covaries_after_the_dilution_is_taken_out():
    rng = np.random.default_rng(1)
    n = 12
    dil = np.exp(rng.normal(0, 0.3, n))
    own = np.exp(rng.normal(0, 0.5, n))
    other = np.exp(rng.normal(0, 0.5, n))
    area = {1: 3.0 * own * dil, 2: 1.0 * own * dil, 3: 2.0 * other * dil}
    met = {'matched_peaks': [{'peak_rank': 1}, {'peak_rank': 2}]}
    stranger = {'matched_peaks': [{'peak_rank': 1}, {'peak_rank': 3}]}
    r_own, _ = ids.analyte_covariation(met, area, dil)
    r_str, _ = ids.analyte_covariation(stranger, area, dil)
    assert r_own > 0.99 and r_str < 0.8


def test_the_score_is_a_probability():
    x = ids.features({'snr': 50, '_identity_level': 3}, None, {}, np.ones(0), {})
    assert x.size == len(ids.SCORE_COEF)
    assert 0.0 <= ids.score(x) <= 1.0


def _row(reported, rejection=None, conc=(1.0, 1.2)):
    return {'hmdb_id': 'X', 'metabolite': 'x', '_not_quantified': not reported,
            '_rejection': rejection, '_confidence': 2 if reported else 0,
            '_concs_per_sample': {i: v for i, v in enumerate(conc)},
            '_status_reasons': []}


def test_off_nothing_changes(monkeypatch):
    monkeypatch.setattr(ids, 'IDENTITY_SCORE_MIN', None)
    rows = [_row(True)]
    assert ids.apply(rows, [], []) == rows and '_identity_score' not in rows[0]


def test_on_a_low_score_withdraws_and_a_high_one_admits(monkeypatch):
    monkeypatch.setattr(ids, 'IDENTITY_SCORE_MIN', 0.3)
    monkeypatch.setattr(ids, 'score', lambda x: 0.1)
    low = ids.apply([_row(True)], [], [])[0]
    assert low['_not_quantified'] and low['_rejection'] == ids.IDENTITY_UNLIKELY
    monkeypatch.setattr(ids, 'score', lambda x: 0.9)
    high = ids.apply([_row(False, 'fit_misplaced')], [], [])[0]
    assert not high['_not_quantified'] and high['_confidence'] == 1
    assert high['conc_mm'] == pytest.approx(1.1)
    assert 'upper bound' in high['_status_reasons'][0]


def test_a_refusal_without_an_amount_is_not_admitted(monkeypatch):
    monkeypatch.setattr(ids, 'IDENTITY_SCORE_MIN', 0.3)
    monkeypatch.setattr(ids, 'score', lambda x: 0.9)
    r = ids.apply([_row(False, 'fit_misplaced', conc=())], [], [])[0]
    assert r['_not_quantified']
    lost = ids.apply([_row(False, 'lost_competition')], [], [])[0]
    assert lost['_not_quantified']


def test_when_it_decides_the_gates_before_it_open(monkeypatch):
    from processing import quant_engine as qe
    monkeypatch.setattr(ids, 'IDENTITY_SCORE_MIN', 0.3)
    assert qe._cascade_score_min({'step': 1}) == 0.0
    assert qe.mstar_compete_min_good() == 1


def test_the_analyst_word_stands_over_the_score(monkeypatch):
    monkeypatch.setattr(ids, 'IDENTITY_SCORE_MIN', 0.3)
    monkeypatch.setattr(ids, 'score', lambda x: 0.05)
    marked = ids.apply([_row(True)], [], [], protected={'X'})[0]
    assert not marked['_not_quantified'] and marked['_identity_score'] == 0.05
    accepted = dict(_row(True), _analyst_decision='accepted')
    assert not ids.apply([accepted], [], [])[0]['_not_quantified']
