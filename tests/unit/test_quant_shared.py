"""A window two accepted analytes share is resolved by their shapes: each
gets what its shape accounts for, an intruder no more than its own clean
signal allows, and what is left is a remainder, not anybody's."""
import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.quant_shared import (
    Component, amplitude_ceiling, component_from_evidence,
    component_from_record, resolve_window, resolve_shared_windows,
    separability)

FIELD = 500.0
PPM = np.linspace(3.1, 3.4, 3000)
LW_PPM = 2.6 / FIELD
J = 6.6


def _triplet(centre, height):
    return height * lines_model(PPM, [(-J, 0.5), (0.0, 1.0), (J, 0.5)],
                                centre, 2.6, FIELD)


def _singlet(centre, height):
    return height * lines_model(PPM, [(0.0, 1.0), (1e-3, 0.0)], centre, 2.6,
                                FIELD)


def _study(tau_h, bet_h, seed=0):
    rng = np.random.default_rng(seed)
    cols = []
    for a, b in zip(tau_h, bet_h):
        cols.append(_triplet(3.25, a) + _singlet(3.254, b)
                    + rng.normal(0, 1.0, PPM.size))
    return np.column_stack(cols)


def _taurine(**kw):
    ev = {'multiplicity': 't', 'predicted_ppms': [3.25 - J / FIELD, 3.25,
                                                   3.25 + J / FIELD]}
    return component_from_evidence('TAU', ev, label='taurine 3.25 t',
                                   linewidth_ppm=LW_PPM, nh=2, **kw)


def _betaine(**kw):
    ev = {'multiplicity': 's', 'peak_ppm': 3.254}
    return component_from_evidence('BET', ev, label='betaine 3.254 s',
                                   linewidth_ppm=LW_PPM, nh=9, **kw)


def test_each_gets_what_its_shape_accounts_for():
    X = _study([100, 120, 90, 110], [60, 50, 70, 65])
    out = resolve_window([_taurine(owner=True), _betaine()], PPM, X,
                         included=[0, 1, 2, 3])
    amps = out['amplitudes']
    # amplitudes are heights of the shape's tallest point: the triplet's
    # centre line, the singlet's apex
    for s, (a, b) in enumerate(zip([100, 120, 90, 110], [60, 50, 70, 65])):
        assert amps['TAU'][s] == pytest.approx(a, rel=0.08)
        assert amps['BET'][s] == pytest.approx(b, rel=0.12)
    assert all(v < 0.1 for v in out['remainder'].values())
    assert out['areas']['TAU'][0] == pytest.approx(
        amps['TAU'][0] * out['components'][0]['shape_area'])
    # a component built again from what was recorded is the same shape
    again = component_from_record(out['components'][1])
    assert again.met_id == 'BET' and again.lines == [3.254]
    assert np.allclose(again.model(PPM), _betaine().model(PPM))


def test_an_intruder_is_held_to_its_own_clean_signal():
    X = _study([100, 100, 100], [60, 60, 60])
    # betaine's clean signal says it is worth a per-proton area far under
    # what fits here: the bound holds, the remainder says the rest
    bet = _betaine(per_proton_hint=1e-6)
    out = resolve_window([_taurine(owner=True), bet], PPM, X, included=[0, 1, 2])
    assert bet.ceiling is not None
    assert out['amplitudes']['BET'][0] <= bet.ceiling + 1e-9
    assert out['components'][1]['ceiling'] == pytest.approx(bet.ceiling)
    # and no floor: an intruder the trace does not hold takes nothing
    X0 = _study([100, 100, 100], [0, 0, 0])
    out = resolve_window([_taurine(owner=True), _betaine(per_proton_hint=50.0)],
                         PPM, X0, included=[0, 1, 2])
    assert out['amplitudes']['BET'][0] < 3.0


def test_what_the_shapes_leave_is_a_remainder():
    # a third signal nobody accepted, under the same window
    X = _study([100, 100, 100], [0, 0, 0])
    X += _singlet(3.262, 80.0)[:, None]
    out = resolve_window([_taurine(owner=True), _betaine()], PPM, X,
                         included=[0, 1, 2])
    assert all(v > 0.1 for v in out['remainder'].values())


def test_fewer_than_two_components_is_nothing_to_resolve():
    X = _study([100], [60])
    assert resolve_window([_taurine(owner=True)], PPM, X) is None
    assert resolve_window([_taurine(owner=True), None], PPM, X) is None


def test_components_from_evidence():
    assert component_from_evidence('X', {}, label='x',
                                   linewidth_ppm=LW_PPM, nh=1) is None
    c = _betaine()
    assert c.lo < 3.254 < c.hi and c.lines == [3.254]
    t = _taurine()
    assert len(t.lines) == 3 and t.lo < t.lines[0] and t.hi > t.lines[-1]
    from processing.pattern_shape import encode_profile
    offs = np.arange(-10, 10, 0.1)
    prof = encode_profile(offs, np.exp(-(offs / 3) ** 2))
    m = component_from_evidence(
        'M', {'multiplicity': 'm*',
              'shape_fit': {'origin_ppm': 3.3, 'pattern_profile': prof,
                            'profile_fwhm_hz': 1.5, 'apex_ppms': [3.3],
                            'field_mhz': FIELD}},
        label='m', linewidth_ppm=LW_PPM, nh=2)
    assert m is not None and m.lo < 3.3 < m.hi


def test_the_ceiling_is_twice_the_clean_signals_prediction():
    assert amplitude_ceiling(None, 2, 0.01) is None
    assert amplitude_ceiling(50.0, 2, 0.01) == pytest.approx(50 * 2 / 0.01 * 2)


def _singlet_at(mid, centre, nh=1, **kw):
    return component_from_evidence(
        mid, {'multiplicity': 's', 'peak_ppm': centre}, label=f'{mid} s',
        linewidth_ppm=LW_PPM, nh=nh, **kw)


def test_a_shape_the_owner_makes_is_not_separable():
    own = _singlet_at('A', 3.25, owner=True)
    same = _singlet_at('B', 3.25)
    off = _singlet_at('C', 3.25 + 8 / FIELD)
    x = PPM
    A = np.column_stack([own.model(x), same.model(x), off.model(x)])
    sep = separability(A, [True, False, False])
    assert sep[0] is None and sep[1] < 0.01 and sep[2] > 0.5
    # the fit leaves such an intruder out and says so
    X = _study([100, 100], [0, 0])
    out = resolve_window([own, same, off], PPM, X, included=[0, 1])
    assert out['inseparable'] == ['B s']
    assert [c['met_id'] for c in out['components']] == ['A', 'C']
    # two owners on one shape: refused outright
    out = resolve_window([_singlet_at('A', 3.25, owner=True),
                          _singlet_at('B', 3.25, owner=True)], PPM, X)
    assert 'refused' in out


class _Cand:
    def __init__(self, met_id, evidence, ranks=()):
        self.met_id, self.evidence = met_id, evidence
        self.line_ranks = list(ranks)
        self.peak_rank = ranks[0] if ranks else None


def test_the_winners_windows_are_resolved_with_accepted_intruders():
    X = _study([100, 120, 90], [60, 50, 70])
    tau = _Cand('TAU', {'multiplicity': 't', 'protons': 2,
                        'predicted_ppms': [3.25 - J / FIELD, 3.25,
                                           3.25 + J / FIELD]},
                ranks=(1, 2, 3))
    bet_here = _Cand('BET', {'multiplicity': 's', 'protons': 9,
                             'predicted_ppms': [3.254]}, ranks=(4,))
    bet_own = _Cand('BET', {'multiplicity': 's', 'protons': 2,
                            'predicted_ppms': [3.90]}, ranks=(9,))
    lines = {1: 3.25 - J / FIELD, 2: 3.25, 3: 3.25 + J / FIELD, 4: 3.254,
             9: 3.90}

    def fresh():
        return {
            'TAU': {'signal': tau, 'nh': 2.0, 'window': (3.22, 3.28),
                    'areas': {0: 1.0, 1: 1.0, 2: 1.0}, 'captured': 0.9},
            'BET': {'signal': bet_own, 'nh': 2.0, 'window': (3.88, 3.92),
                    'areas': {0: 1.0, 1: 1.0, 2: 1.0}, 'captured': 0.9}}
    measured = fresh()
    ppa = {'TAU': {0: 0.5, 1: 0.5, 2: 0.5}, 'BET': {0: 0.5, 1: 0.5, 2: 0.5}}
    res = resolve_shared_windows(
        measured, [tau, bet_here, bet_own], PPM, X, per_proton_areas=ppa,
        included=[0, 1, 2], linewidth_ppm=LW_PPM, names={'BET': 'betaine'},
        lines_of=lambda c: [lines[r] for r in c.line_ranks])
    assert set(res) == {'TAU'}
    rec = measured['TAU']
    assert rec['shared_with'] == ['BET'] and rec['captured'] is None
    assert rec['deconv']['components'][1]['label'].startswith('betaine')
    # the owner's areas are now its shape's, per sample, and per proton
    for s_i, h in enumerate([100, 120, 90]):
        assert rec['areas'][s_i] == pytest.approx(
            h * rec['deconv']['components'][0]['shape_area'], rel=0.1)
        assert ppa['TAU'][s_i] == pytest.approx(rec['areas'][s_i] / 2.0)
    # betaine's own window holds nobody else: untouched
    assert 'deconv' not in measured['BET']
    # an intruder detection did not place is skipped, and the row says so
    measured = fresh()
    res = resolve_shared_windows(
        measured, [tau, bet_here, bet_own], PPM, X, per_proton_areas=ppa,
        included=[0, 1, 2], linewidth_ppm=LW_PPM,
        lines_of=lambda c: ([lines[r] for r in c.line_ranks]
                            if c is not bet_here else []))
    assert res == {}
    assert 'detection placed no line' in measured['TAU']['unshared_reason']
