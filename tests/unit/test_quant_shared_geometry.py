# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The shared-window fit's geometry: the reference's profile for a letter,
a refinement of position and width on the mean trace, an intruder placed
where annotation expects it when detection gave it the owner's line, and
an inseparable intruder fixed at what its own clean signal predicts."""
import json

import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.quant_shared import (
    SEPARABLE_MIN, component_from_evidence, refine_geometry, resolve_window,
    resolve_shared_windows)

FIELD = 500.0
PPM = np.linspace(3.1, 3.4, 6000)
LW_HZ = 2.6
LW_PPM = LW_HZ / FIELD
J = 6.6


def _profile_json(lines, fwhm_hz=1.4):
    """A reference profile: the lines at ``fwhm_hz``, sampled at 0.1 Hz."""
    offs = np.arange(-20.0, 20.0 + 0.05, 0.1)
    h = np.zeros_like(offs)
    hw = fwhm_hz / 2.0
    for o, r in lines:
        h += r * hw ** 2 / ((offs - o) ** 2 + hw ** 2)
    return json.dumps({'start': float(offs[0]), 'step': 0.1,
                       'h': [float(v) for v in h / h.max()]})


# a second-order-leaning triplet: the reference measured 0.7 : 1 : 0.4
TRIPLET = [(-J, 0.7), (0.0, 1.0), (J, 0.4)]
TRIPLET_PROFILE = _profile_json(TRIPLET)


def _trace(centre, height, lines=TRIPLET):
    return height * lines_model(PPM, lines, centre, LW_HZ, FIELD)


def _letter_ev(centre, with_profile=True):
    ev = {'multiplicity': 't',
          'predicted_ppms': [centre - J / FIELD, centre, centre + J / FIELD]}
    if with_profile:
        ev.update({'pattern_profile': TRIPLET_PROFILE, 'profile_fwhm_hz': 1.4,
                   'pattern_field_mhz': FIELD})
    return ev


def test_a_letter_with_a_profile_is_modelled_by_it():
    """1:2:1 Lorentzians cannot make a 0.7:1:0.4 triplet; the profile can."""
    x = PPM
    truth = _trace(3.25, 1.0)
    with_p = component_from_evidence('T', _letter_ev(3.25), label='t',
                                     linewidth_ppm=LW_PPM, nh=2)
    without = component_from_evidence('T', _letter_ev(3.25, False), label='t',
                                      linewidth_ppm=LW_PPM, nh=2)
    err_p = np.abs(with_p.curve(x) - truth / truth.max()).max()
    err_l = np.abs(without.curve(x) - truth / truth.max()).max()
    assert err_p < 0.05 < err_l


def test_the_profiles_window_is_its_lines_reach_not_its_whole_range():
    c = component_from_evidence('T', _letter_ev(3.25), label='t',
                                linewidth_ppm=LW_PPM, nh=2)
    assert 3.25 - J / FIELD - 4 * LW_PPM < c.lo
    assert c.hi < 3.25 + J / FIELD + 4 * LW_PPM


def test_refinement_finds_a_shift_and_a_width_the_annotation_missed():
    """Placed a third of a linewidth off and a fifth too wide, the shape is
    slid and narrowed onto the trace."""
    truth = _trace(3.25, 100.0)
    c = component_from_evidence('T', _letter_ev(3.25 + LW_PPM / 3), label='t',
                                linewidth_ppm=LW_PPM * 1.2, nh=2)
    x = PPM
    before = np.abs(c.curve(x) * 100 - truth).max()
    refine_geometry([c], x, truth)
    after = np.abs(c.curve(x) * 100 - truth).max()
    assert after < before / 3
    assert c.shift_ppm == pytest.approx(-LW_PPM / 3, abs=LW_PPM / 6)
    assert c.width_scale < 1.0


class TestAnIntruderOnTheOwnersLine:
    """Betaine's singlet half a hertz from taurine's central line."""

    def _study(self, tau=(100.0, 200.0), bet=(30.0, 60.0)):
        cols = []
        for a, b in zip(tau, bet):
            cols.append(_trace(3.25, a)
                        + b * lines_model(PPM, [(0.0, 1.0)], 3.251, LW_HZ, FIELD))
        return np.column_stack(cols)

    def _components(self, hint, by_sample=None):
        owner = component_from_evidence('TAU', _letter_ev(3.25), label='taurine t',
                                        linewidth_ppm=LW_PPM, nh=2, owner=True)
        bet = component_from_evidence('BET', {'multiplicity': 's',
                                              'peak_ppm': 3.251},
                                      label='betaine s', linewidth_ppm=LW_PPM,
                                      nh=9, per_proton_hint=hint,
                                      per_proton_by_sample=by_sample)
        return owner, bet

    def test_it_is_inseparable_and_without_a_clean_signal_stays_out(self):
        owner, bet = self._components(None)
        out = resolve_window([owner, bet], PPM, self._study(), included=[0, 1])
        assert out.get('refused') and 'betaine' in out['refused']

    def test_with_a_clean_signal_its_share_is_fixed_and_subtracted(self):
        """The singlet's amplitude is what betaine's own measurement
        predicts; taurine gets the rest, not an arbitrary share."""
        X = self._study()
        owner, bet = self._components(hint=None)
        # the per-proton area betaine's clean singlet would give: its
        # shape's area at amplitude 30 over its 9 protons
        x = PPM[(PPM >= bet.lo) & (PPM <= bet.hi)]
        area_30 = 30.0 * float(np.trapezoid(bet.curve(x), x=x))
        # sample by sample, as its clean signal read it: 30 then 60
        owner, bet = self._components(hint=area_30 / 9.0,
                                      by_sample={0: area_30 / 9.0,
                                                 1: 2 * area_30 / 9.0})
        out = resolve_window([owner, bet], PPM, X, included=[0, 1])
        assert not out.get('refused')
        assert out['fixed'] == ['betaine s']
        assert bet.separability < SEPARABLE_MIN
        # the refinement may narrow the shape a little; the area is the hint's
        assert out['amplitudes']['BET'][0] == pytest.approx(30.0, rel=0.1)
        assert out['amplitudes']['BET'][1] == pytest.approx(60.0, rel=0.1)
        assert out['amplitudes']['TAU'][0] == pytest.approx(100.0, rel=0.05)
        assert out['amplitudes']['TAU'][1] == pytest.approx(200.0, rel=0.05)
        assert max(out['remainder'].values()) < 0.05


class _Cand:
    def __init__(self, met_id, evidence, ranks=()):
        self.met_id = met_id
        self.evidence = evidence
        self.line_ranks = frozenset(ranks)
        self.peak_rank = min(ranks) if ranks else None
        self.payload = {}


def test_an_intruder_detection_gave_the_owners_line_is_placed_where_expected():
    """Detection merged betaine's singlet into taurine's central line and
    reported one apex; annotation expects betaine 0.004 ppm up the axis.
    The component is built there, and the record says so."""
    X = np.column_stack([_trace(3.25, 100.0)
                         + 30.0 * lines_model(PPM, [(0.0, 1.0)], 3.254, LW_HZ, FIELD)])
    tau_sig = _Cand('TAU', _letter_ev(3.25), ranks=(1, 2, 3))
    bet_sig = _Cand('BET', {'multiplicity': 's', 'peak_ppm': 3.25,
                            'expected_ppm': 3.254, 'protons': 9}, ranks=(2,))
    measured = {'TAU': {'window': (3.24, 3.26), 'areas': {0: 1.0}, 'nh': 2.0,
                        'signal': tau_sig},
                'BET': {'window': (3.30, 3.31), 'areas': {0: 1.0}, 'nh': 9.0,
                        'signal': _Cand('BET', {'multiplicity': 's',
                                                'peak_ppm': 3.305}, ranks=(9,))}}
    lines = {1: 3.25 - J / FIELD, 2: 3.25, 3: 3.25 + J / FIELD, 9: 3.305}
    out = resolve_shared_windows(
        measured, [tau_sig, bet_sig], PPM, X,
        per_proton_areas={'BET': {0: 5.0}}, included=[0], linewidth_ppm=LW_PPM,
        lines_of=lambda c: [lines[r] for r in sorted(c.line_ranks)])
    d = measured['TAU'].get('deconv')
    assert d is not None
    bet = next(c for c in d['components'] if c['met_id'] == 'BET')
    assert bet['placed_at_expected'] is True
    assert bet['lines'] == pytest.approx([3.254])
