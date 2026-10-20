# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The remainder as evidence: a winner's window whose accepted shapes leave
something is offered the signals the election did not accept -- but only
one with a detected apex of its own that nobody holds, at the amount that
apex says, and only if that amount fits under the trace and accounts for a
part of what was left, and the rest of its fingerprint can carry it."""
import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.quant_shared import resolve_residuals

FIELD = 500.0
PPM = np.linspace(0.90, 1.10, 8000)
LW_HZ = 2.6
LW_PPM = LW_HZ / FIELD
J = 7.0


def _doublet(centre, height):
    return height * lines_model(PPM, [(-J / 2, 1.0), (J / 2, 1.0)], centre,
                                LW_HZ, FIELD)


class _Cand:
    def __init__(self, met_id, evidence, ranks=(), payload=None):
        self.met_id = met_id
        self.evidence = evidence
        self.line_ranks = frozenset(ranks)
        self.peak_rank = min(ranks) if ranks else None
        self.payload = payload or {}


VAL_C, ILE_C = 0.988, 1.0066            # valine's doublet; isoleucine's, 4 Hz up
VAL_LINES = [VAL_C - J / 2 / FIELD, VAL_C + J / 2 / FIELD]
ILE_LINES = [ILE_C - J / 2 / FIELD, ILE_C + J / 2 / FIELD]
PEAKS = {1: VAL_LINES[0], 2: VAL_LINES[1], 3: ILE_LINES[1]}   # ile's low line merged


def _study(val=(100.0, 100.0), ile=(60.0, 30.0), seed=0):
    rng = np.random.default_rng(seed)
    cols = []
    for a, b in zip(val, ile):
        cols.append(_doublet(VAL_C, a) + _doublet(ILE_C, b)
                    + rng.normal(0, 1.0, PPM.size))
    return np.column_stack(cols)


def _setup(ile_ev_extra=None):
    val = _Cand('VAL', {'multiplicity': 'd', 'predicted_ppms': VAL_LINES,
                        'protons': 3}, ranks=(1, 2))
    ile = _Cand('ILE', {'multiplicity': 'd', 'predicted_ppms': [VAL_LINES[1],
                                                                ILE_LINES[1]],
                        'expected_ppm': ILE_C, 'db_ppm': ILE_C - 0.008,
                        'protons': 3, **(ile_ev_extra or {})},
                ranks=(2, 3), payload={'met': {'db_peaks': []}})
    measured = {'VAL': {'window': (VAL_LINES[0] - 3 * LW_PPM,
                                   VAL_LINES[1] + 3 * LW_PPM),
                        'areas': {0: 1.0, 1: 1.0}, 'nh': 3.0, 'signal': val}}
    return val, ile, measured


def _lines_of(c):
    return [PEAKS[r] for r in sorted(c.line_ranks) if r in PEAKS]


def test_a_signal_with_a_free_apex_is_taken_at_the_amount_that_apex_says():
    X = _study()
    val, ile, measured = _setup()
    per_proton = {}
    taken = resolve_residuals(
        measured, [ile], PPM, X, included=[0, 1], linewidth_ppm=LW_PPM,
        lines_of=_lines_of, per_proton_areas=per_proton, noise_sigma=1.0,
        free_apex_of=lambda c: [PEAKS[r] for r in c.line_ranks if r == 3])
    assert 'ILE' in taken
    d = measured['VAL']['deconv']
    ile_rec = next(c for c in d['components'] if c['met_id'] == 'ILE')
    assert ile_rec['pinned'] and ile_rec['by_residual']
    # the owner is no longer credited with the merged line: its amplitude
    # comes down from what it read alone (about 130 with isoleucine's line
    # absorbed) towards its own 100; the owner's geometry, refined alone,
    # stands, so it does not land exactly
    a_val = d['amplitudes']['VAL']
    assert 60.0 < a_val[0] < 110.0
    # isoleucine's amount follows its free apex, sample by sample: 60 then
    # 30 -- the apex is one noisy point and the shape is refined after it
    # was read, so the scale is approximate; the ratio between samples is
    # the apex's, exactly
    a_ile = d['amplitudes']['ILE']
    assert a_ile[0] == pytest.approx(60.0, rel=0.3)
    assert a_ile[0] / a_ile[1] == pytest.approx(2.0, rel=0.2)
    assert 'VAL' in per_proton


def test_without_an_apex_of_its_own_a_signal_is_not_offered():
    X = _study()
    val, ile, measured = _setup()
    taken = resolve_residuals(
        measured, [ile], PPM, X, included=[0, 1], linewidth_ppm=LW_PPM,
        lines_of=_lines_of, noise_sigma=1.0,
        free_apex_of=lambda c: [])            # every line it has is valine's
    assert taken == {}
    (t,) = measured['VAL']['residual_trials']
    assert t['verdict'] == 'refused' and 'no apex' in t['why']


def test_an_apex_whose_shape_does_not_fit_under_the_trace_is_refused():
    """Isoleucine's free apex is made three times taller than its doublet
    allows: at that amount its other line overshoots the trace."""
    X = _study()
    i_ap = int(np.argmin(np.abs(PPM - ILE_LINES[1])))
    X[i_ap - 3:i_ap + 4, :] += 150.0
    val, ile, measured = _setup()
    taken = resolve_residuals(
        measured, [ile], PPM, X, included=[0, 1], linewidth_ppm=LW_PPM,
        lines_of=_lines_of, noise_sigma=1.0,
        free_apex_of=lambda c: [PEAKS[3]])
    assert taken == {}
    (t,) = measured['VAL']['residual_trials']
    assert t['verdict'] == 'not enough'
    assert t['overshoot'] > t['overshoot_base']


def test_a_window_with_nothing_left_offers_nothing():
    X = _study(ile=(0.0, 0.0))
    val, ile, measured = _setup()
    taken = resolve_residuals(
        measured, [ile], PPM, X, included=[0, 1], linewidth_ppm=LW_PPM,
        lines_of=_lines_of, noise_sigma=1.0,
        free_apex_of=lambda c: [PEAKS[3]])
    assert taken == {}
    assert 'residual_trials' not in measured['VAL']
