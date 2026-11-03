# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The quantification run's progress: its phases in the paper's order, the
engine's reports reaching the window, and Cancel stopping the engine at its
next report (gui.stages._quant_progress, quant_engine._under)."""
import pytest

from gui.stages._quant_progress import (
    PASS, PHASES, EngineProgress, RunCancelled, pass_fraction, run_phases)
from processing.quant_engine import _under, _within


class _Window:
    def __init__(self):
        self.posted = []

    def post(self, *a):
        self.posted.append(a)


def test_a_run_lists_its_phases_in_the_papers_order():
    assert run_phases() == ('prepare', 'windows', 'shapes', 'joint',
                            'recheck', 'cascade', 'results')
    assert run_phases(joint=False, cascade=False, accepted=True) == (
        'prepare', 'windows', 'shapes', 'recheck', 'accepted', 'results')
    assert set(run_phases(accepted=True)) <= set(PHASES)


def test_reports_reach_the_window_and_cancel_stops_the_next_one():
    p = EngineProgress(run_phases(cascade=False))
    p.window = _Window()
    p('joint', 3, 32, 'samples', 'fitting each sample')
    p('cascade', 1, 60)                     # not a phase of this run
    assert p.window.posted == [('joint', 3, 32, 'samples',
                                'fitting each sample', None)]
    p.cancel()
    with pytest.raises(RunCancelled):
        p('results')
    assert p.cancelled


def test_a_repeated_sub_run_reports_under_its_own_phase():
    got = []
    rec = lambda *a, **k: got.append(a + (k,))
    rep = _under(rec, 'recheck', 'round 2')
    rep('joint', 5, 10, 'samples', 'fitting each sample', frac=0.7)
    rep('windows')
    assert got == [('recheck', 5, 10, 'samples',
                    'round 2 · joint fit (B) · fitting each sample',
                    {'frac': 0.7, 'sub': 'joint'}),
                   ('recheck', 0, 0, '', 'round 2 · windows',
                    {'frac': None, 'sub': 'windows'})]
    loop = _within(rec, 'cascade')
    loop(4, 60, 'steps')
    assert got[-1] == ('cascade', 4, 60, 'steps', '', {'frac': None})
    assert _under(None, 'recheck') is None and _within(None, 'joint') is None


def test_a_repeated_pass_moves_forward_through_its_own_phases():
    """'accepted' runs the windows, the shapes, the joint fit and the
    cascade again: its bar moves through them in order, never back to the
    start when the next one begins."""
    p = EngineProgress(run_phases(accepted=True))
    p.window = _Window()
    p('accepted', sub='windows')
    p('accepted', 32, 32, 'samples', frac=1.0, sub='joint')
    p('accepted', 0, 60, 'steps', sub='cascade')
    fr = [x[-1] for x in p.window.posted]
    assert fr == sorted(fr) and fr[0] == 0.0
    assert fr[2] == pass_fraction('cascade', 0.0) < 1.0
    assert pass_fraction(PASS[-1], 1.0) == 1.0
