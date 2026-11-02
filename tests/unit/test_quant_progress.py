# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The quantification run's progress: its phases in the paper's order, the
engine's reports reaching the window, and Cancel stopping the engine at its
next report (gui.stages._quant_progress, quant_engine._under)."""
import pytest

from gui.stages._quant_progress import (
    PHASES, EngineProgress, RunCancelled, run_phases)
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
                                'fitting each sample')]
    p.cancel()
    with pytest.raises(RunCancelled):
        p('results')
    assert p.cancelled


def test_a_repeated_sub_run_reports_under_its_own_phase():
    got = []
    rep = _under(lambda *a: got.append(a), 'recheck', 'round 2')
    rep('joint', 5, 10, 'samples', 'fitting each sample')
    rep('windows')
    assert got == [('recheck', 5, 10, 'samples',
                    'round 2 · joint fit (B) · fitting each sample'),
                   ('recheck', 0, 0, '', 'round 2 · windows')]
    loop = _within(lambda *a: got.append(a), 'cascade')
    loop(4, 60, 'steps')
    assert got[-1] == ('cascade', 4, 60, 'steps', '')
    assert _under(None, 'recheck') is None and _within(None, 'joint') is None
