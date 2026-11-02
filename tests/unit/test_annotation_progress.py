# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The arithmetic of the annotation run's progress window: the bar's share
per phase, the time left, the phases of a run. The window itself is in
tests/integration (tkinter is stubbed here)."""
import pytest

from gui.widgets.annotation_progress import (
    PHASES, bar_percent, format_seconds, run_phases, time_left)



def test_a_run_lists_its_phases_in_order():
    assert run_phases() == ('search', 'per_sample', 'scoring', 'interface')
    assert run_phases(search=False, mcr=True) == (
        'per_sample', 'mcr', 'scoring', 'interface')
    assert set(run_phases(mcr=True)) <= set(PHASES)


def test_the_bar_moves_by_each_phases_measured_share():
    ph = run_phases()
    w = {k: PHASES[k][1] for k in ph}
    total = sum(w.values())
    assert bar_percent(ph, 'search', 0, 10) == 0.0
    assert bar_percent(ph, 'search', 10, 10) == pytest.approx(
        100 * w['search'] / total)
    half = bar_percent(ph, 'per_sample', 5, 10)
    assert half == pytest.approx(
        100 * (w['search'] + w['per_sample'] / 2) / total)
    assert bar_percent(ph, 'interface', 1, 1) == pytest.approx(100.0)
    # no count: the phase sits at its start, and a count past the end stops
    assert bar_percent(ph, 'scoring', 0, 0) == bar_percent(ph, 'scoring', 0, 5)
    assert bar_percent(ph, 'search', 20, 10) == bar_percent(ph, 'search', 10, 10)


def test_time_left_waits_until_the_rate_means_something():
    assert time_left(1.0, 5, 10) is None           # under 2 s
    assert time_left(10.0, 0, 10) is None          # nothing done yet
    assert time_left(10.0, 10, 10) is None         # finished
    assert time_left(10.0, 0, 0) is None           # no count
    assert time_left(10.0, 5, 20) == pytest.approx(30.0)


def test_seconds_read_as_minutes_past_one():
    assert format_seconds(4.4) == '4 s'
    assert format_seconds(59.6) == '1 min 00 s'
    assert format_seconds(125) == '2 min 05 s'
