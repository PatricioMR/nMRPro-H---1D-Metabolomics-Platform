# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Replicate runs are reported as the mean per run with the SD between runs,
never as totals."""
import math

import pytest

from validation.replicates import mean_sd, paired_change, pm, run_figures, summarize


def _row(mid, dots, refused=False, standard=False):
    return {'hmdb_id': mid, '_confidence': dots, '_not_quantified': refused, '_is_is': standard}


def test_run_figures_counts_one_run():
    truth = {'a': {}, 'b': {}, 'c': {}, 'd': {}}
    rows = [_row('a', 3), _row('b', 2), _row('x', 1), _row('y', 2),
            _row('c', 1, refused=True), _row('dss', 3, standard=True)]
    f = run_figures(rows, truth, [0.1, -0.3])
    assert (f['in_tube'], f['found'], f['false']) == (4, 2, 2)
    assert f['recovery_pct'] == 50 and f['precision_pct'] == 50
    assert (f['real3'], f['false3'], f['real2'], f['false2'], f['real1'], f['false1']) == (1, 0, 1, 1, 0, 1)
    assert f['error_median_pct'] == pytest.approx(20)


def test_mean_sd_is_per_run_not_total():
    assert mean_sd([41, 49, 59]) == (49.7, 9.0)
    assert mean_sd([5]) == (5.0, '')
    assert mean_sd(['', None, float('nan')]) == ('', '')


def test_summarize_and_paired_change():
    runs = [{'found': 80, 'false': 20}, {'found': 86, 'false': 16}]
    base = [{'found': 85, 'false': 18}, {'found': 85, 'false': 18}]
    s = summarize(runs, ['found', 'false'])
    assert s == {'runs': 2, 'found_mean': 83.0, 'found_sd': 4.2, 'false_mean': 18.0, 'false_sd': 2.8}
    c = paired_change(runs, base, ['found'])
    assert c['found_change_mean'] == -2.0 and math.isclose(c['found_change_sd'], 4.2)
    with pytest.raises(ValueError):
        paired_change(runs, base[:1], ['found'])


def test_pm_cell():
    assert pm(83.0, 3.5, '%') == '83.0 ± 3.5%'
    assert pm(5.0, '') == '5.0'
    assert pm('', '') == '—'
