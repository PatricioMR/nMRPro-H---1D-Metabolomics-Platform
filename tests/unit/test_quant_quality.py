# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The quantification quality: whether a number can be trusted, per group,
from what the engine already records."""
import numpy as np
import pytest

from processing import quant_quality as qq

PPM = np.linspace(3.0, 1.0, 4001)


def _lor(c, w=0.003):
    return w ** 2 / ((PPM - c) ** 2 + w ** 2)


def _fit(amp=100.0, n=(6, 6)):
    o = np.argsort(PPM)
    unit = _lor(2.0) / np.trapezoid(_lor(2.0)[o], PPM[o])
    return {'ppm': PPM, 'focus': (2.0, 2.0), 'amp_by_group': {'A': amp},
            'unit_by_group': {'A': unit}, 'focus_unit_by_group': {'A': unit},
            'n_by_group': {'A': list(n)}}


def test_a_clean_number_is_good():
    row = {'_fit_curve': _fit(), 'cv_by_group': {'A': 8.0}, '_cascade': {'disagreement': 0.1}}
    q = qq.assess(row, 'A')
    assert q == {'level': 'Good', 'reasons': [], 'cv': 8.0}


@pytest.mark.parametrize('row, level, reason', [
    ({'_fit_curve': _fit(n=(4, 6))}, 'Poor', 'found in 4 of 6 samples'),
    ({'_cascade': {'disagreement': 0.60}}, 'Poor', 'its signals disagree by 60%'),
    ({'cv_by_group': {'A': 45.0}}, 'Poor', 'replicates spread 45%'),
    ({'cv_by_group': {'A': {'cv': 25.0}}}, 'Warn', 'replicates spread 25%'),
    ({'_cascade': {'at_cap': True}}, 'Warn', 'held at its cap'),
    ({'_cascade': {'capped_share': 0.3}}, 'Warn', 'held at its cap'),
])
def test_each_check_sets_its_level(row, level, reason):
    q = qq.assess(row, 'A')
    assert q['level'] == level
    assert reason in q['reasons'][0]


def test_a_fit_holding_more_than_the_trace_is_a_warning():
    trace = 100.0 * _lor(2.0) / np.trapezoid(_lor(2.0)[np.argsort(PPM)], np.sort(PPM))
    assert qq.assess({'_fit_curve': _fit(100.0)}, 'A', ppm=PPM, trace=trace)['level'] == 'Good'
    q = qq.assess({'_fit_curve': _fit(200.0)}, 'A', ppm=PPM, trace=trace)
    assert q['level'] == 'Warn' and 'holds 2.00x' in q['reasons'][0]
    # without the trace the check is not made
    assert qq.assess({'_fit_curve': _fit(200.0)}, 'A')['level'] == 'Good'


def test_the_worst_reason_comes_first_and_decides():
    row = {'cv_by_group': {'A': 25.0}, '_cascade': {'at_cap': True, 'disagreement': 0.9}}
    q = qq.assess(row, 'A')
    assert q['level'] == 'Poor'
    assert q['reasons'][0].startswith('its signals disagree')
    assert len(q['reasons']) == 3


def test_rows_without_a_number_have_no_level():
    assert qq.assess({'_not_quantified': True}, 'A')['level'] is None
    assert qq.assess({'_is_is': True}, 'A')['level'] is None
    assert qq.worst(['Good', None, 'Warn']) == 'Warn'
    assert qq.worst([None]) is None


def test_tiers_sort_best_first():
    levels = [None, qq.POOR, qq.GOOD, qq.WARN]
    assert sorted(levels, key=qq.rank) == [qq.GOOD, qq.WARN, qq.POOR, None]
