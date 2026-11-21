# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Every analyte of a view drawn as what produced its number, in a colour of
its own (plotting.contributions)."""
import numpy as np
import pytest

from plotting import contributions as pc

PPM = np.linspace(2.60, 2.20, 801)


def _lorentz(x, c, w=0.002):
    return w ** 2 / ((x - c) ** 2 + w ** 2)


def _fit_row(mid, centre, amp=2.0, group='A'):
    return {'hmdb_id': mid, 'metabolite': mid, '_joint_fit': True,
            '_fit_curve': {'ppm': PPM, 'focus': (centre, centre),
                           'amp_by_group': {group: amp},
                           'unit_by_group': {group: _lorentz(PPM, centre)}}}


def _integral_row(mid, lo, hi):
    return {'hmdb_id': mid, 'metabolite': mid, '_integ_mode': 'Multiplet',
            '_multiplet_window': (lo, hi), '_local_baseline': False}


@pytest.mark.parametrize('row, want', [
    (_fit_row('a', 2.4), 'fit'),
    ({'_integ_mode': 'Shape', '_shape_fit': {'x': 1}}, 'shape'),
    ({'_deconv': {'window': (2.3, 2.4)}}, 'deconv'),
    (_integral_row('b', 2.3, 2.4), 'integral'),
    # a fit record without a joint fit or cascade did not give the number
    ({'_fit_curve': {'focus': (2.4, 2.4)}, '_deconv': {'window': (2.3, 2.4)}}, 'deconv'),
])
def test_method_is_what_gave_the_number(row, want):
    assert pc.method(row) == want


def test_the_selected_curve_is_named_by_its_method():
    assert pc.method_label(_fit_row('a', 2.4)) == 'Fit'
    row = _fit_row('a', 2.4)
    row['_fit_curve']['label'] = 'Cascade fit'
    assert pc.method_label(row) == 'Cascade fit'
    shape = {'_integ_mode': 'Shape', '_shape_fit': {'x': 1}, '_shape_explained': 0.9}
    assert pc.method_label(shape) == 'Shape fit  (explains 90%)'
    assert pc.method_label(_integral_row('b', 2.3, 2.4)) is None


def test_neighbours_on_the_axis_take_different_colours():
    rows = [_fit_row('a', 2.40), _fit_row('b', 2.41), _integral_row('c', 2.25, 2.26)]
    pal = pc.stable_palette(rows, PPM)
    assert len(set(pal.values())) == 3
    assert pal == pc.stable_palette(list(reversed(rows)), PPM)


def test_trimmed_outline_drops_only_where_the_signal_is_nothing():
    y = _lorentz(PPM, 2.40)
    t = pc._trimmed(y, np.zeros_like(y), y)
    assert np.isnan(t[0]) and np.isnan(t[-1])
    assert t[np.argmax(y)] == pytest.approx(1.0)


def test_the_stage_draws_on_the_fit_group_when_the_snr_group_holds_little():
    tk = pytest.importorskip('tkinter')  # noqa: F841 -- the mixin imports it
    from gui.stages._quant_visualization import _QuantVisMixin
    groups = {'A': [0, 1], 'B': [2, 3]}
    row = _fit_row('a', 2.40)
    row['_fit_curve']['amp_by_group'] = {'A': 0.1, 'B': 2.0}
    row['_fit_curve']['unit_by_group']['B'] = _lorentz(PPM, 2.40)
    row['_snr_group'] = 'A'
    assert _QuantVisMixin._analytes_group(row, groups) == 'B'
    assert _QuantVisMixin._analytes_group({'_snr_group': 'A'}, groups) == 'A'
    assert _QuantVisMixin._analytes_group({}, groups) == 'A'
