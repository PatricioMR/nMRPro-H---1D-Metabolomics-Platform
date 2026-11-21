# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""plotting.contributions on a real matplotlib axes: the analytes of the
view drawn, the selected last, nothing outside the view."""
import sys

import numpy as np
import pytest

from plotting import contributions as pc

PPM = np.linspace(2.60, 2.20, 801)


def _lorentz(x, c, w=0.002):
    return w ** 2 / ((x - c) ** 2 + w ** 2)


def _plt():
    """The real pyplot, or a skip: the unit suite's conftest stubs
    matplotlib for a session that collects both."""
    mpl = sys.modules.get('matplotlib')
    if mpl is not None and not callable(getattr(mpl, 'get_backend', None)):
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _fit_row(mid, centre, amp=2.0, group='A'):
    return {'hmdb_id': mid, 'metabolite': mid, '_joint_fit': True,
            '_fit_curve': {'ppm': PPM, 'focus': (centre, centre),
                           'amp_by_group': {group: amp},
                           'unit_by_group': {group: _lorentz(PPM, centre)}}}


def test_draw_analytes_draws_the_view_and_puts_the_selected_last():
    trace = 2.0 * _lorentz(PPM, 2.40) + 1.5 * _lorentz(PPM, 2.41) + _lorentz(PPM, 2.25)
    a, b, far = _fit_row('a', 2.40), _fit_row('b', 2.41), _fit_row('far', 2.25)
    rows = [a, b, far]
    plt = _plt()
    fig, ax = plt.subplots()
    try:
        drawn = pc.draw_analytes(ax, rows, a, PPM, trace, 'A', 2.35, 2.45,
                                 pc.stable_palette(rows, PPM), lambda amps: 1.0)
        assert [n for n, _c, _s in drawn] == ['b', 'a']
        assert [s for _n, _c, s in drawn] == [False, True]
        # one fill per analyte in view, nothing for the one outside it
        assert len(ax.collections) == 2
    finally:
        plt.close(fig)


def test_draw_fit_says_when_the_group_holds_no_amplitude():
    plt = _plt()
    fig, ax = plt.subplots()
    try:
        assert pc.draw_fit(ax, _fit_row('a', 2.40), 'B', '#000', 2.3, 2.5) is False
        assert pc.draw_fit(ax, _fit_row('a', 2.40), 'A', '#000', 2.3, 2.5) is True
    finally:
        plt.close(fig)
