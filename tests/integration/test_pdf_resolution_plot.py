# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The report's first page shows the RP-Method resolution plot of each
group, drawn as the window draws it, with the share of the spectrum
resolved."""
import sys

import numpy as np
import pytest

from gui.widgets import pdf_report
from processing import resolution_plot as rp

PPM = np.linspace(3.0, 1.0, 2001)


def _lor(c, w=0.003):
    return w ** 2 / ((PPM - c) ** 2 + w ** 2)


@pytest.fixture
def res():
    trace = 100 * _lor(2.0) + 60 * _lor(1.5)
    x = np.linspace(1.9, 2.1, 401)
    rows = [{'hmdb_id': 'X', 'metabolite': 'X', '_joint_fit': True,
             '_fit_curve': {'ppm': x, 'focus': (2.0, 2.0), 'amp_by_group': {'A': 100.0},
                            'unit_by_group': {'A': 0.003 ** 2 / ((x - 2.0) ** 2 + 0.003 ** 2)}}}]
    out = rp.resolution(rows, PPM, trace, 'A', lambda a: 1.0, sigma=1.0, min_snr=5.0)
    out.update(ppm=PPM, trace=trace, excluded=[], rows=rows)
    return out


def test_the_line_under_the_plot_says_how_much_was_resolved(res):
    text = pdf_report.resolved_text('A', res)
    assert f"<b>{100 * res['share']:.0f}%</b>" in text
    assert '1 missing signal(s), 0 over-claim(s)' in text


def test_the_plot_is_the_windows_on_an_inverted_axis(res):
    mpl = sys.modules.get('matplotlib')
    if mpl is not None and not callable(getattr(mpl, 'get_backend', None)):
        pytest.skip('matplotlib is stubbed in this session')
    fig = pdf_report._resolution_fig('A', res)
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    assert lo > hi, 'δ decreases to the right'
    ylo, yhi = ax.get_ylim()
    assert ylo == pytest.approx(-yhi)
    assert 'resolved' in ax.get_title(loc='left')
    assert 'ppm' in ax.get_xlabel()
    pdf_report.plt.close(fig)


def test_the_report_zooms_on_0_8_ppm_at_half_the_internal_standard(res):
    mpl = sys.modules.get('matplotlib')
    if mpl is not None and not callable(getattr(mpl, 'get_backend', None)):
        pytest.skip('matplotlib is stubbed in this session')
    res = dict(res, is_height=500.0)
    fig = pdf_report._resolution_fig('A', res)
    ax = fig.axes[0]
    assert ax.get_xlim() == (8.0, 0.0)
    assert ax.get_ylim() == pytest.approx((-250.0, 250.0))
    assert not ax.collections or all(
        c.get_label() not in ('missing', 'over-claim') for c in ax.collections)
    pdf_report.plt.close(fig)


def test_the_internal_standards_height_is_read_on_its_region():
    trace = 100 * _lor(2.0) + 400 * _lor(1.2)
    rows = [{'hmdb_id': 'IS', 'metabolite': 'DSS', '_is_is': True, 'ppm': 1.2}]
    assert pdf_report.is_height(rows, PPM, trace) == pytest.approx(400.0, rel=0.01)
    assert pdf_report.is_height([], PPM, trace) is None


def test_the_report_lists_the_analytes_good_first(monkeypatch):
    from processing import quant_quality as qq
    levels = {'p': qq.POOR, 'g1': qq.GOOD, 'w': qq.WARN, 'g2': qq.GOOD}
    monkeypatch.setattr(qq, 'assess', lambda r, g, **k: {'level': levels[r['hmdb_id']]})
    rows = [{'hmdb_id': k, 'conc_mm': c} for k, c in
            (('p', 9.0), ('g1', 1.0), ('w', 5.0), ('g2', 3.0))]
    si = [{'label': 'A', 'type': 'Sample'}]
    order = pdf_report.quality_order(rows, PPM, np.ones((PPM.size, 1)), si)
    assert [r['hmdb_id'] for r in order] == ['g2', 'g1', 'w', 'p']
