# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Moving through the samples of "View multiplet across all samples" shows
each sample's signal in the plot.

The table had no selection handler: clicking a sample or walking the rows
with the arrow keys left the overlay of every sample unchanged, so the
signal in the selected sample could not be inspected (reported 2026-11-26).
"""
import numpy as np
import pytest

tk = pytest.importorskip('tkinter')


@pytest.fixture(scope='module')
def root():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'no display: {exc}')
    r.withdraw()
    yield r
    r.destroy()


def _view(root):
    from gui.widgets.multiplet_sample_view import MultipletSampleView
    ppm = np.linspace(2.0, 1.0, 2000)
    cols = np.stack([1000 / (1 + ((ppm - 1.33 - 0.001 * s) / 0.002) ** 2)
                     for s in range(4)], axis=1)
    per_sample = [dict(sample_idx=s, present=s != 2, ppm_centre=1.33 + 0.001 * s,
                       j_obs_hz=7.0, area=1.0) for s in range(4)]
    mp = {'peak_rank': 1, 'peak_ppm': 1.33, 'per_sample': per_sample,
          'predicted_lines': [{'ppm': 1.324}, {'ppm': 1.336}]}
    dp = {'type': 'd', 'j_coupling': '7.0 Hz', 'ppm_center': 1.33}
    info = [{'name': f'S{s}', 'label': 'AB'[s // 2], 'type': 'Sample'}
            for s in range(4)]
    app = type('App', (), {'stages': {}, 'group_colors': {'A': 'red', 'B': 'blue'}})()
    return MultipletSampleView(root, {'name': 'Lactate'}, dp, mp, ppm, cols,
                               info, app)


def _select(v, iid):
    v._tree.selection_set(iid)
    v.win.update()


def _traces(v):
    return [l for l in v._ax.get_lines() if len(l.get_xdata()) > 2][:4]


def test_selecting_a_sample_brings_its_trace_forward(root):
    v = _view(root)
    try:
        widths = {l.get_linewidth() for l in _traces(v)}
        assert widths == {0.8}, 'unselected: every sample drawn alike'
        _select(v, 'sample:3')
        lines = _traces(v)
        assert lines[3].get_linewidth() > lines[0].get_linewidth()
        assert lines[3].get_alpha() == 1.0
        assert 'S3' in v._ax.get_title() and 'detected' in v._ax.get_title()
    finally:
        v.win.destroy()


def test_the_arrow_keys_walk_the_samples(root):
    v = _view(root)
    try:
        _select(v, 'sample:0')
        assert 'S0' in v._ax.get_title()
        v._tree.focus('sample:0')
        v._tree.focus_force()
        v._tree.event_generate('<Down>')
        v.win.update()
        assert v._tree.selection() == ('sample:1',)
        assert 'S1' in v._ax.get_title()
    finally:
        v.win.destroy()


def test_a_sample_the_search_did_not_detect_says_so(root):
    v = _view(root)
    try:
        _select(v, 'sample:2')
        assert 'not detected' in v._ax.get_title()
    finally:
        v.win.destroy()


def test_a_group_row_shows_that_group(root):
    v = _view(root)
    try:
        _select(v, 'group:A')
        lines = _traces(v)
        assert lines[0].get_color() == 'red' and lines[2].get_color() != 'blue'
        assert 'Group A' in v._ax.get_title()
    finally:
        v.win.destroy()
