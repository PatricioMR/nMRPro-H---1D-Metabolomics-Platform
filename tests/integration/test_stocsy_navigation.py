# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_stocsy_navigation.py
"""The STOCSY window, the Annotation Results and the Signal List in step:
going from one to the other, and each told when the others changed."""
from types import SimpleNamespace

import pytest

tk = pytest.importorskip('tkinter')


@pytest.fixture(scope='module')
def root():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    r.withdraw()
    yield r
    r.destroy()


def _annotation_like(root):
    """The Annotation Results' rows: a parent row per signal, a child row per
    listed signal."""
    from tkinter import ttk
    win = tk.Toplevel(root)
    tree = ttk.Treeview(win)
    met = {'id': 'HMDB1', 'matched_peaks': [{'peak_rank': 5}, {'peak_rank': 7}]}
    other = {'id': 'HMDB2', 'matched_peaks': [{'peak_rank': 9}]}
    rows, children, selected = {}, {}, []
    for k, (m, mp) in enumerate(((met, met['matched_peaks'][0]), (other, other['matched_peaks'][0]))):
        iid = tree.insert('', tk.END, text=f'row{k}')
        rows[iid] = (mp['peak_rank'], m, mp, None)
        if m is met:
            child = tree.insert(iid, tk.END, text='child')
            children[child] = (m, {}, met['matched_peaks'][1])
    return SimpleNamespace(tree=tree, win=win, _iid_to_row=rows, _iid_to_child=children,
                           _on_select=lambda: selected.append(tree.selection()[0])), met, selected


def test_the_annotation_window_selects_the_signal_it_is_sent(root):
    from gui.widgets.annotation_results_window import AnnotationResultsWindow
    w, met, selected = _annotation_like(root)
    assert AnnotationResultsWindow.focus_signal(w, 'HMDB1', met['matched_peaks'][1])
    assert w.tree.item(w.tree.selection()[0], 'text') == 'child' and selected
    assert w.tree.focus() == w.tree.selection()[0]
    assert AnnotationResultsWindow.focus_signal(w, 'HMDB1', {'peak_rank': 1})   # the analyte's row
    assert w.tree.item(w.tree.selection()[0], 'text') == 'row0'
    assert not AnnotationResultsWindow.focus_signal(w, 'HMDB3', {})
    w.win.destroy()


def _app(stocsy=None):
    from gui.app import App
    told = []
    app = SimpleNamespace(_stocsy_win=stocsy, root=None, told=told,
                          _stocsy_inputs=lambda: ('analysis', 'browser'))
    for name in ('_open_stocsy_win', 'on_signals_edited', 'on_annotation_quality_changed'):
        setattr(app, name, getattr(App, name).__get__(app))
    return app


def _stocsy_like(root, told):
    return SimpleNamespace(win=tk.Toplevel(root),
                           reload=lambda a, b: told.append(('reload', a, b)),
                           refresh_from_annotation=lambda: told.append('refresh'))


def test_an_open_stocsy_window_follows_the_signal_list_and_the_annotation(root):
    told = []
    app = _app(_stocsy_like(root, told))
    app.on_signals_edited()
    app.on_annotation_quality_changed()
    assert told == [('reload', 'analysis', 'browser'), 'refresh']
    app._stocsy_win.win.destroy()
    app.on_signals_edited()                                   # closed: nothing to follow
    assert len(told) == 2


def test_the_signal_list_tells_the_app_when_its_signals_changed(root, monkeypatch):
    from gui.widgets.peak_list_window import PeakListWindow
    told = []
    w = SimpleNamespace(
        win=tk.Toplevel(root), peaks=[], _stale_lbl=tk.Label(root), tree=None,
        app=SimpleNamespace(stages={}, on_signals_edited=lambda: told.append('edited')),
        _populate=lambda: None, _current_peak=None)
    w.tree = SimpleNamespace(get_children=lambda: ())
    PeakListWindow._signals_changed(w)
    assert told == ['edited']
    w.win.destroy()
