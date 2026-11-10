# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_stocsy_window_reopen.py
"""The STOCSY matrix is derived state: a reopened project must rebuild it.
(Own file, sorted after the window tests: once the real application has run, a
second Tk root cannot be made in the same session.)"""
import pytest

from data import project_file

tk = pytest.importorskip('tkinter')


@pytest.fixture(scope='module')
def real_app():
    try:
        from gui.app import App
        app = App()
    except tk.TclError as e:
        pytest.skip(f'no display: {e}')
    except Exception as e:
        pytest.skip(f'the application cannot be built here: {e}')
    app.root.withdraw()
    try:
        app._build_ui()
    except Exception as e:
        pytest.skip(f'the application cannot be built here: {e}')
    app._init_project_manager()
    yield app
    try:
        app.root.destroy()
    except Exception:
        pass


def test_a_reopened_project_annotates_with_the_same_correlation(
        real_app, tmp_path, monkeypatch):
    """The matrix is derived and not stored. Left empty after opening, the
    annotation run on a reopened project would score without the STOCSY term
    the saved run had."""
    from gui import project_manager as mod
    from tests.unit.test_stocsy_controller import _study
    monkeypatch.setattr(mod.messagebox, 'showwarning', lambda *a, **k: None)
    monkeypatch.setattr(mod.messagebox, 'showerror', lambda *a, **k: None)
    ppm, X, peaks, info = _study(seed=4)
    real_app.ppm = ppm
    real_app.all_intensities = X
    real_app.samples_info = info
    pd = real_app.stages['peak_detection']
    pd.peaks = peaks
    pd.build_peak_correlation()
    before = dict(pd._peak_corr)
    assert before
    path = str(tmp_path / 'stocsy.nmrproj')
    assert real_app._project_mgr.save_project(path)
    pd._peak_corr, pd._n_samples_pk = {}, 0
    real_app._project_mgr._restore_state(project_file.read(path))
    assert pd._n_samples_pk == 12
    assert pd._peak_corr.keys() == before.keys()
    for k, v in before.items():
        assert pd._peak_corr[k] == pytest.approx(v, abs=1e-5)
    # nothing quantified in this project: nothing for the window to audit
    assert str(real_app.btn_stocsy_shortcut.cget('state')) == 'disabled'


def _molecule_one(real_app):
    from tests.unit.test_stocsy_controller import _study
    ppm, X, peaks, info = _study(seed=4)
    real_app.ppm, real_app.all_intensities, real_app.samples_info = ppm, X, info
    pd = real_app.stages['peak_detection']
    pd.peaks = peaks
    ann = real_app.stages['annotation']
    ann._metabolites = [{'id': 'M1', 'name': 'Molecule one', 'matched_peaks': [
        {'peak_rank': 1, 'peak_ppm': 2.0, 'db_ppm': 2.0, 'protons': 3, 'type': 's',
         'predicted_lines': [{'ppm': 2.0}]},
        {'peak_rank': 2, 'peak_ppm': 3.5, 'db_ppm': 3.5, 'protons': 2, 'type': 's',
         'predicted_lines': [{'ppm': 3.5}]}], 'db_peaks': []}]
    ann._quality_marks = {}
    real_app.stages['quantification']._quant_results = [{'hmdb_id': 'M1', '_confidence': 2}]
    real_app.refresh_stocsy_button()
    return pd, ann, peaks


def test_the_bottom_bar_opens_the_window_once_there_is_a_quantification(real_app, monkeypatch):
    """The button every stage shows: on with a quantification, it builds the
    analysis from the project, lists the quantified analytes and comes to the
    front; opened again it returns to where the analyst was."""
    pd, _ann, peaks = _molecule_one(real_app)
    assert str(real_app.btn_stocsy_shortcut.cget('state')) == 'normal'
    real_app.open_stocsy()
    w = real_app._stocsy_win
    real_app.root.update_idletasks()
    assert w.win.winfo_exists() and w.met_id == 'M1' and w.editor is not None
    assert w.an_tree.item('M1')['text'] == 'Molecule one'
    w.hw_var.set('0.3')
    first = w.win
    pd.peaks = peaks                                  # the application's own idle work resets them
    real_app.open_stocsy()
    assert not first.winfo_exists() and real_app._stocsy_win.met_id == 'M1'
    real_app._stocsy_win.win.destroy()


def test_the_lines_chosen_in_the_window_are_saved_with_the_project(real_app, tmp_path, monkeypatch):
    """Lines chosen in the window ride on the annotation and its marks: saved,
    reopened, the signal reads the same lines and keeps its Good mark."""
    from gui import project_manager as pm
    from gui.widgets import stocsy_window as sw
    from processing.multiplet_line_selector import multiplet_line_peaks
    from processing.signal_lines import manual_lines
    monkeypatch.setattr(pm.messagebox, 'showwarning', lambda *a, **k: None)
    monkeypatch.setattr(pm.messagebox, 'showerror', lambda *a, **k: None)
    monkeypatch.setattr(sw.messagebox, 'askyesno', lambda *a, **k: True)
    pd, ann, peaks = _molecule_one(real_app)
    real_app.open_stocsy()
    w = real_app._stocsy_win
    w._set_sought(next(i for i, s in enumerate(w.browser.signals('M1')) if s['db_ppm'] == 3.5))
    w.ticked = [3]                                      # another peak as the 2 H singlet
    w._rebuild()
    w._save_lines()
    w.win.destroy()
    mp = ann._metabolites[0]['matched_peaks'][1]
    chosen = manual_lines(mp)
    assert chosen and ann._quality_marks == {('M1', 3): 'good'}
    path = str(tmp_path / 'lines.nmrproj')
    assert real_app._project_mgr.save_project(path)
    ann._metabolites, ann._quality_marks = None, {}
    real_app._project_mgr._restore_state(project_file.read(path))
    mp2 = ann._metabolites[0]['matched_peaks'][1]
    assert manual_lines(mp2) == pytest.approx(chosen) and mp2['peak_rank'] == 3
    assert ann._quality_marks == {('M1', 3): 'good'} and mp2['annotation_quality'] == 'good'
    by_rank = {int(p['rank']): p for p in pd.peaks}
    assert [p['rank'] for p in multiplet_line_peaks(mp2, by_rank)] == [3]


def test_the_window_asks_for_a_quantification_first(real_app, monkeypatch):
    from gui import app as mod
    shown = []
    monkeypatch.setattr(mod.messagebox, 'showinfo', lambda *a, **k: shown.append(a))
    real_app.stages['quantification']._quant_results = []
    real_app.refresh_stocsy_button()
    real_app._stocsy_win = None
    real_app.open_stocsy()
    assert shown and 'Quantify first' in shown[0][1]
    assert str(real_app.btn_stocsy_shortcut.cget('state')) == 'disabled'
