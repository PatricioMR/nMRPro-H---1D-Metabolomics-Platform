# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Autosave keeps the work a crash or a forgotten save would lose, and gives
it back.

The real application is built without a window. A session with a spectrum is
saved, changed, autosaved, and the autosave is offered when the project is
opened again; work never saved goes to the recovery folder and is offered at
start. A project in the old format opens and saving converts it.
"""
import os
import pickle

import numpy as np
import pytest

from data import project_file

tk = pytest.importorskip('tkinter')


@pytest.fixture(scope='module')
def real_app():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
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


@pytest.fixture
def pm(real_app, monkeypatch, tmp_path):
    from gui import project_manager as mod
    answers = []
    monkeypatch.setattr(mod.messagebox, 'showwarning', lambda *a, **k: None)
    monkeypatch.setattr(mod.messagebox, 'showerror',
                        lambda *a, **k: pytest.fail(f'error shown: {a}'))
    monkeypatch.setattr(mod.messagebox, 'askyesnocancel',
                        lambda *a, **k: answers.pop(0))
    rec = tmp_path / 'recovery'
    rec.mkdir()
    monkeypatch.setattr(mod, 'recovery_dir', lambda: str(rec))
    mgr = real_app._project_mgr
    mgr.project_path = None
    mgr._saved_digest = mgr._saved_content = None
    mgr._autosaved_digest = mgr._recovery_path = None
    mgr._dirty = False
    rng = np.random.default_rng(0)
    real_app.ppm = np.linspace(10.0, -0.5, 2048)
    real_app.all_intensities = rng.normal(0, 1e3, (2048, 4)) + 1e6
    real_app.samples_info = [{'index': i, 'name': f'S{i}', 'label': 'G'}
                             for i in range(4)]
    mgr.answers = answers
    mgr.recovery = rec
    yield mgr
    mgr._stop_autosave()
    mgr._wait_for_writer()


def _quant(app):
    return app.stages['quantification']


def _pretend_saved(pm):
    state = pm._collect_state()
    pm._mark_saved(project_file.snapshot(state), state)


def _autosave(pm):
    t = pm.autosave_now()
    assert t is not None, 'nothing was autosaved'
    pm._wait_for_writer()
    return t


def test_a_change_no_stage_announced_is_still_an_unsaved_change(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.has_unsaved_changes(), 'imported and never saved'
    assert pm.save_project(path)
    assert not pm.has_unsaved_changes()
    _quant(real_app)._quant_results = [{'metabolite': 'x', 'conc_mm': 1.0}]
    assert pm.has_unsaved_changes()
    _quant(real_app)._quant_results = []


def test_the_stage_on_screen_is_not_a_change(real_app, pm, tmp_path):
    assert pm.save_project(str(tmp_path / 'p.nmrproj'))
    was = real_app.active_stage
    real_app.active_stage = 'statistics' if was != 'statistics' else 'annotation'
    try:
        assert not pm.has_unsaved_changes()
    finally:
        real_app.active_stage = was


def test_a_panel_built_after_saving_is_not_a_change_but_an_option_is(
        real_app, pm, tmp_path):
    st = real_app.stages['statistics']
    assert pm.save_project(str(tmp_path / 'p.nmrproj'))
    st._built_late_var = tk.StringVar(master=real_app.root, value='PNG')
    try:
        assert not pm.has_unsaved_changes()
        q = _quant(real_app)
        q.singlet_snr_var.set(q.singlet_snr_var.get() + 1)
        assert pm.has_unsaved_changes()
        q.singlet_snr_var.set(q.singlet_snr_var.get() - 1)
    finally:
        del st._built_late_var


def test_the_autosave_goes_beside_the_project_and_leaves_it_alone(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    saved = open(path, 'rb').read()
    assert pm.autosave_now() is None, 'nothing changed: nothing to write'
    _quant(real_app)._quant_results = [{'metabolite': 'x', 'conc_mm': 2.0}]
    _autosave(pm)
    assert open(path, 'rb').read() == saved
    auto = project_file.read(path + '.autosave')
    assert auto['qnt']['quant_results'] == [{'metabolite': 'x', 'conc_mm': 2.0}]
    assert pm.autosave_now() is None, 'the same state is not written twice'
    _quant(real_app)._quant_results = []


def test_a_newer_autosave_is_offered_when_the_project_is_opened(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    _quant(real_app)._quant_results = [{'metabolite': 'x', 'conc_mm': 3.0}]
    _autosave(pm)
    os.utime(path, (1, 1))                      # the project is the older
    _quant(real_app)._quant_results = []
    _pretend_saved(pm)                          # nothing to ask on leaving
    pm.answers.append(True)                     # yes: open the autosave
    assert pm.open_project(path)
    assert _quant(real_app)._quant_results == [{'metabolite': 'x', 'conc_mm': 3.0}]
    assert pm.project_path == path, 'saving goes to the project, not the autosave'
    assert pm.has_unsaved_changes()
    assert os.path.exists(path + '.autosave'), 'kept until it is saved'
    assert pm.save_project(path)
    assert not os.path.exists(path + '.autosave'), 'the project now holds it'
    assert project_file.read(path)['qnt']['quant_results'][0]['conc_mm'] == 3.0
    _quant(real_app)._quant_results = []


def test_declining_the_autosave_opens_the_project_as_saved(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    _quant(real_app)._quant_results = [{'metabolite': 'x', 'conc_mm': 4.0}]
    _autosave(pm)
    os.utime(path, (1, 1))
    _pretend_saved(pm)
    pm.answers.append(False)
    assert pm.open_project(path)
    assert _quant(real_app)._quant_results == []
    assert not pm.has_unsaved_changes()


def test_work_never_saved_is_recovered_at_start(real_app, pm):
    _quant(real_app)._quant_results = [{'metabolite': 'y', 'conc_mm': 5.0}]
    _autosave(pm)
    files = os.listdir(pm.recovery)
    assert len(files) == 1 and files[0].endswith('.nmrproj.autosave')
    # the application ended without saving; the next session starts empty
    _quant(real_app)._quant_results = []
    pm._recovery_path = pm._autosaved_digest = None
    pm.answers.append(True)
    pm.offer_recovery()
    assert _quant(real_app)._quant_results == [{'metabolite': 'y', 'conc_mm': 5.0}]
    assert pm.project_path is None and pm.has_unsaved_changes()
    pm.shutdown()
    assert os.listdir(pm.recovery) == [], 'declined on close: nothing left to offer'
    _quant(real_app)._quant_results = []


def test_a_recovery_declined_is_deleted_and_one_postponed_stays(real_app, pm):
    _autosave(pm)
    pm._recovery_path = None
    pm.answers.append(None)                     # later
    pm.offer_recovery()
    assert len(os.listdir(pm.recovery)) == 1
    pm.answers.append(False)                    # delete
    pm.offer_recovery()
    assert os.listdir(pm.recovery) == []


def test_closing_leaves_no_autosave_behind(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    _quant(real_app)._quant_results = [{'metabolite': 'z', 'conc_mm': 6.0}]
    _autosave(pm)
    pm.shutdown()
    assert not os.path.exists(path + '.autosave')
    _quant(real_app)._quant_results = []


def test_a_project_in_the_old_format_opens_and_saving_converts_it(real_app, pm, tmp_path):
    path = str(tmp_path / 'old.nmrproj')
    assert pm.save_project(path)
    state = project_file.read(path)
    state['all_intensities'] = real_app.all_intensities.copy()
    with open(path, 'wb') as f:                 # as v0.72 wrote it
        pickle.dump(state, f, protocol=5)
    old_size = os.path.getsize(path)
    assert pm.open_project(path)
    assert not project_file.is_current_format(path)
    assert pm.save_project(path)
    assert project_file.is_current_format(path)
    assert os.path.getsize(path) < old_size
    np.testing.assert_array_equal(
        project_file.read(path)['all_intensities'],
        project_file.stored_spectrum(state['all_intensities']))


def _until(app, cond, secs=10.0):
    import time
    end = time.monotonic() + secs
    while not cond() and time.monotonic() < end:
        app.root.update()
        time.sleep(0.01)
    assert cond(), 'timed out'


def test_a_quantification_run_autosaves_first_and_cancel_keeps_the_table(
        real_app, pm, monkeypatch):
    """A run of minutes: what the analyst had is on disk before it starts,
    and stopping it leaves the table and the run button as they were."""
    q = _quant(real_app)
    calls = []
    monkeypatch.setattr(pm, 'autosave_before_run', lambda: calls.append('saved'))

    def run(progress=None):
        calls.append('run')
        progress('windows')
        progress.cancel()                      # the Cancel button
        progress('shapes')
        calls.append('not reached')
    monkeypatch.setattr(q, '_run', run)
    monkeypatch.setattr(q, '_populate_tree', lambda: calls.append('table'))
    monkeypatch.setattr(real_app.stages['annotation'], '_metabolites',
                        [{'id': 'm'}], raising=False)
    monkeypatch.setattr(q.ext_is_var, 'get', lambda: False)
    before = [{'metabolite': 'x', 'conc_mm': 2.0}]
    q._quant_results = before
    cancelled = []
    q._safe_run(on_done=lambda: calls.append('done'),
                on_cancel=lambda: cancelled.append(True))
    _until(real_app, lambda: bool(cancelled))
    assert calls == ['saved', 'run', 'table']
    assert q._quant_results is before
    assert q.run_status.cget('text').startswith('Cancelled')
    assert str(q.btn_run.cget('state')) == 'normal'
    assert getattr(q, '_prog_win', None) is None
    q._quant_results = []


def test_an_annotation_run_autosaves_first(real_app, pm, monkeypatch):
    from gui.widgets.annotation_progress import run_phases
    calls = []
    monkeypatch.setattr(pm, 'autosave_before_run', lambda: calls.append('saved'))
    prog = real_app.stages['annotation']._run_progress(run_phases())
    prog.close()
    assert calls == ['saved']


def test_a_signal_chosen_for_one_analyte_moves_only_it_and_is_saved(
        real_app, pm, tmp_path, monkeypatch):
    """Quant Signals -> Quantify: the analyte is measured on the doublet the
    analyst chose with the run's conversion, its neighbour's row is left
    exactly as it was, the choice is in the saved project, and the engine's
    choice puts the row back."""
    g = 0.002
    real_app.ppm = ppm = np.linspace(4.0, 1.0, 12001)

    def line(c, area):
        return area * (g / np.pi) / ((ppm - c) ** 2 + g * g)
    spec = line(3.000, 50.0) + line(3.012, 50.0) + line(2.000, 30.0)
    real_app.all_intensities = np.column_stack([spec] * 4)
    q = _quant(real_app)
    peaks = [{'rank': 1, 'ppm': 3.000}, {'rank': 2, 'ppm': 3.012},
             {'rank': 3, 'ppm': 2.000}]
    monkeypatch.setattr(q.peak_stage, 'peaks', peaks, raising=False)
    monkeypatch.setattr(q.peak_stage, 'noise_std', 1.0, raising=False)
    q._last_run_settings = {'local_baseline_var': False, 'icoshift_var': False,
                            'blk_sub_var': False, 'bl_model_quant_var': False}
    q._last_groups = {'G': [0, 1, 2, 3]}
    q._last_blk_indices = []
    neighbour = {'hmdb_id': 'B', 'metabolite': 'B', '_nh': 1.0,
                 '_areas_per_sample': [30.0] * 4,
                 '_concs_per_sample': [3.0] * 4, 'conc_mm': 3.0,
                 '_multiplet_window': [1.99, 2.01]}
    row = {'hmdb_id': 'A', 'metabolite': 'A', '_not_quantified': True,
           'status': 'n.q.', 'conc_mm': float('nan')}
    q._quant_results = [neighbour, row]
    before_b = dict(neighbour)
    choice = {'line_ranks': [2, 1], 'protons': 2, 'line_weights': None,
              'predicted_ppms': [3.012, 3.000], 'label': '3.006 ppm d'}
    assert q.quantify_with_signal(row, choice) is None
    # 100 of area for 2 protons, at the run's 0.1 mM per unit per proton
    assert row['conc_mm'] == pytest.approx(5.0, rel=0.03)
    assert row['status'] == '●○○' and not row['_not_quantified']
    assert neighbour == before_b
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    saved = project_file.read(path)['qnt']
    assert saved['quant_peak_overrides']['A']['line_ranks'] == [2, 1]
    assert saved['quant_results'][1]['_analyst_signal'] == '3.006 ppm d'
    q.undo_signal_choice(row)
    assert row['_not_quantified'] and 'A' not in q._quant_peak_overrides
    q._quant_results = []
    q._quant_peak_overrides = {}


def test_the_spectrum_resolved_is_saved_with_the_results(real_app, pm, tmp_path):
    """The report quotes how much of the spectrum the run explained; a
    reopened project must still say it."""
    q = _quant(real_app)
    cov = {'peaks': 270, 'explained': 111, 'area_fraction': 0.62,
           'orphans': [(3.21, 40.0, 0.1)], 'n_orphans': 159,
           'explained_share': 0.5, 'min_snr': 5.0}
    q._quant_diagnostics = {'coverage': cov}
    q._quant_results = [{'metabolite': 'x', 'conc_mm': 2.0}]
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    q._quant_diagnostics = {}
    _pretend_saved(pm)
    assert pm.open_project(path)
    got = q._quant_diagnostics['coverage']
    assert (got['peaks'], got['explained'], got['n_orphans']) == (270, 111, 159)
    assert got['area_fraction'] == 0.62
    q._quant_results = []
    q._quant_diagnostics = {}
