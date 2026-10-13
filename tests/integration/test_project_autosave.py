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
