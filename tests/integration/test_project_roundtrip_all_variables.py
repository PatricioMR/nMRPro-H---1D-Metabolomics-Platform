"""Save is the same as having stopped where the analyst stopped.

The real application is built without a window, every variable of every stage
is changed, the project is saved to a file, the variables are put back to what
they were and the project is opened: every one has to come back as it was
saved. This is the test that would have caught a parameter the saving code
never listed.
"""
import os
import pickle

import pytest

tk = pytest.importorskip('tkinter')


@pytest.fixture(scope='module')
def real_app():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        from gui.app import App
        app = App()
    except tk.TclError:
        pytest.skip('no display')
    except Exception as e:                       # a stubbed matplotlib, etc.
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
def pm(real_app, monkeypatch):
    from gui import project_manager as mod
    shown = []
    monkeypatch.setattr(mod.messagebox, 'showwarning', lambda *a, **k: shown.append(a))
    monkeypatch.setattr(mod.messagebox, 'showerror', lambda *a, **k: shown.append(('ERROR',) + a))
    mgr = real_app._project_mgr
    mgr.shown = shown
    return mgr


def _targets(app):
    return {'app': app, **app.stages}


def _change_every_variable(app):
    from gui.project_vars import variables_of
    for obj in _targets(app).values():
        for var in variables_of(obj).values():
            v = var.get()
            try:
                if isinstance(v, bool):
                    var.set(not v)
                elif isinstance(v, int):
                    var.set(v + 3)
                elif isinstance(v, float):
                    var.set(round(v + 0.5, 6))
            except Exception:
                pass


def _all(app):
    from gui.project_vars import collect_vars
    return {n: collect_vars(o) for n, o in _targets(app).items()}


def test_every_variable_of_every_stage_comes_back_as_it_was_saved(
        real_app, pm, tmp_path):
    original = _all(real_app)
    _change_every_variable(real_app)
    changed = _all(real_app)
    assert changed != original, 'the test changed nothing'
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    # back to what they were: opening has to move them again
    from gui.project_vars import apply_vars
    for n, obj in _targets(real_app).items():
        apply_vars(obj, original[n])
    with open(path, 'rb') as f:
        state = pickle.load(f)
    pm._restore_state(state)
    restored = _all(real_app)
    wrong = {n: {k: (v, restored[n].get(k)) for k, v in changed[n].items()
                 if restored[n].get(k) != v} for n in changed}
    assert not any(wrong.values()), wrong


def test_the_import_referencing_comes_back_with_the_project(real_app, pm, tmp_path):
    rec = {'target_ppm': 0.0, 'summary': 'Reference standard set',
           'samples': {'S1': {'offset_ppm': 0.0583, 'apex_ppm': 0.0583, 'snr': 9e3},
                       'S2': {'offset_ppm': None, 'apex_ppm': 0.11, 'snr': 2.0}}}
    real_app.import_reference = rec
    path = str(tmp_path / 'ref.nmrproj')
    assert pm.save_project(path)
    real_app.import_reference = None
    with open(path, 'rb') as f:
        pm._restore_state(pickle.load(f))
    assert real_app.import_reference == rec


def test_save_load_save_gives_the_same_variables(real_app, pm, tmp_path):
    _change_every_variable(real_app)
    first = str(tmp_path / 'a.nmrproj')
    second = str(tmp_path / 'b.nmrproj')
    assert pm.save_project(first)
    with open(first, 'rb') as f:
        pm._restore_state(pickle.load(f))
    assert pm.save_project(second)
    with open(first, 'rb') as f:
        a = pickle.load(f)
    with open(second, 'rb') as f:
        b = pickle.load(f)
    assert a['vars'] == b['vars']


def test_the_options_the_results_came_from_are_kept_and_a_change_is_said(
        real_app, pm, tmp_path):
    from gui.project_vars import collect_vars
    q = real_app.stages['quantification']
    q.local_baseline_var.set(False)
    q.singlet_snr_var.set(5.0)
    q._quant_results = [{'metabolite': 'x', 'hmdb_id': 'X', 'conc_mm': 1.0}]
    q._last_run_settings = collect_vars(q)
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    with open(path, 'rb') as f:
        state = pickle.load(f)
    assert state['qnt']['last_run_settings']['local_baseline_var'] is False
    # the analyst changed an option after running, and saved
    state['vars']['quantification']['local_baseline_var'] = True
    state['qnt']['local_baseline'] = True
    pm.shown.clear()
    pm._restore_state(state)
    text = ' '.join(str(a) for call in pm.shown for a in call)
    assert 'not the ones the saved results were produced with' in text
    assert 'local_baseline_var' in text and 'False' in text and 'True' in text
    q._quant_results = []


def test_a_variable_that_cannot_be_put_back_is_said(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    with open(path, 'rb') as f:
        state = pickle.load(f)
    state['vars']['quantification']['joint_fit_var'] = object()   # not settable
    pm.shown.clear()
    failed = pm._restore_all_vars(state)
    assert 'joint_fit_var' in failed['quantification']
    assert any('joint_fit_var' in str(a) for call in pm.shown for a in call)


def test_a_project_saved_before_this_still_opens(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    with open(path, 'rb') as f:
        state = pickle.load(f)
    for k in ('vars', 'software'):
        state.pop(k)
    state['version'] = 6
    pm.shown.clear()
    pm._restore_state(state)
    assert not any(c and c[0] == 'ERROR' for c in pm.shown)


def test_a_save_that_fails_leaves_the_old_project_as_it_was(
        real_app, pm, tmp_path, monkeypatch):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    before = open(path, 'rb').read()
    real_dump = pickle.dump

    def broken(obj, f, *a, **k):
        f.write(b'half a file')
        raise OSError('disk full')
    monkeypatch.setattr(pickle, 'dump', broken)
    assert pm.save_project(path) is False
    monkeypatch.setattr(pickle, 'dump', real_dump)
    assert open(path, 'rb').read() == before
    assert not os.path.exists(path + '.tmp'), 'the half-written file is removed'


def test_the_previous_version_is_kept_for_one_save(real_app, pm, tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    assert pm.save_project(path)
    first = open(path, 'rb').read()
    assert pm.save_project(path)
    assert os.path.exists(path + '.bak') and open(path + '.bak', 'rb').read() == first
    assert not os.path.exists(path + '.tmp')
