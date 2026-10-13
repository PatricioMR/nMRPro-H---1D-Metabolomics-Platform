"""Aligning a region is an edit of the spectrum, and an edit of the spectrum has
to be undoable and on the record.

It used to overwrite the whole matrix with nothing kept: a Revert button the
code looked for on the wrong object, vacated edges filled with zero (a step at
every join), no record of what was shifted, and the baseline cache left
holding the spectrum before it.
"""
import numpy as np
import pytest

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


PPM = np.linspace(4.2, 3.9, 3000)          # high -> low, as the application holds it
STEP = abs(PPM[1] - PPM[0])


def _lorentz(c, h=100.0, w=0.002):
    return h * w ** 2 / ((PPM - c) ** 2 + w ** 2)


def _study(shifts_pts=(0, 6, -5)):
    cols = []
    for k in shifts_pts:
        c = 4.05 + k * STEP
        cols.append(_lorentz(c) + _lorentz(3.95 + k * STEP, 40.0) + 1.0)
    return np.column_stack(cols)


@pytest.fixture
def window(real_app, monkeypatch):
    from gui.widgets import region_alignment_window as mod
    shown = []
    monkeypatch.setattr(mod.messagebox, 'showinfo', lambda *a, **k: shown.append(a))
    monkeypatch.setattr(mod.messagebox, 'showerror', lambda *a, **k: shown.append(('E',) + a))
    vis = real_app.stages['visualization']
    X = _study()
    real_app.ppm = PPM
    real_app.all_intensities = X.copy()
    real_app.samples_info = [{'label': 'A', 'type': 'Sample', 'name': f'S{i}'}
                             for i in range(3)]
    real_app.region_alignments = []
    vis._original_intensities = None
    peak = {'rank': 7, 'ppm': 4.05, 'start_ppm': 4.06, 'end_ppm': 4.04}
    win = mod.RegionAlignmentWindow(real_app.root, PPM, X, peak, real_app)
    win.win.withdraw()
    win._lo_var.set('4.12')
    win._hi_var.set('3.98')
    win.shown = shown
    yield win, X, vis
    try:
        win.win.destroy()
    except Exception:
        pass


def _run(win, mode):
    win._mode_var.set(mode)
    win._max_shift_var.set('0.02')
    win._n_seg_var.set('3')
    win._preview()
    assert win._state['aligned'] is not None, win._status_var.get()
    win._apply()


def test_what_is_changed_is_kept_so_revert_can_put_it_back(real_app, window, monkeypatch):
    win, X, vis = window
    _run(win, 'single')
    assert not np.array_equal(real_app.all_intensities, X), 'nothing was aligned'
    assert np.array_equal(vis._original_intensities, X), 'the original is kept'
    from gui.stages import visualization_stage as vmod
    monkeypatch.setattr('tkinter.messagebox.askyesno', lambda *a, **k: True)
    vis._revert_processing()
    assert np.array_equal(real_app.all_intensities, X)
    assert real_app.region_alignments == []


def test_the_shifts_are_on_the_record(real_app, window):
    win, X, vis = window
    _run(win, 'single')
    rec, = real_app.region_alignments
    assert rec['mode'] == 'single' and rec['signal_rank'] == 7
    assert rec['region_ppm'][0] == pytest.approx(3.98, abs=2 * STEP)
    assert rec['region_ppm'][1] == pytest.approx(4.12, abs=2 * STEP)
    assert any(v != 0 for v in rec['shifts_pts'])
    assert 'applied' in rec and rec['ppm_per_point'] == pytest.approx(STEP)


@pytest.mark.parametrize('mode', ['single', 'icoshift'])
def test_the_edges_a_shift_vacates_do_not_step_to_zero(real_app, window, mode):
    win, X, vis = window
    _run(win, mode)
    got = real_app.all_intensities
    # the baseline is 1.0 everywhere: a zero anywhere is a fabricated step
    assert got.min() > 0.5, 'a vacated edge was filled with zero'


def test_the_spectrum_outside_the_region_is_untouched(real_app, window):
    win, X, vis = window
    _run(win, 'single')
    outside = (PPM > 4.13) | (PPM < 3.97)
    assert np.array_equal(real_app.all_intensities[outside], X[outside])


def test_the_baseline_cache_is_dropped_with_the_spectrum(real_app, window, monkeypatch):
    win, X, vis = window
    calls = []
    real = real_app.invalidate_baseline_cache
    monkeypatch.setattr(real_app, 'invalidate_baseline_cache',
                        lambda: (calls.append(1), real())[1])
    _run(win, 'single')
    assert calls, 'the corrected spectrum would still be the one before'


def test_the_analyst_is_told_what_the_edit_leaves_stale(real_app, window):
    win, X, vis = window
    _run(win, 'single')
    text = ' '.join(str(a) for call in win.shown for a in call)
    assert 'detect the signals again' in text


def test_the_record_travels_with_the_project(real_app, window, tmp_path, monkeypatch):
    from data import project_file
    from gui import project_manager as pmm
    monkeypatch.setattr(pmm.messagebox, 'showwarning', lambda *a, **k: None)
    win, X, vis = window
    _run(win, 'single')
    mgr = real_app._project_mgr
    path = str(tmp_path / 'p.nmrproj')
    assert mgr.save_project(path)
    state = project_file.read(path)
    assert state['region_alignments'] == real_app.region_alignments
    real_app.region_alignments = []
    mgr._restore_state(state)
    assert len(real_app.region_alignments) == 1


def test_the_report_lists_the_regional_alignment(real_app, window):
    from gui.widgets import pdf_report
    import types
    win, X, vis = window
    _run(win, 'icoshift')
    q = real_app.stages['quantification']
    cells = [[getattr(c, 'text', c) for c in row]
             for row in pdf_report._params_table(q, real_app)._cellvalues]
    row = next(r for r in cells if r[0] == 'Regional alignment')
    assert 'icoshift' in row[1] and 'signal #7' in row[1]


def test_a_shifted_piece_holds_its_edge_value():
    from processing.peak_alignment import _shift_1d
    y = np.array([5.0, 6.0, 7.0, 8.0])
    assert list(_shift_1d(y, 1)) == [0.0, 5.0, 6.0, 7.0], 'the default is unchanged'
    assert list(_shift_1d(y, 1, edge=True)) == [5.0, 5.0, 6.0, 7.0]
    assert list(_shift_1d(y, -2, edge=True)) == [7.0, 8.0, 8.0, 8.0]
