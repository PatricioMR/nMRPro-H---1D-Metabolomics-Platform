"""The deconvolution review, on the real application without a window.

It lists everything the quantification deconvolved (not only the windows
resolved by shape), draws the fit of each, and lets the analyst accept, reject
or keep the engine's choice, then applies them together: one re-quantification.
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


PPM = np.linspace(3.0, 1.0, 4000)


def _lor(c, w=0.004):
    return w ** 2 / ((PPM - c) ** 2 + w ** 2)


def _record(centre, amp):
    sel = (PPM > centre - 0.12) & (PPM < centre + 0.12)
    x = PPM[sel]
    u = 1.0 / (1.0 + ((x - centre) / 0.004) ** 2)
    return {'ppm': x.tolist(), 'unit_by_group': {'A': u.tolist()},
            'focus_unit_by_group': {'A': u.tolist()},
            'amp_by_group': {'A': amp}, 'n_by_group': {'A': [3, 3]},
            'baseline_by_group': None, 'label': 'Cascade fit',
            'focus': [centre - 0.01, centre + 0.01], 'focus_nh': 1.0,
            'focus_matches_window': True, 'at_limit': [], 'n_signals': 1}


def _row(mid, name, **kw):
    r = {'hmdb_id': mid, 'metabolite': name, 'ppm': 2.0, '_confidence': 1,
         '_not_quantified': False, '_concs_per_sample': [1.0] * 3,
         '_snr_group': 'A', '_status_reasons': []}
    r.update(kw)
    return r


@pytest.fixture
def window(real_app, monkeypatch):
    from gui.widgets import deconvolution_window as mod
    app = real_app
    Y = np.column_stack([60.0 * _lor(2.0) + 30.0 * _lor(2.1) + 20.0 * _lor(1.5)] * 3)
    app.ppm = PPM
    app.all_intensities = Y.copy()
    app.baseline_corrected = False
    app.samples_info = [{'label': 'A', 'type': 'Sample', 'name': f'S{i}'}
                        for i in range(3)]
    q = app.stages['quantification']
    q._last_groups = {'A': [0, 1, 2]}
    q._last_run_settings = {'bl_model_quant_var': False}
    q._deconv_overrides = {}
    q._quant_results = [
        _row('casc', 'Cascade analyte', _by_cascade=True, _confidence=2,
             _fit_curve=_record(2.0, 60.0), _cascade={'score': 0.7, 'free_share': 0.9}),
        _row('joint', 'Joint analyte', _joint_fit={'group': ['joint', 'casc']},
             _confidence=3, ppm=2.1, _fit_curve=_record(2.1, 30.0),
             _identity_why='2 of its signals confirmed'),
        _row('cand', 'Candidate', _not_quantified=True, _rejection='cascade_crowded',
             _confidence=0, ppm=1.5, _cascade={'signals': [(1.5, 'x', 3.0, 8.0)]}),
        _row('plain', 'Plain integral', _confidence=3),
    ]
    calls = []
    monkeypatch.setattr(q, '_safe_run', lambda on_done=None: calls.append(on_done))
    asked = []
    monkeypatch.setattr(mod.messagebox, 'askyesno',
                        lambda *a, **k: (asked.append(a), True)[1])
    win = mod.DeconvolutionWindow(app.root, q)
    win.win.withdraw()
    win.calls, win.asked, win.q = calls, asked, q
    yield win
    try:
        win.win.destroy()
    except Exception:
        pass
    q._quant_results = []
    q._deconv_overrides = {}


def test_it_lists_every_kind_of_deconvolution_and_not_a_plain_integral(window):
    ids = set(window._tree.get_children())
    assert ids == {'casc', 'joint', 'cand'}
    kinds = {i['id']: i['kind'] for i in window._items}
    assert kinds == {'casc': 'cascade', 'joint': 'joint', 'cand': 'candidate'}
    assert '1 cascade' in window._summary.cget('text')


def test_every_kind_draws_without_error(window):
    for mid in ('casc', 'joint', 'cand'):
        window._tree.selection_set(mid)
        window._show(mid)
        assert window.ax.get_lines(), f'nothing drawn for {mid}'
        assert window._head.cget('text').startswith(window._by_id[mid]['name'])


def test_the_figure_draws_the_analyte_the_others_standing_there_and_what_is_left(window):
    window._show('casc')
    labels = [l.get_label() for l in window.ax.get_lines()]
    assert 'Cascade analyte' in labels        # the owner, by name
    assert 'Joint analyte' in labels          # the other one standing there
    assert 'sum of the fit' in labels and 'what the fit leaves' in labels
    assert any('spectrum (group A)' == l for l in labels)
    assert len(window._ctree.get_children()) >= 2, 'the other analyte standing there is listed'


def test_a_candidate_shows_where_the_cascade_saw_it(window):
    window._show('cand')
    assert window.ax.lines, 'a marker line for its signal'
    assert any(abs(l.get_xdata()[0] - 1.5) < 1e-9 for l in window.ax.lines
               if len(set(l.get_xdata())) == 1)


def test_accepting_and_rejecting_wait_until_applied(window):
    window._tree.selection_set('cand')
    window._decide('accept')
    assert window._pending == {'cand': {'decision': 'accept'}}
    assert window.q._deconv_overrides == {}, 'nothing is sent to the engine yet'
    assert '1 decision' in window._summary.cget('text')
    assert str(window._apply_btn.cget('state')) == 'normal'
    assert '✓ accepted' in window._tree.item('cand', 'values')[5]
    window._tree.selection_set('casc')
    window._decide('reject')
    assert window._pending['casc'] == {'decision': 'reject'}
    assert 'Apply 2 decision' in window._apply_btn.cget('text')


def test_several_rows_are_decided_at_once(window):
    window._tree.selection_set(('casc', 'joint'))
    window._decide('reject')
    assert set(window._pending) == {'casc', 'joint'}
    assert all(v == {'decision': 'reject'} for v in window._pending.values())


def test_the_engines_choice_takes_a_decision_back(window):
    window._tree.selection_set('cand')
    window._decide('accept')
    window._decide(None)
    assert window._pending == {}
    assert str(window._apply_btn.cget('state')) == 'disabled'


def test_apply_sends_every_decision_in_one_run(window):
    window._tree.selection_set('cand')
    window._decide('accept')
    window._tree.selection_set('casc')
    window._decide('reject')
    window._apply()
    assert len(window.calls) == 1, 'one re-quantification for all of them'
    assert window.q._deconv_overrides == {'cand': {'decision': 'accept'},
                                          'casc': {'decision': 'reject'}}
    assert 'EVERY analyte' in window.asked[0][1]
    assert 'Cascade analyte' in window.asked[0][1]


def test_a_decision_not_confirmed_is_not_applied(window, monkeypatch):
    from gui.widgets import deconvolution_window as mod
    monkeypatch.setattr(mod.messagebox, 'askyesno', lambda *a, **k: False)
    window._tree.selection_set('cand')
    window._decide('accept')
    window._apply()
    assert window.calls == [] and window.q._deconv_overrides == {}


def test_discard_goes_back_to_what_was_applied(window):
    window.q._deconv_overrides = {'casc': {'decision': 'reject'}}
    window.refresh()
    window._tree.selection_set('cand')
    window._decide('accept')
    assert len(window._pending) == 2
    window._discard()
    assert window._pending == {'casc': {'decision': 'reject'}}


def test_after_the_run_the_list_shows_what_was_applied(window):
    window.q._deconv_overrides = {'cand': {'decision': 'accept'}}
    window.refresh(select='cand')
    assert '✓ accepted' in window._tree.item('cand', 'values')[5]
    assert '*' not in window._tree.item('cand', 'values')[5]
    assert str(window._apply_btn.cget('state')) == 'disabled'


def test_the_filters_narrow_the_list(window):
    window._kind_vars['cascade'].set(False)
    window._fill_tree()
    assert 'casc' not in window._tree.get_children()
    window._kind_vars['cascade'].set(True)
    window._look_var.set(True)                   # only ● or no number
    window._fill_tree()
    assert set(window._tree.get_children()) == {'cand'}
    window._look_var.set(False)
    window._search_var.set('joint')
    window._fill_tree()
    assert set(window._tree.get_children()) == {'joint'}


def test_with_no_results_it_says_what_it_will_show_and_offers_to_run(real_app, monkeypatch):
    from gui.widgets import deconvolution_window as mod
    q = real_app.stages['quantification']
    q._quant_results = []
    runs = []
    monkeypatch.setattr(q, '_safe_run', lambda on_done=None: runs.append(1))
    win = mod.DeconvolutionWindow(real_app.root, q)
    win.win.withdraw()
    try:
        assert 'No quantification yet' in win._summary.cget('text')
        assert 'Run quantification' in win._run_btn.cget('text')
        assert 'shows here every window' in win._why.get('1.0', 'end')
        win._run_now()
        assert runs == [1]
    finally:
        win.win.destroy()


def test_the_decisions_are_kept_in_the_project(real_app, window, tmp_path, monkeypatch):
    from data import project_file
    from gui import project_manager as pmm
    monkeypatch.setattr(pmm.messagebox, 'showwarning', lambda *a, **k: None)
    window.q._deconv_overrides = {'cand': {'decision': 'accept'}}
    mgr = real_app._project_mgr
    path = str(tmp_path / 'p.nmrproj')
    assert mgr.save_project(path)
    state = project_file.read(path)
    assert state['qnt']['deconv_overrides'] == {'cand': {'decision': 'accept'}}
    window.q._deconv_overrides = {}
    mgr._restore_state(state)
    assert window.q._deconv_overrides == {'cand': {'decision': 'accept'}}


def test_the_toolbar_button_follows_what_there_is_to_review(real_app, window):
    assert real_app.quant_has_deconvolution()
    window.q._quant_results = [_row('plain', 'Plain integral')]
    assert not real_app.quant_has_deconvolution()


def test_after_the_run_it_says_what_the_decisions_changed(window):
    window._tree.selection_set('cand')
    window._decide('accept')
    window._apply()
    # the run: the candidate now has a number, nothing else moves
    rows = window.q._quant_results
    cand = next(r for r in rows if r['hmdb_id'] == 'cand')
    cand.update({'_not_quantified': False, 'conc_mm': 0.5, '_confidence': 1,
                 '_by_cascade': True, '_fit_curve': _record(1.5, 20.0),
                 '_rejection': None})
    window.calls[0]()                                   # the stage's on_done
    assert window.last_change['entered'] == ['Candidate']
    assert 'entered' in window._preview_lbl.cget('text')
