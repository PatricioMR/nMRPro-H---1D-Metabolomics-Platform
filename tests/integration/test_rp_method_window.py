# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""RP-Method, on the real application without a window: the residual of a
signal nothing explains is listed with the annotation's candidates there,
marking one Good is the annotation's mark the next quantification reads,
and re-running goes through the stage's own run."""
import numpy as np
import pytest

tk = pytest.importorskip('tkinter')

PPM = np.linspace(3.0, 1.0, 4000)


def _lor(c, w=0.004):
    return w ** 2 / ((PPM - c) ** 2 + w ** 2)


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


def _record(centre, amp):
    sel = (PPM > centre - 0.1) & (PPM < centre + 0.1)
    x = PPM[sel]
    u = 1.0 / (1.0 + ((x - centre) / 0.004) ** 2)
    return {'ppm': x, 'unit_by_group': {'A': u}, 'amp_by_group': {'A': amp},
            'label': 'Cascade fit', 'focus': [centre, centre]}


@pytest.fixture
def window(real_app, monkeypatch):
    from gui.widgets import rp_method_window as mod
    app = real_app
    # explained at 2.0; at 1.5 a signal nothing explains
    Y = np.column_stack([60.0 * _lor(2.0) + 40.0 * _lor(1.5)] * 3)
    app.ppm = PPM
    app.all_intensities = Y.copy()
    app.baseline_corrected = False
    app.samples_info = [{'label': 'A', 'type': 'Sample', 'name': f'S{i}'} for i in range(3)]
    q = app.stages['quantification']
    q._last_groups = {'A': [0, 1, 2]}
    q._last_run_settings = {'bl_model_quant_var': False}
    q._quant_results = [{'hmdb_id': 'HIT', 'metabolite': 'Explained', '_by_cascade': True,
                         '_fit_curve': _record(2.0, 60.0)}]
    monkeypatch.setattr(q.peak_stage, 'noise_std', 0.5, raising=False)
    ann = app.stages['annotation']
    ann._metabolites = [{'id': 'MISS', 'name': 'Missing one', 'composite_score': 0.8,
                         'db_peaks': [{'ppm_center': 1.501, 'type': 's', 'protons': 3}],
                         'matched_peaks': [{'db_ppm': 1.501, 'peak_ppm': 1.5,
                                            'peak_rank': 12, 'annotation_quality': None}]}]
    ann._quality_marks = {}
    calls = []
    monkeypatch.setattr(q, '_safe_run', lambda on_done=None, on_cancel=None: calls.append(on_done))
    monkeypatch.setattr(mod.messagebox, 'askyesno', lambda *a, **k: True)
    win = mod.RPMethodWindow(app, q, ann)
    win.win.withdraw()
    win.calls, win.ann = calls, ann
    yield win
    try:
        win.win.destroy()
    except Exception:
        pass
    q._quant_results = []
    ann._metabolites = None


def test_the_missing_signal_is_listed_with_its_candidate(window):
    regs = window._shown
    assert [round(g['apex_ppm'], 2) for g in regs if g['kind'] == 'missing'] == [1.5]
    window._rtree.selection_set('0')
    window._on_region()
    assert [c['name'] for c in window._cands] == ['Missing one']
    assert window._ctree.item('0')['values'][0] == 'Missing one'
    # the plot: spectrum above, the resolved analyte mirrored below
    lo, hi = window.ax.get_ylim()
    assert lo < 0 < hi
    assert window.ax.get_xlim()[0] > window.ax.get_xlim()[1], 'δ decreases to the right'


def test_marking_the_candidate_good_is_the_annotations_mark(window):
    window._rtree.selection_set('0')
    window._on_region()
    window._ctree.selection_set('0')
    window._update_buttons()
    assert str(window._good_btn.cget('state')) == 'normal'
    window._mark('good')
    assert window.ann._quality_marks[('MISS', 12)] == 'good'
    assert window.ann._metabolites[0]['matched_peaks'][0]['annotation_quality'] == 'good'
    assert 'Re-run quantification (1)' == window._run_btn.cget('text')
    window._rerun()
    assert len(window.calls) == 1


def test_the_reference_window_is_not_a_missing_signal(window, monkeypatch):
    q = window.qs
    monkeypatch.setattr(q.ref_ppm_var, 'get', lambda: '1.5')
    window.refresh()
    assert not [g for g in window._shown if g['kind'] == 'missing']


def test_picking_candidates_lays_their_fingerprints_on_the_residual(window):
    ann = window.ann
    ann._metabolites[0]['db_peaks'][0].update(
        {'j_coupling': '7.1', 'pattern_lines': '[[-7.1, 0.5], [0.0, 1.0], [7.1, 0.5]]'})
    ann._metabolites[0]['matched_peaks'][0].update(
        {'observed_j_hz': 7.3, 'delta_j_hz': 0.2, '_n_found_lines': 3, '_n_exp_lines': 3})
    window._field = lambda: 600.0
    window._rtree.selection_set('0')
    window._on_region()
    vals = window._ctree.item('0')['values']
    assert [str(v) for v in vals[1:6]] == ['s', '7.10', '7.30', '0.20', '3/3']
    window._ctree.selection_set('0')
    window._on_candidates()
    labels = [t.get_text() for t in window.ax.get_legend().get_texts()]
    assert any(lb.startswith('Missing one s J 7.10') for lb in labels)


def test_previous_and_next_walk_the_regions(window):
    window._step(1)
    assert window._rtree.selection() == ('0',)


def test_a_confirmed_analytes_other_signal_is_subtracted_and_can_be_switched_off(window):
    q, ann = window.qs, window.ann
    q._quant_results[0]['_confidence'] = 2
    # a singlet: taken only because STOCSY linked it to the amount measured
    q._quant_results[0]['_signal_links'] = [{'db_ppm': 1.5, 'type': 's', 'rank': 0,
                                             'r': 0.95, 'gap': 0.02}]
    # the explained analyte's reference also lists a singlet at 1.5: its
    # number predicts it, and with the subtraction on it is taken off
    ann._metabolites.append(
        {'id': 'HIT', 'name': 'Explained',
         'db_peaks': [{'ppm_center': 2.0, 'protons': 1, 'type': 's'},
                      {'ppm_center': 1.5, 'protons': 1, 'type': 's',
                       'pattern_lines': '[[0.0, 1.0]]'}],
         'matched_peaks': [{'db_ppm': 1.5, 'peak_ppm': 1.5, 'study_linewidth_hz': 2.4,
                            'pattern_tolerance_ppm': 0.01, 'pattern_field_mhz': 600.0}]})
    window._secondary_var.set(True)
    window.refresh()
    assert any(c['method'] == 'secondary' for c in window._out['curves'])
    assert 'other signals of 1 confirmed' in window._summary.cget('text')
    window._secondary_var.set(False)
    window.refresh()
    assert not any(c['method'] == 'secondary' for c in window._out['curves'])
    assert [round(g['apex_ppm'], 2) for g in window._shown if g['kind'] == 'missing'] == [1.5]
