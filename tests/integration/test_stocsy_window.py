# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_stocsy_window.py
"""The STOCSY window -- quantified analytes on the left, what the
quantification measured as the reference, a signal read line by line with the
ticked lines whole, the analyst's lines saved and restored -- and the stage
that feeds it, with a real Tk."""
import numpy as np
import pytest

from tests.unit.test_stocsy_analytes import _ab, detection_areas

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


def _browser(quant=False, mets=None, **kw):
    """Over the unit tests' study; ``quant`` gives A a quantification with its
    per-sample areas (the reference the window opens on)."""
    from controllers.stocsy_analytes import AnalyteBrowser
    from tests.unit.test_stocsy_analytes import _mets, _quant_rows, _study
    rows = (_quant_rows() if quant else
            [{'hmdb_id': 'MA', '_confidence': 3}, {'hmdb_id': 'MB', '_confidence': 1}])
    an = _study(**kw)
    return AnalyteBrowser(an, _mets() if mets is None else mets, rows, measure=detection_areas(an))


def _window(root, browser=None, **kw):
    from gui.widgets.stocsy_window import StocsyWindow
    b = browser or _browser()
    w = StocsyWindow(root, b.an, browser=b, view_region=(9.0, 0.5),
                     group_colors={'G1': '#1f77b4', 'G2': '#d62728'}, **kw)
    root.update()
    return w


def _idx(w, ppm):
    return next(i for i, s in enumerate(w.browser.signals(w.met_id)) if abs(s['db_ppm'] - ppm) < 1e-6)


def _click_signal(w, root, ppm):
    w.sig_tree.selection_set(str(_idx(w, ppm)))
    w.sig_tree.event_generate('<<TreeviewSelect>>')
    root.update()


# ── opening ──────────────────────────────────────────────────────────────────

def test_it_opens_on_what_the_quantification_measured_for_the_first_analyte(root):
    w = _window(root, _browser(quant=True))
    assert w.met_id == 'MA' and w.ref_i == 'q' and w.ref.kind == 'quant'
    assert 'what the quantification measured' in w.ref_lbl.cget('text')
    assert 'Cascade' in w.ref_lbl.cget('text') and '7 H' in w.ref_lbl.cget('text')
    first = w.sig_tree.get_children()[0]
    assert first == 'q' and str(w.sig_tree.item(first)['values'][0]) == '◉'
    assert w.sought_i == _idx(w, 1.30)                     # the listed signal at the δ it reported
    w.win.destroy()


def test_only_the_quantified_analytes_are_listed_once_there_is_a_quantification(root):
    from controllers.stocsy_analytes import AnalyteBrowser
    from tests.unit.test_stocsy_analytes import _mets, _study
    b = _ab(_study(), _mets(), [{'hmdb_id': 'MA', '_confidence': 3}])
    w = _window(root, b)
    assert w.only_q_var.get() and w.an_tree.get_children() == ('MA',)
    w.win.destroy()


def test_without_the_quantifications_areas_it_opens_on_the_clearest_signal(root):
    w = _window(root)
    assert w.met_id == 'MA'
    assert [w.an_tree.item(i)['text'] for i in w.an_tree.get_children()] == ['A-like', 'Other']
    assert w.an_tree.item('MA')['values'][0] == '●●●'
    assert w.ref.ranks == [1] and w.ref_i == _idx(w, 1.30)            # the singlet of 3 H
    assert 'A-like' in w.title_lbl.cget('text')
    assert 'Reference' in w.ref_lbl.cget('text') and '3 H' in w.ref_lbl.cget('text')
    assert 'Signal read' in w.sought_lbl.cget('text')
    w.win.destroy()


def test_the_window_comes_to_the_front_and_is_sized_to_the_screen(root):
    w = _window(root)
    assert w.win.state() == 'normal'
    assert w.win.winfo_width() <= root.winfo_screenwidth()
    w.win.destroy()


def test_the_listed_signals_say_which_was_not_found(root):
    w = _window(root)
    rows = [w.sig_tree.item(i)['values'] for i in w.sig_tree.get_children()]
    assert len(rows) == 4
    assert 'not detected' in rows[0][-1]                              # 7.90 ppm, high δ first
    assert [str(r[0]) for r in rows].count('◉') == 1 and [str(r[0]) for r in rows].count('▶') == 1
    w.win.destroy()


def test_the_lines_table_has_the_reference_then_the_ticked_signal_then_the_peaks(root):
    w = _window(root, _browser(quant=True))
    _click_signal(w, root, 4.10)
    rows = w.win_tree.get_children()
    assert rows[:2] == ('ref', 'whole') and set(rows[2:]) == {'p10', 'p8', 'p2'}
    whole = w.win_tree.item('whole')['values']
    assert whole[0] == 'Σ' and whole[1] == 'ticked lines (2)'
    assert w.win_tree.item('p2')['values'][0] == '☑' and w.win_tree.item('p10')['values'][0] == '☐'
    assert float(w.win_tree.item('p2')['values'][4]) == pytest.approx(0.5)   # half the doublet
    assert w.win_tree.item('p10')['values'][4] == '—'                       # on no predicted line
    w.win.destroy()


def test_the_spectrum_runs_from_high_to_low_delta_in_ppm(root):
    w = _window(root)
    lo, hi = sorted(w.ax_spec.get_xlim())
    assert w.ax_spec.get_xlim()[0] > w.ax_spec.get_xlim()[1]
    assert w.ax_spec.get_xlabel() == 'δ (ppm)' and 'au' in w.ax_spec.get_ylabel()
    assert hi - lo == pytest.approx(2 * w._hw(), rel=0.01)
    w.win.destroy()


def test_clicking_the_quantification_row_shows_the_lines_it_integrated(root):
    w = _window(root, _browser(quant=True))
    _click_signal(w, root, 4.10)
    w.sig_tree.selection_set('q')
    w.sig_tree.event_generate('<<TreeviewSelect>>')
    root.update()
    assert w.quant_view and w.sought_i == _idx(w, 1.30) and w.ticked == [1]
    assert w._hw() == pytest.approx(0.01 + w.tolerance / 10, abs=1e-3)   # its window, 1.29-1.31
    assert 'the quantification integrated' in w.sought_lbl.cget('text')
    _click_signal(w, root, 4.10)
    assert not w.quant_view
    w.win.destroy()


def test_the_window_is_the_annotation_tolerance_around_the_pattern(root):
    w = _window(root, tolerance_ppm=0.04)
    _click_signal(w, root, 4.10)                                      # lines 4.095 and 4.105
    assert w._hw() == pytest.approx(0.04 + 0.005, abs=1e-3)
    _click_signal(w, root, 2.05)
    assert w._hw() == pytest.approx(0.04, abs=1e-3)
    w.win.destroy()


def test_the_signals_list_shows_the_annotations_delta_ppm_and_delta_j(root):
    from controllers.stocsy_analytes import AnalyteBrowser
    from tests.unit.test_stocsy_analytes import _mets, _quant_rows, _study
    mets = _mets()
    mets[0]['matched_peaks'][1].update({'ppm_diff': 0.0042, 'delta_j_hz': 0.37})
    w = _window(root, _ab(_study(), mets, _quant_rows()))
    vals = w.sig_tree.item(str(_idx(w, 4.10)))['values']
    cols = list(w.sig_tree['columns'])
    assert vals[cols.index('dd')] == '0.0042' and vals[cols.index('dj')] == '0.37'
    assert w.sig_tree.item(str(_idx(w, 7.90)))['values'][cols.index('dj')] == '—'
    w.win.destroy()


# ── choosing ─────────────────────────────────────────────────────────────────

def test_clicking_another_analyte_changes_the_reference_and_clears_the_typed_protons(root):
    w = _window(root)
    w.h_ref_var.set('9')
    w.an_tree.selection_set('MB')
    w.an_tree.event_generate('<<TreeviewSelect>>')
    root.update()
    assert w.met_id == 'MB' and w.h_ref_var.get() == ''
    assert 'Other' in w.title_lbl.cget('text')
    w.win.destroy()


def test_clicking_a_signal_reads_its_lines_and_the_window_follows_it(root):
    w = _window(root)
    assert w.sought.ppm == pytest.approx(4.10)                       # the clearest after the reference
    _click_signal(w, root, 2.05)
    assert w.sought_i == _idx(w, 2.05) and w.sought.ppm == pytest.approx(2.05)
    assert w.ax_spec.get_xlim()[0] == pytest.approx(2.05 + w._hw(), abs=0.01)
    assert w.cand_rows[0]['label'] == 'ticked lines (1)'
    _click_signal(w, root, 4.10)
    assert w.cand_rows[0]['label'] == 'ticked lines (2)' and sorted(w.ticked) == [2, 8]
    w.win.destroy()


def test_a_signal_not_detected_is_read_around_where_it_is_listed(root):
    w = _window(root, _browser(quant=True))
    _click_signal(w, root, 7.90)
    w.hw_var.set('1.5')                                               # widened by the analyst
    w._rebuild()
    assert w.sought_i == _idx(w, 7.90) and w.ticked == []
    assert all(r['kind'] == 'peak' for r in w.cand_rows)
    assert 7 in {r['rank'] for r in w.cand_rows}                      # 6.5 ppm, within 1.5
    assert '7.900' in w.sought_lbl.cget('text')
    assert 'disabled' in w.btn_save_lines.state()                     # no matched peak to write on
    w.win.destroy()


def test_double_clicking_a_signal_makes_it_the_reference(root):
    w = _window(root)
    target = _idx(w, 2.05)
    w.sig_tree.selection_set(str(target))
    w._ref_from_signal()
    assert w.ref_i == target and w.ref.ranks == [3]
    w.win.destroy()


def test_the_quantification_can_be_made_the_reference_again(root):
    w = _window(root, _browser(quant=True))
    w.sig_tree.selection_set(str(_idx(w, 2.05)))
    w._ref_from_signal()
    assert w.ref.kind == 'signal'
    w.sig_tree.selection_set('q')
    w._ref_from_signal()
    assert w.ref_i == 'q' and w.ref.kind == 'quant'
    w.win.destroy()


def test_a_signal_that_was_not_detected_cannot_be_the_reference(root, monkeypatch):
    from gui.widgets import stocsy_window as mod
    shown = []
    monkeypatch.setattr(mod.messagebox, 'showinfo', lambda *a, **k: shown.append(a))
    w = _window(root)
    before = w.ref_i
    w.sig_tree.selection_set(str(_idx(w, 7.90)))
    w._ref_from_signal()
    assert shown and w.ref_i == before
    w.win.destroy()


def test_a_wider_window_lists_more_peaks(root):
    w = _window(root)
    n = len(w.cand_rows)
    w.hw_var.set('5')
    w._rebuild()
    assert len(w.cand_rows) > n
    w.win.destroy()


def test_the_protons_typed_for_the_reference_change_the_quantification_gap(root):
    w = _window(root)
    gap = w.cand_rows[0]['gap']
    w.h_ref_var.set('6')
    w._rebuild()
    assert w.cand_rows[0]['gap'] != pytest.approx(gap)
    assert 'set by hand' in w.ref_lbl.cget('text')
    w.win.destroy()


# ── ticking lines ────────────────────────────────────────────────────────────

def test_unticking_a_line_of_the_doublet_halves_the_signal_and_warns(root):
    w = _window(root, _browser(quant=True))
    _click_signal(w, root, 4.10)
    assert abs(w.cand_rows[0]['gap']) < 0.08
    w._toggle(8)
    assert w.ticked == [2] and w.cand_rows[0]['gap'] == pytest.approx(-0.5, abs=0.08)
    assert w.cand_rows[0]['share'] == pytest.approx(0.5)
    assert 'under 75%' in w.sought_lbl.cget('text')
    w._toggle(8)
    assert sorted(w.ticked) == [2, 8] and 'under' not in w.sought_lbl.cget('text')
    w.win.destroy()


def test_ticking_another_compounds_peak_moves_the_signal_away(root):
    w = _window(root, _browser(quant=True))
    _click_signal(w, root, 4.10)
    good = w.cand_rows[0]['distance']
    w.win_tree.selection_set('p10')
    w._toggle_selected()
    assert 10 in w.ticked and w.cand_rows[0]['distance'] > 3 * good
    assert w.win_tree.item('p10')['values'][0] == '☑'
    w.win.destroy()


def test_a_click_on_the_in_signal_cell_ticks_the_peak(root):
    w = _window(root, _browser(quant=True))
    _click_signal(w, root, 4.10)
    w.win.deiconify()
    w.nb.select(1)
    root.update()
    tree = w.win_tree
    tree.see('p10')
    root.update()
    bbox = tree.bbox('p10', '#1')
    if not bbox:
        pytest.skip('the row is not drawn on this display')

    class Ev:
        x, y = bbox[0] + 5, bbox[1] + bbox[3] // 2
    assert w._on_win_click(Ev) == 'break' and 10 in w.ticked
    w.win.destroy()


def test_a_flat_peak_cannot_be_ticked(root, monkeypatch):
    from gui.widgets import stocsy_window as mod
    shown = []
    monkeypatch.setattr(mod.messagebox, 'showinfo', lambda *a, **k: shown.append(a))
    w = _window(root)
    _click_signal(w, root, 1.30)
    w._toggle(6)
    assert shown and 6 not in w.ticked
    w.win.destroy()


# ── saving the analyst's lines ───────────────────────────────────────────────

def _editable(root, monkeypatch, answer=True):
    from controllers.signal_lines import SignalLineEditor
    from gui.widgets import stocsy_window as mod
    from tests.unit.test_stocsy_analytes import _mets
    asked, changed = [], []
    monkeypatch.setattr(mod.messagebox, 'askyesno',
                        lambda *a, **k: asked.append((a, k)) or answer)
    mets, marks = _mets(), {}
    w = _window(root, _browser(quant=True, mets=mets), editor=SignalLineEditor(mets, marks),
                on_change=changed.append)
    _click_signal(w, root, 4.10)
    return w, mets, marks, asked, changed


def test_saving_writes_the_ticked_lines_on_the_annotation_and_marks_the_signal_good(root, monkeypatch):
    from processing.signal_lines import manual_lines
    w, mets, marks, asked, changed = _editable(root, monkeypatch)
    w._toggle(8)
    w._toggle(10)                                       # the analyst's choice: 4.115 and 4.095
    w._save_lines()
    mp = mets[0]['matched_peaks'][1]
    assert manual_lines(mp) == pytest.approx([4.115, 4.095])
    assert marks == {('MA', mp['peak_rank']): 'good'} and changed == ['MA']
    assert sorted(w.ticked) == [2, 10]                  # read again from the annotation
    assert '✎' in w.an_tree.item('MA')['text'] and 'by hand' in w.sought_lbl.cget('text')
    assert 'run the Quantification again' in w.sought_lbl.cget('text')
    assert asked and asked[0][1].get('icon') == 'question'
    w.win.destroy()


def test_saving_under_three_quarters_of_the_pattern_warns_that_it_will_not_be_scaled(root, monkeypatch):
    w, mets, _marks, asked, _changed = _editable(root, monkeypatch, answer=False)
    w._toggle(8)
    w._save_lines()
    assert asked[0][1].get('icon') == 'warning' and 'under the 75%' in asked[0][0][1]
    assert 'manual_lines_ppm' not in mets[0]['matched_peaks'][1]        # answered no
    w.win.destroy()


def test_restoring_gives_the_signal_back_its_lines(root, monkeypatch):
    w, mets, marks, _asked, changed = _editable(root, monkeypatch)
    w._toggle(8)
    w._save_lines()
    assert str(w.btn_restore_lines.state()) != "('disabled',)"
    w._restore_lines()
    assert 'manual_lines_ppm' not in mets[0]['matched_peaks'][1] and marks == {}
    assert sorted(w.ticked) == [2, 8] and changed == ['MA', 'MA']
    assert 'disabled' in w.btn_restore_lines.state()
    w.win.destroy()


def test_without_an_editor_the_window_only_reads(root):
    w = _window(root, _browser(quant=True))
    assert 'disabled' in w.btn_save_lines.state() and 'disabled' in w.btn_restore_lines.state()
    w.win.destroy()


# ── looking at a peak ────────────────────────────────────────────────────────

def test_selecting_a_peak_marks_it_on_the_spectrum_and_draws_it_against_the_reference(root):
    w = _window(root)
    _click_signal(w, root, 1.30)
    w.win_tree.selection_set('p4')                                  # the other molecule beside it
    w.win_tree.event_generate('<<TreeviewSelect>>')
    root.update()
    assert w.focus == 4
    dashed = [ln for ln in w.ax_var.lines if ln.get_linestyle() == '--']
    assert dashed and 'peak #4' in dashed[0].get_label()
    w.win.destroy()


def test_a_click_on_a_peak_of_the_spectrum_selects_its_row(root):
    w = _window(root)
    _click_signal(w, root, 1.30)

    class Ev:
        inaxes = w.ax_spec
        xdata = 1.33
        button = 1
    w._on_spec_click(Ev)
    assert w.focus == 4 and w.win_tree.selection() == ('p4',)
    Ev.xdata = 9.9                                                    # nothing there
    w._on_spec_click(Ev)
    assert w.focus == 4
    w.win.destroy()


def test_the_variation_tab_draws_the_reference_the_ticked_signal_and_every_signal(root):
    w = _window(root, _browser(quant=True))
    labels = [ln.get_label() for ln in w.ax_var.lines]
    assert sum('reference' in x for x in labels) == 1
    assert any(x.startswith('ticked lines') for x in labels) and len(labels) >= 4
    assert 'H' in w.ax_var.get_ylabel()
    w.win.destroy()


def test_the_matrix_tab_correlates_the_quantification_and_the_signals_of_the_analyte(root):
    w = _window(root, _browser(quant=True))
    names = [t.get_text() for t in w.ax_mat.get_xticklabels()]
    assert len(names) == 4 and names[0].startswith('quantification')
    w.win.destroy()


# ── without an analyte ───────────────────────────────────────────────────────

def test_without_an_annotation_any_peak_can_be_the_reference(root):
    from controllers.stocsy_analytes import AnalyteBrowser
    from tests.unit.test_stocsy_analytes import _study
    w = _window(root, _ab(_study()))
    assert w.met_id is None and w.ref is not None and w.sought is not None
    assert w.ref.ranks == [1]                                         # the tallest in view
    assert 'No annotation' in w.an_tree.item(w.an_tree.get_children()[0])['text']
    assert 'assumed 1' in w.ref_lbl.cget('text')
    w.win.destroy()


def test_picking_a_peak_as_reference_leaves_the_analyte_and_keeps_the_window(root):
    w = _window(root)
    _click_signal(w, root, 1.30)
    w.win_tree.selection_set('p4')
    w._ref_from_candidate()
    assert w.met_id is None and w.ref.ranks == [4]
    assert w.an_tree.selection() == ()
    assert any(r['kind'] == 'peak' for r in w.cand_rows)
    w.win.destroy()


def test_searching_around_a_peak_moves_the_window_to_it(root):
    w = _window(root)
    _click_signal(w, root, 1.30)
    w.win_tree.selection_set('p4')
    w._sought_from_candidate()
    assert w.met_id is None and w.sought.ranks == [4] and w.ticked == [4]
    w.win.destroy()


def test_the_protons_typed_without_an_analyte_are_the_ones_used(root):
    from controllers.stocsy_analytes import AnalyteBrowser
    from tests.unit.test_stocsy_analytes import _study
    w = _window(root, _ab(_study()))
    w.h_ref_var.set('3')
    w.h_sought_var.set('2')
    w._rebuild()
    assert w.ref.protons == 3 and w.sought.protons == 2 and w.cand_rows[0]['protons'] == 2
    w.win.destroy()


# ── the tables ───────────────────────────────────────────────────────────────

def test_a_heading_sorts_the_peaks_and_a_second_click_reverses_it(root):
    w = _window(root)
    _click_signal(w, root, 2.05)
    w.hw_var.set('5')
    w._rebuild()
    w._sort_by('snr')
    kids = w.win_tree.get_children()
    assert kids[:2] == ('ref', 'whole')
    nums = [float(w.win_tree.item(i)['values'][3]) for i in kids[2:]
            if w.win_tree.item(i)['values'][3] != '—']
    assert nums == sorted(nums)
    w._sort_by('snr')
    again = [float(w.win_tree.item(i)['values'][3]) for i in w.win_tree.get_children()[2:]
             if w.win_tree.item(i)['values'][3] != '—']
    assert again == sorted(again, reverse=True)
    w.win.destroy()


def test_the_values_tab_lists_the_numbers_of_every_sample_with_the_reference_first(root):
    w = _window(root)
    tree = next(c for c in w.val_frame.winfo_children() if c.winfo_class() == 'Treeview')
    assert len(list(tree['columns'])) == 5 + len(w.an.sample_names)
    items = tree.get_children()
    assert tree.item(items[0])['values'][1] == 'group'
    assert tree.item(items[1])['values'][0] == '◉' and tree.item(items[2])['values'][0] == 'Σ'
    w.per_var.set('proton')
    w._fill_values()
    tree = next(c for c in w.val_frame.winfo_children() if c.winfo_class() == 'Treeview')
    assert float(str(tree.item(tree.get_children()[1])['values'][5]).replace(',', '')) > 0
    w.win.destroy()


# ── options, notes, export ───────────────────────────────────────────────────

def test_changing_the_options_redraws_and_the_analyte_list_follows_the_correlation(root):
    w = _window(root)
    mean_r = w.an_tree.item('MA')['values'][3]
    w.norm_var.set('PQN')
    w.method_var.set('Spearman')
    w._on_options()
    assert w.an.method == 'spearman' and w.an.normalize == 'pqn'
    assert w.an_tree.item('MA')['values'][3] != '' and mean_r != ''
    w.win.destroy()


def test_within_group_is_disabled_for_a_study_with_one_group(root):
    from controllers.stocsy_analytes import AnalyteBrowser
    from controllers.stocsy_controller import StocsyAnalysis
    from gui.widgets.stocsy_window import StocsyWindow
    from tests.unit.test_stocsy_controller import _study
    ppm, X, peaks, info = _study()
    for i in info:
        i['label'] = 'G'
    an = StocsyAnalysis(ppm, X, peaks, info)
    w = StocsyWindow(root, an, browser=AnalyteBrowser(an, measure=detection_areas(an)))
    assert 'disabled' in w.within_chk.state()
    w.win.destroy()


def test_export_writes_one_row_per_line_with_every_sample(root, tmp_path, monkeypatch):
    import csv
    from gui.widgets import stocsy_window as mod
    out = tmp_path / 'c.csv'
    monkeypatch.setattr(mod.filedialog, 'asksaveasfilename', lambda **k: str(out))
    w = _window(root)
    w._export()
    rows = list(csv.DictReader(open(out, encoding='utf-8')))
    assert len(rows) == len(w._window_rows()) and rows[0]['reference'] == w.ref.label
    assert any(k.startswith('S0 ') for k in rows[0])
    assert {'distance', 'ratio_to_reference', 'in_signal', 'line_share'} <= set(rows[0])
    w.win.destroy()


def test_reading_and_ticking_does_not_change_the_peaks_or_the_annotation(root):
    import copy
    from tests.unit.test_stocsy_analytes import _mets
    mets = _mets()
    snapshot = copy.deepcopy(mets)
    b = _browser(quant=True, mets=mets)
    before = {k: np.array(p['area_per_sample']) for k, p in b.an.peaks.items()}
    w = _window(root, b)
    _click_signal(w, root, 4.10)
    w._toggle(10)
    w.norm_var.set('PQN')
    w._on_options()
    for k, a in before.items():
        assert np.array_equal(a, b.an.peaks[k]['area_per_sample'], equal_nan=True)
    assert mets == snapshot
    w.win.destroy()


def test_it_reopens_on_the_state_it_was_left_in(root):
    w = _window(root)
    w.sig_tree.selection_set(str(_idx(w, 2.05)))
    w._ref_from_signal()
    state = w.state()
    w.win.destroy()
    w2 = _window(root, _browser(), initial=state)
    assert w2.met_id == 'MA' and w2.ref_i == state['ref_i'] and w2.ref.ranks == [3]
    w2.win.destroy()


# ── the stage: the correlation is rebuilt and the analyst is told ───────────

def _stage_like(samples_info, peaks, n_cols):
    from types import SimpleNamespace
    app = SimpleNamespace(all_intensities=np.zeros((10, n_cols)),
                          samples_info=samples_info)
    return SimpleNamespace(app=app, peaks=peaks, _peak_corr=None,
                           _n_samples_pk=None)


def _rebuild(stage):
    from gui.stages.peak_detection_stage import PeakDetectionStage
    return PeakDetectionStage.build_peak_correlation(stage)


def test_blanks_do_not_count_as_samples_for_the_scoring_weight():
    from tests.unit.test_stocsy_controller import _study
    ppm, X, peaks, info = _study(blanks=3)
    st = _stage_like(info, peaks, X.shape[1])
    _rebuild(st)
    assert X.shape[1] == 15 and st._n_samples_pk == 12
    assert st._peak_corr and (1, 2) in st._peak_corr


def test_too_few_samples_says_so_and_builds_nothing():
    from tests.unit.test_stocsy_controller import _study
    ppm, X, peaks, info = _study()
    few_info = info[:4] + [dict(i, excluded=True) for i in info[4:]]
    st = _stage_like(few_info, peaks, 12)
    warn = _rebuild(st)
    assert 'Only 4 analytical' in warn and 'at least 5' in warn
    assert st._peak_corr == {} and st._n_samples_pk == 4


def test_left_out_signals_are_named_in_the_warning():
    from tests.unit.test_stocsy_controller import _study
    ppm, X, peaks, info = _study()
    st = _stage_like(info, peaks, 12)
    warn = _rebuild(st)
    assert '1 signal(s) left out' in warn and 'NaN' in warn   # rank 5
    assert not any(5 in pair for pair in st._peak_corr)
    assert not any(4 in pair for pair in st._peak_corr)       # flat: no pair


# ── modules and samples ─────────────────────────────────────────────────

def _structured():
    from tests.unit.test_stocsy_structure_controller import _an
    return _an()


def test_the_structure_window_fills_its_three_tables(root):
    from gui.widgets.stocsy_structure_window import StocsyStructureWindow
    an = _structured()
    w = StocsyStructureWindow(root, an)
    w.snr_var.set(10.0)
    w.refresh()
    assert len(w.mod_tree.get_children()) >= 3
    assert len(w.axis_tree.get_children()) == 4
    assert len(w.sam_tree.get_children()) == 32
    assert 'raw areas' in w.basis_lbl.cget('text')
    kinds = {w.mod_tree.item(i)['values'][9] for i in w.mod_tree.get_children()}
    assert {'reference', 'water tail'} <= kinds
    w.win.destroy()


def test_picking_a_module_lists_its_members_and_sends_them(root):
    from gui.widgets.stocsy_structure_window import StocsyStructureWindow
    got = []
    w = StocsyStructureWindow(root, _structured(), on_send=got.append)
    w.snr_var.set(10.0)
    w.refresh()
    first = w.mod_tree.get_children()[0]
    w.mod_tree.selection_set(first)
    w._on_module()
    assert len(w.mem_tree.get_children()) == w._modules[0]['n']
    assert 'Module score by group' in w.means_lbl.cget('text')
    w._send()
    assert got and got[0] == w._modules[0]['ranks']
    w.win.destroy()


def test_the_structure_window_follows_the_options_of_the_stocsy_window(root):
    from gui.widgets.stocsy_structure_window import StocsyStructureWindow
    an = _structured()
    w = StocsyStructureWindow(root, an)
    an.set_options(normalize='pqn', within_groups=True, method='spearman')
    w.refresh()
    text = w.basis_lbl.cget('text')
    assert 'spearman' in text and 'PQN' in text and 'within groups' in text
    w.win.destroy()


def test_sending_a_module_to_the_stocsy_window_shows_its_signals_around_the_clearest(root):
    from controllers.stocsy_analytes import AnalyteBrowser
    from gui.widgets.stocsy_window import StocsyWindow
    an = _structured()
    w = StocsyWindow(root, an, browser=AnalyteBrowser(an, measure=detection_areas(an)), view_region=(9.0, 6.0))
    w._send_signals([6, 7, 8])                       # the reference signals
    assert w.met_id is None and w.ref.ranks == [8]
    assert {r['rank'] for r in w.cand_rows if r['kind'] == 'peak'} >= {6, 7, 8}
    assert w._hw() > 0.1
    w.win.destroy()
