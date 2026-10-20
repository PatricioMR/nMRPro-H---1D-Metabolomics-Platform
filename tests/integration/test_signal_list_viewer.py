# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The Signal List's viewer draws the spectrum the numbers were measured on and
shades the area that was measured.

It used to draw the raw spectrum from a copy taken when the window opened, and
fill from zero up to the highest sample: a region nobody measured, 5.75 times
the Area beside it on the reference study.
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
    except tk.TclError:
        pytest.skip('no display')
    except Exception as e:
        pytest.skip(f'the application cannot be built here: {e}')
    app.root.withdraw()
    try:
        app._build_ui()
    except Exception as e:
        pytest.skip(f'the application cannot be built here: {e}')
    yield app
    try:
        app.root.destroy()
    except Exception:
        pass


PPM = np.linspace(4.0, 0.0, 16000)


def _lorentz(c, h, w=0.003):
    return h * w ** 2 / ((PPM - c) ** 2 + w ** 2)


def _hump():
    return 30.0 * np.exp(-((PPM - 2.0) / 0.5) ** 2)


@pytest.fixture
def listed(real_app):
    """Three samples with a pedestal under a signal, a project baseline model
    that takes it out, and the peak measured the way detection measures it."""
    from processing.baseline_model import BaselineModel
    from processing.peak_detection import peak_measures
    from gui.widgets.peak_list_window import PeakListWindow
    app = real_app
    X = np.column_stack([_lorentz(2.0, 100.0 * k) + _hump() for k in (1, 2, 3)])
    app.ppm = PPM
    app.all_intensities = X.copy()
    app.samples_info = [{'label': 'A', 'type': 'Sample', 'name': f'S{i}'}
                        for i in range(3)]
    app.baseline_corrected = False
    app.baseline_model = BaselineModel(enabled=True, method='snip',
                                       params={'niter': 40})
    app.invalidate_baseline_cache()
    corr = np.asarray(app.corrected_intensities(), dtype=float)
    i = int(np.argmin(np.abs(PPM - 2.0)))
    s, e = i - 150, i + 150
    m = peak_measures(PPM, corr, s, e, 1.0, index=i)
    peak = {'rank': 1, 'ppm': m['ppm'], 'index': m['index'],
            'start_idx': s, 'end_idx': e, 'start_ppm': float(PPM[s]),
            'end_ppm': float(PPM[e]), 'width_ppm': float(abs(PPM[e] - PPM[s])),
            'area': m['area'], 'height': m['height'], 'snr': m['snr'],
            'mean_intensity': m['mean_intensity'],
            'area_per_sample': m['area_per_sample'], 'status': None,
            'signal_quality': 'Good', 'signal_quality_score': 0.9}
    win = PeakListWindow(app.root, [peak], PPM, X, app=app)
    win.win.withdraw()
    yield win, peak, app, X
    try:
        win.win.destroy()
    except Exception:
        pass


def _polygon_area(coll):
    total = 0.0
    for path in coll.get_paths():
        v = path.vertices
        x, y = v[:, 0], v[:, 1]
        total += 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return total


def test_the_shaded_area_is_the_area_in_the_table(listed):
    win, peak, app, X = listed
    win._view_var.set('corrected')
    win._plot_peak(peak)
    shaded = _polygon_area(win.ax.collections[0])
    assert shaded == pytest.approx(peak['area'], rel=0.02)
    # and the number written beside it is the table's, to the digit
    cap = next(x.get_text() for x in win.ax.texts
               if x.get_text().startswith('Area of the shaded region'))
    assert f"{peak['area']:.4g}" in cap and 'differs' not in cap


def test_the_corrected_and_the_raw_views_draw_different_spectra(listed):
    win, peak, app, X = listed
    win._view_var.set('corrected')
    win._plot_peak(peak)
    corr_y = max(np.max(l.get_ydata()) for l in win.ax.get_lines()
                 if len(l.get_ydata()) > 50)
    labels_corr = [l.get_label() for l in win.ax.get_lines()]
    win._view_var.set('raw')
    win._plot_peak(peak)
    raw_y = max(np.max(l.get_ydata()) for l in win.ax.get_lines()
                if len(l.get_ydata()) > 50)
    labels_raw = [l.get_label() for l in win.ax.get_lines()]
    assert raw_y > corr_y, 'the pedestal is in the raw spectrum only'
    assert not any('baseline model' in str(x) for x in labels_corr)
    assert any('baseline model' in str(x) for x in labels_raw)


def test_even_on_the_raw_spectrum_the_shading_is_above_the_line_not_above_zero(listed):
    win, peak, app, X = listed
    win._view_var.set('raw')
    win._plot_peak(peak)
    shaded = _polygon_area(win.ax.collections[0])
    from_zero = abs(np.trapezoid(X[peak['start_idx']:peak['end_idx'] + 1].mean(axis=1),
                                 PPM[peak['start_idx']:peak['end_idx'] + 1]))
    assert shaded < 0.6 * from_zero, 'a pedestal is not area'


def test_the_viewer_reads_the_spectrum_the_application_holds_now(listed):
    win, peak, app, X = listed
    win._view_var.set('raw')
    win._plot_peak(peak)
    before = max(np.max(l.get_ydata()) for l in win.ax.get_lines()
                 if len(l.get_ydata()) > 50)
    app.all_intensities = X * 2.0                # an alignment, a new detection...
    app.invalidate_baseline_cache()
    win._plot_peak(peak)
    after = max(np.max(l.get_ydata()) for l in win.ax.get_lines()
                if len(l.get_ydata()) > 50)
    assert after == pytest.approx(2.0 * before, rel=0.01)


def test_the_caption_says_when_the_shading_and_the_column_differ(listed):
    win, peak, app, X = listed
    win._view_var.set('corrected')
    win._plot_peak(peak)
    texts = [t.get_text() for t in win.ax.texts]
    cap = next(t for t in texts if t.startswith('Area of the shaded region'))
    assert 'differs' not in cap
    peak['area'] = peak['area'] * 3.0
    win._plot_peak(peak)
    cap = next(t.get_text() for t in win.ax.texts if t.get_text().startswith('Area of the shaded region'))
    assert 'differs' in cap


def test_a_moved_boundary_measures_the_signal_again(listed):
    win, peak, app, X = listed
    win._current_peak = peak
    old_area, old_snr = peak['area'], peak['snr']
    i = peak['index']
    lo, hi = float(PPM[i + 60]), float(PPM[i - 60])      # a narrower window
    win._on_span_selected(lo, hi)
    assert peak['boundaries_edited'] is True
    assert peak['area'] < old_area and peak['area'] > 0
    assert peak['start_idx'] > i - 150 and peak['end_idx'] < i + 150
    from processing.peak_detection import peak_measures
    corr = np.asarray(app.corrected_intensities(), dtype=float)
    m = peak_measures(PPM, corr, peak['start_idx'], peak['end_idx'], None,
                      index=peak['index'])
    assert peak['area'] == pytest.approx(m['area'])
    assert 'orig. bounds' in win._quality_text(peak)


def test_a_manual_mark_is_not_called_an_old_grade(listed):
    win, peak, app, X = listed
    peak['boundaries_edited'] = True
    peak['status'] = 'Good'
    assert 'orig. bounds' not in win._quality_text(peak)


class _Click:
    def __init__(self, ax, x):
        self.button, self.inaxes, self.xdata = 1, ax, x


def test_split_cuts_the_selected_window_in_two_and_undo_merges_it(listed):
    win, peak, app, X = listed
    win._current_peak = peak
    s, e = peak['start_idx'], peak['end_idx']
    win._arm_edit('split')
    win._on_plot_click(_Click(win.ax, float(PPM[peak['index'] + 60])))
    assert len(win.peaks) == 2 and win._edit_mode is None
    piece = next(p for p in win.peaks if p is not peak)
    assert peak['rank'] == 1 and piece['rank'] == 2
    assert {peak['start_idx'], piece['end_idx']} == {s, e}
    assert len(win.tree.get_children()) == 2
    assert '✂ from #1' in win._quality_text(piece)
    assert 'run them again' in win._stale_lbl.cget('text')
    win._undo_split(piece)
    assert win.peaks == [peak]
    assert (peak['start_idx'], peak['end_idx']) == (s, e)


def test_a_click_without_split_armed_changes_nothing(listed):
    win, peak, app, X = listed
    win._current_peak = peak
    win._on_plot_click(_Click(win.ax, float(PPM[peak['index'] + 60])))
    assert len(win.peaks) == 1


def test_new_signal_is_drawn_by_a_right_drag_and_refused_over_another(
        listed, monkeypatch):
    import gui.widgets.peak_list_window as plw
    warned = []
    monkeypatch.setattr(plw.messagebox, 'showwarning',
                        lambda *a, **k: warned.append(a))
    win, peak, app, X = listed
    win._current_peak = peak
    win._arm_edit('new')
    win._on_span_selected(1.98, 2.02)               # over signal #1
    assert len(win.peaks) == 1 and warned
    win._on_span_selected(0.98, 1.02)
    assert len(win.peaks) == 2 and win._edit_mode is None
    added = win.peaks[-1]
    assert added['manual_origin'] == 'added' and added['rank'] == 2
    win._remove_added(added)
    assert win.peaks == [peak]


def test_the_axes_keep_the_panel_width_whatever_the_captions_say(listed):
    """tight_layout counted the captions drawn in axes coordinates as part of
    the axes: in a narrow panel the plot collapsed to a sliver beside them."""
    win, peak, app, X = listed
    for w_in in (4.8, 3.6):
        win.fig.set_size_inches(w_in, 4.5)
        win._plot_peak(peak)
        assert win.ax.get_position().width > 0.6, w_in
