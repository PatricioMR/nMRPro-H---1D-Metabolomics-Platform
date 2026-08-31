"""Tests for the Tk-free math of the per-multiplet across-samples view."""

from gui.widgets.multiplet_sample_view import (
    parse_j_hz, multiplet_window,
)

MHZ = 600.0


def test_parse_j_hz():
    assert parse_j_hz('7.1 Hz') == 7.1
    assert parse_j_hz(7) == 7.0
    assert parse_j_hz(None) is None
    assert parse_j_hz('n/a') is None


def test_multiplet_window_covers_predicted_lines():
    mp = {'peak_ppm': 0.90,
          'predicted_lines': [{'ppm': 0.888}, {'ppm': 0.900}, {'ppm': 0.912}]}
    dp = {'type': 't', 'j_coupling': '7.0 Hz', 'ppm_center': 0.90}
    lo, hi, centre, db_j, mult = multiplet_window(mp, None, dp, MHZ)
    assert lo < 0.888 and hi > 0.912
    assert abs(centre - 0.90) < 1e-9
    assert db_j == 7.0 and mult == 't'


def test_plot_width_is_derived_from_the_table_columns():
    """The plot is pinned to the table width so a trace sits above its own row.
    Deriving it from COLS means adding a column moves both."""
    from gui.widgets.multiplet_sample_view import COLS, TABLE_WIDTH_PX
    assert TABLE_WIDTH_PX == sum(c[2] for c in COLS) + 18


def test_each_group_is_labelled_once():
    """A per-sample legend over 32 traces covers the plot it explains."""
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._draw_overlay)
    assert 'labelled' in src
    assert 'label=lbl' in src



def test_the_window_displays_and_does_not_measure():
    """This window renders what the annotation search settled. Nothing here may
    re-derive a number: when it did, the crop-based J it produced disagreed with
    the pipeline's — three signals sharing a region all read 4.76 Hz.
    """
    import inspect
    import gui.widgets.multiplet_sample_view as mod
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V

    src = inspect.getsource(V._precompute)
    assert "self.mp_ref.get('per_sample')" in src, 'rows must come from the signal'
    for banned in ('measure_j', 'measure_sample', 'estimate_noise',
                   'signal_in_sample', 'find_peaks'):
        assert banned not in src, f'_precompute must not call {banned}'

    # The measuring helpers are gone from the module, not merely unused.
    for banned in ('measure_j', 'measure_sample'):
        assert not hasattr(mod, banned), f'{banned} still lives in the view'


def test_alignment_shift_is_read_not_recomputed():
    import inspect
    import gui.widgets.multiplet_sample_view as mod
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._alignment_shifts)
    assert "r['shift']" in src, 'the shift must come from the stored row'
    # The functions that DERIVE a reading are not reachable from here. Summary
    # statistics over what was stored are not a measurement and may be imported.
    for banned in ('signal_in_sample', 'alignment_shifts', 'signal_across_samples'):
        assert not hasattr(mod, banned), f'{banned} still reachable from the view'


def test_a_project_without_readings_says_so():
    """Dashes and a warning beat this window inventing a second set of numbers."""
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._build)
    assert 'has_readings' in src
    assert 'annotation search again' in src


def test_each_measured_quantity_is_followed_by_its_spread():
    from gui.widgets.multiplet_sample_view import COLS
    keys = [c[0] for c in COLS]
    assert keys.index('ppm_dp') == keys.index('ppm') + 1
    assert keys.index('jobs_dp') == keys.index('jobs') + 1


def test_the_table_reports_measured_values_not_differences():
    """The per-sample columns carry what was measured in that sample — its δ
    and its J. The only Δ is the signal's own, decided by the search."""
    from gui.widgets.multiplet_sample_view import COLS
    heads = {c[0]: c[1] for c in COLS}
    assert heads['ppm'] == 'δ (ppm)'
    assert heads['jobs'] == 'J (Hz)'
    assert 'signal' in heads['dj']
    assert 'dppm' not in heads


def test_the_signal_delta_j_is_read_from_the_search():
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._populate)
    assert "self.mp_ref.get('delta_j_hz')" in src
    # Not recomputed from the per-sample J.
    assert "- self.db_j" not in src


def test_the_row_spread_is_distance_from_the_sample_own_group():
    """Chemical shift moves with pH and ionic strength — what the groups differ
    in — so a study-wide mean would call an entire group an outlier."""
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._populate)
    assert "group_means(present_rows, 'ppm_centre')" in src
    assert "group_means(present_rows, 'j_obs')" in src
    assert "percent_deviation(r['ppm_centre'], ppm_mean.get(r['group']))" in src


def test_the_group_summary_reports_that_group_spread():
    """A mean's distance from itself is zero; the cell carries the group CV."""
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._populate_summary)
    assert '_cv(centres)' in src and '_cv(jobs)' in src
    assert '_cv(djs)' not in src


def test_the_symbols_say_which_quantity_is_which():
    """Δ is the difference; lowercase δ is the chemical shift itself."""
    from gui.widgets.multiplet_sample_view import COLS
    heads = {c[0]: c[1] for c in COLS}
    assert heads['ppm'].startswith('δ')      # δ — the shift
    assert heads['dj'].startswith('Δ')       # Δ — the difference



def test_the_window_shows_the_pipeline_verdict_not_one_of_its_own():
    """The signal has ONE verdict, reached by the annotation search. Classifying
    each sample here invented a second one, and the two contradicted each other:
    a signal scored 0.99 and Good showed every sample as ΔJ-rejected.
    """
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    assert not hasattr(V, '_sample_verdict')
    src = inspect.getsource(V._pipeline_verdict)
    assert "self.mp_ref.get('annotation_quality')" in src
    assert "self.mp_ref.get('delta_j_hz')" in src
    # No threshold is applied here — nothing is being decided.
    assert 'dj_gold_hz' not in src and 'dj_bad_hz' not in src


def test_the_detected_column_reports_detection_only():
    """All the lines, some of them, or none — detection's own answer. No
    threshold is applied: the window does not decide anything."""
    import inspect
    from gui.widgets.multiplet_sample_view import COLS, MultipletSampleView as V
    assert dict((c[0], c[1]) for c in COLS)['present'] == 'Detected'
    src = inspect.getsource(V._populate)
    assert "r['partial']" in src
    assert "f\"{r['n_found']}/{r['n_lines']}\"" in src
    assert 'dj_gold_hz' not in src and 'dj_bad_hz' not in src


def test_a_partial_reading_still_shows_its_numbers():
    """Half a multiplet is worth seeing; it just cannot confirm the signal."""
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._populate)
    # δ and J are formatted from the row, not gated on a complete multiplet.
    assert "pptxt = (f\"{r['ppm_centre']:.4f}\"" in src
    assert "if r['ppm_centre'] is not None else '—')" in src


def test_the_group_mean_uses_complete_readings_only():
    """A multiplet missing a leg has its centre displaced by half a coupling;
    letting that into the mean moves the reference the others are judged by."""
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._populate)
    assert "present_rows = [r for r in self._counted_rows() if r['present']]" in src
