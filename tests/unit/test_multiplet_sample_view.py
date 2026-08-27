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
    # The processing helper that derives it is no longer reachable from here.
    assert not hasattr(mod, 'signal_in_sample')
    assert 'processing.per_sample_signal' not in inspect.getsource(mod)


def test_a_project_without_readings_says_so():
    """Dashes and a warning beat this window inventing a second set of numbers."""
    import inspect
    from gui.widgets.multiplet_sample_view import MultipletSampleView as V
    src = inspect.getsource(V._build)
    assert 'has_readings' in src
    assert 'annotation search again' in src
