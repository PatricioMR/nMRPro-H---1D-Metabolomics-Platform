# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The search runs without a window.

That is the whole point of the module: while it was a nested function of
the method that opens the dialogs, the only way to run it was to open
one, the bench could not call it and grew a second annotation of its
own, and two defects lived for months in the path only the window took.
"""
import inspect

import pytest

from controllers.annotation_controller import (
    IncompatibleDatabase, SearchOutcome, search_database)


def test_the_search_takes_data_and_returns_an_answer():
    sig = inspect.signature(search_database)
    for name in ('db_path', 'peaks_to_search', 'ppm', 'intensities',
                 'samples_info', 'field_mhz', 'tolerance_ppm'):
        assert name in sig.parameters, name
    # what detection measured is passed, not read off a stage
    for name in ('peak_correlation', 'n_samples_correlated',
                 'min_distance_ppm', 'noise_std', 'min_snr'):
        assert name in sig.parameters, name
    # and what it says while it works goes out through callbacks
    assert 'progress' in sig.parameters and 'status' in sig.parameters


def test_nothing_of_the_window_is_left_in_it():
    src = inspect.getsource(search_database)
    for forbidden in ('self.', 'messagebox', 'tkinter', '.after(', 'pw.'):
        assert forbidden not in src, forbidden


def test_a_file_that_is_not_a_reference_database_raises(tmp_path):
    db = tmp_path / 'empty.db'
    import sqlite3
    sqlite3.connect(db).execute('CREATE TABLE plain_table (a)')
    with pytest.raises(IncompatibleDatabase):
        search_database(
            db_path=str(db), db_type='hmdb', peaks_to_search=[{'ppm': 1.0}],
            annotatable_peaks=[{'ppm': 1.0}], tiers={'Good'},
            ppm=None, intensities=None, samples_info=[],
            field_mhz=600.0, tolerance_ppm=0.02)


def test_the_outcome_carries_what_it_had_to_work_with():
    out = SearchOutcome(results=[1, 2], peaks_annotatable=10, peaks_matched=7,
                        elapsed_s=1.5)
    assert (len(out.results), out.peaks_annotatable, out.peaks_matched) == (2, 10, 7)


class TestTheApplicationPathRunsHeadless:
    """The bench annotates with the application's own search and passes,
    so its numbers are a statement about the software (ADR-030/032)."""

    def test_the_whole_path_is_reachable_without_a_window(self):
        from validation.app_annotation import annotate, PASSES
        import inspect
        sig = inspect.signature(annotate)
        for name in ('peaks', 'ppm', 'intensities', 'samples_info',
                     'db_path', 'field_mhz', 'tolerance_ppm'):
            assert name in sig.parameters, name
        # the passes the results window runs after the search, in its order
        assert PASSES[0] == '_fetch_fingerprints'
        for needed in ('_compute_scores', '_auto_mark_good',
                       '_promote_singlets', '_apply_fingerprint_gate',
                       '_nnls_spectral_validation'):
            assert needed in PASSES, needed

    def test_the_bench_can_be_told_to_use_it(self):
        import inspect
        from validation import mixture_bench
        assert 'real_annotation' in inspect.signature(mixture_bench.run).parameters

    def test_nothing_of_the_window_is_needed_to_run_them(self):
        """Every pass reads data and writes bookkeeping; if one starts
        touching a widget, this says so before the bench does."""
        import inspect
        from gui.widgets.annotation_results_window import AnnotationResultsWindow
        from validation.app_annotation import PASSES
        for name in PASSES:
            src = inspect.getsource(getattr(AnnotationResultsWindow, name))
            for forbidden in ('self.win', 'self.tree', 'messagebox',
                              'ttk.', 'tk.Toplevel'):
                assert forbidden not in src, f'{name} touches {forbidden}'
