# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Recovering a multiplet that line-matching missed.

The capability is legitimate — look in the DB window for the expected lines when
the first pass found none. What it must not do is accept any set of peaks that
happens to average out to the right spacing.
"""
import pytest

from processing.annotation_scoring import (
    J_SPACING_TOL_HZ, search_multiplet_in_window)

MHZ = 600.71


def _comb(centre, j_hz, n):
    j = j_hz / MHZ
    return [centre + (k - (n - 1) / 2) * j for k in range(n)]


class TestSearchMultipletInWindow:
    def test_finds_a_clean_triplet(self):
        peaks = _comb(3.0, 7.0, 3)
        got = search_multiplet_in_window(
            peaks, centre_ppm=3.0, j_hz=7.0, n_expected=3,
            field_mhz=MHZ, half_width_ppm=0.05)
        assert got is not None and len(got) == 3
        for k in range(2):
            assert (got[k + 1] - got[k]) * MHZ == pytest.approx(7.0, abs=0.1)

    def test_rejects_unequal_spacings_that_average_correctly(self):
        """The hole in the old mean-based test: 1.2 and 12.8 Hz average to 7.0."""
        j = 1.0 / MHZ
        peaks = [3.0, 3.0 + 1.2 * j, 3.0 + 14.0 * j]
        assert search_multiplet_in_window(
            peaks, centre_ppm=3.0, j_hz=7.0, n_expected=3,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_finds_the_comb_among_neighbours(self):
        peaks = _comb(3.0, 7.0, 3) + [2.98, 3.03, 3.041]
        got = search_multiplet_in_window(
            peaks, centre_ppm=3.0, j_hz=7.0, n_expected=3,
            field_mhz=MHZ, half_width_ppm=0.05)
        assert got is not None and len(got) == 3
        for k in range(2):
            assert (got[k + 1] - got[k]) * MHZ == pytest.approx(7.0, abs=J_SPACING_TOL_HZ)

    def test_too_few_peaks_returns_none(self):
        assert search_multiplet_in_window(
            [3.0, 3.01], centre_ppm=3.0, j_hz=7.0, n_expected=4,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_peaks_outside_the_window_are_ignored(self):
        peaks = _comb(3.0, 7.0, 3)
        assert search_multiplet_in_window(
            peaks, centre_ppm=5.0, j_hz=7.0, n_expected=3,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_quintet_needs_all_five(self):
        four = _comb(2.0, 7.0, 5)[:4]
        assert search_multiplet_in_window(
            four, centre_ppm=2.0, j_hz=7.0, n_expected=5,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_no_j_returns_none(self):
        assert search_multiplet_in_window(
            _comb(3.0, 7.0, 3), centre_ppm=3.0, j_hz=0.0, n_expected=3,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_makes_no_assumption_about_intensities(self):
        """Geometry only — the caller may check the pattern separately."""
        import inspect
        src = inspect.getsource(search_multiplet_in_window)
        assert 'rel_intensity' not in src and 'height' not in src


class TestSpacingsMustAgreeWithEachOther:
    """A multiplet is equally spaced; a drifting run of peaks is not one.

    Measured on a real dataset at the looser 2 Hz tolerance: five metabolites in
    one crowded region all matched the same peak triple, and a 'quartet' was
    accepted with spacings 11.65, 14.21 and 16.05 Hz for a DB J of 13.33.
    """

    def test_drifting_spacings_are_rejected(self):
        j = 1.0 / MHZ
        peaks = [2.4, 2.4 + 11.65 * j, 2.4 + 25.86 * j, 2.4 + 41.91 * j]
        assert search_multiplet_in_window(
            peaks, centre_ppm=2.42, j_hz=13.33, n_expected=4,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_uneven_triplet_is_rejected(self):
        j = 1.0 / MHZ
        peaks = [1.9, 1.9 + 7.25 * j, 1.9 + 12.22 * j]   # 7.25 then 4.97 Hz
        assert search_multiplet_in_window(
            peaks, centre_ppm=1.91, j_hz=6.93, n_expected=3,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_doublet_far_from_the_db_j_is_rejected(self):
        j = 1.0 / MHZ
        assert search_multiplet_in_window(
            [5.4, 5.4 + 1.7 * j], centre_ppm=5.4, j_hz=3.60, n_expected=2,
            field_mhz=MHZ, half_width_ppm=0.05) is None

    def test_an_evenly_spaced_multiplet_still_passes(self):
        peaks = _comb(2.4, 13.33, 4)
        got = search_multiplet_in_window(
            peaks, centre_ppm=2.4, j_hz=13.33, n_expected=4,
            field_mhz=MHZ, half_width_ppm=0.05)
        assert got is not None and len(got) == 4

    def test_tolerance_is_tied_to_the_dj_rejection_gate(self):
        from processing.annotation_scoring import SEARCH_SPACING_TOL_HZ
        assert SEARCH_SPACING_TOL_HZ == pytest.approx(0.50)


def test_search_wires_the_recovery_into_the_annotation_pass():
    """The recovery runs where candidates are built, not in the results window.

    An earlier version of this capability lived in the window, took the N
    tallest peaks in the region, and accepted them when the MEAN of their
    spacings matched the DB J.
    """
    import inspect
    import gui.stages._annotation_search as srch
    import controllers.annotation_controller as core
    src = inspect.getsource(srch) + inspect.getsource(core)
    assert 'search_multiplet_in_window(' in src
    assert 'not _meas_sr.complete' in src, 'recovery must trigger on incompleteness'
