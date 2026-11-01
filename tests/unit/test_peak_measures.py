# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The Signal List's Area, height and SNR have one definition, and its viewer
shades exactly that. Detection, the viewer and a re-measure after the analyst
moves a boundary all go through ``foot_line_area`` / ``peak_measures``."""
import numpy as np
import pytest

from processing.peak_detection import (
    detect_peaks, foot_line_area, peak_measures, window_apex)


def _lorentz(ppm, c, h, w=0.002):
    return h * w ** 2 / ((ppm - c) ** 2 + w ** 2)


def test_the_area_is_what_stands_above_the_line_between_the_windows_ends():
    x = np.linspace(2.0, 1.0, 11)               # descending, as a spectrum is held
    y = np.array([1, 1, 1, 3, 5, 3, 1, 1, 1, 1, 1], dtype=float)
    area, line = foot_line_area(x, y)
    assert np.allclose(line, 1.0)
    # triangle of height 4 over 4 steps of 0.1: 0.1*(2+4+2) = 0.8
    assert area == pytest.approx(0.8)


def test_a_sloped_pedestal_is_not_area():
    x = np.linspace(1.0, 2.0, 21)
    y = 10.0 + 5.0 * (x - 1.0)                   # a line and nothing else
    assert foot_line_area(x, y)[0] == pytest.approx(0.0, abs=1e-12)


def test_below_the_line_counts_nothing():
    x = np.linspace(1.0, 2.0, 5)
    y = np.array([2.0, 0.0, 0.0, 0.0, 2.0])
    assert foot_line_area(x, y)[0] == 0.0


def test_a_window_of_one_point_has_no_area():
    area, line = foot_line_area([1.0], [5.0])
    assert area == 0.0 and list(line) == [5.0]


def test_the_measures_are_detections_own():
    ppm = np.linspace(4.0, 0.0, 16000)
    rng = np.random.default_rng(0)
    cols = []
    for k in (1.0, 2.0, 3.0):
        cols.append(_lorentz(ppm, 3.0, 100 * k) + _lorentz(ppm, 1.5, 60 * k)
                    + rng.normal(0, 0.5, ppm.size))
    X = np.column_stack(cols)
    det = detect_peaks(ppm, X, noise_std=0.5, min_snr=6.0, baseline_correct=False,
                       global_align=False, return_details=True,
                       per_sample_detection=False, excluded_regions=())
    peaks = det['peaks']
    assert len(peaks) >= 2
    for p in peaks:
        m = peak_measures(ppm, X, p['start_idx'], p['end_idx'], 0.5, index=p['index'])
        assert m['area'] == pytest.approx(p['area'], rel=1e-9)
        assert m['height'] == pytest.approx(p['height'], rel=1e-9)
        assert m['snr'] == pytest.approx(p['snr'], rel=1e-9)
        assert np.allclose(m['area_per_sample'], p['area_per_sample'])


def test_a_moved_boundary_changes_the_measures():
    ppm = np.linspace(4.0, 0.0, 16000)
    X = np.column_stack([_lorentz(ppm, 2.0, 100.0 * k) for k in (1, 2)])
    i = int(np.argmin(np.abs(ppm - 2.0)))
    wide = peak_measures(ppm, X, i - 300, i + 300, 1.0, index=i)
    narrow = peak_measures(ppm, X, i - 40, i + 40, 1.0, index=i)
    assert narrow['area'] < wide['area']
    assert narrow['height'] == pytest.approx(wide['height'], rel=0.05)


def test_the_window_apex_is_the_most_prominent_interior_maximum():
    y = np.array([9.0, 7.0, 5.0, 5.5, 4.0, 3.0, 3.2, 3.1, 1.0])
    # 5.5 stands 0.5 over its valley; 3.2, the other maximum, only 0.2
    assert window_apex(y) == (3, pytest.approx(0.5))


def test_a_window_without_an_interior_maximum_falls_back_to_the_chord():
    y = np.array([0.0, 3.0, 4.0, 4.5, 4.8, 5.0])     # a bare shoulder
    i, prom = window_apex(y)
    assert 0 < i < y.size - 1 and prom == 0.0
    assert window_apex(np.array([5.0, 4.0, 3.0]))[0] == 0   # nothing above it


def test_the_apex_follows_when_the_old_one_leaves_the_window():
    ppm = np.linspace(4.0, 0.0, 16000)
    X = np.column_stack([_lorentz(ppm, 2.0, 100.0) + _lorentz(ppm, 1.0, 40.0)])
    i = int(np.argmin(np.abs(ppm - 2.0)))
    j = int(np.argmin(np.abs(ppm - 1.0)))
    m = peak_measures(ppm, X, j - 100, j + 100, 1.0, index=i)
    assert m['index'] == pytest.approx(j, abs=3)
    assert m['ppm'] == pytest.approx(1.0, abs=0.002)
