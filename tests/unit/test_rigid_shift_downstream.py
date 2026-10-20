# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Detection's rigid alignment carried to the matrix the later steps read
(processing.peak_alignment.apply_rigid_shifts)."""
import numpy as np

from processing.peak_alignment import apply_rigid_shifts
from processing.peak_detection import detect_peaks


def _lorentz(x, c, a, fwhm):
    return a / (1.0 + ((x - c) / (fwhm / 2.0)) ** 2)


def _drifting_study(n=6, seed=0):
    rng = np.random.default_rng(seed)
    ppm = np.linspace(4.0, 0.0, 16000)
    step = abs(ppm[1] - ppm[0])
    drift = rng.integers(-8, 9, n) * step
    cols = []
    for d in drift:
        y = sum(_lorentz(ppm, c + d, a, 0.002)
                for c, a in ((0.9, 1.0), (1.33, 0.7), (2.1, 0.5), (3.2, 0.8)))
        cols.append(y + rng.normal(0, 0.003, ppm.size))
    return ppm, np.column_stack(cols)


def test_shifts_reproduce_the_aligned_copy_detection_made():
    ppm, X = _drifting_study()
    det = detect_peaks(ppm, X, min_snr=5, baseline_correct=False,
                       global_align=True, return_details=True)
    assert det['alignment_applied'] and np.any(det['shift_points'])
    out = apply_rigid_shifts(X, det['shift_points'])
    assert np.array_equal(out, det['aligned_intensities'])


def test_aligned_samples_line_up():
    ppm, X = _drifting_study()
    det = detect_peaks(ppm, X, min_snr=5, baseline_correct=False,
                       global_align=True, return_details=True)
    out = apply_rigid_shifts(X, det['shift_points'])
    apex = lambda M: np.argmax(M[(ppm > 0.85) & (ppm < 0.95)], axis=0)
    assert np.ptp(apex(X)) >= 8
    assert np.ptp(apex(out)) <= 1


def test_no_shift_is_an_identical_copy():
    X = np.arange(20.0).reshape(10, 2)
    out = apply_rigid_shifts(X, [0, 0])
    assert np.array_equal(out, X) and out is not X
