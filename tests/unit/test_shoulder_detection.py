# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Second-derivative shoulder recovery in peak detection.

A shoulder is a real (overlapping) peak with no intensity maximum of its own.
The apex search (find_peaks) cannot see it; the second-derivative augmentation
should recover it.
"""
import numpy as np

from processing.peak_detection import detect_peaks, _detect_single


def _lorentz(x, amp, cen, hwhm):
    return amp * hwhm ** 2 / ((x - cen) ** 2 + hwhm ** 2)


def _doublet_with_shoulder():
    """Big peak at 0.500 with a smaller shoulder at 0.512 (no local max)."""
    ppm = np.linspace(0.0, 1.0, 2000)
    big      = _lorentz(ppm, 1.00, 0.500, 0.004)
    shoulder = _lorentz(ppm, 0.15, 0.512, 0.004)
    return ppm, big + shoulder


def test_shoulder_has_no_intensity_maximum():
    """Sanity: the shoulder really is a shoulder (one apex only via find_peaks)."""
    from scipy.signal import find_peaks
    _, y = _doublet_with_shoulder()
    apices, _ = find_peaks(y, height=0.03, distance=20, prominence=0.015)
    assert len(apices) == 1, "test fixture is not a true shoulder"


def test_shoulder_recovered_when_enabled():
    ppm, y = _doublet_with_shoulder()
    res = detect_peaks(ppm, y, noise_std=0.01, min_snr=3.0,
                       baseline_correct=False, detect_shoulders=True)
    centers = sorted(p['ppm'] for p in res)
    assert len(res) == 2, f"expected 2 peaks (apex + shoulder), got {len(res)}"
    assert any(abs(c - 0.500) < 0.01 for c in centers)
    assert any(abs(c - 0.512) < 0.012 for c in centers)


def test_shoulder_missed_when_disabled():
    ppm, y = _doublet_with_shoulder()
    res = detect_peaks(ppm, y, noise_std=0.01, min_snr=3.0,
                       baseline_correct=False, detect_shoulders=False)
    assert len(res) == 1, "without shoulder detection only the apex is found"


def test_shoulders_do_not_invent_noise_peaks():
    """Flat noise must not produce curvature-driven false positives."""
    rng = np.random.default_rng(0)
    ppm = np.linspace(0.0, 1.0, 2000)
    noise = rng.normal(0.0, 0.01, ppm.size)
    idx = _detect_single(noise, eff_noise=0.01, min_height=0.05, min_snr=3.0,
                         distance_in_points=20, detect_shoulders=True)
    assert idx == [], f"shoulder detection invented {len(idx)} noise peaks"
