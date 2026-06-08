"""Tests for peak start/end (boundary) determination in separate_overlapping_peaks.

Rule: foot = first valley (signal turns up) OR baseline+3σ; isolated peaks get
symmetric feet (raise lower foot to higher → horizontal integration baseline);
overlapped cluster lines keep the valley boundary and survive even with ~0
height above the local baseline.
"""
import numpy as np

from processing.peak_detection import separate_overlapping_peaks, detect_peaks


def _g(x, c, a, w):
    return a * np.exp(-((x - c) ** 2) / (2.0 * w ** 2))


def test_isolated_peak_has_symmetric_feet():
    ppm = np.linspace(0.0, 1.0, 2000)
    y = _g(ppm, 0.5, 1.0, 0.01)
    seps = separate_overlapping_peaks(y, [1000], ppm, noise_std=0.005,
                                      method='auto', baseline_level=0.0)
    p = seps[0]
    ys, ye = float(y[p['start_idx']]), float(y[p['end_idx']])
    # feet at (approximately) the same intensity → horizontal baseline
    assert abs(ys - ye) < 0.02 * float(y.max())


def test_neighbour_does_not_skew_isolated_baseline():
    """Peak with a higher neighbour on the right: the right foot sits in the
    valley (elevated); symmetric feet raise the left foot to match, so the
    integration baseline is horizontal (start/end at similar Y)."""
    ppm = np.linspace(0.0, 1.0, 2000)
    y = _g(ppm, 0.50, 1.0, 0.008) + _g(ppm, 0.56, 0.8, 0.008)
    # detect both; check the left peak's feet are balanced
    res = detect_peaks(ppm, y, noise_std=0.005, min_snr=3.0,
                       baseline_correct=False, global_align=False)
    left = min(res, key=lambda p: abs(p['ppm'] - 0.50))
    ys, ye = float(y[left['start_idx']]), float(y[left['end_idx']])
    assert abs(ys - ye) < 0.05 * float(y.max())


def test_shoulder_survives_with_low_local_height():
    """An opt-in shoulder (cluster member) is kept even though its height above
    the local linear baseline is ~0 — it is real signal on a peak flank."""
    ppm = np.linspace(0.0, 1.0, 2000)
    y = _g(ppm, 0.500, 1.0, 0.010) + _g(ppm, 0.525, 0.45, 0.010)
    res = detect_peaks(ppm, y, noise_std=0.01, min_snr=3.0,
                       baseline_correct=False, global_align=False,
                       detect_shoulders=True)
    centers = sorted(p['ppm'] for p in res)
    assert any(abs(c - 0.525) < 0.012 for c in centers), centers
    # the shoulder is flagged as part of a 2-peak cluster
    sh = min(res, key=lambda p: abs(p['ppm'] - 0.525))
    assert sh['n_cluster_peaks'] >= 2


def test_isolated_peak_grounds_at_baseline():
    """Foot of a clean isolated peak reaches ~baseline (well below the apex)."""
    ppm = np.linspace(0.0, 1.0, 2000)
    y = _g(ppm, 0.5, 1.0, 0.01)
    seps = separate_overlapping_peaks(y, [1000], ppm, noise_std=0.005,
                                      method='auto', baseline_level=0.0)
    p = seps[0]
    assert float(y[p['start_idx']]) < 0.2 * float(y.max())
    assert float(y[p['end_idx']]) < 0.2 * float(y.max())
