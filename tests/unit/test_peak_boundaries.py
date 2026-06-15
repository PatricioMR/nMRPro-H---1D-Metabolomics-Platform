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


def test_isolated_feet_land_at_5pct_height():
    """Free-flank boundary rule: start/end sit where the flank drops to 5 % of
    the apex height above the local baseline."""
    ppm = np.linspace(0.0, 1.0, 2000)
    y = _g(ppm, 0.5, 1.0, 0.01)
    p = separate_overlapping_peaks(y, [1000], ppm, noise_std=0.005,
                                   baseline_level=0.0)[0]
    apex = float(y[p['index']])
    for idx in (p['start_idx'], p['end_idx']):
        # at (not far below) the 5 % level — discrete sampling lands just above
        assert 0.04 * apex <= float(y[idx]) <= 0.08 * apex, float(y[idx])


def test_5pct_uses_local_baseline_on_pedestal():
    """On a raised pedestal the 5 % is measured ABOVE the local baseline, not
    the absolute zero — feet land near base + 0.05·(apex − base)."""
    ppm = np.linspace(0.0, 1.0, 2000)
    base = 0.30
    y = base + _g(ppm, 0.5, 0.70, 0.01)
    p = separate_overlapping_peaks(y, [1000], ppm, noise_std=0.005,
                                   baseline_level=base)[0]
    expected = base + 0.05 * 0.70
    assert abs(float(y[p['start_idx']]) - expected) < 0.03, float(y[p['start_idx']])


def test_overlap_boundary_stays_at_valley():
    """When a valley is reached before the 5 % level (overlap), the shared inner
    boundary is the valley min — the 5 % rule does not push it past the dip."""
    ppm = np.linspace(0.0, 1.0, 2000)
    y = _g(ppm, 0.500, 1.0, 0.006) + _g(ppm, 0.530, 1.0, 0.006)
    seps = separate_overlapping_peaks(y, [1000, 1060], ppm, noise_std=0.005,
                                      baseline_level=0.0)
    left, right = sorted(seps, key=lambda p: p['ppm'])
    # the shared boundary sits at the 0.515 valley between the two apices
    assert 0.508 < left['end_ppm'] < 0.522
    assert abs(left['end_ppm'] - right['start_ppm']) < 1e-9


def test_isolated_peak_grounds_at_baseline():
    """Foot of a clean isolated peak reaches ~baseline (well below the apex)."""
    ppm = np.linspace(0.0, 1.0, 2000)
    y = _g(ppm, 0.5, 1.0, 0.01)
    seps = separate_overlapping_peaks(y, [1000], ppm, noise_std=0.005,
                                      method='auto', baseline_level=0.0)
    p = seps[0]
    assert float(y[p['start_idx']]) < 0.2 * float(y.max())
    assert float(y[p['end_idx']]) < 0.2 * float(y.max())
