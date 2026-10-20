# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

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


def test_a_noisy_maximum_does_not_cut_a_line_at_a_wiggle():
    """The top of a line on a per-sample maximum follows one sample's noise
    point to point, at about twice the quiet spread the walk is given. The
    feet must still land at the line's feet, not at the first wiggle (the
    walk on the trace itself cut these to 5-39 points and graded one Poor)."""
    ppm = np.linspace(0.0, 1.0, 20000)                 # ~40 points per FWHM
    line = _g(ppm, 0.5, 1.0, 0.00085)
    fwhm_pts = 2.355 * 0.00085 / (ppm[1] - ppm[0])
    for seed in range(4):
        rng = np.random.default_rng(seed)
        X = np.column_stack([line * (1 + 0.1 * k)
                             + rng.normal(0, 0.05, ppm.size)
                             for k in range(12)])
        ref = X.max(axis=1)
        quiet = ref[:5000]
        spread = 1.4826 * np.median(np.abs(quiet - np.median(quiet)))
        p = separate_overlapping_peaks(
            ref, [int(np.argmax(ref))], ppm, noise_std=spread,
            baseline_level=float(np.median(quiet)))[0]
        assert p['end_idx'] - p['start_idx'] > fwhm_pts
        assert p['signal_quality'] != 'Poor'


def test_a_tall_line_keeps_its_height_when_its_top_is_off_the_raw_apex():
    """The smoothed copy's top can sit a point beside the raw maximum; on a
    tall line one point is uphill by more than the walk's margin. Walked
    from the raw apex, the internal standard came out with height 0 and
    was dropped."""
    ppm = np.linspace(0.0, 1.0, 20000)
    line = _g(ppm, 0.5, 100.0, 0.00085)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        X = np.column_stack([line + rng.normal(0, 0.05, ppm.size)
                             for _ in range(12)])
        ref = X.max(axis=1)
        quiet = ref[:5000]
        spread = 1.4826 * np.median(np.abs(quiet - np.median(quiet)))
        p = separate_overlapping_peaks(
            ref, [int(np.argmax(ref))], ppm, noise_std=spread,
            baseline_level=float(np.median(quiet)))[0]
        assert p['height'] > 0.5 * float(ref.max())


def test_a_dip_of_a_few_percent_on_a_tall_top_is_not_a_valley():
    """A tall line's top on a per-sample maximum carries structure far above
    the noise (the internal standard: a 4% dip on 5.4e6). Twice the noise
    alone called it a valley, and the standard lost its left foot."""
    ppm = np.linspace(0.0, 1.0, 20000)
    d = 0.0019
    y = _g(ppm, 0.5 - d / 2, 50.0, 0.00085) + _g(ppm, 0.5 + d / 2, 50.0, 0.00085)
    apex = int(np.argmax(y))
    p = separate_overlapping_peaks(y, [apex], ppm, noise_std=0.005)[0]
    assert p['start_idx'] < 10000 - 40 and p['end_idx'] > 10000 + 40
    assert p['height'] > 0.5 * float(y.max())


def test_a_doublet_line_beside_a_taller_one_keeps_its_own_foot():
    """The smoothed top was the tallest point within half a linewidth -- a
    linewidth read off the tallest lines of the study, broad on a
    per-sample maximum. On a doublet whose other line is taller and within
    that window, the walk started from the other line, stopped in the valley
    between them and put the left foot on the apex itself: valine's 1.052
    line (SNR 50) on the Mice ISS study was measured 0 high and graded Poor."""
    # the study's proportions: lines ~21 points wide, the doublet's two 49
    # apart, the tallest lines of the study ~110 points wide
    ppm = np.linspace(0.0, 1.0, 20000)
    pt = 1.0 / 19999
    broad = sum(_g(ppm, c, 200.0, 47 * pt) for c in (0.15, 0.25, 0.85))
    small, tall = 0.5000, 0.5000 + 49 * pt
    y = broad + _g(ppm, small, 6.0, 9 * pt) + _g(ppm, tall, 10.0, 9 * pt)
    apices = [int(np.argmin(np.abs(ppm - c)))
              for c in (0.15, 0.25, small, tall, 0.85)]
    seps = separate_overlapping_peaks(y, apices, ppm, noise_std=0.01)
    p = min(seps, key=lambda q: abs(q['ppm'] - small))
    assert p['start_idx'] < p['index']
    assert p['height'] > 0.3 * float(y[p['index']])
    assert p['signal_quality'] != 'Poor'
