"""Tests for per-peak cross-sample alignment (processing.peak_alignment +
detect_peaks refinement)."""
import numpy as np

from processing.peak_alignment import (
    refine_peak_centers_per_sample,
    warp_from_peak_shifts,
    _refine_one_center,
)
from processing.peak_detection import detect_peaks


def _gauss(x, c, a, w):
    return a * np.exp(-((x - c) ** 2) / (2.0 * w ** 2))


def test_refine_recovers_per_sample_centers():
    """A peak that sits at a slightly different ppm in each sample → the per-
    sample centres are recovered near the true positions (sub-pixel)."""
    ppm = np.linspace(0.0, 1.0, 4000)
    true = [0.498, 0.500, 0.502, 0.499]
    cols = [_gauss(ppm, c, 1.0, 0.004) for c in true]
    arr = np.column_stack(cols)

    peaks = [{'ppm': 0.500, 'rank': 1}]
    refine_peak_centers_per_sample(peaks, ppm, arr, noise_std=0.01,
                                   min_snr=3.0, max_shift_ppm=0.02)
    got = peaks[0]['ppm_per_sample']
    assert all(peaks[0]['aligned_per_sample'])
    for g, t in zip(got, true):
        assert abs(g - t) < 0.0015, (got, true)


def test_refine_does_not_chase_noise_when_absent():
    """A sample with only noise in the window keeps the reference position and
    is flagged not-aligned (no fabricated peak position)."""
    rng = np.random.default_rng(0)
    ppm = np.linspace(0.0, 1.0, 4000)
    present = _gauss(ppm, 0.500, 1.0, 0.004)
    absent  = rng.normal(0.0, 0.01, ppm.size)
    arr = np.column_stack([present, absent])

    peaks = [{'ppm': 0.500, 'rank': 1}]
    refine_peak_centers_per_sample(peaks, ppm, arr, noise_std=0.01,
                                   min_snr=3.0, max_shift_ppm=0.02)
    ok = peaks[0]['aligned_per_sample']
    assert ok[0] is True
    assert ok[1] is False
    assert peaks[0]['ppm_per_sample'][1] == 0.500       # reference kept
    assert peaks[0]['shift_ppm_per_sample'][1] == 0.0


def test_warp_sharpens_misaligned_max_reference():
    """Aligning samples before the per-sample max narrows a peak that is smeared
    by cross-sample shifts (FWHM of the max reference shrinks)."""
    ppm = np.linspace(0.0, 1.0, 4000)
    shifts = [-0.012, -0.006, 0.0, 0.006, 0.012]
    cols = [_gauss(ppm, 0.500 + d, 1.0, 0.004) for d in shifts]
    arr = np.column_stack(cols)

    def _fwhm(y):
        m = y.max()
        above = np.where(y >= 0.5 * m)[0]
        return abs(ppm[above[-1]] - ppm[above[0]]) if above.size >= 2 else 0.0

    fwhm_before = _fwhm(np.max(arr, axis=1))
    peaks = [{'ppm': 0.500, 'rank': 1}]
    refine_peak_centers_per_sample(peaks, ppm, arr, noise_std=0.01,
                                   min_snr=3.0, max_shift_ppm=0.02)
    warped = warp_from_peak_shifts(ppm, arr, peaks, max_shift_ppm=0.02)
    fwhm_after = _fwhm(np.max(warped, axis=1))
    assert fwhm_after < fwhm_before, (fwhm_before, fwhm_after)


def test_detect_peaks_stores_per_sample_centers():
    """detect_peaks attaches per-sample alignment metadata to each peak."""
    ppm = np.linspace(0.0, 1.0, 4000)
    shifts = [-0.008, 0.0, 0.008]
    arr = np.column_stack([_gauss(ppm, 0.500 + d, 1.0, 0.004) for d in shifts])

    result = detect_peaks(
        ppm, arr,
        noise_std=0.01, min_snr=3.0,
        baseline_correct=False, global_align=False,
        per_peak_align=True,
    )
    target = min(result, key=lambda p: abs(p['ppm'] - 0.50))
    assert 'ppm_per_sample' in target and target['ppm_per_sample'] is not None
    assert len(target['ppm_per_sample']) == 3
    # each stored centre is close to that sample's true shifted position
    for c, d in zip(target['ppm_per_sample'], shifts):
        assert abs(c - (0.500 + d)) < 0.003


def test_refine_one_center_subpixel():
    ppm = np.linspace(0.0, 1.0, 4000)
    y = _gauss(ppm, 0.5007, 1.0, 0.004)
    c, ok = _refine_one_center(ppm, y, 0.500, 0.02, 0.0)
    assert ok is True
    assert abs(c - 0.5007) < 0.0005
