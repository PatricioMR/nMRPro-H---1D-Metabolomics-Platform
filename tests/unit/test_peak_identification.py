"""Tests for peak-identification improvements in processing.peak_detection.

Covers the audit fixes:
  A  shoulder / overlapped-line detection via the second derivative
  D  close multiplet legs are kept as separate apexes (not merged)
  B  robust noise estimation (quiet segment, not blind edges)
  C  detection threshold scaled to the detection spectrum
  E  narrow singlets get a real shape R² (not forced to 0)
  H  single-point spikes are rejected
"""
import numpy as np

from processing.peak_detection import (
    detect_peaks,
    estimate_noise_minimum_segment,
    estimate_noise_region,
    _second_derivative_candidates,
)


def _gauss(x, c, a, w):
    return a * np.exp(-((x - c) ** 2) / (2.0 * w ** 2))


def test_shoulder_without_local_maximum_is_detected():
    """Two heavily overlapped peaks where the smaller is only a shoulder
    (no local maximum). find_peaks alone would report one peak; the second-
    derivative pass must recover the hidden second apex."""
    ppm = np.linspace(0.0, 1.0, 2000)
    main     = _gauss(ppm, 0.500, 1.00, 0.010)
    shoulder = _gauss(ppm, 0.525, 0.45, 0.010)  # buried on the flank
    signal = main + shoulder

    result = detect_peaks(
        ppm, signal,
        noise_std=0.01, min_snr=3.0,
        baseline_correct=False, global_align=False,
        detect_shoulders=True,
    )
    centers = sorted(p['ppm'] for p in result)
    assert len(centers) >= 2, f"shoulder not detected: {centers}"
    assert any(abs(c - 0.500) < 0.01 for c in centers)
    assert any(abs(c - 0.525) < 0.012 for c in centers)


def test_close_doublet_legs_are_not_merged():
    """A doublet (J≈7 Hz @600 MHz → ~0.012 ppm split) must yield two apexes."""
    ppm = np.linspace(0.0, 1.0, 4000)
    split = 0.0117
    d = (_gauss(ppm, 0.500 - split / 2, 1.0, 0.0025)
         + _gauss(ppm, 0.500 + split / 2, 1.0, 0.0025))

    result = detect_peaks(
        ppm, d,
        noise_std=0.01, min_snr=3.0, min_distance=0.005,
        baseline_correct=False, global_align=False,
        detect_shoulders=True,
    )
    centers = sorted(p['ppm'] for p in result)
    assert len(centers) >= 2, f"doublet merged into one: {centers}"


def test_min_segment_noise_robust_to_edge_signal():
    """A broad rolling baseline at the spectrum edge must not inflate the noise
    estimate the way the blind-edge estimator does."""
    rng = np.random.default_rng(0)
    n = 2000
    ppm = np.linspace(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.02, n)
    edge_roll = _gauss(ppm, 0.0, 1.0, 0.08)   # broad baseline hump at the edge
    spectrum = noise + edge_roll

    seg = estimate_noise_minimum_segment(spectrum)
    edge = estimate_noise_region(spectrum)
    assert seg < edge                      # quiet-segment beats blind edges
    assert 0.01 < seg < 0.05               # close to the true 0.02


def test_detection_threshold_uses_detection_spectrum_noise():
    """detect_peaks must not explode with false positives on a noisy multi-
    sample max reference: the threshold is taken on that spectrum."""
    rng = np.random.default_rng(1)
    n = 2000
    ppm = np.linspace(0.0, 1.0, n)
    peak = _gauss(ppm, 0.50, 1.0, 0.01)
    samples = np.column_stack([peak + rng.normal(0, 0.02, n) for _ in range(8)])

    result = detect_peaks(
        ppm, samples,
        min_snr=3.0, baseline_correct=False, global_align=False,
    )
    assert any(abs(p['ppm'] - 0.50) < 0.01 for p in result)
    assert len(result) < 15, f"too many false positives: {len(result)}"


def test_narrow_singlet_gets_real_shape_r2():
    """A clean narrow singlet (few points) must obtain a real Gaussian R²,
    not the automatic 0.0 the old >=8-point rule produced."""
    ppm = np.linspace(0.0, 1.0, 1500)
    s = _gauss(ppm, 0.50, 1.0, 0.004)

    result = detect_peaks(
        ppm, s,
        noise_std=0.01, min_snr=3.0,
        baseline_correct=False, global_align=False,
    )
    target = min(result, key=lambda p: abs(p['ppm'] - 0.50))
    assert target['gaussian_r2'] > 0.0


def test_single_point_spike_is_rejected():
    """A 1-point spike (no width) must not be promoted to a peak."""
    ppm = np.linspace(0.0, 1.0, 2000)
    spectrum = np.zeros(2000)
    spectrum[1000] = 1.0  # lone spike, single sample point

    result = detect_peaks(
        ppm, spectrum,
        noise_std=0.01, min_snr=3.0,
        baseline_correct=False, global_align=False,
    )
    assert all(abs(p['ppm'] - 0.5) > 0.005 for p in result), \
        "single-point spike was detected as a peak"


def test_second_derivative_candidates_basic():
    ppm = np.linspace(0.0, 1.0, 2000)
    sig = _gauss(ppm, 0.40, 1.0, 0.01) + _gauss(ppm, 0.60, 1.0, 0.01)
    idx = _second_derivative_candidates(sig, min_height=0.1, smooth_win=11)
    found = sorted(ppm[i] for i in idx)
    assert any(abs(c - 0.40) < 0.01 for c in found)
    assert any(abs(c - 0.60) < 0.01 for c in found)
