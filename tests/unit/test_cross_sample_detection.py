"""Regression tests for cross-sample detection / quantification honesty.

Covers the failure mode "analyte present in sample X but absent in Y":
  • Fix 1 — peak detection must not dilute a single-sample analyte below SNR
    via the cross-sample mean (processing.peak_detection.detect_peaks now
    detects on the per-sample maximum).
"""
import numpy as np

from processing.peak_detection import detect_peaks


def test_single_sample_analyte_is_not_diluted_below_detection():
    """A resonance present in only 1 of 5 samples is diluted 5x in the mean
    (below SNR) but must still be detected via the per-sample-max reference."""
    ppm = np.linspace(0.0, 1.0, 800)
    n_samples = 5
    peak = 0.25 * np.exp(-((ppm - 0.50) ** 2) / 0.0004)  # height 0.25

    intensities = np.zeros((len(ppm), n_samples))
    intensities[:, 0] = peak  # present in ONE sample only

    # Mean height at apex = 0.25 / 5 = 0.05 < min_height (0.02 * 3 = 0.06):
    # a mean-based detector would miss it. The per-sample max keeps it.
    result = detect_peaks(
        ppm, intensities,
        noise_std=0.02, min_snr=3.0,
        baseline_correct=False, global_align=False,
    )

    assert any(abs(p['ppm'] - 0.50) < 0.01 for p in result), \
        "single-sample analyte was lost (diluted by the cross-sample mean)"
