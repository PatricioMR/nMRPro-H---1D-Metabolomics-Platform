"""Regression tests for cross-sample detection / quantification honesty.

Covers the failure mode "analyte present in sample X but absent in Y":
  • Fix 1 — peak detection must not dilute a single-sample analyte below SNR
    via the cross-sample mean (processing.peak_detection.detect_peaks now
    detects on the per-sample maximum).
  • Fix 2 — per-sample SNR / detection flag must mark samples where the analyte
    is not above the detection limit, so noise is never reported as signal
    (gui.stages._quant_computation._per_sample_snr).
"""
import numpy as np

from processing.peak_detection import detect_peaks
from gui.stages._quant_computation import _per_sample_snr


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


def test_per_sample_snr_flags_undetected_samples():
    """_per_sample_snr returns one SNR per sample; samples with only noise in
    the window fall below min_snr and must be flagged as not detected."""
    ppm = np.linspace(0.0, 1.0, 800)
    win = (ppm >= 0.48) & (ppm <= 0.52)
    lo_i = int(np.argmax(win))
    hi_i = len(win) - 1 - int(np.argmax(win[::-1]))

    shape = np.exp(-((ppm - 0.50) ** 2) / 0.0004)
    all_int = np.column_stack([
        1.00 * shape,                 # strong   → detected
        0.30 * shape,                 # moderate → detected
        np.full(len(ppm), 0.005),     # noise only → NOT detected
    ])

    p = {'start_idx': lo_i, 'end_idx': hi_i, 'noise': 0.02}
    snr_list, noise_lvl = _per_sample_snr(p, all_int, n_s=3)

    assert noise_lvl == 0.02
    assert len(snr_list) == 3
    min_snr = 3.0
    detected = [v >= min_snr for v in snr_list]
    assert detected == [True, True, False], snr_list
