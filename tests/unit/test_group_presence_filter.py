"""Group-presence filter: a peak detected in only one sample is dropped;
a peak seen in >= N samples of a group is kept."""
import numpy as np

from processing.peak_detection import detect_peaks


def _lorentz(x, a, c, w):
    return a * w * w / ((x - c) ** 2 + w * w)


def _make_group_data(spurious_in_one=True):
    """6 samples, one group. Real peak at 3.00 in all; spurious at 2.50 in 1."""
    ppm = np.linspace(2.0, 3.5, 3000)
    cols = []
    rng = np.random.default_rng(1)
    for s in range(6):
        # Very low noise so per-sample detection only finds the real signals
        # (deterministic presence — this is a filter logic test, not a noise test).
        y = _lorentz(ppm, 1.0, 3.00, 0.003) + rng.normal(0, 1e-5, ppm.size)
        if spurious_in_one and s == 0:
            y = y + _lorentz(ppm, 1.0, 2.50, 0.003)   # only sample 0
        cols.append(y)
    return ppm, np.column_stack(cols)


def _samples_info(n):
    return [{'label': 'G1', 'type': 'Sample'} for _ in range(n)]


def test_single_sample_peak_dropped_with_threshold():
    ppm, Y = _make_group_data(spurious_in_one=True)
    si = _samples_info(Y.shape[1])
    res = detect_peaks(ppm, Y, noise_std=None, min_snr=3.0, baseline_correct=False,
                       per_sample_detection=True, global_align=False,
                       per_peak_align=False, samples_info=si,
                       min_group_presence=3)
    centers = sorted(round(p['ppm'], 2) for p in res)
    assert any(abs(c - 3.00) < 0.02 for c in centers)          # real kept
    assert not any(abs(c - 2.50) < 0.02 for c in centers)      # spurious dropped


def test_single_sample_peak_kept_without_threshold():
    ppm, Y = _make_group_data(spurious_in_one=True)
    si = _samples_info(Y.shape[1])
    res = detect_peaks(ppm, Y, noise_std=None, min_snr=3.0, baseline_correct=False,
                       per_sample_detection=True, global_align=False,
                       per_peak_align=False, samples_info=si,
                       min_group_presence=1)            # filter off
    centers = sorted(round(p['ppm'], 2) for p in res)
    assert any(abs(c - 2.50) < 0.02 for c in centers)          # spurious kept
