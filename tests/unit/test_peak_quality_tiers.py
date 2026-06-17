"""3-tier peak quality: green (good), yellow (crooked/overlap), red (linear)."""
import numpy as np

from processing.peak_detection import _classify_peak_quality


def _lorentz(x, amp, cen, hwhm):
    return amp * hwhm ** 2 / ((x - cen) ** 2 + hwhm ** 2)


def test_clean_peak_is_green():
    x = np.linspace(-1, 1, 121)
    y = _lorentz(x, 1.0, 0.0, 0.15)
    tier, score = _classify_peak_quality(
        y, 0, len(y) - 1, 0, len(y) - 1, height=float(y.max()),
        shape_r2=0.97, sym_score=0.95, n_cluster=1)
    assert tier == 'green', (tier, score)


def test_straight_line_is_red():
    y = np.linspace(0.2, 0.8, 121)          # pure ramp, no curvature
    tier, score = _classify_peak_quality(
        y, 0, len(y) - 1, 0, len(y) - 1, height=float(y.max() - y.min()),
        shape_r2=0.05, sym_score=0.3, n_cluster=1)
    assert tier == 'red', (tier, score)


def test_overlapped_crooked_is_yellow():
    """Overlapped peak: mediocre shape fit + tilted baseline → mid score."""
    x = np.linspace(-1, 1, 121)
    y = _lorentz(x, 1.0, 0.0, 0.15) + np.linspace(0.0, 0.6, 121)  # strong tilt
    tier, score = _classify_peak_quality(
        y, 0, len(y) - 1, 0, len(y) - 1, height=float(y.max()),
        shape_r2=0.50, sym_score=0.5, n_cluster=1)
    assert tier == 'yellow', (tier, score)


def test_clean_cluster_member_can_be_green():
    """A well-resolved multiplet line (cluster member) is still high quality."""
    x = np.linspace(-1, 1, 121)
    y = _lorentz(x, 1.0, 0.0, 0.15)
    tier, _ = _classify_peak_quality(
        y, 0, len(y) - 1, 0, len(y) - 1, height=float(y.max()),
        shape_r2=0.99, sym_score=0.99, n_cluster=2)
    assert tier == 'green'
