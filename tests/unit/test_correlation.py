"""Unit tests for processing.correlation (coefficient + OLS confidence band)."""
import numpy as np

from processing.correlation import correlation_coeff, linear_fit_ci


def test_pearson_perfect_line():
    x = np.arange(10.0)
    y = 2.0 * x + 1.0
    r, p = correlation_coeff(x, y, method='pearson')
    assert abs(r - 1.0) < 1e-9
    assert p < 1e-6


def test_spearman_monotonic_nonlinear():
    x = np.arange(1.0, 11.0)
    y = x ** 3                       # monotonic but non-linear
    r, _ = correlation_coeff(x, y, method='spearman')
    assert abs(r - 1.0) < 1e-9


def test_coeff_undefined_cases():
    assert np.isnan(correlation_coeff([1.0, 2.0], [1.0, 2.0])[0])   # <3 pairs
    assert np.isnan(correlation_coeff([1, 1, 1], [1, 2, 3])[0])     # constant x


def test_coeff_drops_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.1, 6.0, np.nan, 10.2])
    r, p = correlation_coeff(x, y, method='pearson')
    assert r > 0.99                  # remaining finite pairs are near-linear


def test_linear_fit_recovers_slope_and_band_brackets_line():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    y = 3.0 * x - 2.0 + rng.normal(0, 0.5, x.size)
    fit = linear_fit_ci(x, y, alpha=0.05)
    assert fit is not None
    assert abs(fit['slope'] - 3.0) < 0.2
    assert np.all(fit['lower'] <= fit['ys'])
    assert np.all(fit['ys'] <= fit['upper'])
    # band is widest at the extremes, narrowest near the mean of x
    half = fit['upper'] - fit['lower']
    assert half[0] > half[len(half) // 2]
    assert half[-1] > half[len(half) // 2]


def test_linear_fit_none_when_undefined():
    assert linear_fit_ci([1.0, 2.0], [1.0, 2.0]) is None      # <3 pairs
    assert linear_fit_ci([1, 1, 1], [1, 2, 3]) is None        # constant x
