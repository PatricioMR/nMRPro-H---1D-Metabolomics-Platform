"""The statistics of validation.composition_bench."""
import numpy as np
import pytest

from validation.composition_bench import inverse, linear_fit, linear_range, run_metrics


def test_an_exact_line_is_recovered_and_inverted():
    x = [10, 10, 20, 20, 30, 30]
    y = [5 + 2 * v for v in x]
    f = linear_fit(x, [v + d for v, d in zip(y, (0.1, -0.1) * 3)])
    assert f['b'] == pytest.approx(2.0, abs=1e-9)
    assert f['a'] == pytest.approx(5.0, abs=1e-9)
    x0, h = inverse(f, 45.0)
    assert x0 == pytest.approx(20.0, abs=1e-9)
    assert 0 < h < 1


def test_curvature_shows_as_lack_of_fit():
    x = np.repeat([10, 20, 30, 40], 3).astype(float)
    rng = np.random.default_rng(0)
    y = 3 * x - 0.04 * x ** 2 + rng.normal(0, 0.2, x.size)
    f = linear_fit(x, y)
    assert f['lof'] is not None and f['lof'][1] < 0.01


def test_a_straight_line_passes_the_lack_of_fit_test():
    x = np.repeat([10, 20, 30, 40], 3).astype(float)
    rng = np.random.default_rng(1)
    f = linear_fit(x, 3 * x + rng.normal(0, 1.0, x.size))
    assert f['lof'][1] > 0.05


def test_the_interval_widens_away_from_the_centre():
    x = np.repeat([10, 20, 30], 3).astype(float)
    rng = np.random.default_rng(2)
    f = linear_fit(x, 4 * x + rng.normal(0, 2.0, x.size))
    _c, h_mid = inverse(f, f['ybar'])
    _e, h_far = inverse(f, f['ybar'] + 200)
    assert h_far > h_mid


def test_run_metrics_counts_real_false_and_errors():
    truth = {'A': {'A': 1.0, 'B': 1.0}, 'B': {'A': 2.0, 'B': 2.0}}
    rows = [{'hmdb_id': 'A', '_confidence': 3, '_concs_by_group': {'A': [1.1], 'B': [0.9]}},
            {'hmdb_id': 'X', '_confidence': 1},
            {'hmdb_id': 'B', '_not_quantified': True}]
    m = run_metrics(rows, truth, 40, {'explained': 30})
    assert (m['found'], m['false'], m['analytes'], m['peaks']) == (1, 1, 2, 40)
    assert m['recall'] == 0.5 and m['prec3'] == 1.0
    assert m['err_median'] == pytest.approx(0.1)


def test_a_saturating_top_level_is_left_out_of_the_line():
    x = np.repeat([10, 20, 30, 40, 80], 3).astype(float)
    rng = np.random.default_rng(3)
    y = np.where(x < 50, 3 * x, 160.0) + rng.normal(0, 0.5, x.size)
    top, f = linear_range(x, y)
    assert top == 40 and f['b'] == pytest.approx(3.0, abs=0.1)


def test_a_curve_with_a_ceiling_is_preferred_when_the_data_saturate():
    from validation.composition_bench import saturation_fit
    x = np.repeat([20, 50, 100, 150, 200], 3).astype(float)
    rng = np.random.default_rng(4)
    y = 400 * (1 - np.exp(-x / 80)) + rng.normal(0, 3, x.size)
    out = saturation_fit(x, y)
    assert min(out, key=lambda k: out[k]['aic']) == 'teto'
    assert out['teto']['ymax'] == pytest.approx(400, rel=0.05)
