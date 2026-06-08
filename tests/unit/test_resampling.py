# tests/unit/test_resampling.py
"""
Testes para os helpers robustos/não-paramétricos de _quant_helpers:
  fold_range, bootstrap_ci_median, bootstrap_fraction_ci.
"""
import numpy as np
import pytest
from gui.stages._quant_helpers import (
    fold_range, bootstrap_ci_median, bootstrap_fraction_ci)


# ── fold_range ───────────────────────────────────────────────────────────────

def test_fold_range_single_value_is_one():
    assert fold_range([5.0]) == 1.0


def test_fold_range_basic_ratio():
    assert fold_range([2.0, 4.0, 3.0]) == pytest.approx(2.0)


def test_fold_range_empty_is_inf():
    assert fold_range([]) == float('inf')


def test_fold_range_ignores_nonpositive():
    # zeros/negativos/nan descartados → [10, 20] → 2.0
    assert fold_range([10.0, 0.0, -3.0, np.nan, 20.0]) == pytest.approx(2.0)


def test_fold_range_all_nonpositive_is_inf():
    assert fold_range([0.0, -1.0]) == float('inf')


def test_fold_range_scale_free():
    # invariante a escala (scale-free)
    assert fold_range([1.0, 2.0]) == pytest.approx(fold_range([100.0, 200.0]))


# ── bootstrap_ci_median ──────────────────────────────────────────────────────

def test_boot_median_small_n_no_ci():
    point, lo, hi = bootstrap_ci_median([10.0, 11.0, 9.0], min_n=4)
    assert point == pytest.approx(10.0)          # mediana
    assert np.isnan(lo) and np.isnan(hi)         # n<min_n → CI indisponível


def test_boot_median_ci_brackets_point():
    rng = np.random.default_rng(1)
    vals = list(rng.normal(100.0, 5.0, size=40))
    point, lo, hi = bootstrap_ci_median(vals, n_boot=500, min_n=4, seed=1)
    assert lo <= point <= hi
    assert hi > lo


def test_boot_median_tight_for_consistent_data():
    point, lo, hi = bootstrap_ci_median([50.0] * 20, min_n=4, seed=1)
    assert point == pytest.approx(50.0)
    assert lo == pytest.approx(50.0) and hi == pytest.approx(50.0)


def test_boot_median_deterministic_seed():
    a = bootstrap_ci_median(list(range(1, 30)), seed=7)
    b = bootstrap_ci_median(list(range(1, 30)), seed=7)
    assert a == b


def test_boot_median_empty():
    point, lo, hi = bootstrap_ci_median([])
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


# ── bootstrap_fraction_ci ────────────────────────────────────────────────────

def test_boot_fraction_small_n_no_ci():
    point, lo, hi = bootstrap_fraction_ci([0.5, 0.6, 0.4], min_n=6)
    assert point == pytest.approx(0.5)
    assert np.isnan(lo) and np.isnan(hi)


def test_boot_fraction_ci_brackets_mean():
    rng = np.random.default_rng(2)
    vals = list(rng.uniform(0.2, 0.8, size=20))
    point, lo, hi = bootstrap_fraction_ci(vals, n_boot=500, min_n=6, seed=2)
    assert lo <= point <= hi


def test_boot_fraction_keeps_negatives_out_of_positivity_assumption():
    # mean inclui todos os finitos (fração pode ser 0); só descarta nan
    point, _, _ = bootstrap_fraction_ci([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert point == pytest.approx(0.0)
