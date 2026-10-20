# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Unit tests for processing.group_stats (Welch t / Mann-Whitney + stars)."""
import numpy as np

from processing.group_stats import (
    two_group_pvalue, significance_stars, pairwise_significance,
    metabolite_min_pvalues)


def test_welch_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = rng.normal(3, 1, 30)
    assert two_group_pvalue(a, b, test='t') < 0.001


def test_mann_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 30)
    b = rng.normal(3, 1, 30)
    assert two_group_pvalue(a, b, test='mann') < 0.001


def test_no_difference_is_nonsignificant():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 40)
    b = rng.normal(0, 1, 40)
    assert two_group_pvalue(a, b, test='t') > 0.05


def test_nan_entries_are_dropped_not_imputed():
    # Separated groups with some not-detected (NaN) values. The test must use
    # only the real measurements and stay significant — never fill NaN with a
    # shared mean (which would flatten the difference to ns).
    a = np.array([1.0, np.nan, 1.1, 0.9, np.nan])
    b = np.array([3.0, 3.1, np.nan, 2.9, 3.2])
    assert two_group_pvalue(a, b, test='t') < 0.001


def test_too_few_points_returns_nan():
    assert np.isnan(two_group_pvalue([1.0], [2.0, 3.0], test='t'))
    assert np.isnan(two_group_pvalue([np.nan, np.nan, 1.0], [1, 2, 3], test='t'))


def test_constant_groups_return_nan():
    assert np.isnan(two_group_pvalue([5, 5, 5], [5, 5, 5], test='t'))
    assert np.isnan(two_group_pvalue([5, 5, 5], [5, 5, 5], test='mann'))


def test_star_thresholds():
    assert significance_stars(0.0005) == '***'
    assert significance_stars(0.005) == '**'
    assert significance_stars(0.03) == '*'
    assert significance_stars(0.2) == 'ns'
    assert significance_stars(float('nan')) == 'ns'
    assert significance_stars(None) == 'ns'


def test_min_pvalues_ranks_separated_metabolite_first():
    rng = np.random.default_rng(5)
    y = np.array([0] * 6 + [1] * 6)
    X = rng.normal(0, 0.3, (12, 3))
    X[6:, 0] += 5.0                       # column 0: strong group separation
    # column 1: no separation; column 2: mild separation
    X[6:, 2] += 1.0
    pvals = metabolite_min_pvalues(X, y, test='t')
    order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
    assert order[0] == 0                  # most separated ranked first
    assert pvals[0] < pvals[1]


def test_min_pvalues_nan_when_untestable():
    y = np.array([0, 0, 1, 1])
    X = np.array([[1.0], [1.0], [1.0], [1.0]])   # constant -> undefined
    pvals = metabolite_min_pvalues(X, y, test='t')
    assert np.isnan(pvals[0])


def test_pairwise_ordered_by_span():
    g0 = np.zeros(10)
    g1 = np.ones(10) * 5
    g2 = np.ones(10) * 10
    res = pairwise_significance([g0, g1, g2], test='t')
    spans = [j - i for i, j, _, _ in res]
    assert spans == sorted(spans)
    assert len(res) == 3          # (0,1),(1,2),(0,2)
