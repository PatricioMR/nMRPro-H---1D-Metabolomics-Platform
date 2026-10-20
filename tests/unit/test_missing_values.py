# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Not-detected samples before a multivariate model, as MetaboAnalyst does it
(processing.missing_values)."""
import numpy as np
import pytest

from processing.missing_values import impute_lod, remove_missing_by_percent

NAN = np.nan


def test_a_metabolite_missing_in_a_fifth_or_more_of_the_samples_is_left_out():
    X = np.array([[1, 1, 1],
                  [1, NAN, 1],
                  [1, NAN, NAN],
                  [1, 1, 1],
                  [1, 1, 1]], dtype=float)
    # 0 %, 40 %, 20 % missing; the default threshold is 20 %
    assert remove_missing_by_percent(X).tolist() == [True, False, False]
    assert remove_missing_by_percent(X, percent=0.5).tolist() == [True, True, True]


def test_group_wise_a_metabolite_present_in_one_group_stays():
    # seen in every LPS sample, in no control: the rule must not drop it
    X = np.array([[NAN, 1], [NAN, 1], [NAN, 1],
                  [2.0, 1], [3.0, 1], [4.0, 1]])
    grp = ['ctrl'] * 3 + ['lps'] * 3
    assert remove_missing_by_percent(X, grp).tolist() == [False, True]
    assert remove_missing_by_percent(X, grp, group_wise=True).tolist() == [True, True]


def test_a_missing_cell_becomes_a_fifth_of_the_metabolites_minimum():
    X = np.array([[10.0, 4.0], [NAN, 2.0], [5.0, 0.0], [20.0, 3.0]])
    out, n = impute_lod(X)
    assert n == 2                         # zero counts as not detected
    assert out[1, 0] == pytest.approx(5.0 / 5)
    assert out[2, 1] == pytest.approx(2.0 / 5)
    assert out[0, 0] == 10.0               # what was measured is untouched


def test_group_wise_the_lod_is_the_groups_own():
    X = np.array([[NAN], [10.0], [NAN], [1.0]])
    out, _n = impute_lod(X, ['a', 'a', 'b', 'b'], group_wise=True)
    assert out[0, 0] == pytest.approx(2.0)
    assert out[2, 0] == pytest.approx(0.2)


def test_a_column_with_nothing_measured_is_not_filled():
    X = np.array([[NAN, 1.0], [NAN, 2.0]])
    out, n = impute_lod(X)
    assert np.isnan(out[:, 0]).all() and n == 0


def test_the_input_is_not_changed():
    X = np.array([[NAN, 1.0], [2.0, 1.0]])
    before = X.copy()
    impute_lod(X)
    assert np.array_equal(np.isnan(X), np.isnan(before))


def test_a_group_that_measured_nothing_takes_the_overall_lod_not_a_zero():
    X = np.array([[NAN], [NAN], [3.0], [4.0]])
    out, n = impute_lod(X, ['ctrl', 'ctrl', 'lps', 'lps'], group_wise=True)
    assert n == 2 and out[0, 0] == pytest.approx(3.0 / 5)
