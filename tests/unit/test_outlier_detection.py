# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Unit tests for processing.outlier_detection (Grubbs' test)."""
import numpy as np
import pytest

from processing.outlier_detection import grubbs_critical_value, grubbs_outliers


class TestCriticalValue:
    def test_undefined_below_3(self):
        assert np.isnan(grubbs_critical_value(2))

    def test_one_sided_smaller_than_two_sided(self):
        # α/n > α/2n → smaller t quantile → smaller critical value one-sided.
        c1 = grubbs_critical_value(10, 0.05, two_sided=False)
        c2 = grubbs_critical_value(10, 0.05, two_sided=True)
        assert c1 < c2

    def test_known_two_sided_value(self):
        # Textbook Grubbs two-sided critical, n=10, α=0.05 ≈ 2.29.
        assert grubbs_critical_value(10, 0.05, two_sided=True) == pytest.approx(2.29, abs=0.02)


class TestGrubbsOutliers:
    def test_clean_group_no_outlier(self):
        vals = [10.0, 10.1, 9.9, 10.2, 9.8]
        flags, _, _ = grubbs_outliers(vals, side='upper')
        assert not flags.any()

    def test_single_upper_outlier_flagged(self):
        vals = [10.0, 10.1, 9.9, 10.2, 25.0]
        flags, _, _ = grubbs_outliers(vals, side='upper')
        assert flags[4]
        assert flags.sum() == 1

    def test_point_close_to_mean_not_flagged(self):
        # One value sits far BELOW the others (small distance). One-sided upper
        # must NOT flag it — being close/typical is not an outlier. A two-sided
        # test would wrongly flag it.
        vals = [10.0, 10.0, 10.0, 10.0, 2.0]
        up, _, _ = grubbs_outliers(vals, side='upper')
        assert not up.any()
        two, _, _ = grubbs_outliers(vals, side='two-sided')
        assert two[4]

    def test_iteration_unmasks_second_outlier(self):
        # Two large outliers: a single pass inflates SD and can mask both.
        # Iterative removal must catch both.
        vals = [10.0, 10.1, 9.9, 10.2, 9.8, 30.0, 31.0]
        flags, _, _ = grubbs_outliers(vals, side='upper')
        assert flags[5] and flags[6]
        assert flags[:5].sum() == 0

    def test_below_3_returns_all_false(self):
        flags, g, c = grubbs_outliers([1.0, 5.0], side='upper')
        assert not flags.any()
        assert np.isnan(g).all() and np.isnan(c).all()

    def test_stats_aligned_to_input(self):
        vals = [10.0, 10.1, 9.9, 10.2, 25.0]
        flags, gstats, gcrits = grubbs_outliers(vals, side='upper')
        assert len(flags) == len(gstats) == len(gcrits) == len(vals)
        # Flagged point's statistic must exceed its critical value.
        assert gstats[4] > gcrits[4]

    def test_zero_variance_no_flag(self):
        flags, _, _ = grubbs_outliers([5.0, 5.0, 5.0, 5.0], side='upper')
        assert not flags.any()
