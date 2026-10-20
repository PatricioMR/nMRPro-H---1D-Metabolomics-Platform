# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_quantification_math.py
"""The math helpers the Quantification stage still uses.

What used to be here went with the old engine: its PQN, its MCR-ALS solver,
its two effective-nH variants and the tier penalty that ranked signals. Those
answers now live in ``processing/quant_concentration.py`` and
``processing/quantification.py``, tested there against the functions that
actually run.
"""
import numpy as np
import pytest

from gui.stages._quant_helpers import _trapz, _cv








class TestTrapz:
    def test_rectangle(self):
        x = np.array([0.0, 1.0])
        y = np.array([2.0, 2.0])
        assert _trapz(y, x) == pytest.approx(2.0)

    def test_triangle(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        assert _trapz(y, x) == pytest.approx(1.0)

    def test_single_point(self):
        assert _trapz(np.array([5.0]), np.array([1.0])) == pytest.approx(0.0)

    def test_ppm_descending(self):
        # NMR ppm arrays are typically high→low
        x = np.array([2.0, 1.5, 1.0])
        y = np.array([0.0, 1.0, 0.0])
        result = abs(_trapz(y, x))
        assert result == pytest.approx(0.5, rel=0.01)


class TestCV:
    def test_identical_values(self):
        assert _cv([10.0, 10.0, 10.0]) == pytest.approx(0.0)

    def test_known_cv(self):
        # mean=10, sample std (ddof=1)=5.0 → CV=50%
        cv = _cv([5.0, 10.0, 15.0])
        assert 48 < cv < 52

    def test_single_value_is_nan(self):
        assert np.isnan(_cv([5.0]))

    def test_empty_is_nan(self):
        assert np.isnan(_cv([]))

    def test_negative_values_excluded(self):
        assert np.isnan(_cv([-1.0, -2.0]))

    def test_uses_sample_sd_ddof1_not_population(self):
        # Locks the convention: CV% must use sample SD (ddof=1), the same SD the
        # Groups ±SD bar now uses. A population SD (ddof=0) would be smaller and
        # make the reported CV% disagree with the visible error bar.
        vals = [8.0, 10.0, 12.0, 9.0, 11.0]
        m = float(np.mean(vals))
        expected = float(np.std(vals, ddof=1) / m * 100)
        population = float(np.std(vals, ddof=0) / m * 100)
        assert _cv(vals) == pytest.approx(expected)
        assert _cv(vals) != pytest.approx(population)




