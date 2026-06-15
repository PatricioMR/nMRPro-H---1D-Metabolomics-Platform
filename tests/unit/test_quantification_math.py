# tests/unit/test_quantification_math.py
"""Tests for pure math helpers in gui/stages/_quant_helpers.py"""
import numpy as np
import pytest

# Import helpers directly (they're module-level functions)
from gui.stages._quant_helpers import _trapz, _cv, _pqn, _mcr_als


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


class TestPQN:
    def test_no_normalization_when_too_few(self):
        areas = {'A': 10.0, 'B': 20.0}
        ref   = {'A': 10.0, 'B': 20.0}
        result, f = _pqn(areas, ref)
        assert f == pytest.approx(1.0)

    def test_scales_correctly(self):
        areas = {'A': 20.0, 'B': 40.0, 'C': 60.0}
        ref   = {'A': 10.0, 'B': 20.0, 'C': 30.0}
        result, f = _pqn(areas, ref)
        # All quotients = 2.0, so median = 2.0 → each area halved
        assert f == pytest.approx(2.0)
        assert result['A'] == pytest.approx(10.0)

    def test_missing_ref_key_skipped(self):
        areas = {'A': 10.0, 'B': 20.0, 'C': 30.0}
        ref   = {'A': 10.0, 'C': 30.0}  # B missing
        result, f = _pqn(areas, ref)
        assert 'B' in result


class TestMCRALS:
    def test_output_shape(self):
        X = np.random.default_rng(0).random((50, 6))
        C, S, _ = _mcr_als(X, n_components=1)
        assert C.shape == (6, 1)
        assert S.shape == (50, 1)

    def test_non_negative(self):
        X = np.abs(np.random.default_rng(0).random((30, 4)))
        C, S, _ = _mcr_als(X, n_components=1)
        assert (C >= 0).all()
        assert (S >= 0).all()

    def test_reconstruction_quality(self):
        # Single Lorentzian, 4 identical samples → perfect reconstruction
        x   = np.linspace(0, 1, 100)
        col = (0.01**2 / ((x - 0.5)**2 + 0.01**2)).reshape(-1, 1)
        X   = np.hstack([col] * 4)
        C, S, _ = _mcr_als(X, n_components=1, max_iter=200)
        recon = S @ C.T
        rel_err = np.linalg.norm(X - recon) / np.linalg.norm(X)
        assert rel_err < 0.05, f'Reconstruction error too high: {rel_err:.3f}'
