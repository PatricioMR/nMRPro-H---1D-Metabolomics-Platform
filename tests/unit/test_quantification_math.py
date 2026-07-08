# tests/unit/test_quantification_math.py
"""Tests for pure math helpers in gui/stages/_quant_helpers.py"""
import numpy as np
import pytest

# Import helpers directly (they're module-level functions)
from gui.stages._quant_helpers import (
    _trapz, _cv, _pqn, _mcr_als, _effective_nh, _effective_nh_source,
    multiplet_tier_penalty, multiplet_sdj_missing, SINGLET_TIER_PENALTY)

_SINGLET_MULTS = {'s', 'br s', 'brs', 'bs', ''}


class TestMultipletTierPenalty:
    """#2: continuous tier penalty replacing the binary s_dj>=1.0 cliff."""

    def test_perfect_multiplet_near_zero(self):
        assert multiplet_tier_penalty(False, 1.0) == pytest.approx(0.0)

    def test_poor_multiplet_near_one(self):
        assert multiplet_tier_penalty(False, 0.0) == pytest.approx(1.0)

    def test_singlet_at_boundary(self):
        assert multiplet_tier_penalty(True, 1.0) == pytest.approx(SINGLET_TIER_PENALTY)

    def test_near_good_multiplet_beats_singlet(self):
        # s_dj=0.98 → penalty 0.02 < 0.5 (singlet) → multiplet wins (no more cliff)
        assert multiplet_tier_penalty(False, 0.98) < multiplet_tier_penalty(True, 1.0)

    def test_poor_multiplet_loses_to_singlet(self):
        # s_dj=0.3 → penalty 0.7 > 0.5 → singlet wins
        assert multiplet_tier_penalty(False, 0.3) > multiplet_tier_penalty(True, 1.0)

    def test_monotonic_in_sdj(self):
        assert (multiplet_tier_penalty(False, 0.9)
                < multiplet_tier_penalty(False, 0.5)
                < multiplet_tier_penalty(False, 0.1))

    def test_clamps_out_of_range(self):
        assert multiplet_tier_penalty(False, 1.5) == pytest.approx(0.0)
        assert multiplet_tier_penalty(False, -0.5) == pytest.approx(1.0)

    def test_non_numeric_sdj_treated_as_zero(self):
        assert multiplet_tier_penalty(False, None) == pytest.approx(1.0)


class TestMultipletSdjMissing:
    """#1: multiplets with unknown s_dj are blocked; singlets never are."""

    def test_singlet_never_blocked(self):
        assert multiplet_sdj_missing('s', False, _SINGLET_MULTS) is False
        assert multiplet_sdj_missing('', False, _SINGLET_MULTS) is False

    def test_multiplet_with_sdj_allowed(self):
        assert multiplet_sdj_missing('d', True, _SINGLET_MULTS) is False

    def test_multiplet_without_sdj_blocked(self):
        assert multiplet_sdj_missing('d', False, _SINGLET_MULTS) is True
        assert multiplet_sdj_missing('t', False, _SINGLET_MULTS) is True

    def test_case_insensitive(self):
        assert multiplet_sdj_missing('  D  ', False, _SINGLET_MULTS) is True


class TestEffectiveNhFractionalProtons:
    """F1: fractional per-signal nH from the DB must be preserved, not
    truncated/floored to 1.0 (which biased eff_nH up to ~3×)."""

    def test_singlet_fractional_protons_preserved(self):
        hit = {'protons': 0.5, 'multiplicity': 's', 'predicted_lines': []}
        assert abs(_effective_nh(hit, 1.0) - 0.5) < 1e-9

    def test_source_variant_preserves_fraction(self):
        hit = {'protons': 0.33, 'multiplicity': 's', 'predicted_lines': []}
        nh, _src = _effective_nh_source(hit, 1.0)
        assert abs(nh - 0.33) < 1e-9

    def test_nonpositive_protons_floored_to_one(self):
        hit = {'protons': 0.0, 'multiplicity': 's', 'predicted_lines': []}
        assert _effective_nh(hit, 1.0) == 1.0


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
