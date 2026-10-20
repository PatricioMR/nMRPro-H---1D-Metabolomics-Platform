# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy.py
"""Unit tests for processing.stocsy — Graph STOCSY and Seed STOCSY."""
import math
import numpy as np
import pytest

from processing.stocsy import (
    compute_peak_correlation_matrix,
    compute_stocsy_spectrum,
)
from gui.stages._annotation_helpers import _stocsy_coherence


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_peaks(*area_vecs):
    """Build minimal peak dicts for compute_peak_correlation_matrix."""
    return [
        {'rank': i + 1, 'area_per_sample': np.array(v, dtype=float)}
        for i, v in enumerate(area_vecs)
    ]


# ── Grupo 1: compute_peak_correlation_matrix ─────────────────────────────────

class TestComputePeakCorrMatrix:

    def test_identical_vectors_r1(self):
        v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        peaks = _make_peaks(v, v)
        corr = compute_peak_correlation_matrix(peaks)
        assert (1, 2) in corr
        assert abs(corr[(1, 2)] - 1.0) < 1e-9

    def test_anticorrelated_vectors_r_neg1(self):
        v  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        nv = [-x for x in v]
        peaks = _make_peaks(v, nv)
        corr = compute_peak_correlation_matrix(peaks)
        assert (1, 2) in corr
        assert abs(corr[(1, 2)] - (-1.0)) < 1e-9

    def test_independent_vectors_small_r(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        peaks = _make_peaks(a.tolist(), b.tolist())
        corr = compute_peak_correlation_matrix(peaks)
        r = corr.get((1, 2), 0.0)
        assert abs(r) < 0.4   # random vectors should not be strongly correlated

    def test_fewer_than_min_samples_returns_empty(self):
        # 2 samples < min_samples=3 → empty dict
        peaks = _make_peaks([1.0, 2.0], [3.0, 4.0])
        corr = compute_peak_correlation_matrix(peaks, min_samples=3)
        assert corr == {}

    def test_exactly_min_samples_included(self):
        peaks = _make_peaks([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        corr = compute_peak_correlation_matrix(peaks, min_samples=3)
        assert (1, 2) in corr
        assert abs(corr[(1, 2)] - 1.0) < 1e-9

    def test_std_zero_peak_skipped_no_crash(self):
        # Constant peak → std=0 → pair omitted, no ZeroDivisionError
        v_const = [5.0] * 10
        v_vary  = list(range(1, 11))
        peaks = _make_peaks(v_const, v_vary)
        corr = compute_peak_correlation_matrix(peaks)
        assert (1, 2) not in corr   # pair must be omitted
        assert corr == {}

    def test_key_ordering_lower_rank_first(self):
        # Keys must always have the smaller rank first (upper triangle)
        v = list(range(1, 11))
        peaks = [
            {'rank': 5, 'area_per_sample': np.array(v, dtype=float)},
            {'rank': 2, 'area_per_sample': np.array(v, dtype=float)},
        ]
        corr = compute_peak_correlation_matrix(peaks)
        assert (2, 5) in corr
        assert (5, 2) not in corr

    def test_multiple_pairs(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])   # = 2a
        c = np.array([5.0, 4.0, 3.0, 2.0, 1.0])    # anticorrelated
        peaks = _make_peaks(a, b, c)
        corr = compute_peak_correlation_matrix(peaks, min_samples=3)
        assert (1, 2) in corr
        assert (1, 3) in corr
        assert (2, 3) in corr
        assert abs(corr[(1, 2)] - 1.0) < 1e-9
        assert abs(corr[(1, 3)] - (-1.0)) < 1e-9


# ── Grupo 2: _stocsy_coherence ────────────────────────────────────────────────

def _obs(rank, ppm):
    return {'rank': rank, 'ppm': ppm}


class TestStocyCoherence:

    def test_empty_peak_corr_returns_none(self):
        all_peaks_by_id = {'M1': [(1.0, 1.0), (2.0, 1.0)]}
        obs = [_obs(1, 1.0), _obs(2, 2.0)]
        assert _stocsy_coherence('M1', all_peaks_by_id, obs, 0.05, {}) is None

    def test_singleton_db_peak_returns_none(self):
        all_peaks_by_id = {'M1': [(1.0, 1.0)]}
        obs = [_obs(1, 1.0)]
        corr = {(1, 2): 0.95}
        assert _stocsy_coherence('M1', all_peaks_by_id, obs, 0.05, corr) is None

    def test_two_matched_peaks_returns_r(self):
        all_peaks_by_id = {'M1': [(1.0, 1.0), (2.0, 1.0)]}
        obs = [_obs(1, 1.0), _obs(2, 2.0)]
        corr = {(1, 2): 0.95}
        result = _stocsy_coherence('M1', all_peaks_by_id, obs, 0.05, corr)
        assert result is not None
        assert abs(result - 0.95) < 1e-9

    def test_three_matched_peaks_mean_of_three_pairs(self):
        all_peaks_by_id = {'M1': [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]}
        obs = [_obs(1, 1.0), _obs(2, 2.0), _obs(3, 3.0)]
        corr = {(1, 2): 0.90, (1, 3): 0.80, (2, 3): 0.70}
        result = _stocsy_coherence('M1', all_peaks_by_id, obs, 0.05, corr)
        assert result is not None
        assert abs(result - np.mean([0.90, 0.80, 0.70])) < 1e-9

    def test_only_one_rank_matched_returns_none(self):
        # One DB peak out of tolerance → only 1 matched rank → None
        all_peaks_by_id = {'M1': [(1.0, 1.0), (10.0, 1.0)]}
        obs = [_obs(1, 1.0)]   # only peak 1 in range
        corr = {(1, 2): 0.95}
        assert _stocsy_coherence('M1', all_peaks_by_id, obs, 0.05, corr) is None

    def test_greedy_no_double_match(self):
        # Two DB peaks very close → should only consume each observed once.
        # If greedy works correctly, the second DB peak has no available match
        # → matched_ranks has only 1 rank → returns None.
        all_peaks_by_id = {'M1': [(1.00, 1.0), (1.01, 1.0)]}
        obs = [_obs(1, 1.00)]   # only one observed peak
        corr = {(1, 1): 1.0}    # dummy — shouldn't be reached
        result = _stocsy_coherence('M1', all_peaks_by_id, obs, 0.05, corr)
        assert result is None

    def test_no_corr_entry_for_pair_returns_none(self):
        # Peaks matched but their rank pair not in peak_corr (e.g. n_samples<3)
        all_peaks_by_id = {'M1': [(1.0, 1.0), (2.0, 1.0)]}
        obs = [_obs(1, 1.0), _obs(2, 2.0)]
        corr = {(3, 4): 0.99}   # different ranks — not relevant
        result = _stocsy_coherence('M1', all_peaks_by_id, obs, 0.05, corr)
        assert result is None

    def test_unknown_met_id_returns_none(self):
        all_peaks_by_id = {}
        obs = [_obs(1, 1.0)]
        corr = {(1, 2): 0.9}
        assert _stocsy_coherence('UNKNOWN', all_peaks_by_id, obs, 0.05, corr) is None


# ── Grupo 3: compute_stocsy_spectrum ─────────────────────────────────────────

class TestComputeStocsySpectrum:

    def _build(self, n_pts=100, n_s=10, seed_idx=50):
        ppm = np.linspace(10.0, 0.0, n_pts)
        # All spectral points scale with sample index → perfect correlation
        conc = np.arange(1, n_s + 1, dtype=float)
        all_int = np.outer(np.ones(n_pts), conc)   # shape (n_pts, n_s)
        seed_ppm = float(ppm[seed_idx])
        return ppm, all_int, seed_ppm

    def test_perfect_covariation_r1_at_seed(self):
        ppm, all_int, seed_ppm = self._build()
        r = compute_stocsy_spectrum(ppm, all_int, seed_ppm, seed_window_ppm=0.1)
        # All points should have r≈1 because all_int is rank-1
        assert np.all(np.abs(r[np.isfinite(r)] - 1.0) < 1e-9)

    def test_no_seed_points_returns_nan(self):
        ppm = np.array([1.0, 2.0, 3.0])
        all_int = np.ones((3, 5))
        r = compute_stocsy_spectrum(ppm, all_int, seed_ppm=9.0,
                                    seed_window_ppm=0.01)
        assert np.all(np.isnan(r))

    def test_constant_seed_returns_nan(self):
        ppm = np.array([1.0, 2.0, 3.0])
        all_int = np.ones((3, 5))   # constant everywhere → std=0
        r = compute_stocsy_spectrum(ppm, all_int, seed_ppm=2.0,
                                    seed_window_ppm=0.5)
        assert np.all(np.isnan(r))

    def test_constant_column_gives_nan_at_that_point(self):
        ppm = np.array([3.0, 2.0, 1.0])
        all_int = np.tile(np.arange(1, 6, dtype=float), (3, 1))
        all_int[2, :] = 5.0   # point at 1.0 ppm is constant
        r = compute_stocsy_spectrum(ppm, all_int, seed_ppm=3.0,
                                    seed_window_ppm=0.1)
        assert np.isfinite(r[0])   # seed point itself
        assert np.isfinite(r[1])   # 2.0 ppm — varies
        assert np.isnan(r[2])      # 1.0 ppm — constant → NaN

    def test_output_shape_matches_ppm(self):
        ppm = np.linspace(10, 0, 200)
        all_int = np.random.default_rng(0).normal(0, 1, (200, 8))
        r = compute_stocsy_spectrum(ppm, all_int, seed_ppm=5.0)
        assert r.shape == (200,)
