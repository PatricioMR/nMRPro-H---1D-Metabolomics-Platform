# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy_statistics.py
"""What r is measured on, and what it is worth: significance, normalisation,
within-group centring, the vectorised Seed STOCSY, and the reasons a peak is
left out."""
import numpy as np
import pytest
from scipy import stats

from processing.annotation_scoring import corroborating_partners
from processing.stocsy import (
    Basis, build_basis, compute_peak_correlation_matrix,
    compute_stocsy_spectrum, critical_r, dilution_spread, peak_correlation_report,
    peak_vectors, pqn_factors, r_interval, r_pvalue, rank_partners,
    stocsy_trace)


def _peaks(*vectors):
    return [{'rank': i + 1, 'area_per_sample': np.asarray(v, dtype=float)}
            for i, v in enumerate(vectors)]


# ── inference on r ───────────────────────────────────────────────────────────

class TestInference:

    @pytest.mark.parametrize('n, expected', [(5, 0.878), (6, 0.811),
                                             (8, 0.707), (12, 0.576)])
    def test_critical_r_matches_the_textbook_table(self, n, expected):
        assert critical_r(n - 2) == pytest.approx(expected, abs=1e-3)

    def test_critical_r_without_degrees_of_freedom_is_unreachable(self):
        assert critical_r(0) == 1.0
        assert critical_r(None) == 1.0

    def test_pvalue_agrees_with_scipy(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=14), rng.normal(size=14)
        r, p = stats.pearsonr(x, y)
        assert float(r_pvalue(r, 12)) == pytest.approx(p, rel=1e-9)

    def test_perfect_correlation_has_p_zero_not_nan(self):
        assert float(r_pvalue(1.0, 8)) == 0.0

    def test_interval_is_fishers_z(self):
        lo, hi = r_interval(0.8, 10)
        z, se = np.arctanh(0.8), 1 / np.sqrt(9)
        assert lo == pytest.approx(np.tanh(z - 1.959964 * se), abs=1e-5)
        assert hi == pytest.approx(np.tanh(z + 1.959964 * se), abs=1e-5)

    def test_interval_widens_with_few_samples(self):
        lo5, hi5 = r_interval(0.9, 3)
        lo30, hi30 = r_interval(0.9, 28)
        assert (hi5 - lo5) > (hi30 - lo30)

    def test_interval_is_nan_where_the_error_does_not_exist(self):
        lo, hi = r_interval(0.9, 1)
        assert np.isnan(lo) and np.isnan(hi)

    def test_spearman_interval_is_wider(self):
        lp, hp = r_interval(0.7, 10, 'pearson')
        ls, hs = r_interval(0.7, 10, 'spearman')
        assert (hs - ls) > (hp - lp)


# ── Basis ────────────────────────────────────────────────────────────────────

class TestBasis:

    def test_raw_pearson_is_numpys(self):
        rng = np.random.default_rng(2)
        M = rng.normal(size=(4, 12))
        R = Basis(12).correlate(M, M)
        assert np.allclose(R, np.corrcoef(M))

    def test_spearman_ignores_a_monotone_transform(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=15)
        y = x + rng.normal(scale=0.5, size=15)
        b = Basis(15, method='spearman')
        r1 = b.correlate(x[None], y[None])[0, 0]
        r2 = b.correlate(np.exp(3 * x)[None], y[None])[0, 0]
        assert r1 == pytest.approx(r2)
        assert r1 == pytest.approx(stats.spearmanr(x, y)[0])

    def test_unknown_method_is_refused(self):
        with pytest.raises(ValueError):
            Basis(10, method='kendall')
        with pytest.raises(ValueError):
            build_basis(np.ones((3, 5)), normalize='total')

    def test_pqn_removes_a_pure_dilution(self):
        """Independent molecules read r ~ 1 through a shared dilution; PQN
        gives it back."""
        rng = np.random.default_rng(4)
        base = rng.uniform(1, 5, size=(40, 1)) * np.ones((40, 14))
        base *= np.exp(rng.normal(0, 0.05, base.shape))   # each molecule's own
        dilution = np.exp(rng.normal(0, 0.4, 14))
        M = base * dilution
        raw = Basis(14).correlate(M[:2], M[:2])[0, 1]
        pqn = build_basis(M, normalize='pqn').correlate(M[:2], M[:2])[0, 1]
        assert raw > 0.9
        assert abs(pqn) < 0.7
        assert dilution_spread(M) == pytest.approx(0.4, abs=0.12)

    def test_pqn_factors_are_one_without_enough_signals(self):
        M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        assert np.allclose(pqn_factors(M), 1.0)

    def test_within_groups_removes_a_difference_between_groups(self):
        """Two unrelated molecules both higher in group B: r is high over all
        samples and gone within the groups."""
        rng = np.random.default_rng(5)
        g = ['A'] * 8 + ['B'] * 8
        step = np.array([0.0] * 8 + [3.0] * 8)
        x = step + rng.normal(0, 0.5, 16)
        y = step + rng.normal(0, 0.5, 16)
        raw = Basis(16).correlate(x[None], y[None])[0, 0]
        b = Basis(16, groups=g, within_groups=True)
        within = b.correlate(x[None], y[None])[0, 0]
        assert raw > 0.8
        assert abs(within) < 0.6
        assert b.within_groups and b.n_groups == 2
        assert b.dof == 16 - 2 - 1

    def test_within_groups_keeps_a_real_covariation(self):
        rng = np.random.default_rng(6)
        g = ['A'] * 8 + ['B'] * 8
        step = np.array([0.0] * 8 + [3.0] * 8)
        shared = rng.normal(0, 1.0, 16)
        x = step + shared + rng.normal(0, 0.1, 16)
        y = 2 * step + 2 * shared + rng.normal(0, 0.1, 16)
        r = Basis(16, groups=g, within_groups=True).correlate(
            x[None], y[None])[0, 0]
        assert r > 0.95

    def test_within_groups_without_groups_is_refused_and_said(self):
        b = Basis(10, within_groups=True)
        assert not b.within_groups
        assert any('groups' in n for n in b.notes)

    def test_within_groups_with_one_group_is_refused_and_said(self):
        b = Basis(10, groups=['A'] * 10, within_groups=True)
        assert not b.within_groups
        assert any('two groups' in n for n in b.notes)

    def test_a_group_of_one_is_left_out_and_said(self):
        g = ['A'] * 6 + ['B'] * 6 + ['C']
        b = Basis(13, groups=g, within_groups=True)
        assert b.within_groups and b.n == 12 and b.n_groups == 2
        assert 12 not in b.keep
        assert any('groups of one' in n for n in b.notes)

    def test_few_samples_are_flagged_with_the_bar_they_must_clear(self):
        note = ' '.join(Basis(4).notes)
        assert 'Only 4 samples' in note and '0.95' in note
        assert '0.88' in ' '.join(Basis(5).notes)
        assert '0.81' in ' '.join(Basis(6).notes)
        assert not Basis(8).notes          # 0.80 is already significant

    def test_a_flat_vector_correlates_with_nothing(self):
        M = np.array([[1.0, 2, 3, 4, 5, 6], [2.0, 2, 2, 2, 2, 2]])
        R = Basis(6).correlate(M, M)
        assert np.isnan(R[1]).all() and np.isnan(R[:, 1]).all()
        assert R[0, 0] == pytest.approx(1.0)


# ── Graph STOCSY: what goes in, what is said ─────────────────────────────────

class TestGraphStocsy:

    def test_default_minimum_is_five_samples(self):
        v = [1.0, 2.0, 3.0, 4.0]
        assert compute_peak_correlation_matrix(_peaks(v, v)) == {}
        v5 = v + [5.0]
        assert (1, 2) in compute_peak_correlation_matrix(_peaks(v5, v5))

    def test_every_reason_for_leaving_a_peak_out_is_named(self):
        ok = [1.0, 2, 3, 4, 5, 6]
        peaks = _peaks(ok, ok)
        peaks.append({'rank': 3, 'area_per_sample': [1, 2, np.nan, 4, 5, 6]})
        peaks.append({'rank': 4})
        peaks.append({'rank': 5, 'area_per_sample': [1.0, 2.0]})
        peaks.append({'rank': 6, 'area_per_sample': [1.0] * 6})
        peaks.append({'rank': 7, 'area_per_sample': [1, 2, 3, 4, 5, 6, 7.0]})
        rep = peak_correlation_report(peaks)
        assert rep['skipped'] == {3: 'non_finite', 4: 'no_areas',
                                  5: 'too_few_samples', 6: 'flat',
                                  7: 'length_mismatch'}
        assert (1, 2) in rep['corr'] and rep['n_samples'] == 6

    def test_a_nan_peak_does_not_disturb_the_others(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=8), rng.normal(size=8)
        with_nan = _peaks(a, b) + [{'rank': 9,
                                    'area_per_sample': [np.nan] * 8}]
        assert (compute_peak_correlation_matrix(with_nan)
                == compute_peak_correlation_matrix(_peaks(a, b)))

    def test_blanks_stay_out(self):
        rng = np.random.default_rng(8)
        a = np.append(rng.uniform(5, 6, 8), 0.0)
        b = np.append(rng.uniform(5, 6, 8), 0.0)
        peaks = _peaks(a, b)
        with_blank = compute_peak_correlation_matrix(peaks)[(1, 2)]
        without = compute_peak_correlation_matrix(
            peaks, included=list(range(8)))[(1, 2)]
        assert with_blank > 0.9 and abs(without) < 0.8

    def test_peak_vectors_match_included_columns(self):
        ranks, M, _ = peak_vectors(_peaks(range(10), range(10)),
                                   included=[0, 2, 4, 6, 8])
        assert M.shape == (2, 5) and list(M[0]) == [0, 2, 4, 6, 8]

    def test_rank_partners_orders_by_r_and_leaves_the_seed_out(self):
        rng = np.random.default_rng(9)
        seed = rng.normal(size=12)
        M = np.vstack([seed, seed + rng.normal(0, 0.3, 12),
                       -seed, rng.normal(size=12), np.ones(12)])
        out = rank_partners(seed, [10, 11, 12, 13, 14], M, Basis(12),
                            seed_rank=10)
        assert [d['rank'] for d in out][:3] == [11, 13, 12] \
            or out[0]['rank'] == 11
        assert 10 not in [d['rank'] for d in out]
        assert out[-1]['rank'] == 14 and np.isnan(out[-1]['r'])
        assert out[0]['n'] == 12 and out[0]['dof'] == 10
        assert out[0]['ci_low'] < out[0]['r'] < out[0]['ci_high']


# ── Seed STOCSY ──────────────────────────────────────────────────────────────

def _spectrum(n_samples=12, seed=0):
    """Two molecules with independent concentrations (A at 2.0 and 3.5 ppm,
    B at 1.0), a flat region, and noise."""
    rng = np.random.default_rng(seed)
    ppm = np.linspace(4.0, 0.5, 700)
    ca = rng.uniform(1, 4, n_samples)
    cb = rng.uniform(1, 4, n_samples)
    g = lambda c: np.exp(-0.5 * ((ppm - c) / 0.01) ** 2)
    X = (np.outer(g(2.0) + 0.5 * g(3.5), ca) + np.outer(g(1.0), cb)
         + rng.normal(0, 0.01, (ppm.size, n_samples)))
    return ppm, X, ca, cb


class TestSeedStocsy:

    def test_the_seeds_own_molecule_lights_up_and_the_other_does_not(self):
        ppm, X, ca, cb = _spectrum()
        r = compute_stocsy_spectrum(ppm, X, 2.0, 0.02)
        at = lambda c: r[np.argmin(abs(ppm - c))]
        assert at(2.0) > 0.99 and at(3.5) > 0.99
        assert abs(at(1.0)) < 0.8

    def test_covariance_is_the_height_and_is_zero_where_nothing_varies(self):
        ppm, X, ca, cb = _spectrum()
        seed = X[np.abs(ppm - 2.0) <= 0.02].mean(axis=0)
        tr = stocsy_trace(ppm, X, seed, Basis(12))
        i = np.argmin(abs(ppm - 2.0))
        assert tr['cov'][i] == pytest.approx(
            np.cov(seed, X[i], ddof=1)[0, 1], rel=1e-9)
        assert tr['std'].shape == ppm.shape

    def test_noise_floor_points_are_nan_not_a_correlation_with_noise(self):
        ppm, X, *_ = _spectrum()
        seed = X[np.abs(ppm - 2.0) <= 0.02].mean(axis=0)
        loose = stocsy_trace(ppm, X, seed, Basis(12))
        tight = stocsy_trace(ppm, X, seed, Basis(12), min_std_factor=3.0)
        floor = (ppm < 3.0) & (ppm > 2.5)
        assert np.isfinite(loose['r'][floor]).all()
        assert np.isnan(tight['r'][floor]).all()
        i = np.argmin(abs(ppm - 2.0))
        assert tight['r'][i] == pytest.approx(loose['r'][i])

    def test_masked_regions_are_nan(self):
        ppm, X, *_ = _spectrum()
        seed = X[np.abs(ppm - 2.0) <= 0.02].mean(axis=0)
        tr = stocsy_trace(ppm, X, seed, Basis(12),
                          masked_regions=[(3.0, 3.8)])
        inside = (ppm >= 3.0) & (ppm <= 3.8)
        assert np.isnan(tr['r'][inside]).all() and np.isnan(tr['cov'][inside]).all()
        assert np.isfinite(tr['r'][~inside]).any()

    def test_blank_columns_are_left_out_by_included(self):
        ppm, X, *_ = _spectrum()
        blank = np.zeros((ppm.size, 5))
        both = np.hstack([X, blank])
        r_all = compute_stocsy_spectrum(ppm, both, 1.0, 0.02)
        r_in = compute_stocsy_spectrum(ppm, both, 1.0, 0.02,
                                       included=list(range(12)))
        i = np.argmin(abs(ppm - 2.0))      # unrelated molecule A
        assert r_all[i] > 0.65             # the blanks manufactured it
        assert abs(r_in[i]) < 0.6
        assert r_in[i] == pytest.approx(
            compute_stocsy_spectrum(ppm, X, 1.0, 0.02)[i])

    def test_a_constant_seed_is_all_nan(self):
        ppm, X, *_ = _spectrum()
        tr = stocsy_trace(ppm, X, np.ones(12), Basis(12))
        assert np.isnan(tr['r']).all()

    def test_it_is_fast_on_a_real_sized_spectrum(self):
        import time
        rng = np.random.default_rng(0)
        ppm = np.linspace(10, -0.5, 65536)
        X = rng.normal(size=(65536, 24))
        t = time.perf_counter()
        stocsy_trace(ppm, X, X[30000], Basis(24))
        assert time.perf_counter() - t < 2.0


# ── the scoring reads the significance bar ───────────────────────────────────

class TestCorroborationBar:

    CORR = {(1, 2): 0.85}

    def test_five_samples_do_not_corroborate_at_0_85(self):
        assert corroborating_partners(1, [2], self.CORR, n_samples=5) == 0

    def test_twelve_samples_do(self):
        assert corroborating_partners(1, [2], self.CORR, n_samples=12) == 1

    def test_without_n_the_bar_is_unchanged(self):
        assert corroborating_partners(1, [2], self.CORR) == 1

    @pytest.mark.parametrize('n', [8, 12, 40])
    def test_from_eight_samples_nothing_changes(self, n):
        assert corroborating_partners(1, [2], {(1, 2): 0.80}, n_samples=n) == 1
