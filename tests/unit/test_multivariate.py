"""Unit tests for processing.multivariate (PLS R2X, OPLS-DA)."""
import numpy as np
import pytest

from processing.multivariate import (
    pls_r2x_per_component, opls_da,
    plsda_binarize, plsda_cv_q2, plsda_permutation_p, permanova)


def _two_class(n=24, p=30, seed=0):
    """Synthetic 2-class data: a class shift on a few features + noise."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=float)
    X = rng.normal(size=(n, p))
    X[:, :5] += y[:, None] * 3.0          # class signal on first 5 features
    X[:, 5] += rng.normal(size=n) * 5.0   # large variance ORTHOGONAL to class
    X -= X.mean(axis=0)
    return X, y


class TestR2X:
    def test_matches_manual_formula(self):
        from sklearn.cross_decomposition import PLSRegression
        X, y = _two_class()
        pls = PLSRegression(n_components=3, scale=False).fit(X, y.reshape(-1, 1))
        T, P = pls.x_scores_, pls.x_loadings_
        r2x = pls_r2x_per_component(T, P, X)
        ssx = np.sum(X ** 2)
        manual = [np.sum(T[:, k] ** 2) * np.sum(P[:, k] ** 2) / ssx * 100
                  for k in range(3)]
        assert np.allclose(r2x, manual)

    def test_exceeds_naive_var_estimate(self):
        # The corrected R2X must differ from (and here exceed) the old
        # var(t_k)/Σvar(X) under-estimate that dropped the loading norm.
        from sklearn.cross_decomposition import PLSRegression
        X, y = _two_class()
        pls = PLSRegression(n_components=2, scale=False).fit(X, y.reshape(-1, 1))
        T, P = pls.x_scores_, pls.x_loadings_
        r2x = pls_r2x_per_component(T, P, X)
        naive = [np.var(T[:, k]) / np.sum(np.var(X, axis=0)) * 100 for k in range(2)]
        assert not np.allclose(r2x, naive)

    def test_zero_variance_X(self):
        T = np.ones((5, 2)); P = np.ones((4, 2))
        assert np.all(pls_r2x_per_component(T, P, np.zeros((5, 4))) == 0)


class TestOPLSDA:
    def test_predictive_correlates_with_class(self):
        X, y = _two_class()
        res = opls_da(X, y, n_ortho=1)
        r = abs(np.corrcoef(res['t_pred'], y)[0, 1])
        assert r > 0.8, f'predictive score should track the class, got r={r:.2f}'

    def test_orthogonal_decorrelated_from_class(self):
        # The orthogonal component captures Y-unrelated variance → must NOT
        # correlate with the class label (this is what a real OPLS delivers and
        # the old fake Gram-Schmidt did not).
        X, y = _two_class()
        res = opls_da(X, y, n_ortho=1)
        assert res['t_ortho'].shape[1] == 1
        r = abs(np.corrcoef(res['t_ortho'][:, 0], y)[0, 1])
        assert r < 0.3, f'orthogonal score must be ~uncorrelated with class, got r={r:.2f}'

    def test_predictive_orthogonal_to_orthogonal(self):
        X, y = _two_class()
        res = opls_da(X, y, n_ortho=1)
        tp, to = res['t_pred'], res['t_ortho'][:, 0]
        cos = abs(np.dot(tp, to) / (np.linalg.norm(tp) * np.linalg.norm(to)))
        assert cos < 1e-6

    def test_not_identical_to_plain_pls_lv2(self):
        # Guard against the old no-op: OPLS orthogonal score must genuinely
        # differ from ordinary PLS LV2.
        from sklearn.cross_decomposition import PLSRegression
        X, y = _two_class()
        res = opls_da(X, y, n_ortho=1)
        pls = PLSRegression(n_components=2, scale=False).fit(X, y.reshape(-1, 1))
        lv2 = pls.x_scores_[:, 1]
        r = abs(np.corrcoef(res['t_ortho'][:, 0], lv2)[0, 1])
        assert r < 0.999

    def test_r2x_values_in_range(self):
        X, y = _two_class()
        res = opls_da(X, y, n_ortho=2)
        assert 0 <= res['r2x_pred'] <= 100
        assert np.all((res['r2x_ortho'] >= 0) & (res['r2x_ortho'] <= 100))

    def test_zero_ortho_components(self):
        X, y = _two_class()
        res = opls_da(X, y, n_ortho=0)
        assert res['t_ortho'].shape[1] == 0
        assert abs(np.corrcoef(res['t_pred'], y)[0, 1]) > 0.8

    def test_no_covariance_raises(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 4)); X -= X.mean(0)
        y = np.zeros(10)              # constant class → no covariance
        with pytest.raises(ValueError):
            opls_da(X, y)


class TestPLSDAValidation:
    def test_binarize_two_class_single_column(self):
        Y = plsda_binarize([0, 0, 1, 1])
        assert Y.shape == (4, 1)
        assert np.array_equal(Y.ravel(), [0, 0, 1, 1])

    def test_binarize_three_class_one_hot(self):
        Y = plsda_binarize([0, 1, 2, 0])
        assert Y.shape == (4, 3)
        assert np.array_equal(Y.sum(axis=1), [1, 1, 1, 1])

    def test_q2_high_for_separable_classes(self):
        X, y = _two_class(seed=2)
        q2 = plsda_cv_q2(X, y, n_comp=2, n_splits=4)
        assert q2 > 0.3, f'separable data should give positive Q², got {q2:.2f}'

    def test_q2_low_for_random_labels(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(24, 30)); X -= X.mean(0)
        y = np.array([0] * 12 + [1] * 12)
        q2 = plsda_cv_q2(X, y, n_comp=2, n_splits=4)
        assert q2 < 0.3, f'noise should give low Q², got {q2:.2f}'

    def test_permutation_p_significant_for_separable(self):
        X, y = _two_class(seed=4)
        _, p = plsda_permutation_p(X, y, n_comp=2, n_splits=4, n_perm=50)
        assert p < 0.1

    def test_no_constant_y_warning(self):
        # Stratified folds + per-fold n_comp clamp must not trip sklearn's
        # "y residual is constant" UserWarning that the old KFold path produced.
        import warnings
        X, y = _two_class(seed=5)
        with warnings.catch_warnings():
            warnings.simplefilter('error')          # any warning → failure
            plsda_cv_q2(X, y, n_comp=2, n_splits=4)
            plsda_permutation_p(X, y, n_comp=2, n_splits=4, n_perm=10)

    def test_three_class_cv_runs(self):
        rng = np.random.default_rng(6)
        y = np.array([0] * 8 + [1] * 8 + [2] * 8)
        X = rng.normal(size=(24, 30)); X[:, :5] += y[:, None] * 2.0; X -= X.mean(0)
        q2 = plsda_cv_q2(X, y, n_comp=2, n_splits=4)
        assert np.isfinite(q2)


def _anderson_f_r2(X, labels):
    """Pseudo-F and R² straight from the distance matrix (Anderson 2001, eqs.
    2-4): SS_T = Σ d²/N over pairs, SS_W = Σ_g Σ d²/n_g over within-group pairs."""
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    n = len(labels)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    sst = d2[np.triu_indices(n, 1)].sum() / n
    ssw = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        sub = d2[np.ix_(idx, idx)]
        ssw += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    g = len(np.unique(labels))
    return ((sst - ssw) / (g - 1)) / (ssw / (n - g)), (sst - ssw) / sst


class TestPermanova:
    def test_f_and_r2_match_anderson_distance_formula(self):
        X, y = _two_class(seed=7)
        res = permanova(X, y, n_perm=0)
        f, r2 = _anderson_f_r2(X, y)
        assert res['F'] == pytest.approx(f)
        assert res['R2'] == pytest.approx(r2)

    def test_three_groups_unequal_sizes(self):
        rng = np.random.default_rng(8)
        y = np.array([0] * 5 + [1] * 8 + [2] * 3)
        X = rng.normal(size=(16, 10)) + y[:, None] * 0.5
        res = permanova(X, y, n_perm=0)
        f, r2 = _anderson_f_r2(X, y)
        assert res['F'] == pytest.approx(f)
        assert res['R2'] == pytest.approx(r2)
        assert res['groups'] == 3 and res['n'] == 16

    def test_significant_for_separated_groups(self):
        X, y = _two_class(seed=9)
        assert permanova(X, y, n_perm=199)['p'] < 0.05

    def test_not_significant_for_random_labels(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(24, 30))
        y = np.array([0] * 12 + [1] * 12)
        assert permanova(X, y, n_perm=199)['p'] > 0.05

    def test_p_floor_is_one_over_n_perm_plus_one(self):
        X = np.vstack([np.zeros((6, 3)), np.full((6, 3), 100.0)])
        X += np.random.default_rng(11).normal(size=X.shape) * 0.01
        y = np.array([0] * 6 + [1] * 6)
        assert permanova(X, y, n_perm=99)['p'] == pytest.approx(1 / 100)

    def test_same_seed_same_p(self):
        X, y = _two_class(seed=12)
        assert permanova(X, y, n_perm=99)['p'] == permanova(X, y, n_perm=99)['p']

    def test_string_labels(self):
        X, y = _two_class(seed=13)
        names = np.where(y == 0, 'CTRL', 'MSCs EVs')
        assert permanova(X, names, n_perm=0)['F'] == pytest.approx(
            permanova(X, y, n_perm=0)['F'])

    def test_undefined_cases_return_none(self):
        X = np.random.default_rng(14).normal(size=(6, 4))
        assert permanova(X, [0] * 6) is None                    # one group
        assert permanova(X[:3], [0, 1, 2]) is None              # no within-group df
        assert permanova(np.zeros((6, 4)), [0, 0, 0, 1, 1, 1]) is None  # no variance
