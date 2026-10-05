"""
Multivariate chemometrics for the Statistics stage.

Pure, dependency-light (numpy only) routines so the projection maths is unit
tested independently of the GUI.

OPLS-DA reference:
    Trygg, J.; Wold, S. "Orthogonal projections to latent structures (O-PLS)."
    Journal of Chemometrics 2002, 16 (3), 119-128.  doi:10.1002/cem.695

PERMANOVA reference:
    Anderson, M. J. "A new method for non-parametric multivariate analysis of
    variance." Austral Ecology 2001, 26 (1), 32-46.
    doi:10.1111/j.1442-9993.2001.01070.pp.x
"""
from __future__ import annotations

import numpy as np


def pls_r2x_per_component(T, P, X):
    """Fraction of total X variance captured by each latent component.

    For PLS/PCA with score matrix ``T`` (n_samples × A) and loading matrix
    ``P`` (n_features × A), the X-variance explained by component k is

        R2X_k = ||t_k||² · ||p_k||² / ||X||²

    ``X`` must be the SAME (centred/scaled) matrix the model was fitted on.
    The naive ``var(t_k) / Σ var(X_j)`` used previously drops the ``||p_k||²``
    loading-norm factor and therefore under-reports the explained variance.

    Returns an array of percentages, one per component.
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    ssx = float(np.sum(np.asarray(X, dtype=float) ** 2))
    if ssx <= 0 or T.ndim != 2 or P.ndim != 2:
        return np.zeros(T.shape[1] if T.ndim == 2 else 0)
    a = min(T.shape[1], P.shape[1])
    return np.array([float(np.sum(T[:, k] ** 2) * np.sum(P[:, k] ** 2) / ssx * 100.0)
                     for k in range(a)])


def plsda_binarize(labels):
    """Class labels → PLS-DA response Y.

    Two classes → a single 0/1 column; more → one-hot columns. Building Y from
    the labels (instead of regressing on ordinal 0,1,2,…) avoids imposing a false
    ordering between classes.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size <= 2:
        ref = classes[-1] if classes.size else None
        return (labels == ref).astype(float).reshape(-1, 1)
    Y = np.zeros((labels.size, classes.size), dtype=float)
    for i, c in enumerate(classes):
        Y[labels == c, i] = 1.0
    return Y


def plsda_cv_q2(X, labels, n_comp, n_splits, seed=0):
    """Stratified-CV Q² for PLS-DA, fitting the binarized Y.

    Stratifying on the class labels keeps every training fold's Y non-constant
    (so the PLS deflation never hits the "y residual is constant" degeneracy),
    and the number of components is clamped to what each fold can support. Q² is
    the cross-validated predictive R²:  1 − PRESS / TSS.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    Y = plsda_binarize(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y)
    for tr, te in skf.split(X, labels):
        nc = max(1, min(int(n_comp), len(tr) - 1, X.shape[1]))
        m = PLSRegression(n_components=nc, scale=False).fit(X[tr], Y[tr])
        preds[te] = m.predict(X[te])
    ss_res = float(np.sum((Y - preds) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def plsda_permutation_p(X, labels, n_comp, n_splits, n_perm, seed=0):
    """Permutation test for PLS-DA Q² (stratified CV, label shuffling).

    Returns ``(q2_observed, p_value)`` where p = (1 + #{Q²_perm >= Q²_obs}) /
    (n_perm + 1). Permuting the labels preserves class sizes, so every permuted
    fit stays stratified and non-degenerate.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    q2_obs = plsda_cv_q2(X, labels, n_comp, n_splits, seed=seed)
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(int(n_perm)):
        perm = rng.permutation(labels)
        if plsda_cv_q2(X, perm, n_comp, n_splits, seed=seed) >= q2_obs:
            ge += 1
    return q2_obs, ge / (int(n_perm) + 1)


def _within_group_ss(X, codes, n_groups):
    """Σ over groups of the squared Euclidean distance to the group centroid."""
    ssw = 0.0
    for g in range(n_groups):
        Xg = X[codes == g]
        ssw += float(np.sum((Xg - Xg.mean(axis=0)) ** 2))
    return ssw


def permanova(X, labels, n_perm=999, seed=0):
    """One-factor PERMANOVA on Euclidean distances (Anderson 2001).

    With Euclidean distance the distance-matrix sums of squares equal the
    ordinary ones around the centroids, so they are computed on ``X`` directly:

        F  = (SS_between / (g-1)) / (SS_within / (N-g))
        R² = SS_between / SS_total

    The p-value is the share of label permutations whose F reaches the observed
    one, counting the observed arrangement: (1 + #{F_perm >= F_obs}) /
    (n_perm + 1). Permuting labels keeps the group sizes. A fixed ``seed`` makes
    the same data give the same p every time the plot is redrawn.

    Returns ``{'F', 'R2', 'p', 'n_perm', 'n', 'groups'}``, or ``None`` when the
    test is not defined (fewer than 2 groups, or no sample beyond one per group,
    so there is no within-group variation to compare against).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    classes, codes = np.unique(labels, return_inverse=True)
    n, g = X.shape[0], classes.size
    if g < 2 or n - g < 1:
        return None
    sst = float(np.sum((X - X.mean(axis=0)) ** 2))
    if sst <= 0:
        return None

    def pseudo_f(c):
        ssw = _within_group_ss(X, c, g)
        return ((sst - ssw) / (g - 1)) / (ssw / (n - g)) if ssw > 0 else np.inf

    f_obs = pseudo_f(codes)
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(int(n_perm)):
        if pseudo_f(rng.permutation(codes)) >= f_obs * (1 - 1e-12):
            ge += 1
    ssw_obs = _within_group_ss(X, codes, g)
    return {'F': float(f_obs), 'R2': 1.0 - ssw_obs / sst,
            'p': ge / (int(n_perm) + 1), 'n_perm': int(n_perm),
            'n': n, 'groups': g}


def opls_da(X, y, n_ortho=1):
    """Orthogonal PLS-DA for a single (binary) response — Trygg & Wold 2002.

    Separates the X variation correlated with ``y`` (one *predictive* component)
    from the variation orthogonal to ``y`` (``n_ortho`` *orthogonal* components),
    which is the whole point of OPLS and is impossible to recover by merely
    Gram-Schmidt-ing two ordinary PLS scores (those are already orthogonal, so
    the correction is a no-op).

    Parameters
    ----------
    X : (n_samples, n_features) — assumed centred/scaled (re-centred defensively).
    y : (n_samples,) class indicator (e.g. 0/1); centred internally.
    n_ortho : number of orthogonal components to remove (>= 0).

    Returns dict with:
        't_pred'    : (n,)            predictive score
        't_ortho'   : (n, n_ortho)    orthogonal scores (may have 0 columns)
        'r2x_pred'  : float           % X variance of the predictive component
        'r2x_ortho' : (n_ortho,)      % X variance per orthogonal component
        'w_pred'    : (n_features,)   predictive weight
        'p_pred'    : (n_features,)   predictive loading
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    ssx = float(np.sum(Xc ** 2))
    if ssx <= 0:
        raise ValueError('OPLS-DA: X has no variance.')

    # Predictive weight from the original X–y covariance (fixed throughout).
    w = Xc.T @ yc
    nw = float(np.linalg.norm(w))
    if nw < 1e-12:
        raise ValueError('OPLS-DA: no covariance between X and the class vector.')
    w = w / nw

    ww = float(np.sum(w * w))
    Xdef = Xc.copy()
    t_orthos, p_orthos, r2x_orth = [], [], []
    for _ in range(max(0, int(n_ortho))):
        t = Xdef @ w / ww
        p = (Xdef.T @ t) / float(np.sum(t * t))
        # Orthogonal weight: part of the loading not aligned with w.
        w_o = p - (float(np.sum(w * p)) / ww) * w
        no = float(np.linalg.norm(w_o))
        if no < 1e-12:
            break
        w_o = w_o / no
        t_o = Xdef @ w_o / float(np.sum(w_o * w_o))
        p_o = (Xdef.T @ t_o) / float(np.sum(t_o * t_o))
        Xdef = Xdef - t_o @ p_o.T                     # deflate orthogonal variation
        t_orthos.append(t_o.ravel())
        p_orthos.append(p_o.ravel())
        r2x_orth.append(float(np.sum(t_o ** 2) * np.sum(p_o ** 2) / ssx * 100.0))

    # Predictive component on the Y-orthogonal-deflated X.
    t_p = Xdef @ w / ww
    p_p = (Xdef.T @ t_p) / float(np.sum(t_p * t_p))
    r2x_pred = float(np.sum(t_p ** 2) * np.sum(p_p ** 2) / ssx * 100.0)

    return {
        't_pred':   t_p.ravel(),
        't_ortho':  (np.array(t_orthos).T if t_orthos
                     else np.zeros((X.shape[0], 0))),
        'r2x_pred':  r2x_pred,
        'r2x_ortho': np.array(r2x_orth),
        'w_pred':    w.ravel(),
        'p_pred':    p_p.ravel(),
    }
