# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/signal_structure.py
"""
How a study's signals and samples are organised.

STOCSY asks what co-varies with one signal. This asks the whole matrix: which
signals move together as a block (a molecule, a pool of molecules that rise and
fall with the tissue, a reference or water-tail artefact), what each block does
between the groups, and which axes of variation separate the samples.

    Cloarec, O. et al. (2005). Anal Chem 77(5):1282-1289 (STOCSY).
    Robinette, S.L. et al. -- CLASSY, cluster analysis statistical
    spectroscopy (reviewed in Robinette, Lindon & Nicholson (2013), Anal Chem
    85:5297-5303): clusters the peaks of one molecule by the correlation of the
    spectral variables -- the clustering used here, average linkage on 1 - r.
    Posma, J.M. et al. (2012). Subset optimization by reference matching
    (STORM), Anal Chem 84(24):10694-10701: a correlation between two signals
    can be driven by a difference between populations rather than by the
    molecule -- why every block here carries its group test, and why the
    within-group correlation exists.
    Wei, S. et al. (2011). Anal Chem 83(20):7616-7623 (RANSY): the ratio of two
    signals of one molecule is fixed.

Nothing here imports a GUI layer, and nothing changes the data it is handed.
"""
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# A module needs at least this many signals: two are a pair, which the STOCSY
# table already shows.
MODULE_MIN_SIZE = 3

# Signals read for modules and components must stand over the noise of the
# mean trace by this much: below it the per-sample areas are mostly noise.
STRUCTURE_MIN_SNR = 15.0


# ── groups ───────────────────────────────────────────────────────────────────

def split_factors(labels, sep='_'):
    """Two factors from labels like ``Flight_Younger``: when every label has
    the same number of tokens, at least two groups exist and each token
    position takes at least two values. Returns ``{}`` otherwise (the labels
    are then one factor)."""
    toks = [str(x).split(sep) for x in labels]
    n = {len(t) for t in toks}
    if len(n) != 1 or n.pop() != 2:
        return {}
    cols = [[t[i] for t in toks] for i in range(2)]
    if any(len(set(c)) < 2 for c in cols):
        return {}
    return {'A': cols[0], 'B': cols[1]}


def _rss(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return float(r @ r), np.linalg.matrix_rank(X)


def _dummies(labels):
    lab = np.asarray([str(x) for x in labels])
    names = sorted(set(lab))
    return np.column_stack([(lab == n).astype(float) for n in names[1:]]) \
        if len(names) > 1 else np.empty((len(lab), 0))


def _f_test(X_full, X_red, y):
    rf, kf = _rss(X_full, y)
    rr, kr = _rss(X_red, y)
    dfn, dfd = kf - kr, len(y) - kf
    if dfn <= 0 or dfd <= 0 or rf <= 0:
        return float('nan')
    F = ((rr - rf) / dfn) / (rf / dfd)
    return float(stats.f.sf(F, dfn, dfd))


def group_tests(y, labels):
    """p-values of a variable ``y`` (one value per sample) against the groups.

    ``{'group': p}`` from a one-way ANOVA; when the labels split in two factors
    (``split_factors``) also ``'A'``, ``'B'`` (each given the other) and
    ``'AxB'`` (interaction) from nested F tests. NaN where a test cannot be
    made (a group of one, no variation)."""
    y = np.asarray(y, dtype=float)
    lab = np.asarray([str(x) for x in labels])
    out = {}
    groups = [y[lab == u] for u in sorted(set(lab))]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups) and np.ptp(y) > 0:
        out['group'] = float(stats.f_oneway(*groups).pvalue)
    else:
        out['group'] = float('nan')
    fac = split_factors(labels)
    if fac:
        one = np.ones((len(y), 1))
        A, B = _dummies(fac['A']), _dummies(fac['B'])
        AB = np.column_stack([A[:, i] * B[:, j] for i in range(A.shape[1])
                              for j in range(B.shape[1])]) \
            if A.shape[1] and B.shape[1] else np.empty((len(y), 0))
        full = np.column_stack([one, A, B, AB])
        add = np.column_stack([one, A, B])
        out['AxB'] = _f_test(full, add, y)
        out['A'] = _f_test(add, np.column_stack([one, B]), y)
        out['B'] = _f_test(add, np.column_stack([one, A]), y)
    return out


def group_means(y, labels):
    """``{label: mean}`` of ``y`` by group, in sorted label order."""
    y = np.asarray(y, dtype=float)
    lab = np.asarray([str(x) for x in labels])
    return {u: float(y[lab == u].mean()) for u in sorted(set(lab))}


# ── modules ──────────────────────────────────────────────────────────────────

def signal_modules(R, *, r_cut=0.8, min_size=MODULE_MIN_SIZE):
    """Blocks of signals that move together.

    ``R`` is the correlation matrix of the signals read (k x k, NaN allowed:
    a flat signal relates to nothing). Average linkage on 1 - r, cut where
    the mean r inside a block falls under ``r_cut`` -- CLASSY. Returns the
    index lists of the blocks of at least ``min_size`` signals, biggest first,
    and the labels of every signal (singletons included)."""
    R = np.nan_to_num(np.asarray(R, dtype=float), nan=0.0)
    k = R.shape[0]
    if k < 2:
        return [], np.arange(1, k + 1)
    D = 1.0 - (R + R.T) / 2.0
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(D, 0.0, 2.0), checks=False), method='average')
    lab = fcluster(Z, t=1.0 - float(r_cut), criterion='distance')
    blocks = [np.flatnonzero(lab == c) for c in np.unique(lab)]
    blocks = [b for b in blocks if b.size >= min_size]
    blocks.sort(key=lambda b: -b.size)
    return blocks, lab


def mean_pairwise_r(R, members):
    """Mean r over the pairs of ``members``."""
    sub = np.asarray(R, dtype=float)[np.ix_(members, members)]
    n = len(members)
    if n < 2:
        return float('nan')
    return float((np.nansum(sub) - np.nansum(np.diag(sub))) / (n * (n - 1)))


def first_component(X):
    """Scores of the first principal component of the rows of ``X`` (signals x
    samples, each row standardised), its share of the variance and the signs
    of the loadings. Rows that do not vary are left out of the share."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = Xc.std(axis=1, keepdims=True)
    ok = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
    if ok.sum() < 1:
        return np.full(X.shape[1], np.nan), float('nan'), np.zeros(X.shape[0])
    Z = Xc[ok] / sd[ok]
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    load = np.zeros(X.shape[0])
    load[ok] = U[:, 0]
    # sign: most of the block rises with the score
    if load.sum() < 0:
        load, Vt = -load, -Vt
    return Vt[0] * S[0], float(S[0] ** 2 / np.sum(S ** 2)), load


# ── sample axes ──────────────────────────────────────────────────────────────

def sample_components(X, k=4):
    """Principal components of the samples over the signals ``X`` (signals x
    samples), each signal standardised. Returns ``{'scores': (samples x k),
    'loadings': (signals x k), 'variance': (k,)}`` -- variance as fractions."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = Xc.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    U, S, Vt = np.linalg.svd(Xc / sd, full_matrices=False)
    k = int(min(k, S.size))
    return {'scores': (Vt[:k] * S[:k, None]).T, 'loadings': U[:, :k],
            'variance': (S[:k] ** 2 / np.sum(S ** 2))}


def sample_distance(scores):
    """How far each sample stands from the others on its components, in
    standard deviations summed over the components (a screening number, not a
    p-value)."""
    sc = np.asarray(scores, dtype=float)
    sd = sc.std(axis=0)
    sd[sd == 0] = 1.0
    return np.abs((sc - sc.mean(axis=0)) / sd).sum(axis=1)
