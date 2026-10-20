# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/baseline_correction.py
"""
Baseline correction algorithms for NMR spectroscopy.

Implements:
  - arPLS (asymmetrically reweighted Penalized Least Squares, Baek 2015)
  - SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping, Ryan 1988)
  - Rolling ball minimum
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def arPLS(y, lam=1e5, p=0.01, niter=100, return_weights=False):
    """
    arPLS baseline — asymmetrically reweighted Penalized Least Squares.

    Parameters
    ----------
    y    : ndarray (n,)  — input signal
    lam  : float         — smoothing parameter (larger = smoother); 1e4–1e7 for NMR
    p    : float         — asymmetry weight for peaks (0.001–0.05 for NMR)
    niter: int           — maximum iterations

    Returns
    -------
    baseline : ndarray (n,)
    weights  : ndarray (n,)  (only if return_weights=True)

    References
    ----------
    Baek, S.-J., et al. (2015). Analyst, 140, 250–257.
    """
    n = len(y)
    if n < 10:
        z = np.zeros(n)
        return (z, np.ones(n)) if return_weights else z

    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n),
                     format='csc', dtype=float)
    H = lam * D.T @ D
    w = np.ones(n)

    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(n, n), format='csc')
        z = spsolve(W + H, w * y)

        d = y - z
        d_neg = d[d < 0]
        if len(d_neg) < 2:
            break
        sigma = max(d_neg.std(), 1e-10)
        w_new = np.clip(
            np.where(d >= 0, p, (1 - p) * np.exp(2 * d / sigma)),
            1e-8, 1.0,
        )
        if np.sum(np.abs(w_new - w)) / n < 1e-4:
            w = w_new
            break
        w = w_new

    # baseline must not exceed signal
    z = np.minimum(z, y)
    return (z, w) if return_weights else z


# This function was called airPLS for years, but the implementation is arPLS
# (Baek 2015, asymmetrically reweighted) — a different algorithm from Zhang's
# airPLS (2010, adaptive iteratively reweighted). The alias keeps old imports
# and saved projects working.
airPLS = arPLS

# Canonical method names, and the spellings accepted for them.
_METHOD_ALIASES = {
    'arpls':    'arPLS',
    'airpls':   'arPLS',      # historical misnomer
    'snip':     'snip',
    'rolling':  'rolling',
    'constant': 'constant',
}


def normalize_method(method):
    """Canonical spelling of a baseline method name.

    Accepts the historical 'airPLS' spelling so projects saved before the
    rename still open on the method they were built with.
    """
    return _METHOD_ALIASES.get(str(method or '').strip().lower(), 'snip')


def snip(y, niter=40, **_kwargs):
    """
    SNIP — Statistics-sensitive Non-linear Iterative Peak-clipping (Ryan 1988).

    Uses the log-log-sqrt transform so the algorithm operates in a linearised
    space and is robust to sloping baselines.

    Parameters
    ----------
    y     : ndarray (n,)  — input signal
    niter : int           — clipping iterations (10–200; more = deeper)

    Returns
    -------
    baseline : ndarray (n,)

    References
    ----------
    Ryan, C. G., et al. (1988). Nucl. Instr. Meth. B, 34, 396–402.
    """
    n = len(y)
    if n < 5:
        return np.zeros(n)

    y_shift = y - y.min()
    z = np.log(np.log(np.sqrt(y_shift + 1) + 1) + 1)

    for p in range(1, niter + 1):
        z_new = z.copy()
        z_new[p:-p] = np.minimum(z[p:-p], (z[:n - 2 * p] + z[2 * p:]) / 2)
        z = z_new

    # Exact inverse of the LLS transform above:
    #   z = log(log(sqrt(v + 1) + 1) + 1)  ⇒  v = (exp(exp(z) - 1) - 1)² - 1
    # The trailing -1 is not optional. Dropping it inflates the baseline by one
    # intensity unit, which the clamp below then turns into baseline == y — the
    # correction returns an all-zero spectrum. Invisible on raw spectra with
    # large counts, fatal on normalised ones where 1 is the full scale.
    baseline = (np.exp(np.exp(z) - 1) - 1) ** 2 - 1 + y.min()
    return np.minimum(baseline, y)


def rolling_ball_minimum(y, ball_radius=50):
    """
    Rolling ball minimum baseline estimation (morphological erosion + smooth).

    Parameters
    ----------
    y           : ndarray (n,)
    ball_radius : int  — radius in data points

    Returns
    -------
    baseline : ndarray (n,)
    """
    n = len(y)
    if n < ball_radius:
        return np.zeros(n)

    from scipy.ndimage import minimum_filter1d, gaussian_filter1d
    eroded   = minimum_filter1d(y, size=ball_radius * 2 + 1)
    baseline = gaussian_filter1d(eroded, sigma=max(1, ball_radius // 4))
    return baseline


def estimate_baseline_level(y, method='median', edge_fraction=0.1):
    """
    Estimate a constant baseline level.

    Parameters
    ----------
    y             : ndarray (n,)
    method        : 'median', 'mean', 'mode', or 'edge'
    edge_fraction : fraction of data used by 'edge' method

    Returns
    -------
    float
    """
    if method == 'median':
        return float(np.median(y))
    elif method == 'mean':
        return float(np.mean(y))
    elif method == 'mode':
        hist, bin_edges = np.histogram(y, bins=100)
        idx = np.argmax(hist)
        return float((bin_edges[idx] + bin_edges[idx + 1]) / 2)
    elif method == 'edge':
        n_edge = max(5, int(len(y) * edge_fraction))
        edge_values = np.concatenate([y[:n_edge], y[-n_edge:]])
        return float(np.median(edge_values))
    return 0.0


def correct_baseline(ppm, intensity, method='arPLS', **kwargs):
    """
    Apply baseline correction to an NMR spectrum.

    Parameters
    ----------
    ppm       : ndarray (n,)
    intensity : ndarray (n,)
    method    : 'arPLS', 'snip', 'rolling', or 'constant'
                ('airPLS' is accepted as a historical alias of 'arPLS')
    **kwargs  : forwarded to the chosen method

    Returns
    -------
    corrected : ndarray (n,)
    baseline  : ndarray (n,)
    """
    if len(intensity) != len(ppm):
        raise ValueError("ppm and intensity must have the same length")

    intensity = np.asarray(intensity, dtype=float)
    method = normalize_method(method)

    if method == 'arPLS':
        lam   = kwargs.get('lam',   1e5)
        p     = kwargs.get('p',     0.01)
        niter = kwargs.get('niter', 100)
        baseline = arPLS(intensity, lam=lam, p=p, niter=niter)
    elif method == 'snip':
        niter    = kwargs.get('niter', kwargs.get('window', 40))
        baseline = snip(intensity, niter=niter)
    elif method == 'rolling':
        ball_radius = kwargs.get('ball_radius', 50)
        baseline    = rolling_ball_minimum(intensity, ball_radius=ball_radius)
    elif method == 'constant':
        est_method = kwargs.get('est_method',    'median')
        edge_frac  = kwargs.get('edge_fraction', 0.1)
        baseline   = np.full_like(
            intensity,
            estimate_baseline_level(intensity, method=est_method,
                                    edge_fraction=edge_frac),
        )
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    return intensity - baseline, baseline
