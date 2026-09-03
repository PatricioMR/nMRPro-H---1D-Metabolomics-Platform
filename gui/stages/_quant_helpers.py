# gui/stages/_quant_helpers.py
"""
Standalone helper functions used by the Quantification stage.

Extracted from quantification_stage.py to keep each file focused:
  - _quant_helpers.py  → pure functions + constants (this file)
  - _quant_visualization.py → _QuantVisMixin  (detail plots / tooltips)
  - _quant_computation.py   → _QuantCompMixin (_run, VIP, update_sample_area, export)
"""
import numpy as np

# Integration & lineshape DSP now lives in the processing/ layer (MVC).
# Re-exported here so existing callers (`from ._quant_helpers import _trapz, …`)
# keep working unchanged.
from processing.integration import (
    _trapz, _lorentz_shape, _lorentz_func, _lorentz_amplitude,
    _linear_baseline, _fit_lorentz_mean, _fit_lorentz_mean_with_bl,
    _fit_lorentz_sample, _fit_multi_lorentz, jr_pointwise_multiplet_area,
)






# ── Robust / non-parametric statistics for J-Resolve ─────────────────────────
# qNMR fundamental: every peak of a compound yields the same concentration
# (area ∝ nH·conc).  Per-peak concentration probes should therefore agree.  CV
# (std/mean) assumes symmetric dispersion and is fragile with 2–3 probes — poor
# for skewed NMR data.  fold_range (max/min) is scale-free, distribution-free,
# robust at small n, and its tolerance is physically motivated by T1 differential
# saturation (≈ the 0.35–2.50 anchor_ratio window → ~2.5× fold).

def fold_range(values):
    """Max/min ratio of strictly-positive values.

    Returns 1.0 for a single positive value (perfectly consistent by
    definition) and inf if any non-positive/empty (cannot form a ratio).
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0],
                   dtype=float)
    if v.size == 0:
        return float('inf')
    if v.size == 1:
        return 1.0
    lo = float(v.min())
    hi = float(v.max())
    return hi / lo if lo > 0 else float('inf')


def bootstrap_ci_median(values, n_boot=1000, ci=0.95, min_n=4, seed=0):
    """Non-parametric percentile bootstrap CI for the median of `values`.

    Resamples `values` with replacement n_boot times, takes the median of each
    resample, and returns the percentile CI.

    Returns (point_median, lo, hi).  When fewer than `min_n` finite-positive
    values are available the bootstrap is unreliable, so (median, nan, nan) is
    returned — the caller must treat the CI as unavailable, never fabricate it.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0],
                   dtype=float)
    if v.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    point = float(np.median(v))
    if v.size < min_n:
        return (point, float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    meds = np.median(v[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(meds, alpha * 100.0))
    hi = float(np.percentile(meds, (1.0 - alpha) * 100.0))
    return (point, lo, hi)


def bootstrap_fraction_ci(values, n_boot=1000, ci=0.95, min_n=6, seed=0):
    """Non-parametric percentile bootstrap CI for the MEAN of a sample cohort.

    Used for cohort-level uncertainty (J-Resolve fractions / concentrations per
    group), resampling sample columns with replacement.  Distribution-free, so
    valid for skewed/log-normal metabolomics data.

    Returns (point_mean, lo, hi); (mean, nan, nan) when n < min_n (too few
    samples to bootstrap — report 'unavailable', do not invent an interval).
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    point = float(np.mean(v))
    if v.size < min_n:
        return (point, float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = np.mean(v[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, alpha * 100.0))
    hi = float(np.percentile(means, (1.0 - alpha) * 100.0))
    return (point, lo, hi)


def _pls_vip(X, y, n_components=2):
    """Compute PLS-DA VIP scores (no sklearn required).

    X : (n_samples, n_features)  — concentrations, NaN columns are skipped
    y : (n_samples,)             — group labels encoded as integers
    Returns array of shape (n_features,) with VIP scores (NaN for skipped cols).
    """
    n, p = X.shape
    n_comp = min(n_components, min(n - 1, p))
    if n_comp < 1 or n < 3:
        return np.full(p, float('nan'))

    col_means = np.nanmean(X, axis=0)
    nan_cols   = np.all(np.isnan(X), axis=0)
    Xc = np.where(np.isnan(X), col_means, X)
    x_std = Xc.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    Xc = (Xc - col_means) / x_std

    yc = y.astype(float)
    yc -= yc.mean()
    y_std = yc.std()
    if y_std < 1e-12:
        return np.full(p, float('nan'))
    yc /= y_std

    W = np.zeros((p, n_comp))
    T = np.zeros((n, n_comp))
    SS = np.zeros(n_comp)

    Xr, yr = Xc.copy(), yc.copy()
    for k in range(n_comp):
        w = Xr.T @ yr
        wn = np.linalg.norm(w)
        if wn < 1e-12:
            break
        w /= wn
        t = Xr @ w
        tt = float(t @ t)
        if tt < 1e-12:
            break
        q = float(yr @ t) / tt
        p_k = Xr.T @ t / tt
        W[:, k] = w
        T[:, k] = t
        SS[k] = q * q * tt
        Xr -= np.outer(t, p_k)
        yr -= q * t

    SS_total = SS.sum()
    if SS_total < 1e-12:
        return np.full(p, float('nan'))

    vip = np.sqrt(p * (W ** 2 @ SS) / SS_total)
    vip[nan_cols] = float('nan')
    return vip


def _cv(vals):
    """CV% of a 1-D array (sample std, ddof=1). NaN if < 2 finite positive values."""
    v = [x for x in vals if np.isfinite(x) and x > 0]
    if len(v) < 2:
        return float('nan')
    m = float(np.mean(v))
    return float(np.std(v, ddof=1) / m * 100) if m > 0 else float('nan')

























_STD_PRESETS = {
    'TSP':         {'ppm': '0.0',  'nh': '9'},
    'DSS':         {'ppm': '0.0',  'nh': '9'},
    'TMSP':        {'ppm': '0.0',  'nh': '9'},
    'Maleic acid': {'ppm': '6.0',  'nh': '2'},
    'Formate':     {'ppm': '8.46', 'nh': '1'},
    'Custom':      {'ppm': '',     'nh': ''},
}
