"""
Bivariate correlation with an OLS trend line and a confidence band.

Pearson product-moment correlation:
  Pearson, K. (1895), "Notes on regression and inheritance in the case of two
  parents", Proc. R. Soc. Lond. 58:240-242.
Spearman rank correlation:
  Spearman, C. (1904), "The proof and measurement of association between two
  things", Am. J. Psychol. 15(1):72-101.

The trend line is ordinary least squares (y ~ a + b·x) and the shaded band is
the (1-alpha) confidence interval for the regression *mean* response, from
Student's t with n-2 degrees of freedom. The reported coefficient (r/rho and
p) follows the chosen method — Pearson (parametric) or Spearman (rank-based,
non-parametric). Pairs with a non-finite x or y are dropped, never imputed.
"""
import numpy as np


def _finite_pairs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def correlation_coeff(x, y, method='pearson'):
    """Return (r, p) for the chosen method, or (nan, nan) when undefined
    (fewer than 3 finite pairs, or a constant input)."""
    from scipy import stats

    x, y = _finite_pairs(x, y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), float('nan')
    if method == 'spearman':
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def linear_fit_ci(x, y, alpha=0.05, n_points=100):
    """OLS fit of y on x with a confidence band for the mean response.

    Returns a dict with slope, intercept, the sample x/y used, the dense grid
    xs, the fitted line ys, and band bounds lower/upper. Returns None when the
    fit is undefined (fewer than 3 finite pairs, or x is constant).
    """
    from scipy import stats

    x, y = _finite_pairs(x, y)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    dof = n - 2
    resid = y - yhat
    s_err = np.sqrt(np.sum(resid ** 2) / dof) if dof > 0 else 0.0
    xbar = np.mean(x)
    sxx = np.sum((x - xbar) ** 2)

    xs = np.linspace(np.min(x), np.max(x), n_points)
    ys = slope * xs + intercept
    tval = stats.t.ppf(1.0 - alpha / 2.0, dof) if dof > 0 else 0.0
    se_mean = s_err * np.sqrt(1.0 / n + (xs - xbar) ** 2 / sxx)
    half = tval * se_mean
    return {
        'slope': float(slope), 'intercept': float(intercept),
        'x': x, 'y': y, 'xs': xs, 'ys': ys,
        'lower': ys - half, 'upper': ys + half,
    }
