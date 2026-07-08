"""
Two-group hypothesis tests and significance annotation for box/violin panels.

Welch's t-test (unequal variances):
  Welch, B. L. (1947), "The generalization of 'Student's' problem when several
  different population variances are involved", Biometrika 34(1-2):28-35.

Mann-Whitney U (rank-sum, distribution-free):
  Mann, H. B. & Whitney, D. R. (1947), "On a Test of Whether one of Two Random
  Variables is Stochastically Larger than the Other", Ann. Math. Statist.
  18(1):50-60.

These are the two tests exposed in the Box/Violin panel: parametric (Welch t)
and non-parametric (Mann-Whitney). No test is silently substituted — the caller
picks explicitly, and a NaN p-value is returned (never fabricated) whenever the
sample sizes or variability make the test undefined.
"""
from itertools import combinations

import numpy as np


def two_group_pvalue(a, b, test='t'):
    """Two-sided p-value comparing groups `a` and `b`.

    `test` is 't' (Welch) or 'mann' (Mann-Whitney U). Returns NaN when either
    group has fewer than 2 finite values, or (for Mann-Whitney) when the pooled
    values are all tied — cases where the test is undefined. Never invents data.
    """
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')

    if test == 'mann':
        if np.unique(np.concatenate([a, b])).size < 2:
            return float('nan')
        try:
            return float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
        except ValueError:
            return float('nan')

    # Welch t-test (does not assume equal variances)
    if np.ptp(a) == 0 and np.ptp(b) == 0:
        return float('nan')          # both constant -> undefined
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def significance_stars(p):
    """Map a p-value to the conventional star code. NaN/None -> 'ns'."""
    if p is None or not np.isfinite(p):
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def metabolite_min_pvalues(X, y, test='t'):
    """Smallest pairwise group p-value per metabolite (column of X).

    X is (n_samples, n_features); `y` is an integer group array aligned to the
    rows. Returns a length-n_features array where entry c is the minimum finite
    pairwise p-value across groups for that metabolite, or NaN when no pair is
    testable. Useful to rank metabolites by how strongly groups separate.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = sorted(set(y.tolist()))
    out = np.full(X.shape[1], np.nan)
    for c in range(X.shape[1]):
        vbg = [X[y == g, c] for g in groups]
        ps = [p for _, _, p, _ in pairwise_significance(vbg, test=test)
              if np.isfinite(p)]
        if ps:
            out[c] = min(ps)
    return out


def pairwise_significance(values_by_group, test='t'):
    """Compare every pair of groups.

    `values_by_group` is a list of 1D arrays (one per group, in display order).
    Returns a list of (i, j, p, stars) for i < j, ordered by ascending span
    (|j - i|) so the caller can stack shorter brackets below longer ones.
    """
    out = []
    for i, j in combinations(range(len(values_by_group)), 2):
        p = two_group_pvalue(values_by_group[i], values_by_group[j], test=test)
        out.append((i, j, p, significance_stars(p)))
    out.sort(key=lambda t: (t[1] - t[0]))
    return out
