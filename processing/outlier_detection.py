# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Grubbs-family outlier test for univariate samples (generalized ESD / Rosner).

Grubbs, F. E. (1969), "Procedures for Detecting Outlying Observations in
Samples", Technometrics 11(1):1-21.
Rosner, B. (1983), "Percentage Points for a Generalized ESD Many-Outlier
Procedure", Technometrics 25(2):165-172.

Grubbs' single-outlier test assumes the tested variable is approximately normal
and detects ONE outlier. Two properties matter for correct use here:

  • Masking — two nearby outliers each inflate the SD enough that neither is
    individually significant, so plain iterative Grubbs (which stops at the
    first non-rejection) misses both. The generalized ESD procedure peels the r
    most extreme points, records the statistic R_i and its critical value λ_i at
    every step, and declares the number of outliers to be the LARGEST i with
    R_i > λ_i — so a masked pair is still recovered. λ_i equals the Grubbs
    critical value at the current sample size.

  • Sidedness — the two-sided form flags BOTH tails. When the tested quantity
    only has a meaningful upper tail (e.g. a distance-to-centroid, which is
    non-negative), the two-sided form wrongly flags points that are unusually
    SMALL (very close to the centroid). Use the one-sided upper form there.

Bonferroni significance per step: alpha/n (one-sided) or alpha/(2n) (two-sided).
"""
from __future__ import annotations

import numpy as np


def grubbs_critical_value(n: int, alpha: float = 0.05,
                          two_sided: bool = False) -> float:
    """Critical G for a sample of size *n* at significance *alpha*.

    ``two_sided=False`` (default) returns the one-sided (upper or lower) value;
    ``True`` returns the two-sided value. NaN when n < 3 (test undefined).
    """
    if n < 3:
        return float('nan')
    from scipy import stats as spstats
    p = alpha / (2.0 * n) if two_sided else alpha / n
    t = spstats.t.ppf(1.0 - p, df=n - 2)
    return float(((n - 1) / np.sqrt(n))
                 * np.sqrt(t ** 2 / (n - 2 + t ** 2)))


def _one_step_extreme(sub, m, s, side):
    """Return (relative index, R statistic) of the extreme point for one ESD step."""
    if side == 'lower':
        i_rel = int(np.argmin(sub))
        return i_rel, (m - sub[i_rel]) / s
    if side == 'two-sided':
        dev = np.abs(sub - m)
        i_rel = int(np.argmax(dev))
        return i_rel, dev[i_rel] / s
    i_rel = int(np.argmax(sub))            # 'upper'
    return i_rel, (sub[i_rel] - m) / s


def grubbs_outliers(values, alpha: float = 0.05, side: str = 'upper',
                    max_outliers: int | None = None):
    """Generalized ESD (Rosner) outlier test — the masking-robust Grubbs.

    Parameters
    ----------
    values       : 1-D array-like.
    alpha        : family significance level.
    side         : 'upper' (only large values are outliers — the correct choice
                   for distances/spreads), 'lower', or 'two-sided'.
    max_outliers : cap on how many points may be declared outliers. Defaults to
                   ``max(1, (n - 1) // 2)`` — at most about half the sample, the
                   usual upper bound for a meaningful outlier procedure.

    Returns
    -------
    (flags, g_stats, g_crits) each an array aligned to *values*:
      flags   — bool, True where the point is an outlier.
      g_stats — the ESD statistic R_i recorded for that point (at its peel step
                for peeled points; against the clean set for the rest).
      g_crits — the critical value λ_i in force at that evaluation.

    Groups with fewer than 3 points cannot be tested → all False / NaN.
    """
    vals = np.asarray(values, dtype=float)
    n0 = vals.size
    flags = np.zeros(n0, dtype=bool)
    g_stats = np.full(n0, float('nan'))
    g_crits = np.full(n0, float('nan'))
    if n0 < 3:
        return flags, g_stats, g_crits

    two_sided = (side == 'two-sided')
    r = max(1, (n0 - 1) // 2) if max_outliers is None else max(1, int(max_outliers))
    r = min(r, n0 - 2)   # need ≥ 2 points left to keep a defined SD

    # Peel the r most extreme points, recording R_i and λ_i at each step.
    active = list(range(n0))
    peeled: list[int] = []        # original indices, in peel order
    R_steps: list[float] = []
    lam_steps: list[float] = []
    for _ in range(r):
        if len(active) < 3:
            break
        sub = vals[active]
        m = float(np.mean(sub))
        s = float(np.std(sub, ddof=1))
        lam = grubbs_critical_value(len(active), alpha, two_sided=two_sided)
        if s <= 0 or not np.isfinite(lam):
            break
        i_rel, R = _one_step_extreme(sub, m, s, side)
        gi = active[i_rel]
        peeled.append(gi)
        R_steps.append(float(R))
        lam_steps.append(float(lam))
        g_stats[gi] = round(float(R), 4)
        g_crits[gi] = round(float(lam), 4)
        active.pop(i_rel)

    # Number of outliers = largest i (1-based) with R_i > λ_i.
    n_out = 0
    for i in range(len(R_steps)):
        if R_steps[i] > lam_steps[i]:
            n_out = i + 1
    for k in range(n_out):
        flags[peeled[k]] = True

    # Reference statistic/critical for every non-outlier point, vs the clean set.
    clean_idx = [i for i in range(n0) if not flags[i]]
    if len(clean_idx) >= 3:
        sub = vals[clean_idx]
        m = float(np.mean(sub))
        s = float(np.std(sub, ddof=1))
        lam = grubbs_critical_value(len(clean_idx), alpha, two_sided=two_sided)
        if s > 0 and np.isfinite(lam):
            for gi in clean_idx:
                if side == 'lower':
                    R = (m - vals[gi]) / s
                elif two_sided:
                    R = abs(vals[gi] - m) / s
                else:
                    R = (vals[gi] - m) / s
                g_stats[gi] = round(float(R), 4)
                g_crits[gi] = round(float(lam), 4)

    return flags, g_stats, g_crits
