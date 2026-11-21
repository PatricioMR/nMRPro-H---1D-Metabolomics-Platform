# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/quant_quality.py
"""Quantification quality: whether a number can be trusted.

The confidence dots say whether an analyte is there (identity). This says
whether its amount is: read per group, as a tier built the way the
annotation's quality is (ADR-001) -- a few hard checks, each from what the
engine already records and against a limit the software already has, the
worst of them decides. Nothing here measures anything again.

Poor -- the number is not a measurement:
  * found in fewer samples of the group than it has (``_fit_curve``
    ``n_by_group``): the group's figure is a mean over the samples it was
    found in, not over the group;
  * its own signals disagree beyond ``max_disagreement_pct`` (the cascade's
    ``disagreement``: each signal measured on its own, the widest departure
    from their median) -- the amount was assigned, not measured;
  * its replicates spread beyond ``CV_POOR_PCT``.
Warn -- a measurement with a known weakness:
  * its replicates spread beyond ``CV_WARN_PCT``;
  * the fit holds more than ``FIT_ABOVE_TRACE`` the area the trace has under
    the signal (``plotting.fit_curve.fit_over_trace``, the report's warning);
  * the amplitude was held at its physical cap (cascade ``at_cap`` /
    ``capped_share``): an upper bound, not an estimate.

Measured on the reference bench (ADR-105): tube of 100 analytes, 600 MHz,
each real analyte in each group. Chosen on five compositions (200-204):
median |error| Good 14 %, Warn 26 %, Poor 68 %. Checked on nine others no
choice saw (the v0.84 baseline, seeds 3-11): Good 14 %, Warn 23 %, Poor
55 %; Poor holds two thirds of the numbers in the worst quarter of error. Quantities that did not
separate good numbers from bad there (baseline share, missing lines, a
shape's explained share, signals at the position limit, the secondary
signals' room) are left out.
"""
from __future__ import annotations

import numpy as np

GOOD, WARN, POOR = 'Good', 'Warn', 'Poor'
_RANK = {GOOD: 0, WARN: 1, POOR: 2}
# the replicate spread the card has always graded on
CV_WARN_PCT = 20.0
CV_POOR_PCT = 40.0


def _cv(row, group):
    cv = (row.get('cv_by_group') or {}).get(group)
    if isinstance(cv, dict):
        cv = cv.get('cv')
    try:
        cv = float(cv)
    except (TypeError, ValueError):
        return None
    return cv if np.isfinite(cv) else None


def assess(row, group, *, ppm=None, trace=None) -> dict:
    """``{'level', 'reasons', 'cv'}`` for ``row`` in ``group``: the tier
    (None for a row without a number) and, worst first, why. ``trace`` is
    the group's mean spectrum on ``ppm``, for the fit-over-trace check; the
    check is skipped without it."""
    from controllers.analysis_defaults import QUANTIFICATION
    from plotting.fit_curve import FIT_ABOVE_TRACE, fit_over_trace
    if row.get('_not_quantified') or row.get('_is_is'):
        return {'level': None, 'reasons': [], 'cv': None}
    g = str(group)
    found = []                                  # (level, reason)
    fit = row.get('_fit_curve') or {}
    n = (fit.get('n_by_group') or {}).get(g)
    if n and n[0] is not None and n[1] and int(n[0]) < int(n[1]):
        found.append((POOR, f'found in {int(n[0])} of {int(n[1])} samples of the group'))
    casc = row.get('_cascade') or {}
    dis = casc.get('disagreement')
    limit = float(QUANTIFICATION['max_disagreement_pct'])
    if dis is not None and np.isfinite(float(dis)) and 100 * float(dis) > limit:
        found.append((POOR, f'its signals disagree by {100 * float(dis):.0f}% '
                            f'(limit {limit:.0f}%)'))
    cv = _cv(row, g)
    if cv is not None and cv >= CV_POOR_PCT:
        found.append((POOR, f'replicates spread {cv:.0f}% (CV ≥ {CV_POOR_PCT:.0f}%)'))
    elif cv is not None and cv >= CV_WARN_PCT:
        found.append((WARN, f'replicates spread {cv:.0f}% (CV ≥ {CV_WARN_PCT:.0f}%)'))
    if fit and ppm is not None and trace is not None:
        over = fit_over_trace(fit, g, ppm, trace)
        if over is not None and over > FIT_ABOVE_TRACE:
            found.append((WARN, f'the fit holds {over:.2f}x the area the trace has '
                                f'under the signal'))
    if casc.get('at_cap') or (casc.get('capped_share') or 0) > 0:
        found.append((WARN, 'amplitude held at its cap: an upper bound'))
    found.sort(key=lambda t: -_RANK[t[0]])
    level = found[0][0] if found else GOOD
    return {'level': level, 'reasons': [r for _l, r in found], 'cv': cv}


def worst(levels) -> str | None:
    """The worst of several groups' tiers; None when none has one."""
    ls = [lv for lv in levels if lv is not None]
    return max(ls, key=_RANK.get) if ls else None
