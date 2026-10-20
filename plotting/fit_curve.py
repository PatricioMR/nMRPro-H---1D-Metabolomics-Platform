# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# plotting/fit_curve.py
"""The drawing of an amount that came from a fit (joint fit, cascade).

A row whose number a fit produced carries ``_fit_curve``
(``processing.quant_cascade.fit_curve_record``): the analyte's shape at one
unit per proton, the fitted amplitude of each group and the baseline the
fit stood on. Drawn here once, for the quantification stage and for the PDF,
so both show the number in the table -- the shape that was fitted, standing
on the line that was subtracted -- and not a window nobody integrated.
"""
from __future__ import annotations

import numpy as np

FIT_LINE = '#2e7d32'
BASELINE_LINE = '#ff6666'
# A figure keeps the group the row's SNR was read on while the fit holds at
# least this share of its highest group's amount there.
FIT_GROUP_SWITCH = 0.5
# The y axis of a fit made on a residual is cut at this multiple of the
# highest of the fit and the residual, when that is far below the trace.
VIEW_MARGIN = 1.3
VIEW_ZOOM_BELOW = 0.5
VIEW_RESIDUAL_PERCENTILE = 90

# A fit standing this much above the trace it was fitted to, over the signal
# the number is read from, is not that trace's amount: the figure says so.
FIT_ABOVE_TRACE = 1.25
# The part of the focus signal's height that counts as the signal's support.
FOCUS_SUPPORT = 0.02


def draw_fit_curve(ax, fit, group, fill_colour, *, label=None,
                   baseline_label='baseline', alpha=0.35):
    """Fill the fitted shape of ``group`` over its baseline on ``ax``, at
    the amplitude the table reports for the group (mean over the samples
    where the fit found it), the label saying in how many samples that
    was. Returns False when the record holds no amplitude for the group."""
    g = str(group)
    a = (fit.get('amp_by_group') or {}).get(g)
    unit = (fit.get('unit_by_group') or {}).get(g)
    if a is None or unit is None:
        return False
    x = np.asarray(fit['ppm'], dtype=float)
    unit = np.asarray(unit, dtype=float)
    base = (fit.get('baseline_by_group') or {}).get(g)
    base = np.zeros_like(x) if base is None else np.asarray(base, dtype=float)
    order = np.argsort(x)
    top = base + float(a) * unit
    focus = (fit.get('focus_unit_by_group') or {}).get(g)
    focus_top = (base + float(a) * np.asarray(focus, dtype=float)
                 if focus is not None else top)
    n_pos, n_all = (fit.get('n_by_group') or {}).get(g, (None, None))
    text = label if label is not None else (fit.get('label') or 'Fit')
    if n_pos is not None and n_all and n_pos < n_all:
        text += f' (found in {n_pos} of {n_all})'
    # the shaded area is the signal the table's area is read from; the dashed
    # line is the whole fit, the compound's other signals in view included
    ax.fill_between(x[order], base[order], focus_top[order], alpha=alpha,
                    color=fill_colour, linewidth=0, zorder=2)
    if focus is not None and not np.allclose(top, focus_top):
        text += ' (shaded: the signal the area is read from)'
    ax.plot(x[order], top[order], color=FIT_LINE, lw=1.4, ls='--', zorder=5,
            label=text)
    if fit.get('baseline_by_group'):
        ax.plot(x[order], base[order], color=BASELINE_LINE, lw=0.8, ls='--',
                alpha=0.7, zorder=3, label=baseline_label)
    return True


def fit_group(fit, prefer=None):
    """The group a figure of this fit draws: ``prefer`` (the group the row's
    signal-to-noise was read on) unless the fit holds less than
    ``FIT_GROUP_SWITCH`` of its highest group's amount there, in which case
    the shaded area would be a line on the axis -- the group where the fit
    found the most."""
    amps = {str(k): float(v) for k, v in (fit.get('amp_by_group') or {}).items()
            if v is not None and np.isfinite(v)}
    if not amps:
        return prefer
    best = max(amps, key=amps.get)
    if prefer is not None and amps.get(str(prefer), 0.0) >= FIT_GROUP_SWITCH * amps[best]:
        return prefer
    return best


def fit_view_top(fit, group) -> float | None:
    """How high the y axis has to reach to show a fit that stands under a
    far taller neighbour: the highest of the fit and, when the fit was made on
    a residual, of that residual where the focus signal stands, with a
    margin. None when the record holds no amplitude for the group."""
    g = str(group)
    resid = (fit.get('residual_by_group') or {}).get(g)
    a = (fit.get('amp_by_group') or {}).get(g)
    unit = (fit.get('unit_by_group') or {}).get(g)
    focus = (fit.get('focus_unit_by_group') or {}).get(g)
    if a is None or unit is None:
        return None
    base = (fit.get('baseline_by_group') or {}).get(g)
    x = np.asarray(fit['ppm'], dtype=float)
    base = np.zeros_like(x) if base is None else np.asarray(base, dtype=float)
    top = float((base + float(a) * np.asarray(unit, dtype=float)).max())
    if resid is not None and focus is not None:
        focus = np.asarray(focus, dtype=float)
        support = focus >= FOCUS_SUPPORT * max(float(focus.max()), 1e-30)
        # a high percentile, not the maximum: what a neighbour's imperfect
        # subtraction leaves in a point or two is not what the fit was made on
        top = max(top, float(np.percentile(
            (base + np.asarray(resid, dtype=float))[support],
            VIEW_RESIDUAL_PERCENTILE)))
    return top * VIEW_MARGIN


def fit_over_trace(fit, group, ppm, trace) -> float | None:
    """The focus signal's fitted area over the area of ``trace`` (the group
    mean the figure draws) where that signal stands, above the baseline the
    fit stood on; None when the record holds no focus curve or the trace has
    nothing there. Over ``FIT_ABOVE_TRACE`` the fit claims more than the
    spectrum holds under it."""
    g = str(group)
    a = (fit.get('amp_by_group') or {}).get(g)
    unit = (fit.get('focus_unit_by_group') or {}).get(g)
    if a is None or unit is None:
        return None
    x = np.asarray(fit['ppm'], dtype=float)
    unit = np.asarray(unit, dtype=float)
    if not unit.size or unit.max() <= 0:
        return None
    base = (fit.get('baseline_by_group') or {}).get(g)
    base = np.zeros_like(x) if base is None else np.asarray(base, dtype=float)
    ppm = np.asarray(ppm, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    resid = (fit.get('residual_by_group') or {}).get(g)
    if resid is not None:
        # the fit was made on the residual: held against that, not against
        # a trace that also holds its neighbours
        y = np.asarray(resid, dtype=float)[order]
        base = np.zeros_like(base)
    else:
        tr = np.asarray(trace, dtype=float)
        y = (np.interp(xs, ppm[::-1], tr[::-1]) if ppm[0] > ppm[-1]
             else np.interp(xs, ppm, tr))
    support = unit[order] >= FOCUS_SUPPORT * unit.max()
    if support.sum() < 2:
        return None
    seen = np.clip(y - base[order], 0.0, None)[support]
    if np.trapezoid(seen, xs[support]) <= 0:
        return None
    return float(np.trapezoid(float(a) * unit[order][support], xs[support])
                 / np.trapezoid(seen, xs[support]))


def fit_warnings(fit) -> list:
    """What the reader must know before trusting the drawn fit: the fitted
    signal is not the one the election found, or the fit left signals at
    the edge of how far it may move them (a misplaced shape)."""
    out = []
    if fit and fit.get('focus_matches_window') is False:
        lo, hi = fit.get('focus') or (0.0, 0.0)
        out.append(f'fitted signal at {lo:.3f}-{hi:.3f} ppm is not the '
                   f'one the election found')
    lim = (fit or {}).get('at_limit') or []
    if lim:
        out.append(f'{len(lim)} of {fit.get("n_signals", len(lim))} signals '
                   f'held at the position limit of the fit (misplaced shape)')
    return out
