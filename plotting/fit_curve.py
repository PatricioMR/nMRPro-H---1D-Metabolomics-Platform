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
    n_pos, n_all = (fit.get('n_by_group') or {}).get(g, (None, None))
    text = label if label is not None else (fit.get('label') or 'Fit')
    if n_pos is not None and n_all and n_pos < n_all:
        text += f' (found in {n_pos} of {n_all})'
    ax.fill_between(x[order], base[order], top[order], alpha=alpha,
                    color=fill_colour, linewidth=0, zorder=2)
    ax.plot(x[order], top[order], color=FIT_LINE, lw=1.4, ls='--', zorder=5,
            label=text)
    if fit.get('baseline_by_group'):
        ax.plot(x[order], base[order], color=BASELINE_LINE, lw=0.8, ls='--',
                alpha=0.7, zorder=3, label=baseline_label)
    return True


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
