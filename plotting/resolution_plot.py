# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# plotting/resolution_plot.py
"""Drawing of the resolution plot (RP-Method): the spectrum above the axis,
everything the quantification resolved mirrored below it, each analyte in
its colour, and the residual. The numbers come from
``processing.resolution_plot``; the RP-Method window and the PDF report draw
them with these functions, so the report shows what the window shows."""
from __future__ import annotations

import numpy as np

from plotting import contributions
from processing.resolution_plot import MISSING, OVER, excluded_mask

INK = '#1a1a19'
RESIDUAL = '#c62828'
OVER_INK = '#1b3f8e'
RESOLVED_LABEL = 'resolved ↓ (hatched: integral · dotted: predicted other signals)'


def draw_resolution(ax, ppm, trace, out, palette, regions=()):
    """``out`` (``processing.resolution_plot.resolution``) on ``ax``: the
    spectrum, the residual, every analyte's curve below the axis, their sum,
    and a marker over each residual region of ``regions``. Labels and
    spines too; the limits are ``view_limits``'s."""
    ppm = np.asarray(ppm, dtype=float)
    o = np.argsort(ppm)
    x = ppm[o]
    for kind, col, mk in ((MISSING, RESIDUAL, 'v'), (OVER, OVER_INK, '^')):
        xs = [g['apex_ppm'] for g in regions or () if g['kind'] == kind]
        if xs:
            ax.scatter(xs, [0.97] * len(xs), marker=mk, s=18, color=col, zorder=6,
                       transform=ax.get_xaxis_transform(), clip_on=False,
                       label='missing' if kind == MISSING else 'over-claim')
    ax.plot(x, np.asarray(trace)[o], color=INK, lw=0.8, zorder=4, label='spectrum')
    ax.plot(x, out['residual'][o], color=RESIDUAL, lw=0.7, zorder=5, label='residual')
    for c in out['curves']:
        col = (contributions.MUTED if c['method'] == 'unreported'
               else palette.get(c['id'], contributions.MUTED))
        if c['method'] == 'secondary':
            # predicted by the analyte's own number, not fitted: lighter, dotted
            ax.fill_between(c['x'], 0, -c['y'], color=col, lw=0, zorder=2, alpha=0.25)
            ax.plot(c['x'], -c['y'], color=col, lw=0.9, ls=':', zorder=3)
            continue
        data = c['method'] == 'integral'
        ax.fill_between(c['x'], 0, -c['y'], color=col, lw=0, zorder=2,
                        alpha=0.25 if data else contributions.FILL_ALPHA,
                        hatch='////' if data else None,
                        edgecolor=col if data else None)
    ax.plot(x, -out['resolved'][o], color='#555555', lw=0.6, zorder=3, label=RESOLVED_LABEL)
    ax.axhline(0, color='#999999', lw=0.6, zorder=1)
    ax.set_xlabel('δ ¹H (ppm)')
    ax.set_ylabel('Intensity (a.u.)  —  ↑ spectrum   ↓ resolved')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def view_limits(ppm, trace, resolved, excluded=(), view=None):
    """``(lo, hi, m)``: the ppm range shown (``view``, else the whole axis)
    and the half-height of a symmetric y range that holds the spectrum and
    the resolved sum there, the excluded regions (water, reference) left out
    of the scale."""
    ppm = np.asarray(ppm, dtype=float)
    lo, hi = (float(ppm.min()), float(ppm.max())) if view is None else view
    sel = (ppm >= lo) & (ppm <= hi) & ~excluded_mask(ppm, excluded)
    top = float(np.max(np.asarray(trace)[sel])) if sel.any() else 1.0
    bot = float(np.max(np.asarray(resolved)[sel])) if sel.any() else 1.0
    return lo, hi, max(top, bot, 1e-12) * 1.1


def apply_limits(ax, lo, hi, m):
    ax.set_xlim(hi, lo)                     # δ decreases to the right
    ax.set_ylim(-m, m)
