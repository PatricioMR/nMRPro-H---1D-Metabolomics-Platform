# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# plotting/contributions.py
"""Each analyte drawn as what produced its number, in a colour of its own.

One quantified row is drawn by the method that gave its amount, as
``gui.widgets.pdf_report`` reads it:

* ``fit`` -- a joint fit or a cascade entry (``_fit_curve``): the signal the
  area is read from, standing on the baseline the fit stood on;
* ``shape`` -- a multiplet quantified by its shape (ADR-013): the copy of
  the reference shape that fits under the trace;
* ``deconv`` -- a window shared by several analytes, resolved by shapes
  (ADR-015): the window shaded and every component as the decomposition
  placed it;
* ``integral`` -- the integrated window above the baseline the integration
  subtracted, on the pieces it summed.

``draw_analytes`` draws every analyte of a view this way, the selected one
heavier: the quantification stage and the PDF report (ADR-103), so each
analyte stands next to the neighbours it shares the region with.
``validation.steps_spectrum_figure`` adds a bar per analyte over the
region selected for it (``region_bar``).
"""
from __future__ import annotations

import numpy as np

from plotting.fit_curve import BASELINE_LINE
# the curves are computed in processing; this module only draws them (and
# keeps the names the figure scripts and the report import from here)
from processing.analyte_models import (  # noqa: F401
    fit_curves, inside, integral_piece, method, own_line_marks, shape_copy, span,
    window_components)

# categorical slots in fixed order (dataviz reference palette)
SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
         '#6250d6', '#e34948')
WINDOW_FILL = '#ecebe6'
MUTED = '#6b6a64'
FILL_ALPHA = 0.45
# a region narrower than this is a line position, not a region
MIN_REGION_PPM = 0.002
# below this share of its height, a fitted signal's outline is not drawn
TRIM_SHARE = 0.005


def _trimmed(y, base, ref):
    """``y`` where ``ref`` stands above ``TRIM_SHARE`` of its height over
    ``base``, NaN elsewhere: an outline along zero marks nothing."""
    h = ref - base
    hmax = float(h.max()) if h.size else 0.0
    return y if hmax <= 0 else np.where(h >= TRIM_SHARE * hmax, y, np.nan)


def draw_fit(ax, row, group, colour, lo, hi, *, kept=False, false=False,
             whole=False, lw=1.0, trim=False, label=None):
    """A joint fit or a cascade entry: the signal the area is read from,
    filled (hatched when ``false``); outlined only when ``kept``. ``whole``
    adds the whole fit, dashed -- the compound's other signals in view --
    and the baseline the fit stood on; ``trim`` drops the outline where the
    signal is next to nothing.
    Returns False when the record holds no amplitude for ``group``."""
    cur = fit_curves(row['_fit_curve'], group)
    if cur is None:
        return False
    xs, base, top, full = cur
    keep = (xs >= lo - 0.05) & (xs <= hi + 0.05)
    xs, base, top, full = xs[keep], base[keep], top[keep], full[keep]
    if kept:
        ax.plot(xs, top, color=colour, lw=1.1, ls='--', zorder=3)
        return True
    ax.fill_between(xs, base, top, color=colour, alpha=FILL_ALPHA, lw=0, zorder=2,
                    hatch='////' if false else None,
                    edgecolor=colour if false else None)
    ax.plot(xs, _trimmed(top, base, top) if trim else top, color=colour, lw=lw,
            zorder=3, label=label)
    if whole and not np.allclose(full, top):
        ax.plot(xs, _trimmed(full, base, full) if trim else full, color=colour,
                lw=0.9, ls='--', zorder=3)
    if whole and row['_fit_curve'].get('baseline_by_group'):
        ax.plot(xs, base, color=BASELINE_LINE, lw=0.8, ls='--', alpha=0.7, zorder=3,
                label='baseline')
    return True


def draw_shape(ax, row, ppm, trace, colour, *, lw=1.0, label=None):
    """A multiplet quantified by its shape: the copy of the reference shape
    that fits under the trace (``pdf_report._peak_crop_fig``)."""
    cur = shape_copy(row, ppm, trace)
    if cur is None:
        return False
    xs, y = cur
    ax.fill_between(xs, 0, y, color=colour, alpha=FILL_ALPHA, lw=0, zorder=2)
    ax.plot(xs, y, color=colour, lw=lw, zorder=3, label=label)
    return True


def draw_integral(ax, row, ppm, trace, colour, *, alpha=0.5, baseline=False):
    """The window above the baseline the integration subtracted, on the
    pieces it summed (``pdf_report._peak_crop_fig``); ``baseline`` draws
    that baseline too."""
    cur = integral_piece(row, ppm, trace)
    if cur is None:
        return False
    xf, base, yf, where = cur
    ax.fill_between(xf, base, yf, where=where, color=colour, alpha=alpha, lw=0, zorder=2)
    if baseline and row.get('_local_baseline', True):
        ax.plot(xf, base, color=BASELINE_LINE, lw=0.8, ls='--', alpha=0.7, zorder=3,
                label='baseline')
    return True


def draw_window(ax, dc, ppm, amp_of, colour_of, *, label_at=None, lw_of=None):
    """A shared window resolved by shapes: the window shaded and every
    component as the decomposition placed it, at ``amp_of(amplitudes)``; a
    component ``colour_of`` gives no colour is drawn as a grey dotted
    line."""
    wlo, whi = dc['window']
    ax.axvspan(wlo, whi, color=WINDOW_FILL, zorder=0)
    if label_at is not None:
        ax.text(label_at + 0.002, 0.45, f'shared window\n{wlo:.3f}–{whi:.3f} ppm',
                transform=ax.get_xaxis_transform(), ha='right', va='top',
                fontsize=8.5, color=MUTED)
    for rec, xs, y in window_components(dc, ppm, amp_of):
        col = colour_of(rec['met_id'])
        if col is None:
            ax.plot(xs, y, color=MUTED, lw=0.8, ls=':', zorder=3, label=rec.get('label', ''))
            continue
        ax.fill_between(xs, 0, y, color=col, alpha=FILL_ALPHA, lw=0, zorder=2)
        ax.plot(xs, y, color=col, lw=(lw_of(rec['met_id']) if lw_of else 1.0), zorder=3,
                label=rec.get('label', ''))


def region_bar(ax, sp, colour, level, lo, hi, *, lw=3.0):
    """The region selected for an analyte, a bar at the top of the axes, cut
    at the edges of the view."""
    a, b = max(min(sp), lo), min(max(sp), hi)
    if a < b:
        y = 0.97 - 0.035 * level
        ax.plot([a, b], [y, y], transform=ax.get_xaxis_transform(), color=colour,
                lw=lw, solid_capstyle='butt', zorder=6)


def draw_row(ax, row, ppm, trace, group, colour, lo, hi, *, kept=False,
             false=False, whole=False, lw=1.0, trim=False, label=None):
    """Draw what produced one row's number; a shared window is drawn by
    ``draw_window``, once for all its components. ``whole`` adds what the
    selected row shows on top: the whole fit, the baseline subtracted."""
    m = method(row)
    if m == 'fit':
        return draw_fit(ax, row, group, colour, lo, hi, kept=kept, false=false,
                        whole=whole, lw=lw, trim=trim, label=label)
    if m == 'shape':
        return draw_shape(ax, row, ppm, trace, colour, lw=lw, label=label)
    if m == 'integral':
        return draw_integral(ax, row, ppm, trace, colour, baseline=whole)
    return False


def method_label(row):
    """The legend entry of the curve that gave the selected row's number:
    the fit's own label, or the shape fit with the share it explains."""
    m = method(row)
    if m == 'fit':
        return row['_fit_curve'].get('label') or 'Fit'
    if m == 'shape':
        ex = row.get('_shape_explained')
        return 'Shape fit' + (f'  (explains {100 * float(ex):.0f}%)' if ex is not None else '')
    return None


def method_name(row):
    """What gave the row's number, in words, when it is not the election's
    integration mode: a fit's own label, the shape, the shared window. A row
    elected as a shape whose number a joint fit gave is a joint fit."""
    m = method(row)
    if m == 'fit':
        return row['_fit_curve'].get('label') or 'Fit'
    if m == 'shape':
        return 'Shape'
    if m == 'deconv':
        return 'Shared window'
    return None


def number_region(row, ppm, group=None):
    """``(lo, hi)`` the row's number was read from: ``span``; for a fit
    focused on one line position, where the focus signal stands above
    ``fit_curve.FOCUS_SUPPORT`` of its height -- the support
    ``fit_over_trace`` holds it against. None without one."""
    from plotting.fit_curve import FOCUS_SUPPORT
    sp = span(row, ppm)
    if sp is None:
        return None
    lo, hi = float(min(sp)), float(max(sp))
    if method(row) == 'fit' and hi - lo < MIN_REGION_PPM and group is not None:
        cur = fit_curves(row['_fit_curve'], group)
        if cur is not None:
            xs, base, top, _full = cur
            h = top - base
            if h.size and h.max() > 0:
                on = h >= FOCUS_SUPPORT * h.max()
                lo, hi = float(xs[on].min()), float(xs[on].max())
    return lo, hi


def stable_palette(rows, ppm) -> dict:
    """``{met_id: colour}`` for every quantified row (the internal standard
    and the rows without a number left out), by the rank of its region's
    centre on the shift axis: neighbours on the spectrum take consecutive
    slots, and an analyte keeps its colour from one view to the next, in the
    stage, the deconvolution review and the report alike."""
    where = {}
    for r in rows:
        if r.get('_not_quantified') or r.get('_is_is'):
            continue
        sp = span(r, ppm)
        if sp is not None:
            where.setdefault(r.get('hmdb_id', r.get('metabolite')), float(np.mean(sp)))
    order = sorted(where, key=lambda m: -where[m])
    return {m: SLOTS[i % len(SLOTS)] for i, m in enumerate(order)}


def draw_analytes(ax, rows, selected, ppm, trace, group, lo, hi, palette, amp_of,
                  only_selected=False):
    """Every quantified row whose region falls in ``[lo, hi]`` drawn on
    ``trace`` (the mean of ``group``), each in its ``palette`` colour; the
    ``selected`` row drawn last, with a heavier outline, its whole fit and
    the baseline subtracted. With ``only_selected``, the selected row
    alone in colour: of a window it shares, the other components grey and
    dotted (``draw_window``), the other analytes of the view not drawn.
    Returns ``[(name, colour, is_selected)]``, the selected last, for a
    legend."""
    key = lambda r: r.get('hmdb_id', r.get('metabolite'))
    sel_id = key(selected)
    here = [r for r in rows if inside(span(r, ppm), lo, hi)
            and (not only_selected or key(r) == sel_id)]
    here.sort(key=lambda r: (key(r) == sel_id, -float(np.mean(span(r, ppm)))))
    windows, drawn = {}, []
    for r in here:
        mid = key(r)
        col = palette.get(mid, MUTED)
        is_sel = mid == sel_id
        if method(r) == 'deconv':
            windows[tuple(r['_deconv']['window'])] = r['_deconv']
        else:
            draw_row(ax, r, ppm, trace, group, col, lo, hi, whole=is_sel,
                     lw=2.0 if is_sel else 1.0, trim=True,
                     label=method_label(r) if is_sel else None)
        drawn.append((r, col, is_sel))
    colour_of = ((lambda m: palette.get(m) if str(m) == str(sel_id) else None)
                 if only_selected else palette.get)
    for dc in windows.values():
        draw_window(ax, dc, ppm, amp_of, colour_of,
                    lw_of=lambda m: 2.0 if m == sel_id else 1.0)
    return [(r['metabolite'], col, is_sel) for r, col, is_sel in drawn]

