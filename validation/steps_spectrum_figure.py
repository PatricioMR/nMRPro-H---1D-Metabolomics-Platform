# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The base and strategies A, B and C on one region of a simulated spectrum.

Four rows, one per step of ``validation.quant_steps`` (base, + A, + B, + C),
on the same trace: the mean of one sample group of a tube of 100 analytes at
600 MHz (composition 200 by default). Every number is read from that step's
cached rows, never re-computed.

Left, the spectrum, each analyte drawn as what produced its number, as
``gui.widgets.pdf_report`` draws it: the integrated window above the
baseline the integration subtracted (base); the analyte's component in a
shared window, the window shaded (A); a multiplet's fitted shape; the signal
of a joint fit or of a cascade entry the area is read from (B, C; in C the
joint fits kept from B are outlined). A bar at the top marks the region
selected for each analyte. An analyte of the region whose number a step read
from another of its signals gets an inset with that signal, drawn the same
way.

Right, every analyte drawn in the region in any step: its error, the group's
median against the true amount of the tube (a filled dot when the number
came from this region, a hollow one when from the signal of its inset),
"not reported" for a true analyte the step does not report, and the amount
of a false one; the title counts the region's true, false and not reported.

The region is chosen as one where the mechanisms act together; it is an
illustration, the steps over all tubes are ``validation.quant_steps``.

    python -m validation.steps_spectrum_figure --dest "<validation folder>"
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.quant_steps import FIELD, _path
from validation.replicates import reported
from validation.validation_folder import subfolder

INK, MUTED, SURFACE = '#1a1a19', '#6b6a64', '#ffffff'
# categorical slots in fixed order (dataviz reference palette), one per analyte
SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
WINDOW_FILL = '#ecebe6'
BAND_FILL = '#e9f1fc'
STEPS = (('base', 'Base: window integration with neighbour correction'),
         ('plus_a', '+ A: shared-window decomposition'),
         ('plus_b', '+ B: joint fit of overlapping analytes'),
         ('full', '+ C: residual cascade (complete engine)'))
# the room the spectrum panels keep above the trace for the insets
HEADROOM = 1.55


def _method(row):
    """What produced the row's number, in the order ``pdf_report`` reads it."""
    if row.get('_fit_curve') and (row.get('_joint_fit') or row.get('_by_cascade')):
        return 'fit'
    if row.get('_integ_mode') == 'Shape' and row.get('_shape_fit'):
        return 'shape'
    if row.get('_deconv'):
        return 'deconv'
    return 'integral'


def _span(row, ppm):
    if _method(row) == 'fit':
        return row['_fit_curve']['focus']
    mw = row.get('_multiplet_window')
    if mw:
        return mw
    return ppm[row['start_idx']], ppm[row['end_idx']]


def _inside(span, lo, hi):
    return span is not None and min(span) < hi and max(span) > lo


def _where(row, ppm):
    """The signal the number was read from, as ``lo–hi ppm``."""
    a, b = sorted(row.get('_multiplet_window') or _span(row, ppm))
    return f'{a:.2f} ppm' if b - a < 0.005 else f'{a:.2f}–{b:.2f} ppm'


def _error(row, truth, group):
    """measured / true - 1 for ``group`` (the median of its samples), and
    the measured amount."""
    vals = (row.get('_concs_by_group') or {}).get(group) or []
    got = float(np.median(vals)) if vals else np.nan
    want = (truth.get(row['hmdb_id']) or {}).get(group)
    return (got / float(want) - 1.0 if want else np.nan), got


def _fit_curves(fit, group):
    """(x, base, focus top) of a fit record for ``group``, as
    ``plotting.fit_curve.draw_fit_curve`` builds them; None without an
    amplitude there."""
    g = str(group)
    a = (fit.get('amp_by_group') or {}).get(g)
    unit = (fit.get('unit_by_group') or {}).get(g)
    if a is None or unit is None:
        return None
    x = np.asarray(fit['ppm'], dtype=float)
    o = np.argsort(x)
    base = (fit.get('baseline_by_group') or {}).get(g)
    base = np.zeros_like(x) if base is None else np.asarray(base, dtype=float)
    focus = (fit.get('focus_unit_by_group') or {}).get(g, unit)
    return x[o], base[o], (base + float(a) * np.asarray(focus, dtype=float))[o]


def _draw_fit(ax, row, truth, col, group, lo, hi, kept=False):
    """A joint fit or a cascade entry: the signal the area is read from,
    filled (hatched when the analyte is not in the tube); outlined only when
    it is a joint fit the cascade step kept."""
    cur = _fit_curves(row['_fit_curve'], group)
    if cur is None:
        return
    xs, base, top = cur
    keep = (xs >= lo - 0.05) & (xs <= hi + 0.05)
    xs, base, top = xs[keep], base[keep], top[keep]
    if kept:
        ax.plot(xs, top, color=col, lw=1.1, ls='--', zorder=3)
        return
    false = row['hmdb_id'] not in truth
    ax.fill_between(xs, base, top, color=col, alpha=0.45, lw=0, zorder=2,
                    hatch='////' if false else None, edgecolor=col if false else None)
    ax.plot(xs, top, color=col, lw=1.0, zorder=3)


def _draw_shape(ax, row, ppm, trace, col):
    """A multiplet quantified by its shape: the copy of the reference shape
    that fits under the trace (``pdf_report._peak_crop_fig``)."""
    from processing.pattern_shape import _amplitude
    from processing.quant_shape import shape_model
    built = shape_model({'shape_fit': row['_shape_fit']}, ppm)
    if built is None:
        return
    xs, ms, (slo, shi) = built
    t = trace[(ppm >= slo) & (ppm <= shi)]
    sig = float(row.get('_noise_sigma') or 0.0) or float(np.std(t[:max(3, len(t) // 10)]) or 1.0)
    a, _sup = _amplitude(t, ms, sig)
    o = np.argsort(xs)
    ax.fill_between(xs[o], 0, a * ms[o], color=col, alpha=0.45, lw=0, zorder=2)
    ax.plot(xs[o], a * ms[o], color=col, lw=1.0, zorder=3)


def _draw_integral(ax, row, ppm, trace, col):
    """The window above the baseline the integration subtracted, on the
    pieces it summed (``pdf_report._peak_crop_fig``)."""
    from processing.quant_integration import baseline_and_signal, baseline_under
    lo, hi = sorted(_span(row, ppm))
    sel = (ppm >= lo) & (ppm <= hi)
    xf, yf = ppm[sel], trace[sel]
    use_bl = row.get('_local_baseline', True)
    if row.get('_baseline_anchors') and use_bl:
        base = baseline_under(xf, ppm, trace, row['_baseline_anchors'])
    else:
        o = np.argsort(xf)
        base = np.empty_like(yf)
        base[o], _ = baseline_and_signal(yf[o], local_baseline=use_bl)
    where = yf >= base
    segs = row.get('_integ_segments') or []
    if len(segs) > 1:
        piece = np.zeros(xf.shape, dtype=bool)
        for a, b in segs:
            piece |= (xf >= min(a, b)) & (xf <= max(a, b))
        where &= piece
    ax.fill_between(xf, base, yf, where=where, color=col, alpha=0.5, lw=0, zorder=2)


def _draw_window(ax, dc, ppm, group, colour_of, label_at=None):
    """A shared window resolved by shapes: the window shaded and every
    component as the decomposition placed it (``pdf_report``); a component
    ``colour_of`` gives no colour is drawn as a grey dotted line."""
    from gui.widgets.pdf_report import _group_amplitude
    from processing.quant_shared import component_from_record
    from validation import mixture_bench as mb
    wlo, whi = dc['window']
    ax.axvspan(wlo, whi, color=WINDOW_FILL, zorder=0)
    if label_at is not None:
        ax.text(label_at + 0.002, 0.45, f'shared window\n{wlo:.3f}–{whi:.3f} ppm',
                transform=ax.get_xaxis_transform(), ha='right', va='top', fontsize=8.5, color=MUTED)
    sel = (ppm >= wlo) & (ppm <= whi)
    xs = ppm[sel]
    for rec in dc.get('components', ()):
        c = component_from_record(rec)
        amps = (dc.get('amplitudes') or {}).get(rec['met_id']) or {}
        if c is None or not amps:
            continue
        y = _group_amplitude(amps, mb.SAMPLES, group) * np.asarray(c.curve(xs), dtype=float)
        col = colour_of(rec['met_id'])
        if col is None:
            ax.plot(xs, y, color=MUTED, lw=0.8, ls=':', zorder=3)
            continue
        ax.fill_between(xs, 0, y, color=col, alpha=0.45, lw=0, zorder=2)
        ax.plot(xs, y, color=col, lw=1.0, zorder=3)


def _bar(ax, span, col, level, lo, hi):
    """The region selected for an analyte, a bar at the top of the panel, cut
    at the edges of the view."""
    a, b = max(min(span), lo), min(max(span), hi)
    if a < b:
        y = 0.97 - 0.035 * level
        ax.plot([a, b], [y, y], transform=ax.get_xaxis_transform(), color=col, lw=3.0,
                solid_capstyle='butt', zorder=6)


def _draw_row(ax, x, vid, truth, ppm, trace, col, group, lo, hi):
    """Draw what produced one row's number; a shared window is drawn by the
    caller, once for all its components."""
    m = _method(x)
    if m == 'fit':
        _draw_fit(ax, x, truth, col, group, lo, hi, kept=vid == 'full' and not x.get('_by_cascade'))
    elif m == 'shape':
        _draw_shape(ax, x, ppm, trace, col)
    elif m == 'integral':
        _draw_integral(ax, x, ppm, trace, col)


def _inset(ax, slot, x, vid, truth, ppm, trace, col, group, name):
    """The other signal an analyte's number was read from, in an inset at
    the panel's upper right: the trace around it and what produced the
    number, drawn as in the panel."""
    a, b = sorted(x.get('_multiplet_window') or _span(x, ppm))
    pad = max(0.015, 0.4 * (b - a))
    lo, hi = a - pad, b + pad
    sel = (ppm >= lo) & (ppm <= hi)
    ins = ax.inset_axes([0.82 - 0.23 * slot, 0.50, 0.17, 0.34])
    ins.plot(ppm[sel], trace[sel], color=INK, lw=0.9, zorder=4)
    if _method(x) == 'deconv':
        own = x['hmdb_id']
        _draw_window(ins, x['_deconv'], ppm, group, lambda m: col if m == own else None)
    else:
        _draw_row(ins, x, vid, truth, ppm, trace, col, group, lo, hi)
    ins.set_xlim(hi, lo)
    # scaled on the analyte's own signal: a taller neighbour runs off the top
    ins.set_ylim(0, float(trace[(ppm >= a) & (ppm <= b)].max()) * 1.5)
    ins.set_yticks([])
    ins.set_xticks([round(a, 2), round(b, 2)] if b - a >= 0.02 else [round((a + b) / 2, 2)])
    ins.tick_params(axis='x', labelsize=7, length=2, pad=1)
    for s in ins.spines.values():
        s.set_edgecolor(col)
        s.set_linewidth(1.3)
    ins.set_title(name, fontsize=7.5, pad=2, color=INK)


def _spectrum_panel(ax, vid, rows, truth, ppm, trace, view, palette, lo, hi, group, names):
    """One step's spectrum; returns {met_id: number read in this region}."""
    ax.plot(ppm[view], trace[view], color=INK, lw=1.1, zorder=4)
    here, bars, windows, insets = {}, [], {}, []
    for x in rows:
        mid = x['hmdb_id']
        if mid not in palette:
            continue
        col = palette[mid]
        if not _inside(_span(x, ppm), lo, hi):
            insets.append(x)
            continue
        here[mid] = True
        if _method(x) == 'deconv':
            windows[tuple(x['_deconv']['window'])] = x['_deconv']
        else:
            _draw_row(ax, x, vid, truth, ppm, trace, col, group, lo, hi)
        if not (vid == 'full' and _method(x) == 'fit' and not x.get('_by_cascade')):
            bars.append((x.get('_multiplet_window') or _span(x, ppm), col))
    for dc in windows.values():
        _draw_window(ax, dc, ppm, group, palette.get, label_at=max(dc['window'][0], lo))
    for k, (span, col) in enumerate(bars):
        _bar(ax, span, col, k, lo, hi)
    for slot, x in enumerate(insets):
        name = names[x['hmdb_id']]
        _inset(ax, slot, x, vid, truth, ppm, trace, palette[x['hmdb_id']], group,
               name if len(name) <= 20 else name[:19] + '…')
    return here


def _error_panel(ax, order, names, rows, truth, here, palette, group, lim, ppm):
    """One step's analytes of the region: error, not reported or false."""
    by_id = {x['hmdb_id']: x for x in rows}
    ax.axvspan(-20, 20, color=BAND_FILL, zorder=0)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for i, mid in enumerate(order):
        col = palette[mid]
        x = by_id.get(mid)
        if x is None:
            if mid in truth:
                ax.text(0, i, 'not reported', ha='center', va='center', fontsize=8.5, color=MUTED,
                        bbox=dict(fc=SURFACE, ec='none', pad=1.0), zorder=3)
            continue
        e, got = _error(x, truth, group)
        if mid not in truth:
            ax.scatter([lim * 0.62], [i], marker='X', s=70, color=col, edgecolor=INK, lw=0.6, zorder=4)
            ax.text(lim * 0.68, i, f'false, {got:.3f} mM', va='center', fontsize=8.5, color=INK)
            continue
        inside = here.get(mid, False)
        ex = float(np.clip(100 * e, -lim * 0.95, lim * 0.95))
        ax.scatter([ex], [i], s=60, zorder=4, color=col if inside else SURFACE, edgecolor=col, lw=1.6)
        right = ex < lim * 0.35
        ax.annotate(f'{100 * e:+.0f}%' + ('' if inside else f' (from {_where(x, ppm)})'), (ex, i),
                    xytext=(8 if right else -8, 0), textcoords='offset points',
                    ha='left' if right else 'right', va='center', fontsize=8.5, color=INK,
                    bbox=dict(fc=SURFACE, ec='none', pad=0.5, alpha=0.85), zorder=5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)), [names[m] for m in order], fontsize=9.5, color=INK)
    # identity: a swatch of the analyte's colour beside its name
    ax.scatter(np.full(len(order), -lim), range(len(order)), marker='s', s=55,
               color=[palette[m] for m in order], clip_on=False, zorder=5)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.grid(axis='x', color='#e4e3dd', lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)


def figure(path, seed=200, lo=2.33, hi=2.52, group='A'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb

    prep, _t = gb.prepared(gb.FIELDS[FIELD], seed, 'r100')
    ppm = np.asarray(prep['ppm'], dtype=float)
    cols = [s['index'] for s in mb.SAMPLES if s['label'] == group]
    trace = np.asarray(prep['raw'], dtype=float)[:, cols].mean(axis=1)
    view = (ppm >= lo) & (ppm <= hi)
    steps = {}
    for vid, _l in STEPS:
        d = pickle.load(open(_path(vid, 'r100', seed), 'rb'))
        steps[vid] = (reported(d['rows']), d['truth'])

    # the analytes of the region over every step, left to right as the
    # spectrum reads (decreasing shift); colours follow that order
    where, names = {}, {}
    for vid, _l in STEPS:
        rows, truth = steps[vid]
        for x in rows:
            span = _span(x, ppm)
            if not _inside(span, lo, hi):
                continue
            where.setdefault(x['hmdb_id'], float(np.mean(span)))
            names[x['hmdb_id']] = x['metabolite']
            if _method(x) == 'deconv':
                for rec in x['_deconv'].get('components', ()):
                    where.setdefault(rec['met_id'], float(np.mean(x['_deconv']['window'])))
                    names.setdefault(rec['met_id'], (truth.get(rec['met_id']) or {}).get('name')
                                     or rec['label'].rsplit(' ', 1)[0])
    order = sorted(where, key=lambda m: -where[m])
    palette = {m: SLOTS[i % len(SLOTS)] for i, m in enumerate(order)}
    short = {m: (n if len(n) <= 26 else n[:25] + '…') for m, n in names.items()}

    errs = [abs(100 * _error(x, steps[v][1], group)[0]) for v, _l in STEPS for x in steps[v][0]
            if x['hmdb_id'] in palette and x['hmdb_id'] in steps[v][1]]
    lim = max(60.0, 1.25 * max([e for e in errs if np.isfinite(e)] or [0]))

    fig, axes = plt.subplots(4, 2, figsize=(17.0, 17.0), dpi=300,
                             gridspec_kw={'width_ratios': [1.55, 1.0], 'hspace': 0.42, 'wspace': 0.62})
    y_top = float(trace[view].max()) * HEADROOM
    for r, (vid, title) in enumerate(STEPS):
        rows, truth = steps[vid]
        ax_s, ax_e = axes[r]
        here = _spectrum_panel(ax_s, vid, rows, truth, ppm, trace, view, palette, lo, hi, group, short)
        ax_s.set_xlim(hi, lo)                      # δ decreases to the right
        ax_s.set_ylim(0, y_top)
        ax_s.set_ylabel('Intensity (a.u.)', fontsize=11)
        ax_s.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax_s.tick_params(labelsize=10)
        for side in ('top', 'right'):
            ax_s.spines[side].set_visible(False)
        ax_s.set_title(title, loc='left', fontsize=12.5, pad=10)
        ax_s.set_xlabel('δ ¹H (ppm)', fontsize=10.5)

        _error_panel(ax_e, order, short, rows, truth, here, palette, group, lim, ppm)
        by_id = {x['hmdb_id'] for x in rows}
        n_true = sum(1 for m in order if m in by_id and m in truth)
        n_false = sum(1 for m in order if m in by_id and m not in truth)
        n_miss = sum(1 for m in order if m not in by_id and m in truth)
        ax_e.set_title(f'{n_true} true · {n_false} false · {n_miss} not reported',
                       loc='left', fontsize=10.5, pad=10, color=INK)
        ax_e.set_xlabel('Error, measured / true − 1 (%)', fontsize=10.5)
        ax_e.tick_params(axis='x', labelsize=10)
        for ax, k in ((ax_s, 2 * r), (ax_e, 2 * r + 1)):
            ax.text(-0.10 if ax is ax_s else -0.66, 1.04, 'ABCDEFGH'[k], transform=ax.transAxes,
                    fontsize=20, fontweight='bold', va='bottom')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--dest', required=True, help='the validation folder')
    ap.add_argument('--seed', type=int, default=200)
    ap.add_argument('--lo', type=float, default=2.33)
    ap.add_argument('--hi', type=float, default=2.52)
    ap.add_argument('--group', default='A')
    a = ap.parse_args()
    from app_paths import APP_VERSION
    version = APP_VERSION.split(' v')[1].split()[0]
    out = subfolder(a.dest, 'figures')
    os.makedirs(out, exist_ok=True)
    print(figure(os.path.join(out, f'Estrategias_ABC_espectro_nMRPro-H_v{version}.png'),
                 a.seed, a.lo, a.hi, a.group))


if __name__ == '__main__':
    main()
