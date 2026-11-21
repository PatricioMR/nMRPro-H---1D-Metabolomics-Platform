# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/analyte_models.py
"""What produced each quantified row's number, as curves on the ppm axis.

One row's amount came from one of four methods (in the order
``gui.widgets.pdf_report`` reads them):

* ``fit`` -- a joint fit or a cascade entry (``_fit_curve``,
  ``processing.quant_cascade.fit_curve_record``): the analyte's shape at one
  unit per proton times the amplitude of a group, on the baseline the fit
  stood on;
* ``shape`` -- a multiplet quantified by its shape (ADR-013,
  ``processing.quant_shape``): the largest copy of the reference shape that
  fits under the trace. Targeted profiling: Weljie et al., Anal. Chem. 2006,
  78, 4430;
* ``deconv`` -- a window shared by several analytes, resolved by shapes
  (ADR-015, ``processing.quant_shared``): every component as the fit placed
  it, at the amplitude of a group. One-sided amplitudes: Lawson & Hanson,
  Solving Least Squares Problems, 1974 (NNLS);
* ``integral`` -- the integrated window above the baseline the integration
  subtracted, on the pieces it summed (``processing.quant_integration``):
  the data itself, not a model.

The drawing (``plotting.contributions``) and the resolution plot
(``processing.resolution_plot``) both read the curves from here, so the
picture and the residual are made of the same curves as the number.
"""
from __future__ import annotations

import numpy as np


def method(row) -> str:
    """What produced the row's number, in the order ``pdf_report`` reads it."""
    if row.get('_fit_curve') and (row.get('_joint_fit') or row.get('_by_cascade')):
        return 'fit'
    if row.get('_integ_mode') == 'Shape' and row.get('_shape_fit'):
        return 'shape'
    if row.get('_deconv'):
        return 'deconv'
    return 'integral'


def span(row, ppm):
    """The region the row's number was read from, ``(a, b)`` in ppm; None
    when the row holds none."""
    if method(row) == 'fit':
        return tuple(row['_fit_curve']['focus'])
    mw = row.get('_multiplet_window')
    if mw and mw[0] is not None and mw[1] is not None:
        return float(mw[0]), float(mw[1])
    p = row if row.get('start_idx') is not None else (row.get('_peak') or {})
    if p.get('start_idx') is None or p.get('end_idx') is None:
        return None
    return float(ppm[p['start_idx']]), float(ppm[p['end_idx']])


def is_singlet_row(row) -> bool:
    """Whether the row's signal is a singlet: what the quantification
    recorded (``_is_singlet``), else the multiplicity of the signal it was
    matched on, read as the annotation reads it (``is_singlet_like``)."""
    flag = row.get('_is_singlet')
    if flag is not None:
        return bool(flag)
    from processing.annotation_scoring import is_singlet_like
    hit = (row.get('_hits') or [{}])[0] or {}
    kind = (row.get('_multiplicity') or hit.get('multiplicity') or hit.get('type') or '')
    return is_singlet_like(str(kind).strip().lower() or 's', strict=True)


def own_line_marks(row) -> list:
    """The positions a figure marks with the software's peak marker (red,
    dotted): a singlet's line only. A multiplet is drawn as its filled shape,
    and its lines marked one by one only crowded the figure."""
    if not is_singlet_row(row):
        return []
    lines = row.get('_line_ppms')
    if lines:
        return [float(v) for v in lines]
    p = row.get('_peak') or {}
    return [float(p['ppm'])] if p.get('ppm') is not None else []


def inside(sp, lo, hi) -> bool:
    return sp is not None and min(sp) < hi and max(sp) > lo


def fit_curves(fit, group):
    """``(x, base, focus top, whole top)`` of a fit record for ``group``, as
    ``plotting.fit_curve.draw_fit_curve`` builds them, sorted on x; None
    without an amplitude there."""
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
    top = base + float(a) * np.asarray(unit, dtype=float)
    focus_top = base + float(a) * np.asarray(focus, dtype=float)
    return x[o], base[o], focus_top[o], top[o]


def shape_copy(row, ppm, trace):
    """``(x, y)`` of the copy of the row's reference shape that fits under
    ``trace``, x ascending; None without a shape."""
    from processing.pattern_shape import _amplitude
    from processing.quant_shape import shape_model
    built = shape_model({'shape_fit': row['_shape_fit']}, ppm)
    if built is None:
        return None
    xs, ms, (slo, shi) = built
    t = trace[(ppm >= slo) & (ppm <= shi)]
    sig = (float(row.get('_noise_sigma') or 0.0)
           or float(np.std(t[:max(3, len(t) // 10)]) or 1.0))
    a, _sup = _amplitude(t, ms, sig)
    o = np.argsort(xs)
    return xs[o], a * ms[o]


def integral_piece(row, ppm, trace):
    """``(x, base, y, where)`` of the integrated window: the trace over the
    window, the baseline the integration subtracted under it, and where the
    area was summed (above the baseline, on the pieces kept). None without a
    window."""
    from processing.quant_integration import baseline_and_signal, baseline_under
    sp = span(row, ppm)
    if sp is None:
        return None
    lo, hi = sorted(sp)
    sel = (ppm >= lo) & (ppm <= hi)
    if sel.sum() < 2:
        return None
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
    return xf, base, yf, where


def window_components(dc, ppm, amp_of):
    """``[(record, x, y)]``: every component of a shared window with an
    amplitude, as the decomposition placed it, at ``amp_of(amplitudes)``;
    x ascending over the window."""
    from processing.quant_shared import component_from_record
    wlo, whi = dc['window']
    xs = np.sort(ppm[(ppm >= wlo) & (ppm <= whi)])
    out = []
    for rec in dc.get('components', ()):
        c = component_from_record(rec)
        amps = (dc.get('amplitudes') or {}).get(rec['met_id']) or {}
        if c is None or not amps:
            continue
        out.append((rec, xs, amp_of(amps) * np.asarray(c.curve(xs), dtype=float)))
    return out
