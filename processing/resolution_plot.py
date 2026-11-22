# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/resolution_plot.py
"""The resolution plot (RP-Method): the spectrum against what the
quantification resolved of it.

Every quantified row is laid on the ppm axis as the curve that gave its
number (``processing.analyte_models``): a fit as the whole fitted compound
(every signal of it the fit placed), a shape as the copy that fits under the
trace, a shared window as its components, an integral as the data it summed
-- the last one is not a model, and the residual under it is nothing by
construction. Their sum is what the quantification resolved; the spectrum
less that sum, and less the baselines the fits stood on, is the residual.

A residual standing above the noise is a signal nothing explains (a
**missing** signal: an analyte absent from the result, or another signal of
a quantified one that only its window was measured on). One standing below
is a fit claiming more than the spectrum holds (**over**). Every missing
signal is listed with the annotation's matches whose listed signal falls
there -- the candidates the analyst can include by marking the signal Good
(``controllers.signal_lines.SignalLineEditor.classify``), which the next
quantification takes into the election.

No threshold is new: the noise is the one detection measured, a region runs
while the residual stays above ``pattern_shape.SLACK_SIGMA`` noise (the
slack a shape is allowed over the trace) and is kept when it reaches the
lone-singlet minimum signal-to-noise of the quantification
(``analysis_defaults.QUANTIFICATION['singlet_min_snr']``); candidates are
searched within the annotation's own match tolerance.
"""
from __future__ import annotations

import numpy as np

from processing.analyte_models import (
    fit_curves, integral_piece, method, shape_copy, span, window_components)
from processing.pattern_shape import SLACK_SIGMA

MISSING = 'missing'
OVER = 'over'


def _key(r):
    return str(r.get('hmdb_id', r.get('metabolite')))


def _reported(r) -> bool:
    return not (r.get('_not_quantified') or r.get('_is_is'))


def analyte_curves(rows, ppm, trace, group, amp_of):
    """``(curves, baselines)``: every reported row as what gave its number,
    ``[{'id', 'name', 'method', 'x', 'y'}]`` (x ascending, y the analyte
    alone, without baseline), and the fits' baselines ``[(x, base)]``.

    A shared window is decomposed once; a component whose analyte has no
    number in the table is kept with ``method`` ``'unreported'``: it took
    its part of the window all the same."""
    ppm = np.asarray(ppm, dtype=float)
    trace = np.asarray(trace, dtype=float)
    ids = {_key(r) for r in rows if _reported(r)}
    names = {_key(r): r.get('metabolite', _key(r)) for r in rows}
    curves, baselines, windows = [], [], set()
    for r in rows:
        if not _reported(r):
            continue
        m = method(r)
        mid = _key(r)
        if m == 'fit':
            cur = fit_curves(r['_fit_curve'], group)
            if cur is None:
                continue
            x, base, _focus, whole = cur
            curves.append({'id': mid, 'name': names[mid], 'method': m,
                           'x': x, 'y': np.clip(whole - base, 0.0, None)})
            if r['_fit_curve'].get('baseline_by_group'):
                baselines.append((x, base))
        elif m == 'shape':
            cur = shape_copy(r, ppm, trace)
            if cur is not None:
                curves.append({'id': mid, 'name': names[mid], 'method': m,
                               'x': cur[0], 'y': cur[1]})
        elif m == 'deconv':
            dc = r['_deconv']
            w = tuple(dc['window'])
            if w in windows:
                continue
            windows.add(w)
            for rec, x, y in window_components(dc, ppm, amp_of):
                cid = str(rec['met_id'])
                curves.append({'id': cid,
                               'name': names.get(cid) or rec.get('label', cid),
                               'method': m if cid in ids else 'unreported',
                               'x': x, 'y': np.clip(y, 0.0, None)})
        else:
            cur = integral_piece(r, ppm, trace)
            if cur is None:
                continue
            x, base, y, where = cur
            o = np.argsort(x)
            curves.append({'id': mid, 'name': names[mid], 'method': m,
                           'x': x[o], 'y': np.where(where, np.clip(y - base, 0.0, None), 0.0)[o]})
    return curves, baselines


def on_axis(x, y, ppm):
    """``y`` given on ascending ``x``, laid on ``ppm`` (either order), zero
    outside ``x``."""
    ppm = np.asarray(ppm, dtype=float)
    if len(x) < 2:
        return np.zeros_like(ppm)
    return np.interp(ppm, x, y, left=0.0, right=0.0)


def resolve(rows, ppm, trace, group, amp_of) -> dict:
    """The spectrum against what was resolved of it, on ``ppm``:
    ``{'curves', 'resolved', 'baseline', 'residual'}``. ``baseline`` is the
    mean of the fits' baselines where any stands, zero elsewhere;
    ``residual = trace - baseline - resolved``."""
    ppm = np.asarray(ppm, dtype=float)
    trace = np.asarray(trace, dtype=float)
    curves, baselines = analyte_curves(rows, ppm, trace, group, amp_of)
    resolved = np.zeros_like(ppm)
    for c in curves:
        resolved += on_axis(c['x'], c['y'], ppm)
    base_sum, base_n = np.zeros_like(ppm), np.zeros_like(ppm)
    for x, b in baselines:
        inside = (ppm >= x[0]) & (ppm <= x[-1])
        base_sum[inside] += np.interp(ppm[inside], x, b)
        base_n[inside] += 1
    baseline = np.divide(base_sum, base_n, out=np.zeros_like(ppm), where=base_n > 0)
    return {'curves': curves, 'resolved': resolved, 'baseline': baseline,
            'residual': trace - baseline - resolved}


def excluded_mask(ppm, regions) -> np.ndarray:
    """True on the points inside any ``(lo, hi)`` of ``regions``."""
    ppm = np.asarray(ppm, dtype=float)
    out = np.zeros(ppm.shape, dtype=bool)
    for lo, hi in regions or ():
        out |= (ppm >= min(lo, hi)) & (ppm <= max(lo, hi))
    return out


def _runs(mask):
    """``[(i, j)]`` index ranges (inclusive) of the True runs of ``mask``."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def _pieces(seg, height):
    """``[(a, b)]`` index ranges of ``seg`` (inclusive), one per peak that
    reaches ``height`` with a prominence of ``height``, cut at the lowest
    point between neighbours: a run that holds several signals is several
    regions."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(np.concatenate(([0.0], seg, [0.0])), height=height,
                       prominence=height)
    pk = pk - 1
    if len(pk) < 2:
        return [(0, len(seg) - 1)]
    cuts = [int(a + np.argmin(seg[a:b + 1])) for a, b in zip(pk[:-1], pk[1:])]
    starts = [0] + [c + 1 for c in cuts]
    ends = cuts + [len(seg) - 1]
    return list(zip(starts, ends))


def residual_regions(ppm, trace, residual, sigma, *, min_snr, excluded=(),
                     edge_sigma=SLACK_SIGMA) -> list:
    """The stretches where the residual stands beyond ``edge_sigma`` noise
    and reaches ``min_snr`` noise at its apex, outside ``excluded``: above
    the axis a ``MISSING`` signal, below it an ``OVER`` claim. A stretch
    holding several peaks of ``min_snr`` prominence is cut between them.

    ``[{'kind', 'lo', 'hi', 'apex_ppm', 'snr', 'area', 'share'}]``, the
    tallest first; ``area`` in the trace's units times ppm, ``share`` the
    part of the spectrum's own area there (0-1)."""
    ppm = np.asarray(ppm, dtype=float)
    trace = np.asarray(trace, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if not sigma or sigma <= 0:
        return []
    keep = ~excluded_mask(ppm, excluded)
    out = []
    for kind, r in ((MISSING, residual), (OVER, -residual)):
        for i0, j0 in _runs((r > edge_sigma * sigma) & keep):
            if r[i0:j0 + 1].max() < min_snr * sigma or j0 <= i0:
                continue
            for a, b in _pieces(r[i0:j0 + 1], min_snr * sigma):
                i, j = i0 + a, i0 + b
                seg = r[i:j + 1]
                k = i + int(np.argmax(seg))
                if r[k] < min_snr * sigma or j <= i:
                    continue
                x = ppm[i:j + 1]
                o = np.argsort(x)
                area = float(abs(np.trapezoid(seg[o], x[o])))
                own = float(abs(np.trapezoid(np.clip(trace[i:j + 1], 0, None)[o], x[o])))
                out.append({'kind': kind, 'lo': float(x.min()), 'hi': float(x.max()),
                            'apex_ppm': float(ppm[k]), 'snr': float(r[k] / sigma),
                            'area': area, 'share': (area / own) if own > 0 else None})
    out.sort(key=lambda g: -g['snr'])
    return out


def resolved_share(ppm, trace, resolved, baseline, excluded=()) -> float | None:
    """The part of the spectrum's area above its baselines (outside
    ``excluded``) that the resolved curves account for, 0-1."""
    ppm = np.asarray(ppm, dtype=float)
    keep = ~excluded_mask(ppm, excluded)
    o = np.argsort(ppm[keep])
    x = ppm[keep][o]
    real = np.clip((np.asarray(trace) - np.asarray(baseline))[keep][o], 0, None)
    res = np.minimum(np.asarray(resolved)[keep][o], real)
    tot = float(np.trapezoid(real, x))
    return float(np.trapezoid(res, x)) / tot if tot > 0 else None


def modelled_extent(row, ppm):
    """``(lo, hi)`` of the spectrum the row's curve covers here: a fit's
    record (``fit_curve_record`` keeps the fitted compound only around the
    signal in focus), else the window its number came from."""
    if method(row) == 'fit':
        x = row['_fit_curve'].get('ppm')
        x = np.asarray(() if x is None else x, dtype=float)
        return (float(x.min()), float(x.max())) if x.size else None
    if method(row) == 'deconv':
        return tuple(float(v) for v in row['_deconv']['window'])
    return span(row, ppm)


def quantified_signals_in(region, metabolites, rows, ppm) -> list:
    """The quantified analytes a residual in ``region`` is most likely
    another signal of: the annotation matched one of their peaks there, and
    the curve their number left (``modelled_extent``) does not reach it --
    an integral, a shape or a shared window models only its own window, and
    a fit's record only the region around its signal in focus. A residual
    on what is modelled is that analyte's fit falling short, and stays to be
    looked at. Names, in the annotation's order."""
    lo, hi = float(region['lo']), float(region['hi'])
    by_id = {_key(r): r for r in rows or () if _reported(r)}
    out = []
    for met in metabolites or ():
        r = by_id.get(str(met.get('id')))
        if r is None:
            continue
        ext = modelled_extent(r, ppm)
        if ext is not None and min(ext) <= hi and max(ext) >= lo:
            continue
        if any(q.get('peak_ppm') is not None and lo <= float(q['peak_ppm']) <= hi
               for q in met.get('matched_peaks') or ()):
            out.append(met.get('name', str(met.get('id'))))
    return out


def fingerprint(cand, field_mhz):
    """``(x_ppm, height, kind)`` of a candidate's reference fingerprint laid
    where the annotation placed it, x ascending, tallest point 1: the
    profile the database stores (``pattern_profile``, in Hz at the
    reference's own width; ``kind`` ``'profile'``), else its pattern lines
    (``'lines'``, to be drawn as sticks). Hz to ppm by the field; None
    without either, or without a field."""
    import json
    from processing.pattern_shape import parse_profile
    f = float(cand.get('field_mhz') or field_mhz or 0)
    if f <= 0:
        return None
    prof = parse_profile(cand.get('profile'))
    if prof is not None:
        offs, h = prof
        x = cand['centre'] + offs / f
        o = np.argsort(x)
        return x[o], h[o] / h.max(), 'profile'
    lines = cand.get('pattern_lines')
    if isinstance(lines, str):
        try:
            lines = json.loads(lines)
        except ValueError:
            lines = None
    if not lines:
        return None
    offs = np.asarray([float(a) for a, _h in lines])
    h = np.asarray([float(b) for _a, b in lines])
    x = cand['centre'] + offs / f
    o = np.argsort(x)
    return x[o], h[o] / h.max(), 'lines'


def _lines(raw):
    """A signal's pattern lines ``[(offset_hz, rel)]``. A singlet is one
    line; the shape builders (``fit_shape``, ``shape_model``) are written
    for multiplets and want two, so a second of height zero is added -- it
    changes nothing in the shape."""
    import json
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    lines = [(float(a), float(b)) for a, b in raw or ()]
    if len(lines) == 1:
        lines.append((lines[0][0], 0.0))
    return lines


def fit_in_residual(cand, ppm, residual, sigma, region, field_mhz=None, study_lw_hz=None,
                    tolerance_ppm=None):
    """The candidate's reference shape fitted into the residual: slid
    within the annotation's own tolerance around where it placed it
    (``pattern_tolerance_ppm``, else ``tolerance_ppm``) to the position
    where the largest copy of it, at the study's linewidth, fits under the
    residual -- ``pattern_shape.fit_shape``, the rule the annotation
    placed it with on the spectrum (Weljie et al., Anal. Chem. 2006, 78,
    4430), applied to what the quantification left.

    ``{'x', 'y', 'share', 'origin', 'shift_ppm'}``: ``share`` the part of
    the residual's area over ``region`` the copy accounts for (0-1),
    ``shift_ppm`` how far from the annotation's placement it fell. None
    without a shape, a field, a linewidth or a tolerance."""
    from processing.pattern_shape import fit_shape, parse_profile
    from processing.quant_shape import shape_model
    f = cand.get('field_mhz') or field_mhz
    lw = cand.get('study_lw_hz') or study_lw_hz
    tol = cand.get('tolerance_ppm') or tolerance_ppm
    if not f or not lw or not sigma or not tol:
        return None
    ppm = np.asarray(ppm, dtype=float)
    res = np.asarray(residual, dtype=float)
    fit = fit_shape(ppm, res, db_ppm=float(cand['centre']), field_mhz=float(f),
                    fwhm_hz=float(lw), tolerance_ppm=float(tol), sigma=float(sigma),
                    profile=parse_profile(cand.get('profile')),
                    profile_fwhm_hz=cand.get('profile_fwhm_hz'),
                    lines=_lines(cand.get('pattern_lines')))
    if fit is None:
        return None
    built = shape_model(_shape_ev(dict(cand, centre=fit['origin_ppm']), f, lw), ppm)
    if built is None:
        return None
    x, m, (lo, hi) = built
    o = np.argsort(x)
    x, m = x[o], m[o]
    r = np.interp(x, np.sort(ppm), res[np.argsort(ppm)])
    y = float(fit['amplitude']) * m
    inside = (x >= region['lo']) & (x <= region['hi'])
    own = np.clip(r, 0, None)
    tot = float(np.trapezoid(own[inside], x[inside])) if inside.sum() > 1 else 0.0
    got = float(np.trapezoid(np.minimum(y, own)[inside], x[inside])) if inside.sum() > 1 else 0.0
    return {'x': x, 'y': y, 'share': (got / tot) if tot > 0 else None,
            'origin': fit['origin_ppm'], 'shift_ppm': fit['shift_ppm']}


def _shape_ev(cand, f, lw):
    return {'shape_fit': {'origin_ppm': cand['centre'], 'field_mhz': float(f),
                          'study_linewidth_hz': float(lw),
                          'pattern_profile': cand.get('profile'),
                          'profile_fwhm_hz': cand.get('profile_fwhm_hz'),
                          'pattern_lines': _lines(cand.get('pattern_lines'))}}


def expected_from_number(cand, row, ppm, residual, group, region, field_mhz=None,
                         study_lw_hz=None, origin=None):
    """What a quantified candidate's own number says this signal holds: the
    fit's area per proton for ``group`` (``amp_by_group`` -- the record's
    curve is one unit of area per proton) times the protons the reference
    lists for this signal, in the reference's shape at the study's width,
    at ``origin`` (where ``fit_in_residual`` found it), else where the
    annotation placed it. ``{'x', 'y', 'ratio'}`` with ``ratio``
    that area over the residual's area in ``region`` (1 = the number
    accounts for the residual exactly); None for a row whose number is not
    a fit, or without a shape."""
    from processing.quant_shape import shape_model
    fit = (row or {}).get('_fit_curve')
    if not fit or method(row) != 'fit' or not cand.get('protons'):
        return None
    a = (fit.get('amp_by_group') or {}).get(str(group))
    f = cand.get('field_mhz') or field_mhz
    lw = cand.get('study_lw_hz') or study_lw_hz
    if a is None or not f or not lw:
        return None
    ppm = np.asarray(ppm, dtype=float)
    at = dict(cand, centre=origin) if origin is not None else cand
    built = shape_model(_shape_ev(at, f, lw), ppm)
    if built is None:
        return None
    x, m, (lo, hi) = built
    o = np.argsort(x)
    x, m = x[o], m[o]
    unit = float(np.trapezoid(m, x))
    if unit <= 0:
        return None
    y = float(a) * float(cand['protons']) * m / unit
    sel = (ppm >= region['lo']) & (ppm <= region['hi'])
    xs = ppm[sel]
    oo = np.argsort(xs)
    own = float(np.trapezoid(np.clip(np.asarray(residual, dtype=float)[sel][oo], 0, None), xs[oo]))
    inside = (x >= region['lo']) & (x <= region['hi'])
    got = float(np.trapezoid(y[inside], x[inside])) if inside.sum() > 1 else 0.0
    return {'x': x, 'y': y, 'ratio': (got / own) if own > 0 else None}


def secondary_confirmed(row, db_signal, mp):
    """Why a listed signal of a confirmed analyte may be taken off the
    residual at what the analyte's number predicts, or None when it may not.

    A multiplet only when the annotation confirmed it (``mp['_confirmed']``,
    ``annotation_scoring.is_confirmed``: every expected line found, ΔJ under
    ``dj_bad_hz``, Δδ under ``dppm_gold``). A singlet has no coupling to
    confirm it: only when STOCSY linked it to the amount measured, in the
    proportion its protons call for (``row['_signal_links']``,
    ``processing.signal_links``, ADR-101) -- the way the engine itself
    decides on singlets."""
    from processing.annotation_scoring import is_singlet_like
    kind = str((mp or {}).get('type') or db_signal.get('type') or 's').strip().lower()
    if is_singlet_like(kind, strict=True):
        c = _signal_centre(db_signal)
        for link in row.get('_signal_links') or ():
            if c is not None and abs(float(link.get('db_ppm', 1e9)) - c) < 1e-3:
                return f"singlet linked by STOCSY (r {float(link.get('r', float('nan'))):.2f})"
        return None
    if mp and mp.get('_confirmed'):
        dj = mp.get('delta_j_hz')
        return (f"{kind} confirmed: {mp.get('_n_found_lines')}/{mp.get('_n_exp_lines')} lines"
                + (f", ΔJ {float(dj):.2f} Hz" if dj is not None else '')
                + f", Δδ {1000 * float(mp.get('ppm_diff') or 0):.0f} mppm")
    return None


def secondary_signals(rows, metabolites, ppm, residual, group, sigma, *, field_mhz=None,
                      study_lw_hz=None, tolerance_ppm=None, min_confidence=2):
    """The other signals of the confirmed analytes, taken off the residual
    at what each analyte's own number says they hold -- the idea: a signal
    confirmed and quantified is the reference for the rest of its analyte.

    For every analyte with a number from a fit at ``min_confidence`` dots
    or more, every signal the reference lists with protons that its fit's
    curve does not reach (``modelled_extent``) and that is itself confirmed
    (``secondary_confirmed``: a multiplet with all its lines, ΔJ and Δδ
    within the annotation's quality parameters; a singlet only when STOCSY
    linked it to the amount measured): predicted by
    ``expected_from_number``, at the position ``fit_in_residual`` finds on
    the residual, and taken only up to what the residual still holds there
    plus ``SLACK_SIGMA`` noise -- a signal that is not in the spectrum
    cannot be subtracted from it. In the annotation's order, each taking
    what the ones before it left.

    ``[{'id', 'name', 'method': 'secondary', 'signal_ppm', 'why', 'x', 'y',
    'predicted', 'taken'}]``: ``y`` what was taken, on ``ppm``'s points
    under the signal; ``predicted`` and ``taken`` areas; ``why`` the
    confirmation that let it be taken."""
    ppm = np.asarray(ppm, dtype=float)
    left = np.asarray(residual, dtype=float).copy()
    by_id = {_key(r): r for r in rows or () if _reported(r)}
    out = []
    for met in metabolites or ():
        r = by_id.get(str(met.get('id')))
        if (r is None or method(r) != 'fit'
                or (r.get('_confidence') or 0) < min_confidence):
            continue
        ext = modelled_extent(r, ppm)
        for p in met.get('db_peaks') or ():
            c = _signal_centre(p)
            if c is None or not p.get('protons'):
                continue
            if ext and min(ext) <= c <= max(ext):
                continue
            here = {'lo': c, 'hi': c, 'apex_ppm': c}
            cands = candidates(here, [met], rows, 0.0, ppm)
            if not cands:
                continue
            cand = cands[0]
            mp = (met.get('matched_peaks') or [None])[cand['mp_index']]                 if cand['mp_index'] is not None else None
            why = secondary_confirmed(r, p, mp)
            if why is None:
                continue
            fit = fit_in_residual(cand, ppm, left, sigma, here, field_mhz, study_lw_hz,
                                  tolerance_ppm)
            exp = expected_from_number(cand, r, ppm, left, group, here, field_mhz,
                                       study_lw_hz, origin=fit['origin'] if fit else None)
            if exp is None:
                continue
            on = (ppm >= exp['x'][0]) & (ppm <= exp['x'][-1])
            want = np.interp(ppm[on], exp['x'], exp['y'])
            take = np.minimum(want, np.clip(left[on], 0.0, None) + SLACK_SIGMA * float(sigma))
            left[on] -= take
            xs, ts, ws = ppm[on], take, want
            k = np.argsort(xs)
            out.append({'id': _key(r), 'name': r.get('metabolite', _key(r)),
                        'method': 'secondary', 'signal_ppm': float(cand['listed_ppm']),
                        'why': why,
                        'x': xs[k], 'y': ts[k],
                        'predicted': float(np.trapezoid(ws[k], xs[k])),
                        'taken': float(np.trapezoid(ts[k], xs[k]))})
    return out


def study_linewidth(metabolites):
    """The study's linewidth (Hz) as the annotation measured it -- any
    matched signal carries it -- or None: for a candidate whose own signal
    has none."""
    for met in metabolites or ():
        for q in met.get('matched_peaks') or ():
            if q.get('study_linewidth_hz'):
                return float(q['study_linewidth_hz'])
    return None


def resolution(rows, ppm, trace, group, amp_of, *, metabolites=(), sigma=None,
               field_mhz=None, tolerance_ppm=None, excluded=(), secondary=True,
               min_snr=None) -> dict:
    """The resolution plot of one group, as the RP-Method window and the
    report both show it: ``resolve``, then -- with ``secondary`` and a noise
    level -- the confirmed analytes' other signals taken off the residual
    (``secondary_signals``), the share of the spectrum resolved
    (``resolved_share``) and, with a noise level and ``min_snr``, the
    residual's regions (``residual_regions``). ``resolve``'s keys plus
    ``'secondary'`` (the signals taken), ``'share'`` (0-1, or None) and
    ``'regions'``."""
    ppm = np.asarray(ppm, dtype=float)
    out = resolve(rows, ppm, trace, group, amp_of)
    sec = []
    if secondary and sigma:
        sec = secondary_signals(rows, metabolites, ppm, out['residual'], group, sigma,
                                field_mhz=field_mhz,
                                study_lw_hz=study_linewidth(metabolites),
                                tolerance_ppm=tolerance_ppm)
        for c in sec:
            v = on_axis(c['x'], c['y'], ppm)
            out['resolved'] = out['resolved'] + v
            out['residual'] = out['residual'] - v
        out['curves'] = out['curves'] + sec
    out['secondary'] = sec
    out['share'] = resolved_share(ppm, trace, out['resolved'], out['baseline'], excluded)
    out['regions'] = (residual_regions(ppm, trace, out['residual'], sigma, min_snr=min_snr,
                                       excluded=excluded)
                      if sigma and min_snr is not None else [])
    return out


def claimed_by(region, curves, sigma, edge_sigma=SLACK_SIGMA) -> list:
    """The analytes whose resolved curve stands above ``edge_sigma`` noise
    inside ``region``: on an over-claim, the ones claiming more than the
    spectrum holds there. Names, tallest first."""
    out = []
    for c in curves:
        inside = (c['x'] >= region['lo']) & (c['x'] <= region['hi'])
        if inside.any():
            top = float(c['y'][inside].max())
            if top > edge_sigma * sigma:
                out.append((top, c['name']))
    return [n for _t, n in sorted(out, reverse=True)]


def _signal_centre(p) -> float | None:
    for k in ('ppm_center', 'ppm_listed'):
        if p.get(k) is not None:
            return float(p[k])
    if p.get('ppm_start') is not None and p.get('ppm_end') is not None:
        return (float(p['ppm_start']) + float(p['ppm_end'])) / 2.0
    return None


def candidates(region, metabolites, rows, tolerance, ppm=None) -> list:
    """The annotation's analytes with a listed signal within ``tolerance``
    of ``region``: one entry per analyte, its nearest such signal.

    ``{'met_id', 'name', 'listed_ppm', 'delta_ppm', 'mult', 'protons',
    'matched_ppm', 'mp_index', 'quality', 'score', 'status', 'detail',
    'j_ref', 'j_obs', 'j_exp', 'delta_j', 'lines', 'centre', 'profile',
    'pattern_lines', 'field_mhz', 'predicted'}`` -- all of it read from the
    annotation, nothing measured again: the reference's J as listed, the
    J the annotation observed and expected at this field and their
    difference, its lines found over expected, where it placed the pattern
    (``centre``, ``predicted``) and the reference's own fingerprint.
    ``mp_index`` the matched peak that signal rode on (what marking it Good
    acts on), None when the annotation matched no peak to it; ``status``
    ``'quantified'``, ``'no number'`` or ``'not in the result'``; ``detail``
    says when a quantified analyte's number was read from another signal,
    which is then the one this residual could be. Nearest first, the ones
    without a number before the quantified."""
    lo, hi, apex = float(region['lo']), float(region['hi']), float(region['apex_ppm'])
    by_id = {_key(r): r for r in rows or ()}
    out = []
    for met in metabolites or ():
        mid = str(met.get('id'))
        best = None
        for p in met.get('db_peaks') or ():
            c = _signal_centre(p)
            if c is None:
                continue
            a = float(p.get('ppm_start', c))
            b = float(p.get('ppm_end', c))
            gap = max(0.0, min(a, b) - hi, lo - max(a, b))
            if gap > tolerance:
                continue
            d = c - apex
            if best is None or abs(d) < abs(best[1]):
                best = (p, d, c)
        if best is None:
            continue
        p, d, c = best
        mp_index, mp = None, None
        for i, q in enumerate(met.get('matched_peaks') or ()):
            if q.get('db_ppm') is not None and abs(float(q['db_ppm']) - c) < 1e-3:
                mp_index, mp = i, q
                break
        r = by_id.get(mid)
        if r is None:
            status, detail = 'not in the result', ''
        elif not _reported(r):
            status = 'no number'
            detail = str(r.get('_rejection') or '')
        else:
            status = 'quantified'
            sp = span(r, ppm) if ppm is not None else None
            read_here = sp is not None and min(sp) <= hi + tolerance and max(sp) >= lo - tolerance
            detail = ('' if read_here or method(r) == 'fit' else
                      'number read from another signal; this one is not modelled')
        mp_ = mp or {}
        found, exp = mp_.get('_n_found_lines'), mp_.get('_n_exp_lines')
        predicted = mp_.get('predicted_lines') or [
            {'ppm': float(x), 'rel_intensity': float(h)} for x, h in p.get('sticks') or ()]
        out.append({'met_id': mid, 'name': met.get('name', mid),
                    'listed_ppm': c, 'delta_ppm': d,
                    'j_ref': p.get('j_coupling'),
                    'j_obs': mp_.get('observed_j_hz'),
                    'j_exp': mp_.get('expected_j_hz'),
                    'delta_j': mp_.get('delta_j_hz'),
                    'lines': (found, exp) if found is not None and exp is not None else None,
                    'centre': next((float(mp_[k]) for k in ('ppm_center', 'peak_ppm')
                                    if mp_.get(k) is not None), c),
                    'profile': p.get('pattern_profile'),
                    'pattern_lines': p.get('pattern_lines'),
                    'field_mhz': mp_.get('pattern_field_mhz'),
                    'study_lw_hz': mp_.get('study_linewidth_hz'),
                    'tolerance_ppm': mp_.get('pattern_tolerance_ppm'),
                    'profile_fwhm_hz': p.get('profile_fwhm_hz', mp_.get('profile_fwhm_hz')),
                    'predicted': predicted,
                    'mult': p.get('type') or '', 'protons': p.get('protons'),
                    'matched_ppm': (float(mp['peak_ppm']) if mp and mp.get('peak_ppm') is not None
                                    else None),
                    'mp_index': mp_index,
                    'quality': (mp or {}).get('annotation_quality'),
                    'score': met.get('composite_score', met.get('best_score')),
                    'status': status, 'detail': detail})
    out.sort(key=lambda e: (e['status'] == 'quantified', abs(e['delta_ppm'])))
    return out
