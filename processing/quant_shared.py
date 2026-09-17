"""A window two accepted analytes share, resolved by their shapes.

Integration reads a window as one signal's. Where a second analyte's signal
lies inside it -- betaine's singlet under taurine's triplet at 3.25 ppm on
the mixture bench, isoleucine's line inside valine's doublet at 1.00 -- the
window's area is both of theirs, and the one it was credited to read 86%
and 45% high.

The trace over the window is modelled, sample by sample, as a sum of the
components' shapes at non-negative amplitudes:

    trace(x) = sum_i a_i * shape_i(x),   a_i >= 0

with the shapes the reference database measured (a profile for a complex
multiplet, Lorentzians on the found lines for a first-order one, one
Lorentzian for a singlet) at the study's linewidth and at the positions
annotation found. What may enter the sum is settled BEFORE it is fitted:
the window's owner, and the signals of other analytes the election has
already accepted on their own evidence. A candidate that exists only in
this window does not get a share; what the accepted shapes leave
unexplained is reported as a remainder, not given to anybody.

An intruder whose analyte is measured cleanly elsewhere is held to what
that measurement says: its amplitude here may be up to twice what the
per-proton area its own clean signal gave predicts, and no more -- a
shared window cannot make betaine five times what betaine's own singlet
is. It may be less, down to nothing: the analyte may be wrong, and a floor
would force its shape into the window (measured: with a floor of half the
prediction a false positive took the whole of taurine's triplet).

Non-negative least squares: Lawson, C.L. & Hanson, R.J., Solving Least
Squares Problems, Prentice-Hall 1974, ch. 23; bounded variant via
scipy.optimize.lsq_linear. Fitting a region as a sum of known lineshapes is
targeted profiling (Weljie et al., Anal. Chem. 2006, 78, 4430).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from processing.pattern_shape import (
    lines_model, parse_profile, profile_model)
from processing.quant_integration import pascal_weights

# How far the window reaches past a component's outer line, in linewidths
# (FWHM): the same reach integration gives a signal on its own.
PAD_LINEWIDTHS = 3.0

# An intruder's amplitude is held to this factor of what its own clean
# signal predicts.
BOUND_FACTOR = 2.0

# What of a shape the shapes it is fitted against cannot make (1 - R^2 of
# its column on theirs) for the fit to tell them apart. Measured on the
# mixture bench: a singlet placed on another singlet 0.0, a doublet on a
# doublet 0.0 (each took an arbitrary half), a sugar's lines over lactate's
# quartet 0.42 (took 58% of the lactate), betaine's singlet on taurine's
# outer line 0.63 (the fit found it).
SEPARABLE_MIN = 0.5

# A remainder over this fraction of the window's area is said on the row:
# something the accepted shapes do not account for is here.
REMAINDER_NOTE = 0.10


@dataclass
class Component:
    met_id: str
    label: str
    model: object                    # callable ppm array -> shape, max 1
    lo: float
    hi: float
    nh: float
    owner: bool = False
    # the per-proton area the analyte's own clean signal gave (median over
    # samples); the amplitude here is held to BOUND_FACTOR times what that
    # predicts. None leaves the amplitude free.
    per_proton_hint: float | None = None
    ceiling: float | None = None     # on the amplitude, set by the fit
    shape_area: float = 0.0          # of the model on the window, ppm units
    # what of this shape the owners' shapes cannot make: 1 - R^2 of its
    # column on theirs. Zero is a shape the fit cannot tell from the owner.
    separability: float | None = None
    lines: list = field(default_factory=list)
    # what the model was built from, so a view can build it again
    evidence: dict = field(default_factory=dict)
    linewidth_ppm: float = 0.0


def component_from_evidence(met_id, ev, *, label, linewidth_ppm, nh,
                            owner=False, per_proton_hint=None):
    """A component from a signal's evidence, as the adapter hands it to the
    engine: ``predicted_ppms`` (the lines annotation found, in ppm) with
    ``line_weights`` where the pattern gave them, a ``shape_fit`` for a
    measured shape (carrying its own field), ``peak_ppm`` for a singlet.
    None when nothing places it."""
    ev = ev or {}
    lw = float(linewidth_ppm)
    fit = ev.get('shape_fit') or {}
    if (fit.get('origin_ppm') is not None and fit.get('field_mhz')
            and (fit.get('pattern_profile') or fit.get('pattern_lines'))):
        f = float(fit['field_mhz'])
        origin = float(fit['origin_ppm'])
        prof = parse_profile(fit.get('pattern_profile'))
        if prof is not None:
            extra = max(0.0, lw * f - float(fit.get('profile_fwhm_hz') or 0.0))
            offs = np.asarray(prof[0], dtype=float)

            def model(x, _p=prof, _o=origin, _e=extra):
                return profile_model(x, _p, _o, _e, f)
            lo = origin + (offs.min() - PAD_LINEWIDTHS * lw * f) / f
            hi = origin + (offs.max() + PAD_LINEWIDTHS * lw * f) / f
            lines = [origin + o / f for o in (fit.get('apex_ppms') or ())]
        else:
            pl = [(float(o), float(r)) for o, r in fit['pattern_lines']]

            def model(x, _l=pl, _o=origin):
                return lines_model(x, _l, _o, lw * f, f)
            offs = [o for o, _ in pl]
            lo = origin + (min(offs) - PAD_LINEWIDTHS * lw * f) / f
            hi = origin + (max(offs) + PAD_LINEWIDTHS * lw * f) / f
            lines = [origin + o / f for o in offs]
        return Component(met_id, label, model, float(lo), float(hi),
                         float(nh), owner=owner, lines=lines,
                         per_proton_hint=per_proton_hint,
                         evidence=_shape_evidence(ev), linewidth_ppm=lw)
    ppms = [float(p) for p in (ev.get('predicted_ppms') or ()) if p is not None]
    if not ppms and ev.get('peak_ppm') is not None:
        ppms = [float(ev['peak_ppm'])]
    if not ppms:
        return None
    weights = (list(ev.get('line_weights') or ())
               if ev.get('line_weights') and len(ev['line_weights']) == len(ppms)
               else pascal_weights(len(ppms)))
    centre = float(np.mean(ppms))
    # offsets and width in ppm, with a "field" of 1: lines_model divides
    # its hertz by the field, and a field of one leaves ppm as ppm
    pl = [((p - centre), float(w)) for p, w in zip(ppms, weights)]

    def model(x, _l=pl, _c=centre):
        return lines_model(x, _l, _c, lw, 1.0)
    lo = min(ppms) - PAD_LINEWIDTHS * lw
    hi = max(ppms) + PAD_LINEWIDTHS * lw
    return Component(met_id, label, model, float(lo), float(hi), float(nh),
                     owner=owner, lines=list(ppms),
                     per_proton_hint=per_proton_hint,
                     evidence=_shape_evidence(ev), linewidth_ppm=lw)


def _shape_evidence(ev) -> dict:
    """The part of a signal's evidence its component is built from."""
    return {k: ev[k] for k in ('multiplicity', 'shape_fit', 'predicted_ppms',
                               'line_weights', 'peak_ppm') if ev.get(k)}


def component_from_record(rec, *, owner=None):
    """The component a ``resolve_window`` result recorded, built again --
    for a view that draws the fit or edits it."""
    return component_from_evidence(
        rec['met_id'], rec.get('evidence') or {}, label=rec.get('label', ''),
        linewidth_ppm=rec['linewidth_ppm'], nh=rec.get('nh') or 1.0,
        owner=rec.get('owner') if owner is None else owner,
        per_proton_hint=rec.get('per_proton_hint'))


def resolve_window(components, ppm, intensities, *, included=None,
                   noise_sigma=None) -> dict | None:
    """Fit the components to every sample over the union of their windows.

    Returns ``{'window': (lo, hi), 'areas': {met_id: {sample: area}},
    'amplitudes': {met_id: {sample: a}}, 'remainder': {sample: fraction},
    'components': [...]}`` -- areas in the integration's units (intensity
    x ppm), the remainder the fraction of the window's area the shapes
    leave unexplained (clipped at zero: what the fit overshoots is not a
    remainder), ``inseparable`` the intruders left out because the owner's
    shape makes theirs (``SEPARABLE_MIN``). ``{'refused': why}`` when an
    owner cannot be told from another owner, or nothing separable is left.
    None with fewer than two components or no trace.
    """
    comps = [c for c in components if c is not None]
    if len(comps) < 2:
        return None
    p = np.asarray(ppm, dtype=float)
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    lo = min(c.lo for c in comps)
    hi = max(c.hi for c in comps)
    sel = (p >= lo) & (p <= hi)
    if sel.sum() < 8:
        return None
    x = p[sel]
    order = np.argsort(x)
    cols = []
    for c in comps:
        m = np.asarray(c.model(x), dtype=float)
        top = float(m.max()) if m.size else 0.0
        m = m / top if top > 0 else m
        c.shape_area = float(np.trapezoid(m[order], x=x[order]))
        cols.append(m)
    A = np.column_stack(cols)
    sep = separability(A, [c.owner for c in comps])
    for c, s_i in zip(comps, sep):
        c.separability = s_i
    # An owner the other owners' shapes can make is not resolvable: neither
    # number would mean anything.
    for c in comps:
        if c.owner and c.separability is not None                 and c.separability < SEPARABLE_MIN:
            return {'refused': f"{c.label} cannot be told by shape from the "
                               f"other analyte measured on this window"}
    inseparable = [c for c in comps if not c.owner
                   and c.separability is not None
                   and c.separability < SEPARABLE_MIN]
    keep = [i for i, c in enumerate(comps) if c not in inseparable]
    comps = [comps[i] for i in keep]
    if len(comps) < 2:
        return {'refused': f"{', '.join(c.label for c in inseparable)} "
                           f"cannot be told by shape from this signal"}
    A = A[:, keep]
    for c in comps:
        c.ceiling = amplitude_ceiling(c.per_proton_hint, c.nh, c.shape_area)
    n_s = arr.shape[1]
    idx = range(n_s) if included is None else [i for i in included
                                              if 0 <= i < n_s]
    lb = np.zeros(len(comps))
    ub = np.array([c.ceiling if c.ceiling is not None else np.inf
                   for c in comps])
    areas = {c.met_id: {} for c in comps}
    amps = {c.met_id: {} for c in comps}
    remainder = {}
    for s in idx:
        t = arr[sel, s]
        res = lsq_linear(A, t, bounds=(lb, ub), lsmr_tol='auto')
        a = np.asarray(res.x, dtype=float)
        fitted = A @ a
        total = float(np.trapezoid(np.clip(t[order], 0, None), x=x[order]))
        left = float(np.trapezoid(np.clip((t - fitted)[order], 0, None),
                                  x=x[order]))
        remainder[int(s)] = float(left / total) if total > 0 else 0.0
        for c, ai in zip(comps, a):
            amps[c.met_id][int(s)] = float(ai)
            areas[c.met_id][int(s)] = float(ai) * c.shape_area
    return {'window': (float(lo), float(hi)), 'areas': areas,
            'amplitudes': amps, 'remainder': remainder,
            'inseparable': [c.label for c in inseparable],
            'components': [{'met_id': c.met_id, 'label': c.label,
                            'owner': c.owner, 'nh': c.nh,
                            'lines': list(c.lines),
                            'ceiling': c.ceiling,
                            'separability': c.separability,
                            'shape_area': c.shape_area,
                            'per_proton_hint': c.per_proton_hint,
                            'evidence': c.evidence,
                            'linewidth_ppm': c.linewidth_ppm}
                           for c in comps]}


def separability(A, owner_flags) -> list:
    """For each column of ``A``, the fraction of it the owners' columns
    cannot make: 1 - R^2 of a least-squares fit of the column on the owner
    columns -- for an owner, on the other owners' columns. None where
    there is nothing to fit against. Two shapes on the same lines give
    zero, and a fit between them hands either an arbitrary share."""
    A = np.asarray(A, dtype=float)
    own = [i for i, f in enumerate(owner_flags) if f]
    out = []
    for i in range(A.shape[1]):
        against = [j for j in own if j != i]
        if not against:
            out.append(None)
            continue
        y = A[:, i]
        X = A[:, against]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        ss = float(np.dot(y, y))
        out.append(float(np.dot(resid, resid) / ss) if ss > 0 else 0.0)
    return out


def amplitude_ceiling(per_proton_hint, nh, shape_area, *,
                      factor=BOUND_FACTOR) -> float | None:
    """The most an intruder may be here, given the per-proton area its own
    clean signal gave: times its protons here, over the shape's area,
    times ``factor``. None without a measurement to hold it to."""
    if (per_proton_hint is None or not np.isfinite(per_proton_hint)
            or per_proton_hint <= 0 or not shape_area or shape_area <= 0
            or not nh):
        return None
    return float(per_proton_hint) * float(nh) / float(shape_area) * float(factor)


def _lines_of(ev) -> list:
    fit = (ev or {}).get('shape_fit') or {}
    if fit.get('apex_ppms'):
        return [float(v) for v in fit['apex_ppms']]
    out = [float(p) for p in ((ev or {}).get('predicted_ppms') or ())
           if p is not None]
    if not out and (ev or {}).get('peak_ppm') is not None:
        out = [float(ev['peak_ppm'])]
    return out


def resolve_shared_windows(measured, candidates, ppm, intensities, *,
                           per_proton_areas, included=None, linewidth_ppm,
                           names=None, is_accepted=None, lines_of=None,
                           overrides=None, noise_sigma=None) -> dict:
    """Resolve, by their shapes, the windows of the winners that hold another
    accepted analyte's signal.

    ``measured`` is the engine's per-winner record (``window``, ``areas``,
    ``nh``, ``signal``); ``candidates`` every signal the annotation handed
    over; ``per_proton_areas`` the per-proton areas per sample the winners
    were measured at. A signal of a winner OTHER than the window's owner
    whose lines fall inside the window, verified (``is_accepted``) and not
    the signal that winner is measured on, is an intruder, held to what its
    analyte's own measurement predicts. Two winners whose windows overlap
    are resolved together, each free.

    ``lines_of(signal)`` says where the peaks detection found for a signal
    are. An intruder enters only where detection placed it (a listing is
    not a measurement) and only where its shape can be told from the
    owner's (``SEPARABLE_MIN``): a singlet matched to the owner's own line
    is the owner's line, and a fit between the two hands the intruder an
    arbitrary share. What was left out is recorded in ``skipped`` on the
    result and said on the row.

    ``overrides`` is what the analyst decided in the review window, by
    owner: ``{'off': True}`` leaves that owner's window whole, and
    ``{'exclude': [met_id, ...]}`` keeps those analytes out of it. Either
    is said on the row: an edit that leaves no trace is a number nobody
    can question.

    Returns ``{owner_met_id: result}`` with the owner's areas replaced in
    ``measured`` and ``per_proton_areas``, and ``shared_with``,
    ``deconv`` (the fit's components and remainder) and
    ``captured=None`` written on the record: a shape's area is exact.
    """
    names = names or {}
    accept = is_accepted or (lambda ev: True)
    where = lines_of or (lambda c: [])
    overrides = overrides or {}

    def _evidence(c):
        ev = dict(c.evidence or {})
        found = where(c)
        if found and not ev.get('shape_fit'):
            ev['predicted_ppms'] = list(found)
            if ev.get('line_weights') and len(ev['line_weights']) != len(found):
                ev.pop('line_weights')
        return ev, bool(found)
    lw = float(linewidth_ppm)
    winners = {mid: rec for mid, rec in measured.items()
               if rec.get('window') and not rec.get('shape')}
    if not winners:
        return {}
    # union-find over overlapping owner windows
    mids = list(winners)
    parent = {m: m for m in mids}

    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    for i, a in enumerate(mids):
        lo_a, hi_a = winners[a]['window']
        for b in mids[i + 1:]:
            lo_b, hi_b = winners[b]['window']
            if lo_a <= hi_b and lo_b <= hi_a:
                parent[find(a)] = find(b)
    groups: dict = {}
    for m in mids:
        groups.setdefault(find(m), []).append(m)

    # the signals each winner is measured on, so its own signal is not its
    # own intruder
    own_signal = {mid: id(rec['signal']) for mid, rec in winners.items()}
    results = {}
    skipped: dict = {}
    for members in groups.values():
        if any((overrides.get(m) or {}).get('off') for m in members):
            for m in members:
                winners[m]['unshared_reason'] = (
                    'the analyst switched the resolution of this window off')
            continue
        excluded = set()
        for m in members:
            excluded |= {str(x) for x in
                         (overrides.get(m) or {}).get('exclude', ())}
        lo = min(winners[m]['window'][0] for m in members)
        hi = max(winners[m]['window'][1] for m in members)
        comps = []
        for m in members:
            rec = winners[m]
            ev, _ = _evidence(rec['signal'])
            comps.append(component_from_evidence(
                m, ev, label=f"{names.get(m, m)} {str(ev.get('multiplicity') or '')}",
                linewidth_ppm=lw, nh=float(rec['nh']), owner=True))
        seen = set(members)
        for c in candidates or ():
            m = str(c.met_id)
            if m in seen or m not in winners or id(c) == own_signal.get(m):
                continue
            ev, found = _evidence(c)
            if not accept(ev):
                continue
            lines = _lines_of(ev)
            if not lines or not any(lo <= v <= hi for v in lines):
                continue
            if not found or m in excluded:
                skipped.setdefault(tuple(members), []).append(
                    f"{names.get(m, m)} {str(ev.get('multiplicity') or '')}"
                    + (' (left out by the analyst)' if m in excluded else ''))
                seen.add(m)
                continue
            hint = None
            pp = per_proton_areas.get(m) or {}
            vals = [v for v in pp.values() if v is not None and np.isfinite(v) and v > 0]
            if vals:
                hint = float(np.median(vals))
            nh = ev.get('protons')
            try:
                nh = float(nh) if nh else None
            except (TypeError, ValueError):
                nh = None
            if not nh:
                continue
            comps.append(component_from_evidence(
                m, ev, label=f"{names.get(m, m)} {str(ev.get('multiplicity') or '')}",
                linewidth_ppm=lw, nh=nh, owner=False, per_proton_hint=hint))
            seen.add(m)
        comps = [c for c in comps if c is not None]
        left_out = skipped.get(tuple(members))
        if len(comps) < 2:
            if left_out:
                for m in members:
                    winners[m]['unshared_reason'] = (
                        f"{', '.join(left_out)} is listed inside it but "
                        f"took no share: not placed by detection, or left "
                        f"out by the analyst")
            continue
        out = resolve_window(comps, ppm, intensities, included=included,
                             noise_sigma=noise_sigma)
        if out is None:
            continue
        if out.get('refused'):
            for m in members:
                winners[m]['unshared_reason'] = out['refused']
            continue
        out['skipped'] = list(left_out or ()) + list(out['inseparable'])
        for m in members:
            rec = winners[m]
            rec['areas'] = dict(out['areas'][m])
            rec['area_median'] = (float(np.median(list(rec['areas'].values())))
                                  if rec['areas'] else float('nan'))
            rec['captured'] = None
            rec['shared_with'] = [c.met_id for c in comps if c.met_id != m]
            rec['deconv'] = {'window': out['window'],
                             'remainder': out['remainder'],
                             'components': out['components'],
                             'amplitudes': out['amplitudes'],
                             'skipped': out['skipped']}
            per_proton_areas[m] = {s_i: a / rec['nh']
                                   for s_i, a in rec['areas'].items()}
            results[m] = out
    return results


def refit_record(deconv, ppm, intensities, *, exclude=(), included=None,
                 noise_sigma=None):
    """The fit a recorded resolution would give without the analytes in
    ``exclude`` -- what the review window shows before the analyst applies
    an edit and the quantification is run again. Returns what
    ``resolve_window`` returns, with the owners' windows as the record
    had them: None when fewer than two components are left."""
    exclude = {str(m) for m in exclude}
    comps = [component_from_record(rec)
             for rec in (deconv or {}).get('components', ())
             if str(rec.get('met_id')) not in exclude]
    comps = [c for c in comps if c is not None]
    if len(comps) < 2:
        return None
    return resolve_window(comps, ppm, intensities, included=included,
                          noise_sigma=noise_sigma)


def component_curves(deconv, ppm, sample=None) -> list:
    """Each recorded component's curve over the record's window at the
    amplitude the fit gave it -- for ``sample`` (an index), or at the median
    amplitude over samples when None. ``[(record, x, y), ...]``, x ascending
    in ppm."""
    if not deconv:
        return []
    p = np.asarray(ppm, dtype=float)
    lo, hi = deconv['window']
    sel = (p >= lo) & (p <= hi)
    x = p[sel]
    order = np.argsort(x)
    x = x[order]
    out = []
    for rec in deconv.get('components', ()):
        c = component_from_record(rec)
        if c is None:
            continue
        amps = (deconv.get('amplitudes') or {}).get(rec['met_id']) or {}
        if sample is not None:
            a = amps.get(int(sample), amps.get(str(sample)))
        else:
            a = float(np.median(list(amps.values()))) if amps else None
        if a is None:
            continue
        m = np.asarray(c.model(x), dtype=float)
        top = float(m.max()) if m.size else 0.0
        m = m / top if top > 0 else m
        out.append((rec, x, float(a) * m))
    return out
