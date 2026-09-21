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

An intruder the owner's shape can make (``SEPARABLE_MIN``) cannot be
fitted: the least squares hands either an arbitrary share. Betaine's
singlet on the mixture bench sits half a hertz from taurine's central
line. Where such an intruder has a clean measurement of its own, its share
here is not fitted but FIXED at what that measurement predicts, and
subtracted before the owner is fitted: the number is the clean signal's,
the shape is the reference's, and nothing is decided by a fit that could
not decide. Without a clean measurement it stays out, as before.

A first-order signal is modelled by the profile the reference database
measured for it (BMRB_Metabolomics_v03 ``pattern_profile``) whenever it
has one -- a triplet is what the pure compound showed, second-order
leaning included -- and by Lorentzians on its found lines only without a
profile. Before the amplitudes are read, each component's position (within
half a linewidth) and the study's width (within a factor) are refined on
the mean of the measured samples, so an apex the neighbour pulled does not
misplace the shape; the geometry is then shared by every sample. Measured
on the mixture bench: taurine under betaine read -21% with Lorentzians at
the detected apices and no refinement.

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

# The geometry refinement: a component may slide this far, in linewidths,
# from where annotation placed it (an apex pulled by a neighbour moves a
# fraction of a width, never a whole one), and the study's width may be
# scaled within this range. Tried on a grid, coordinate-wise, on the mean
# trace; a shift never crosses into another line.
SHIFT_MAX_LINEWIDTHS = 0.5
SHIFT_STEPS = 9
WIDTH_SCALES = (0.8, 0.9, 1.0, 1.15, 1.3)
REFINE_ROUNDS = 2


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
    # the same, sample by sample: what a fixed intruder takes in each
    per_proton_by_sample: dict = field(default_factory=dict)
    ceiling: float | None = None     # on the amplitude, set by the fit
    shape_area: float = 0.0          # of the model on the window, ppm units
    # what of this shape the owners' shapes cannot make: 1 - R^2 of its
    # column on theirs. Zero is a shape the fit cannot tell from the owner.
    separability: float | None = None
    # an inseparable intruder whose amplitude was fixed at what its own
    # clean signal predicts, and subtracted, rather than fitted
    fixed: bool = False
    lines: list = field(default_factory=list)
    # what the model was built from, so a view can build it again
    evidence: dict = field(default_factory=dict)
    linewidth_ppm: float = 0.0
    # the refinement's answer: how far the shape slid from where annotation
    # placed it, and the width it was drawn at, as a factor of the study's
    shift_ppm: float = 0.0
    width_scale: float = 1.0

    def curve(self, x):
        """The shape at its refined geometry, tallest point 1."""
        m = np.asarray(self.model(x, self.shift_ppm, self.width_scale),
                       dtype=float)
        top = float(m.max()) if m.size else 0.0
        return m / top if top > 0 else m


def _profile_apices_hz(prof) -> list:
    """Where the reference's profile peaks, in hertz from its origin."""
    offs, h = prof
    offs = np.asarray(offs, dtype=float)
    h = np.asarray(h, dtype=float)
    if h.size < 3:
        return []
    top = float(h.max())
    if top <= 0:
        return []
    idx = [i for i in range(1, h.size - 1)
           if h[i] >= h[i - 1] and h[i] > h[i + 1] and h[i] >= 0.2 * top]
    return [float(offs[i]) for i in idx]


def component_from_evidence(met_id, ev, *, label, linewidth_ppm, nh,
                            owner=False, per_proton_hint=None,
                            shift_ppm=0.0, width_scale=1.0,
                            per_proton_by_sample=None):
    """A component from a signal's evidence, as the adapter hands it to the
    engine: ``predicted_ppms`` (the lines annotation found, in ppm) with
    ``line_weights`` where the pattern gave them, a ``shape_fit`` for a
    measured shape (carrying its own field), ``peak_ppm`` for a singlet.
    A first-order signal or singlet with a measured ``pattern_profile``
    (and the field to place it, ``pattern_field_mhz``) is modelled by that
    profile, put where its found lines say it is. None when nothing places
    it. The model takes ``(x, shift_ppm, width_scale)``: the refinement's
    two parameters, applied over the annotation's geometry."""
    ev = ev or {}
    lw = float(linewidth_ppm)
    fit = ev.get('shape_fit') or {}
    if (fit.get('origin_ppm') is not None and fit.get('field_mhz')
            and (fit.get('pattern_profile') or fit.get('pattern_lines'))):
        f = float(fit['field_mhz'])
        origin = float(fit['origin_ppm'])
        prof = parse_profile(fit.get('pattern_profile'))
        if prof is not None:
            ref_w = float(fit.get('profile_fwhm_hz') or 0.0)
            offs = np.asarray(prof[0], dtype=float)

            def model(x, shift=0.0, scale=1.0, _p=prof, _o=origin, _w=ref_w):
                extra = max(0.0, lw * f * float(scale) - _w)
                return profile_model(x, _p, _o + shift, extra, f)
            lo = origin + (offs.min() - PAD_LINEWIDTHS * lw * f) / f
            hi = origin + (offs.max() + PAD_LINEWIDTHS * lw * f) / f
            lines = [origin + o / f for o in (fit.get('apex_ppms') or ())]
        else:
            pl = [(float(o), float(r)) for o, r in fit['pattern_lines']]

            def model(x, shift=0.0, scale=1.0, _l=pl, _o=origin):
                return lines_model(x, _l, _o + shift, lw * f * float(scale), f)
            offs = [o for o, _ in pl]
            lo = origin + (min(offs) - PAD_LINEWIDTHS * lw * f) / f
            hi = origin + (max(offs) + PAD_LINEWIDTHS * lw * f) / f
            lines = [origin + o / f for o in offs]
        return Component(met_id, label, model, float(lo), float(hi),
                         float(nh), owner=owner, lines=lines,
                         per_proton_hint=per_proton_hint,
                         per_proton_by_sample=dict(per_proton_by_sample or {}),
                         evidence=_shape_evidence(ev), linewidth_ppm=lw,
                         shift_ppm=float(shift_ppm),
                         width_scale=float(width_scale))
    ppms = [float(p) for p in (ev.get('predicted_ppms') or ()) if p is not None]
    if not ppms and ev.get('peak_ppm') is not None:
        ppms = [float(ev['peak_ppm'])]
    if not ppms:
        return None
    centre = float(np.mean(ppms))
    prof = parse_profile(ev.get('pattern_profile'))
    f_ref = ev.get('pattern_field_mhz')
    if prof is not None and f_ref:
        # The reference's own shape, placed so its apices sit where the
        # found lines are: the drift is what the lines say, the shape is
        # the pure compound's.
        f = float(f_ref)
        apices = _profile_apices_hz(prof)
        centre_hz = float(np.mean(apices)) if apices else 0.0
        origin = centre - centre_hz / f
        ref_w = float(ev.get('profile_fwhm_hz') or 0.0)
        offs = np.asarray(prof[0], dtype=float)

        def model(x, shift=0.0, scale=1.0, _p=prof, _o=origin, _w=ref_w):
            extra = max(0.0, lw * f * float(scale) - _w)
            return profile_model(x, _p, _o + shift, extra, f)
        # the window is the lines' reach, as for Lorentzians: the profile's
        # own range runs tens of hertz into the neighbours
        ap = [origin + a / f for a in apices] or ppms
        lo = min(min(ppms), min(ap)) - PAD_LINEWIDTHS * lw
        hi = max(max(ppms), max(ap)) + PAD_LINEWIDTHS * lw
        return Component(met_id, label, model, float(lo), float(hi),
                         float(nh), owner=owner, lines=list(ppms),
                         per_proton_hint=per_proton_hint,
                         per_proton_by_sample=dict(per_proton_by_sample or {}),
                         evidence=_shape_evidence(ev), linewidth_ppm=lw,
                         shift_ppm=float(shift_ppm),
                         width_scale=float(width_scale))
    weights = (list(ev.get('line_weights') or ())
               if ev.get('line_weights') and len(ev['line_weights']) == len(ppms)
               else pascal_weights(len(ppms)))
    # offsets and width in ppm, with a "field" of 1: lines_model divides
    # its hertz by the field, and a field of one leaves ppm as ppm
    pl = [((p - centre), float(w)) for p, w in zip(ppms, weights)]

    def model(x, shift=0.0, scale=1.0, _l=pl, _c=centre):
        return lines_model(x, _l, _c + shift, lw * float(scale), 1.0)
    lo = min(ppms) - PAD_LINEWIDTHS * lw
    hi = max(ppms) + PAD_LINEWIDTHS * lw
    return Component(met_id, label, model, float(lo), float(hi), float(nh),
                     owner=owner, lines=list(ppms),
                     per_proton_hint=per_proton_hint,
                     per_proton_by_sample=dict(per_proton_by_sample or {}),
                     evidence=_shape_evidence(ev), linewidth_ppm=lw,
                     shift_ppm=float(shift_ppm), width_scale=float(width_scale))


def _shape_evidence(ev) -> dict:
    """The part of a signal's evidence its component is built from."""
    return {k: ev[k] for k in ('multiplicity', 'shape_fit', 'predicted_ppms',
                               'line_weights', 'peak_ppm', 'pattern_profile',
                               'profile_fwhm_hz', 'pattern_field_mhz',
                               'placed_at_expected')
            if ev.get(k)}


def refine_geometry(comps, x, trace, *, lb=None, ub=None):
    """Slide each component and scale the width, on one trace, to where
    the non-negative fit leaves the least residual.

    Coordinate descent on a grid: for each component in turn, every shift
    within SHIFT_MAX_LINEWIDTHS of its linewidth (SHIFT_STEPS of them),
    then every width scale in WIDTH_SCALES for all components at once;
    REFINE_ROUNDS times. The geometry found is written on the components
    (``shift_ppm``, ``width_scale``) and shared by every sample's fit.
    Returns the final residual sum of squares.
    """
    comps = list(comps)
    n = len(comps)
    if n == 0:
        return float('inf')
    t = np.asarray(trace, dtype=float)
    lb = np.zeros(n) if lb is None else np.asarray(lb, dtype=float)
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float)

    def _rss():
        A = np.column_stack([c.curve(x) for c in comps])
        res = lsq_linear(A, t, bounds=(lb, ub), lsmr_tol='auto')
        r = t - A @ res.x
        return float(np.dot(r, r))

    best = _rss()
    for _round in range(REFINE_ROUNDS):
        for c in comps:
            reach = SHIFT_MAX_LINEWIDTHS * float(c.linewidth_ppm)
            if reach <= 0:
                continue
            keep = c.shift_ppm
            for sh in np.linspace(-reach, reach, SHIFT_STEPS):
                c.shift_ppm = float(sh)
                v = _rss()
                if v < best:
                    best, keep = v, float(sh)
            c.shift_ppm = keep
        keep_scale = [c.width_scale for c in comps]
        for sc in WIDTH_SCALES:
            for c in comps:
                c.width_scale = float(sc)
            v = _rss()
            if v < best:
                best, keep_scale = v, [float(sc)] * n
        for c, sc in zip(comps, keep_scale):
            c.width_scale = sc
    return best


def component_from_record(rec, *, owner=None):
    """The component a ``resolve_window`` result recorded, built again --
    for a view that draws the fit or edits it."""
    return component_from_evidence(
        rec['met_id'], rec.get('evidence') or {}, label=rec.get('label', ''),
        linewidth_ppm=rec['linewidth_ppm'], nh=rec.get('nh') or 1.0,
        owner=rec.get('owner') if owner is None else owner,
        per_proton_hint=rec.get('per_proton_hint'),
        per_proton_by_sample={int(k): float(v) for k, v in
                              (rec.get('per_proton_by_sample') or {}).items()},
        shift_ppm=float(rec.get('shift_ppm') or 0.0),
        width_scale=float(rec.get('width_scale') or 1.0))


def resolve_window(components, ppm, intensities, *, included=None,
                   noise_sigma=None, refine=True) -> dict | None:
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
    n_s = arr.shape[1]
    idx = [i for i in (range(n_s) if included is None else included)
           if 0 <= i < n_s]
    if not idx:
        return None
    # The geometry is settled once, on the mean of the measured samples,
    # and shared: twelve traces refined one by one are twelve chances for
    # noise to move a shape, and a resonance does not drift within a study
    # by a fraction of a linewidth from sample to sample.
    if refine:
        refine_geometry(comps, x, arr[sel][:, idx].mean(axis=1))
    cols = []
    for c in comps:
        m = c.curve(x)
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
    # An inseparable intruder with a clean measurement of its own takes
    # exactly what that measurement predicts; one without stays out.
    fixed = [c for c in inseparable
             if amplitude_ceiling(c.per_proton_hint, c.nh, c.shape_area,
                                  factor=1.0) is not None]
    for c in fixed:
        c.fixed = True
        c.ceiling = amplitude_ceiling(c.per_proton_hint, c.nh,
                                      c.shape_area, factor=1.0)
    left_out = [c for c in inseparable if c not in fixed]
    free_idx = [i for i, c in enumerate(comps) if c not in inseparable]
    fixed_idx = [i for i, c in enumerate(comps) if c in fixed]
    free = [comps[i] for i in free_idx]
    if len(free) < 2 and not fixed:
        return {'refused': f"{', '.join(c.label for c in inseparable)} "
                           f"cannot be told by shape from this signal"}
    if not free:
        return {'refused': 'nothing here can be fitted: every shape is '
                           'fixed by its own clean signal'}
    A_free = A[:, free_idx]
    A_fixed = A[:, fixed_idx] if fixed_idx else None

    def _fixed_amplitudes(s):
        # what the intruder's own clean signal read in THIS sample; the
        # median over samples only where that sample gave nothing
        out = []
        for i in fixed_idx:
            c = comps[i]
            v = amplitude_ceiling(
                c.per_proton_by_sample.get(int(s), c.per_proton_hint),
                c.nh, c.shape_area, factor=1.0)
            out.append(c.ceiling if v is None else v)
        return np.array(out, dtype=float)
    for c in free:
        c.ceiling = amplitude_ceiling(c.per_proton_hint, c.nh, c.shape_area)
    lb = np.zeros(len(free))
    ub = np.array([c.ceiling if c.ceiling is not None else np.inf
                   for c in free])
    comps = free + fixed
    areas = {c.met_id: {} for c in comps}
    amps = {c.met_id: {} for c in comps}
    remainder = {}
    for s in idx:
        t = arr[sel, s]
        a_fixed = _fixed_amplitudes(s)
        taken = A_fixed @ a_fixed if A_fixed is not None else 0.0
        res = lsq_linear(A_free, t - taken, bounds=(lb, ub), lsmr_tol='auto')
        a = np.asarray(res.x, dtype=float)
        fitted = A_free @ a + taken
        total = float(np.trapezoid(np.clip(t[order], 0, None), x=x[order]))
        left = float(np.trapezoid(np.clip((t - fitted)[order], 0, None),
                                  x=x[order]))
        remainder[int(s)] = float(left / total) if total > 0 else 0.0
        for c, ai in zip(free, a):
            amps[c.met_id][int(s)] = float(ai)
            areas[c.met_id][int(s)] = float(ai) * c.shape_area
        for c, ai in zip(fixed, a_fixed):
            amps[c.met_id][int(s)] = float(ai)
            areas[c.met_id][int(s)] = float(ai) * c.shape_area
    return {'window': (float(lo), float(hi)), 'areas': areas,
            'amplitudes': amps, 'remainder': remainder,
            'inseparable': [c.label for c in left_out],
            'fixed': [c.label for c in fixed],
            'components': [{'met_id': c.met_id, 'label': c.label,
                            'owner': c.owner, 'nh': c.nh,
                            'lines': list(c.lines),
                            'ceiling': c.ceiling,
                            'separability': c.separability,
                            'shape_area': c.shape_area,
                            'per_proton_hint': c.per_proton_hint,
                            'evidence': c.evidence,
                            'linewidth_ppm': c.linewidth_ppm,
                            'shift_ppm': c.shift_ppm,
                            'width_scale': c.width_scale,
                            'fixed': c.fixed,
                            'per_proton_by_sample': dict(c.per_proton_by_sample),
                            'placed_at_expected': bool(
                                c.evidence.get('placed_at_expected'))}
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
    owner's (``SEPARABLE_MIN``). One exception, measured: an intruder whose
    every detected line IS one of the owner's lines was not placed by
    detection at all -- the two merged and detection reported one apex,
    the owner's. Betaine's singlet under taurine's triplet on the mixture
    bench was anchored on taurine's central line, 4 Hz from where it sat;
    fitted there it was collinear with that line and took an arbitrary
    share of it. Such an intruder is placed where annotation expects it
    (``expected_ppm``: the database position at the study's drift) and the
    refinement slides it from there; the record says so
    (``placed_at_expected``). What was left out is recorded in ``skipped``
    on the result and said on the row.

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

    lw = float(linewidth_ppm)

    def _evidence(c, owner_lines=()):
        ev = dict(c.evidence or {})
        found = where(c)
        if found and not ev.get('shape_fit'):
            ev['predicted_ppms'] = list(found)
            if ev.get('line_weights') and len(ev['line_weights']) != len(found):
                ev.pop('line_weights')
            stolen = owner_lines and all(
                any(abs(v - o) <= lw / 2.0 for o in owner_lines) for v in found)
            exp = ev.get('expected_ppm')
            if stolen and exp is not None:
                move = float(exp) - float(np.mean(found))
                ev['predicted_ppms'] = [float(v) + move for v in found]
                ev['placed_at_expected'] = True
        return ev, bool(found)
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
        owner_lines = [v for oc in comps if oc is not None for v in oc.lines]
        for c in candidates or ():
            m = str(c.met_id)
            if m in seen or m not in winners or id(c) == own_signal.get(m):
                continue
            ev, found = _evidence(c, owner_lines)
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
                linewidth_ppm=lw, nh=nh, owner=False, per_proton_hint=hint,
                per_proton_by_sample={int(k): float(v) for k, v in pp.items()
                                      if v is not None and np.isfinite(v)
                                      and v > 0}))
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
                             'skipped': out['skipped'],
                             'fixed': list(out.get('fixed') or ())}
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
                          noise_sigma=noise_sigma, refine=False)


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
        out.append((rec, x, float(a) * c.curve(x)))
    return out
