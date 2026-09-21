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

# The remainder as evidence (resolve_residuals). A window whose accepted
# shapes leave REMAINDER_NOTE or more is offered the signals of analytes the
# election did NOT accept whose lines fall in it, one at a time. A signal
# is offered only where the spectrum shows an apex of its own that no
# accepted analyte holds: its amplitude is read from THAT apex, sample by
# sample, and pinned -- the window then only tests whether the shape so
# scaled accounts for what was left. Taken when, with it, the remainder
# falls under RESIDUAL_ACCEPT (or the noise floor), when it accounts for at
# least RESIDUAL_EXPLAINED of what stood above the floor, and when the
# amount implied is one the rest of its fingerprint can carry. Measured
# without the apex rule: a seven-line pattern was taken into a singlet's
# window and two shapes into windows of analytes that were themselves
# wrong -- a free shape fits any remainder, which is what made a shape
# fitted alone worth 58 false positives.
RESIDUAL_ACCEPT = 0.05
RESIDUAL_EXPLAINED = 0.5
# The remainder of a window never reaches zero: the positive half of the
# noise integrates to sigma/sqrt(2 pi) per point whatever the shapes do.
# The conditions are read against that floor (``remainder_floor`` on the
# result): a window on a signal of SNR 26 has a floor near 6%, and an
# absolute 5% was asking the impossible of it. Accepted below
# RESIDUAL_FLOOR_MULTIPLE floors, or below RESIDUAL_ACCEPT, whichever is
# looser; explained is the fall over what stood above the floor.
RESIDUAL_FLOOR_MULTIPLE = 1.5
# A pinned shape -- its amount read off its own free apex -- is taken when
# it accounts for at least this share of what stood above the floor, and
# refused when, at that amount, it overshoots the trace in the owner's
# window by more than the noise allows (RESIDUAL_FLOOR_MULTIPLE floors):
# an apex whose shape does not fit under the spectrum is not this
# analyte's. Several may be taken into one window, each subtracted before
# the next is tried; a window with two intruders' tails in it is the case
# that made "explains half of the remainder" the wrong question.
RESIDUAL_MIN_SHARE = 0.2
# An owner left with under this share of its window's area, once the
# signals other quantified analytes list inside it are subtracted, had no
# window: what was measured was theirs.
OWNER_MIN_SHARE = 0.05
# On a free apex with no owner (resolve_free_apices) the shape has the
# whole region to account for: it must explain at least this share of the
# area over its own lines' reach. A shape that leaves a quarter of its own
# region is sharing the apex with somebody, and an apex shared is not an
# apex of its own. Set with the mixture bench in view: glutamate's pattern
# explained 0.77-0.79 of its region, the wrong claimants on other free
# apices (a sugar's doublet on myo-inositol's triplet, GABA's triplet on
# glutamate's) 0.60-0.68.
STANDALONE_MIN_EXPLAINED = 0.75

# The fourth condition: the amount the remainder implies must be consistent
# with the rest of the analyte's fingerprint (ADR-012, "a measurable
# absence contradicts"). Every other listed signal of a taken analyte, at
# the implied per-proton amount, must find a region holding at least a
# RESIDUAL_FINGERPRINT_RATIO-th of the height it predicts; a signal that
# would stand under FINGERPRINT_MIN_SNR is not asked. Measured on the
# mixture bench without it: a seven-line pattern was taken into pyruvate's
# singlet window and adenosine, at 1.7 mM, into a window at 3.88 while
# nothing stood at its aromatic lines.
RESIDUAL_FINGERPRINT_RATIO = 2.0
FINGERPRINT_MIN_SNR = 20.0
# A signal predicted under FINGERPRINT_MIN_SNR is not asked to contradict;
# one predicted over this is still asked whether it is there, as support.
# Glutamate's other patterns, at the amount its 2.34 ppm pattern implied
# on the mixture bench, were predicted at 8-9 sigma: too faint to refute,
# tall enough to look for.
FINGERPRINT_SUPPORT_SNR = 5.0

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
    # fixed whatever its separability: the amplitude was read off the
    # spectrum (a free apex of its own) and is not the fit's to decide.
    # ``pin_apex_ppm`` is that apex and ``pin_heights`` the trace there in
    # every sample; the amplitude is what puts the shape's curve, at its
    # final geometry, through that height.
    pinned: bool = False
    pin_apex_ppm: float | None = None
    pin_heights: dict = field(default_factory=dict)
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
                            per_proton_by_sample=None, pinned=False):
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
                         width_scale=float(width_scale), pinned=bool(pinned))
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
                         width_scale=float(width_scale), pinned=bool(pinned))
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
                     shift_ppm=float(shift_ppm), width_scale=float(width_scale),
                     pinned=bool(pinned))


def _shape_evidence(ev) -> dict:
    """The part of a signal's evidence its component is built from."""
    return {k: ev[k] for k in ('multiplicity', 'shape_fit', 'predicted_ppms',
                               'line_weights', 'peak_ppm', 'pattern_profile',
                               'profile_fwhm_hz', 'pattern_field_mhz',
                               'placed_at_expected')
            if ev.get(k)}


def refine_geometry(comps, x, trace, *, lb=None, ub=None, movable=None):
    """Slide each component and scale the width, on one trace, to where
    the non-negative fit leaves the least residual.

    Coordinate descent on a grid: for each component in turn, every shift
    within SHIFT_MAX_LINEWIDTHS of its linewidth (SHIFT_STEPS of them),
    then every width scale in WIDTH_SCALES for all components at once;
    REFINE_ROUNDS times. The geometry found is written on the components
    (``shift_ppm``, ``width_scale``) and shared by every sample's fit.
    ``movable`` restricts which components may move (the others keep the
    geometry they have; the width scale then applies to the movable ones
    only). Returns the final residual sum of squares.
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

    move = list(comps) if movable is None else [c for c in comps if c in movable]
    best = _rss()
    for _round in range(REFINE_ROUNDS):
        for c in move:
            reach = SHIFT_MAX_LINEWIDTHS * float(c.linewidth_ppm)
            # a shape pinned on an apex stays on it: slid off, its curve
            # at the pin falls and the amplitude that passes through the
            # pin's height rises, and the shape overshoots where its own
            # apex went
            if reach <= 0 or c.pin_apex_ppm is not None:
                continue
            keep = c.shift_ppm
            for sh in np.linspace(-reach, reach, SHIFT_STEPS):
                c.shift_ppm = float(sh)
                v = _rss()
                if v < best:
                    best, keep = v, float(sh)
            c.shift_ppm = keep
        keep_scale = [c.width_scale for c in move]
        for sc in WIDTH_SCALES:
            for c in move:
                c.width_scale = float(sc)
            v = _rss()
            if v < best:
                best, keep_scale = v, [float(sc)] * len(move)
        for c, sc in zip(move, keep_scale):
            c.width_scale = sc
    return best


def component_from_record(rec, *, owner=None):
    """The component a ``resolve_window`` result recorded, built again --
    for a view that draws the fit or edits it."""
    c = component_from_evidence(
        rec['met_id'], rec.get('evidence') or {}, label=rec.get('label', ''),
        linewidth_ppm=rec['linewidth_ppm'], nh=rec.get('nh') or 1.0,
        owner=rec.get('owner') if owner is None else owner,
        per_proton_hint=rec.get('per_proton_hint'),
        per_proton_by_sample={int(k): float(v) for k, v in
                              (rec.get('per_proton_by_sample') or {}).items()},
        shift_ppm=float(rec.get('shift_ppm') or 0.0),
        width_scale=float(rec.get('width_scale') or 1.0),
        pinned=bool(rec.get('pinned')))
    if c is not None and rec.get('pin_apex_ppm') is not None:
        c.pin_apex_ppm = float(rec['pin_apex_ppm'])
        c.pin_heights = {int(k): float(v) for k, v in
                         (rec.get('pin_heights') or {}).items()}
    return c


def resolve_window(components, ppm, intensities, *, included=None,
                   noise_sigma=None, refine=True,
                   allow_single=False, movable=None,
                   remainder_window=None) -> dict | None:
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
    if len(comps) < (1 if allow_single else 2):
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
        refine_geometry(comps, x, arr[sel][:, idx].mean(axis=1),
                        movable=movable)
    # the remainder is read over the owner's window when one is named:
    # a component that reaches past it must not bring new ground into
    # the count
    if remainder_window is not None:
        r_lo, r_hi = min(remainder_window), max(remainder_window)
        rsel = (x >= r_lo) & (x <= r_hi)
        if rsel.sum() < 4:
            rsel = np.ones(x.shape, dtype=bool)
    else:
        rsel = np.ones(x.shape, dtype=bool)
    r_order = np.argsort(x[rsel])
    # the remainder pure noise would leave over this range, per unit of the
    # window's area: read below, once the total is known
    _noise_left = (float(noise_sigma) / np.sqrt(2.0 * np.pi)
                   * float(abs(x[rsel].max() - x[rsel].min()))
                   if noise_sigma and rsel.sum() > 1 else 0.0)
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
    fixed = [c for c in comps
             if (c.pinned and c.pin_apex_ppm is not None and c.pin_heights)
             or ((c.pinned or c in inseparable)
                 and amplitude_ceiling(c.per_proton_hint, c.nh, c.shape_area,
                                       factor=1.0) is not None)]
    for c in fixed:
        c.fixed = True
        c.ceiling = amplitude_ceiling(c.per_proton_hint, c.nh,
                                      c.shape_area, factor=1.0)
    left_out = [c for c in inseparable if c not in fixed]
    free_idx = [i for i, c in enumerate(comps)
                if c not in inseparable and c not in fixed]
    fixed_idx = [i for i, c in enumerate(comps) if c in fixed]
    free = [comps[i] for i in free_idx]
    if len(free) < 2 and not fixed and not allow_single:
        return {'refused': f"{', '.join(c.label for c in inseparable)} "
                           f"cannot be told by shape from this signal"}
    if not free and not allow_single:
        return {'refused': 'nothing here can be fitted: every shape is '
                           'fixed by its own clean signal'}
    A_free = A[:, free_idx]
    A_fixed = A[:, fixed_idx] if fixed_idx else None

    def _fixed_amplitudes(s):
        # what the intruder's own clean signal read in THIS sample; the
        # median over samples only where that sample gave nothing -- or,
        # for a shape pinned on its own apex, what puts its curve (at the
        # geometry the refinement settled) through the trace at that apex
        out = []
        for i in fixed_idx:
            c = comps[i]
            if c.pin_apex_ppm is not None and c.pin_heights:
                at = float(np.interp(c.pin_apex_ppm, x[order],
                                     A[:, i][order]))
                h = c.pin_heights.get(int(s))
                out.append(max(float(h), 0.0) / at
                           if at > 0.05 and h is not None else 0.0)
                continue
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
    floors = {}
    overshoot = {}
    for s in idx:
        t = arr[sel, s]
        a_fixed = _fixed_amplitudes(s)
        taken = A_fixed @ a_fixed if A_fixed is not None else 0.0
        if free:
            res = lsq_linear(A_free, t - taken, bounds=(lb, ub),
                             lsmr_tol='auto')
            a = np.asarray(res.x, dtype=float)
            fitted = A_free @ a + taken
        else:
            a = np.zeros(0)
            fitted = taken
        _t, _r, _x = t[rsel], (t - fitted)[rsel], x[rsel]
        total = float(np.trapezoid(np.clip(_t[r_order], 0, None),
                                   x=_x[r_order]))
        left = float(np.trapezoid(np.clip(_r[r_order], 0, None),
                                  x=_x[r_order]))
        remainder[int(s)] = float(left / total) if total > 0 else 0.0
        floors[int(s)] = float(_noise_left / total) if total > 0 else 0.0
        over = float(np.trapezoid(np.clip(-_r[r_order], 0, None),
                                  x=_x[r_order]))
        overshoot[int(s)] = float(over / total) if total > 0 else 0.0
        for c, ai in zip(free, a):
            amps[c.met_id][int(s)] = float(ai)
            areas[c.met_id][int(s)] = float(ai) * c.shape_area
        for c, ai in zip(fixed, a_fixed):
            amps[c.met_id][int(s)] = float(ai)
            areas[c.met_id][int(s)] = float(ai) * c.shape_area
    return {'window': (float(lo), float(hi)), 'areas': areas,
            'amplitudes': amps, 'remainder': remainder,
            'remainder_floor': floors,
            'overshoot': overshoot,
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
                            'pinned': c.pinned,
                            'pin_apex_ppm': c.pin_apex_ppm,
                            'pin_heights': dict(c.pin_heights),
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


def _placed_evidence(c, linewidth_ppm, owner_lines, lines_of):
    """A pool candidate's evidence with its lines placed: detection's own
    lines when they are its own, otherwise the annotation's line geometry
    moved to where the study's drift puts the signal (``expected_ppm``)."""
    ev = dict(c.evidence or {})
    found = list(lines_of(c) or ())
    lw = float(linewidth_ppm)
    stolen = found and owner_lines and all(
        any(abs(v - o) <= lw / 2.0 for o in owner_lines) for v in found)
    exp = ev.get('expected_ppm')
    if found and not stolen:
        ev['predicted_ppms'] = found
        if ev.get('line_weights') and len(ev['line_weights']) != len(found):
            ev.pop('line_weights')
        return ev
    geom = [float(v) for v in (ev.get('predicted_ppms') or ()) if v is not None]
    if not geom and ev.get('peak_ppm') is not None:
        geom = [float(ev['peak_ppm'])]
    if not geom:
        return None
    if exp is None:
        return None
    move = float(exp) - float(np.mean(geom))
    ev['predicted_ppms'] = [v + move for v in geom]
    ev['placed_at_expected'] = True
    return ev


def _lorentz_height(area_ppm, hwhm_ppm) -> float:
    """The apex height of a Lorentzian line of this area and half-width."""
    return float(area_ppm) / (np.pi * float(hwhm_ppm)) if hwhm_ppm > 0 else 0.0


def fingerprint_holds(db_peaks, per_proton_area, ppm, trace, *, noise,
                      linewidth_ppm, drift_ppm=0.0, tolerance_ppm=0.02,
                      skip_ppm=None, ratio=RESIDUAL_FINGERPRINT_RATIO,
                      min_snr=FINGERPRINT_MIN_SNR,
                      excluded_regions=()) -> tuple[bool, list]:
    """Whether the rest of an analyte's fingerprint can hold the amount a
    window implies.

    ``db_peaks`` are the reference's signals (``ppm_center``, ``type``,
    ``protons``); ``per_proton_area`` the area per proton the window gave
    (intensity x ppm); ``trace`` the mean spectrum the areas were read on.
    For every listed signal but the one at ``skip_ppm`` (the window's
    own) and the unresolved ('m') ones, the tallest line's height is
    predicted -- protons times the line's share of a first-order
    multiplet, as one Lorentzian of the study's width -- and the region
    around the drifted position must hold, peak to trough, at least a
    ``ratio``-th of it. Returns ``(holds, verdicts)``.
    """
    from processing.annotation_detectability import _line_share, proton_count
    from processing.annotation_scoring import pattern_offsets_hz
    x = np.asarray(ppm, dtype=float)
    y = np.asarray(trace, dtype=float)
    hwhm = float(linewidth_ppm) / 2.0
    out, holds = [], True
    for dp in db_peaks or ():
        mult = str(dp.get('type') or dp.get('multiplicity') or '').lower().strip()
        # an unresolved 'm' the reference measured as a pattern is checkable
        # by that pattern; one it did not is not
        pat = pattern_offsets_hz(dp) if mult in ('m', 'm*', 'multiplet') else []
        if (mult in ('peak', '') or (mult in ('m', 'm*', 'multiplet')
                                      and not pat)):
            continue
        centre = float(dp.get('ppm_center', dp.get('ppm', 0.0)))
        if skip_ppm is not None and abs(centre - float(skip_ppm)) < 0.005:
            continue
        nh = proton_count(dp)
        # a measured pattern's tallest line carries its share of the
        # pattern's height, as the reference measured it
        share = (max(r for _o, r in pat) / sum(r for _o, r in pat)
                 if pat else _line_share(mult))
        expected = _lorentz_height(float(per_proton_area) * nh * share, hwhm)
        exp_snr = expected / float(noise) if noise else float('inf')
        if exp_snr < FINGERPRINT_SUPPORT_SNR:
            out.append((centre, mult, exp_snr, 'faint'))
            continue
        here = centre + float(drift_ppm)
        if excluded_regions:
            from processing.spectral_regions import in_excluded_region
            if (in_excluded_region(here - tolerance_ppm, excluded_regions)
                    or in_excluded_region(here + tolerance_ppm,
                                          excluded_regions)):
                out.append((centre, mult, exp_snr, 'excluded'))
                continue
        m = (x >= here - tolerance_ppm) & (x <= here + tolerance_ppm)
        if m.sum() < 3:
            out.append((centre, mult, exp_snr, 'off the axis'))
            continue
        seg = y[m]
        measured = (float(seg.max()) - float(seg.min())) / float(noise) if noise else float('inf')
        if measured * float(ratio) < exp_snr:
            if exp_snr >= min_snr:
                holds = False
                out.append((centre, mult, exp_snr,
                            f'contradicted: region holds {measured:.0f}'))
            else:
                out.append((centre, mult, exp_snr, 'faint'))
        else:
            out.append((centre, mult, exp_snr, 'present'))
    return holds, out


def fingerprint_counts(db_peaks, *, own_ppm, own_mult, own_nh, ppm, trace,
                       noise, linewidth_ppm, drift_ppm=0.0, skip_ppm=None,
                       excluded_regions=(), ratio=3.0,
                       min_snr=FINGERPRINT_MIN_SNR, own_share=None) -> dict:
    """What the rest of an analyte's fingerprint says about a candidate
    peak, read from that peak's own height: the amount the peak implies,
    the height every other listed signal would then have (protons times
    the tallest line's share), and whether its region holds it.

    Returns ``{'contradicted': n, 'consistent': n, 'checked': n,
    'verdicts': [...]}`` -- the counts the election ranks on. A region
    holding under a ``ratio``-th of the predicted height contradicts; one
    holding at least half of it is consistent; the rest (faint, excluded,
    off the axis) are neither. Consistency is weak evidence in a crowded
    region, where a neighbour stands in; contradiction is strong. The
    ratio is three, as the amount check's (quant_engine.AMOUNT_RATIO): the
    peak's height is read with whatever sits under it -- taurine's centre
    line with betaine's singlet on it implied 2.0x taurine's other
    triplet -- and twice is overlap, not absence.
    """
    from processing.annotation_detectability import _line_share
    x = np.asarray(ppm, dtype=float)
    y = np.asarray(trace, dtype=float)
    hwhm = float(linewidth_ppm) / 2.0
    empty = {'contradicted': 0, 'consistent': 0, 'checked': 0, 'verdicts': []}
    if not own_nh or hwhm <= 0 or not noise:
        return empty
    m = (x >= own_ppm - hwhm) & (x <= own_ppm + hwhm)
    if m.sum() < 1:
        return empty
    own_h = max(float(y[m].max()), 0.0)
    mult = str(own_mult or '').lower().strip()
    share = (float(own_share) if own_share else
             _line_share(mult) if mult not in ('m', 'm*', 'multiplet') else 0.5)
    if own_h <= 0 or share <= 0:
        return empty
    per_proton_area = own_h * np.pi * hwhm / (float(own_nh) * share)
    held, verdicts = fingerprint_holds(
        db_peaks, per_proton_area, ppm, trace, noise=noise,
        linewidth_ppm=linewidth_ppm, drift_ppm=drift_ppm, skip_ppm=skip_ppm,
        excluded_regions=excluded_regions, ratio=ratio, min_snr=min_snr)
    n_c = sum(1 for v in verdicts if v[3].startswith('contradicted'))
    n_p = sum(1 for v in verdicts if v[3] == 'present')
    return {'contradicted': n_c, 'consistent': n_p, 'checked': n_c + n_p,
            'verdicts': verdicts}


def annotate_fingerprint_counts(metabolites, ppm, trace, *, noise,
                                linewidth_ppm, drift_ppm=0.0,
                                excluded_regions=()) -> int:
    """Write ``fp_contradicted`` / ``fp_consistent`` on every matched peak
    of every metabolite, from the peak's own height (``fingerprint_counts``),
    so the election can rank on them. Returns how many were written."""
    from processing.annotation_scoring import pattern_offsets_hz
    n = 0
    for met in metabolites or ():
        db_peaks = met.get('db_peaks') or ()
        for mp in met.get('matched_peaks') or ():
            if mp.get('peak_ppm') is None:
                continue
            own_share = None
            pat = pattern_offsets_hz(mp)
            if pat:
                own_share = max(r for _o, r in pat) / sum(r for _o, r in pat)
            nh = mp.get('protons')
            try:
                nh = float(nh) if nh else None
            except (TypeError, ValueError):
                nh = None
            if not nh:
                continue
            out = fingerprint_counts(
                db_peaks, own_ppm=float(mp['peak_ppm']),
                own_mult=mp.get('type') or mp.get('multiplicity'), own_nh=nh,
                ppm=ppm, trace=trace, noise=noise, linewidth_ppm=linewidth_ppm,
                drift_ppm=drift_ppm, skip_ppm=mp.get('db_ppm'),
                excluded_regions=excluded_regions, own_share=own_share)
            mp['fp_contradicted'] = out['contradicted']
            mp['fp_consistent'] = out['consistent']
            mp['fp_checked'] = out['checked']
            mp['fp_verdicts'] = out['verdicts']
            n += 1
    return n


def resolve_residuals(measured, pool, ppm, intensities, *, included=None,
                      linewidth_ppm, names=None, lines_of=None,
                      per_proton_areas=None, noise_sigma=None,
                      accept=RESIDUAL_ACCEPT, explained=RESIDUAL_EXPLAINED,
                      note=REMAINDER_NOTE, fingerprint_of=None,
                      drift_ppm=0.0, free_apex_of=None,
                      offer_if=None) -> dict:
    """Offer each winner's window, where its accepted shapes leave a
    remainder, the signals the election did not accept -- and take one in
    only on the three conditions above.

    ``measured`` is the engine's per-winner record after
    ``resolve_shared_windows`` (a ``deconv`` record where a window was
    resolved); ``pool`` the candidates the election did not accept, with the
    evidence the adapter built (``expected_ppm``, ``pattern_profile``,
    ``protons``). The taken signal's analyte gets what its shape accounts
    for in that window; the owner's areas are replaced. Returns
    ``{taken_met_id: {'owner': mid, 'areas': {...}, 'nh': nh, 'window':
    (lo, hi), 'remainder_before': f, 'remainder_after': f, 'signal': c}}``.
    Every window tried, taken or not, is recorded on the owner's record as
    ``residual_trials`` so the row can say what was offered and refused.

    ``fingerprint_of(candidate)`` returns the reference's signals for the
    candidate's analyte (``db_peaks``); with it, the fourth condition is
    asked (``fingerprint_holds``): the amount the remainder implies must
    be one the rest of the fingerprint can carry, on the mean of the
    measured samples, at the study's ``drift_ppm``.

    ``free_apex_of(candidate)`` returns the ppm of the candidate's detected
    lines that no accepted analyte holds. A candidate without one is not
    offered: a shape with no apex of its own is a shoulder, and a shoulder
    identifies nobody. The amplitude is read off the tallest free apex in
    every sample and pinned.

    ``offer_if(record)`` says whether a winner's window is offered at all.
    A singlet's window with a remainder is more often the singlet being
    wrong than an intruder being right: on the mixture bench three false
    singlets' windows took in three more analytes, one of them right by
    accident. The engine offers only the windows of owners whose own
    structure was confirmed.
    """
    names = names or {}
    where = lines_of or (lambda c: [])
    lw = float(linewidth_ppm)
    taken: dict = {}
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    _idx = [i for i in (range(arr.shape[1]) if included is None else included)
            if 0 <= i < arr.shape[1]]
    mean_trace = arr[:, _idx].mean(axis=1) if _idx else arr.mean(axis=1)
    winners = {mid: rec for mid, rec in measured.items()
               if rec.get('window') and not rec.get('shape')}
    own_ids = set(winners)
    for mid, rec in winners.items():
        if offer_if is not None and not offer_if(rec):
            continue
        d = rec.get('deconv')
        if d:
            base = [component_from_record(c) for c in d['components']]
            base = [c for c in base if c is not None]
            again = resolve_window(base, ppm, intensities, included=included,
                                   noise_sigma=noise_sigma, refine=False,
                                   allow_single=True,
                                   remainder_window=rec['window'])
            if not again or again.get('refused'):
                continue
            before = float(np.median(list(again['remainder'].values())))
            floor = float(np.median(list(again['remainder_floor'].values())))
            over_base = float(np.median(list(again['overshoot'].values())))
        else:
            ev = dict(rec['signal'].evidence or {})
            found = list(where(rec['signal']) or ())
            if found and not ev.get('shape_fit'):
                ev['predicted_ppms'] = found
                if ev.get('line_weights') and len(ev['line_weights']) != len(found):
                    ev.pop('line_weights')
            owner = component_from_evidence(
                mid, ev, label=f"{names.get(mid, mid)} "
                               f"{str(ev.get('multiplicity') or '')}",
                linewidth_ppm=lw, nh=float(rec['nh']), owner=True)
            if owner is None:
                continue
            base = [owner]
            alone = resolve_window(base, ppm, intensities, included=included,
                                   noise_sigma=noise_sigma, allow_single=True,
                                   remainder_window=rec['window'])
            if not alone or alone.get('refused'):
                continue
            before = float(np.median(list(alone['remainder'].values())))
            floor = float(np.median(list(alone['remainder_floor'].values())))
            over_base = float(np.median(list(alone['overshoot'].values())))
        if before < max(note, RESIDUAL_FLOOR_MULTIPLE * floor):
            continue
        lo, hi = rec['window']
        owner_lines = [v for c in base if c.owner for v in c.lines]
        trials = []
        best = None
        taken_here = []          # (cm, c, comp, out, crec, after)
        current = before
        for c in pool or ():
            cm = str(c.met_id)
            if cm in own_ids or cm in taken:
                continue
            ev = _placed_evidence(c, lw, owner_lines, where)
            if ev is None:
                continue
            lines = [float(v) for v in (ev.get('predicted_ppms') or ())]
            if not lines or not any(lo <= v <= hi for v in lines):
                continue
            nh = ev.get('protons')
            try:
                nh = float(nh) if nh else None
            except (TypeError, ValueError):
                nh = None
            if not nh:
                continue
            label = f"{names.get(cm, cm)} {str(ev.get('multiplicity') or '')}"
            free_apex = list(free_apex_of(c) or ()) if free_apex_of else []
            if free_apex_of is not None and not free_apex:
                trials.append({'label': label, 'verdict': 'refused',
                               'why': 'no apex of its own'})
                continue
            comp = component_from_evidence(cm, ev, label=label, linewidth_ppm=lw,
                                           nh=nh, owner=False,
                                           pinned=bool(free_apex))
            if comp is None:
                continue
            if free_apex:
                # pinned on its tallest free apex: the trace there, in
                # every sample, is what the shape must pass through at
                # whatever geometry the refinement settles on
                xs = np.asarray(ppm, dtype=float)
                csel = (xs >= comp.lo) & (xs <= comp.hi)
                cx = xs[csel]
                if cx.size < 4:
                    continue
                curve = comp.curve(cx)
                c_order = np.argsort(cx)
                apex = max(free_apex, key=lambda v: float(
                    np.interp(v, cx[c_order], curve[c_order])))
                if float(np.interp(apex, cx[c_order], curve[c_order])) <= 0.3:
                    trials.append({'label': label, 'verdict': 'refused',
                                   'why': 'its free apex is not where its shape peaks'})
                    continue
                i_ap = int(np.argmin(np.abs(xs - apex)))
                comp.pin_apex_ppm = float(apex)
                comp.pin_heights = {int(s_i): max(float(arr[i_ap, s_i]), 0.0)
                                    for s_i in _idx}
            # the shapes already in the window keep their geometry; the
            # candidate is refined into what they leave
            trial_comps = ([component_from_record(cr) for cr in
                            [_record(b) for b in base]]
                           + [component_from_record(_record(t[2]))
                              for t in taken_here] + [comp])
            # the shapes already in the window keep their geometry and only
            # the candidate is refined: letting the owner move again with a
            # pinned shape beside it took in a real quartet at twice its
            # amount and a wrong one with it (mixture bench, p90 18 -> 55%)
            out = resolve_window(trial_comps, ppm, intensities,
                                 included=included, noise_sigma=noise_sigma,
                                 allow_single=True,
                                 movable=[trial_comps[-1]],
                                 remainder_window=rec['window'])
            if not out or out.get('refused'):
                trials.append({'label': label, 'verdict': 'refused',
                               'why': (out or {}).get('refused')})
                continue
            crec = next((x for x in out['components'] if x['met_id'] == cm),
                        None)
            if crec is None or (crec.get('fixed') and not crec.get('pinned')):
                trials.append({'label': label, 'verdict': 'refused',
                               'why': 'left out'})
                continue
            after = float(np.median(list(out['remainder'].values())))
            over = float(np.median(list(out['overshoot'].values())))
            sep = crec.get('separability')
            above = current - floor
            drop = (current - after) / above if above > 0 else 0.0
            if crec.get('pinned'):
                # not fitted: it took what its own apex says, and the
                # window says whether that still fits under the trace --
                # the least squares already overshoot as much as they
                # leave, so the test is against the base, not zero --
                # and accounts for a part of what was left
                ok = (over <= over_base + RESIDUAL_FLOOR_MULTIPLE * floor
                      and drop >= RESIDUAL_MIN_SHARE)
            else:
                ok = (sep is not None and sep >= SEPARABLE_MIN
                      and after <= max(accept, RESIDUAL_FLOOR_MULTIPLE * floor)
                      and drop >= explained)
            fp_verdicts = None
            if ok and fingerprint_of is not None and noise_sigma:
                areas_c = out['areas'].get(cm) or {}
                med_area = (float(np.median(list(areas_c.values())))
                            if areas_c else 0.0)
                held, fp_verdicts = fingerprint_holds(
                    fingerprint_of(c) or (), med_area / nh, ppm, mean_trace,
                    noise=noise_sigma, linewidth_ppm=lw, drift_ppm=drift_ppm,
                    skip_ppm=ev.get('db_ppm'))
                if not held:
                    ok = False
            trials.append({'label': label, 'remainder_before': current,
                           'remainder_after': after, 'remainder_floor': floor,
                           'overshoot': over, 'overshoot_base': over_base,
                           'separability': sep,
                           'explained': drop, 'pinned': bool(crec.get('pinned')),
                           'fingerprint': fp_verdicts,
                           'verdict': ('taken' if ok else
                                       'fingerprint contradicted'
                                       if fp_verdicts is not None
                                       and not ok
                                       else 'not enough')})
            if ok and crec.get('pinned'):
                # taken now, and subtracted before the next is tried
                crec['by_residual'] = True
                taken_here.append((cm, c, comp, out, crec, after))
                current = after
                over_base = over
                taken[cm] = None      # reserved; filled below
            elif ok and (best is None or after < best[0]):
                best = (after, cm, c, comp, out, crec)
        rec['residual_trials'] = trials
        if not taken_here and best is None:
            continue
        if taken_here:
            cm, c, comp, out, crec, after = taken_here[-1]
        else:
            after, cm, c, comp, out, crec = best
            crec['by_residual'] = True
        rec['areas'] = dict(out['areas'][mid])
        rec['area_median'] = (float(np.median(list(rec['areas'].values())))
                              if rec['areas'] else float('nan'))
        rec['captured'] = None
        rec['shared_with'] = sorted(set(list(rec.get('shared_with') or ())
                                        + [x['met_id'] for x in out['components']
                                           if x['met_id'] != mid]))
        got = taken_here or [(cm, c, comp, out, crec, after)]
        rec['deconv'] = {'window': out['window'],
                         'remainder': out['remainder'],
                         'components': out['components'],
                         'amplitudes': out['amplitudes'],
                         'skipped': list(out.get('inseparable') or ()),
                         'fixed': [f for f in (out.get('fixed') or ())
                                   if f not in {g[4]['label'] for g in got}],
                         'residual': [g[4]['label'] for g in got]}
        for x in rec['deconv']['components']:
            if x['met_id'] in {g[0] for g in got}:
                x['by_residual'] = True
        if per_proton_areas is not None:
            per_proton_areas[mid] = {s_i: a / rec['nh']
                                     for s_i, a in rec['areas'].items()}
        for g_cm, g_c, g_comp, _o, g_crec, g_after in got:
            taken[g_cm] = {'owner': mid, 'signal': g_c, 'nh': g_comp.nh,
                           'areas': dict(out['areas'].get(g_cm) or {}),
                           'window': out['window'],
                           'remainder_before': before,
                           'remainder_after': after,
                           'separability': g_crec.get('separability'),
                           'deconv': rec['deconv']}
    return {k: v for k, v in taken.items() if v is not None}


def resolve_free_apices(measured, pool, free_peaks, ppm, intensities, *,
                        included=None, linewidth_ppm, names=None,
                        lines_of=None, ranks_of=None, noise_sigma=None,
                        fingerprint_of=None, drift_ppm=0.0, skip=(),
                        trials_out=None) -> dict:
    """The spectrum's own remainder: a detected apex no accepted analyte
    holds. Each refused signal with such an apex is tried alone on it --
    its amount read off the apex and pinned, its whole shape required to
    fit under the trace, to account for STANDALONE_MIN_EXPLAINED of the
    area over its lines' reach, and the rest of its fingerprint to carry
    the amount. One signal per apex, the one that explains most; an apex
    a taken signal used is not offered again.

    This is how a complex multiplet (``m*``) is identified when the shapes
    switch is on: not by fitting its shape alone under the trace, which
    any shape does, but on the same terms a refused letter enters a
    winner's window. Returns ``{met_id: {...}}`` as ``resolve_residuals``
    does, with ``'owner': 'spectrum'``.
    """
    names = names or {}
    where = lines_of or (lambda c: [])
    ranks = ranks_of or (lambda c: [])
    lw = float(linewidth_ppm)
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    _idx = [i for i in (range(arr.shape[1]) if included is None else included)
            if 0 <= i < arr.shape[1]]
    mean_trace = arr[:, _idx].mean(axis=1) if _idx else arr.mean(axis=1)
    xs = np.asarray(ppm, dtype=float)
    free = dict(free_peaks or {})
    taken: dict = {}
    seen_mid = set(skip or ()) | set(measured)
    # tallest apex first: the strongest evidence gets the first claim
    order = sorted(free, key=lambda r: -float(free[r].get('height', 0.0) or 0.0))
    for r in order:
        if r not in free:
            continue
        apex_ppm = float(free[r].get('ppm', 0.0))
        best = None
        trials = []
        for c in pool or ():
            cm = str(c.met_id)
            if cm in seen_mid:
                continue
            if r not in set(ranks(c) or ()):
                continue
            ev = _placed_evidence(c, lw, [], where)
            if ev is None:
                continue
            nh = ev.get('protons')
            try:
                nh = float(nh) if nh else None
            except (TypeError, ValueError):
                nh = None
            if not nh:
                continue
            label = f"{names.get(cm, cm)} {str(ev.get('multiplicity') or '')}"
            comp = component_from_evidence(cm, ev, label=label, linewidth_ppm=lw,
                                           nh=nh, owner=True, pinned=True)
            if comp is None:
                continue
            csel = (xs >= comp.lo) & (xs <= comp.hi)
            cx = xs[csel]
            if cx.size < 4:
                continue
            curve = comp.curve(cx)
            c_order = np.argsort(cx)
            if float(np.interp(apex_ppm, cx[c_order], curve[c_order])) <= 0.3:
                trials.append({'label': label, 'verdict': 'refused',
                               'why': 'the apex is not one of its lines'})
                continue
            i_ap = int(np.argmin(np.abs(xs - apex_ppm)))
            comp.pin_apex_ppm = float(apex_ppm)
            comp.pin_heights = {int(s_i): max(float(arr[i_ap, s_i]), 0.0)
                                for s_i in _idx}
            out = resolve_window([comp], ppm, intensities, included=included,
                                 noise_sigma=noise_sigma, allow_single=True,
                                 refine=True, movable=[comp])
            if not out or out.get('refused'):
                trials.append({'label': label, 'verdict': 'refused',
                               'why': (out or {}).get('refused')})
                continue
            after = float(np.median(list(out['remainder'].values())))
            over = float(np.median(list(out['overshoot'].values())))
            floor = float(np.median(list(out['remainder_floor'].values())))
            ok = (over <= RESIDUAL_FLOOR_MULTIPLE * floor
                  and (1.0 - after) >= STANDALONE_MIN_EXPLAINED)
            fp_verdicts = None
            if ok and fingerprint_of is not None and noise_sigma:
                areas_c = out['areas'].get(cm) or {}
                med_area = (float(np.median(list(areas_c.values())))
                            if areas_c else 0.0)
                held, fp_verdicts = fingerprint_holds(
                    fingerprint_of(c) or (), med_area / nh, ppm, mean_trace,
                    noise=noise_sigma, linewidth_ppm=lw, drift_ppm=drift_ppm,
                    skip_ppm=ev.get('db_ppm'))
                # alone on one apex, the analyte needs a second signal seen
                # at the amount this one implies; "nothing contradicts it"
                # is not identification when nothing could be checked
                seen = any(v[3] == 'present' for v in fp_verdicts)
                if not held or not seen:
                    ok = False
            elif ok and fingerprint_of is not None:
                ok = False
            trials.append({'label': label, 'remainder_after': after,
                           'remainder_floor': floor, 'overshoot': over,
                           'explained': 1.0 - after, 'pinned': True,
                           'fingerprint': fp_verdicts,
                           'verdict': ('taken' if ok else
                                       'fingerprint contradicted'
                                       if fp_verdicts is not None else
                                       'not enough')})
            if ok and (best is None or after < best[0]):
                best = (after, cm, c, comp, out)
        if trials and trials_out is not None:
            trials_out[int(r) if isinstance(r, (int, np.integer)) else r] = {
                'apex_ppm': apex_ppm, 'trials': trials}
        if best is None:
            continue
        after, cm, c, comp, out = best
        crec = out['components'][0]
        crec['by_residual'] = True
        for t in trials:
            if t['verdict'] == 'taken' and t['label'] != crec['label']:
                t['verdict'] = 'not the best'
        deconv = {'window': out['window'], 'remainder': out['remainder'],
                  'components': out['components'],
                  'amplitudes': out['amplitudes'], 'skipped': [], 'fixed': [],
                  'residual': [crec['label']]}
        taken[cm] = {'owner': 'spectrum', 'signal': c, 'nh': comp.nh,
                     'areas': dict(out['areas'].get(cm) or {}),
                     'window': out['window'],
                     'remainder_before': 1.0, 'remainder_after': after,
                     'separability': None, 'deconv': deconv,
                     'apex_rank': r}
        seen_mid.add(cm)
        for rr in ranks(c) or ():
            free.pop(rr, None)
    return taken


def listed_signal_evidence(dp, *, centre_ppm, field_mhz) -> dict | None:
    """The evidence a component is built from for a signal the reference
    lists but annotation never matched: its lines laid at ``centre_ppm``
    (the listed position at the study's drift) -- the measured pattern's
    offsets and heights where the reference has them, a first-order comb
    from the listed coupling otherwise, one line for a singlet -- with the
    reference's profile when there is one. None when nothing places it."""
    from processing.annotation_scoring import (
        expected_line_count, is_singlet_like, parse_first_j,
        pattern_offsets_hz, synthetic_lines)
    mult = str(dp.get('type') or dp.get('multiplicity') or '').lower().strip()
    f = float(field_mhz)
    pat = pattern_offsets_hz(dp)
    ev = {'multiplicity': mult, 'protons': dp.get('protons'),
          'db_ppm': dp.get('ppm_center', dp.get('ppm')),
          'pattern_profile': dp.get('pattern_profile'),
          'profile_fwhm_hz': dp.get('profile_fwhm_hz'),
          'pattern_field_mhz': f, 'listed': True}
    if pat:
        offs = [o for o, _r in pat]
        mean_off = float(np.mean(offs))
        ev['predicted_ppms'] = [centre_ppm + (o - mean_off) / f for o in offs]
        ev['line_weights'] = [float(r) for _o, r in pat]
        return ev
    if is_singlet_like(mult) or mult in ('m', 'multiplet'):
        if mult in ('m', 'multiplet'):
            return None                # nothing says what shape it has
        ev['predicted_ppms'] = [float(centre_ppm)]
        ev['peak_ppm'] = float(centre_ppm)
        return ev
    j = parse_first_j(dp.get('j_coupling'))
    n = expected_line_count(mult, fallback=2)
    if not j or n < 2:
        return None
    ev['predicted_ppms'] = [l['ppm'] for l in
                            synthetic_lines(float(centre_ppm), j, n, f)]
    return ev


def resolve_listed_intruders(measured, ppm, intensities, *, included=None,
                             linewidth_ppm, names=None, lines_of=None,
                             per_proton_areas, noise_sigma=None,
                             field_mhz=None) -> dict:
    """The signals of analytes already quantified, wherever the reference
    lists them inside another winner's window, subtracted at the amount
    that analyte was measured at.

    ADR-015 kept out a signal the listing places and detection did not: a
    listing is not a measurement. Once the ANALYTE is measured, its other
    signals are: the amount is known from its own clean signal, the shape
    is the reference's, and only the position is the listing's -- at the
    study's drift, refined within half a linewidth. Each such signal is a
    fixed component (ADR-022); it is kept only if, at that amount, it fits
    under the trace (the window's overshoot does not rise beyond the noise
    over what it was without it), else the listing was wrong here and the
    window stands. Glutamine's triplet on the mixture bench read twice its
    amount with glutamate's alpha-H inside it -- glutamate measured on its
    own pattern, its alpha-H never a candidate.

    Returns ``{owner_met_id: [labels taken in]}``; the owners' areas are
    replaced in ``measured`` and ``per_proton_areas``.
    """
    names = names or {}
    where = lines_of or (lambda c: [])
    lw = float(linewidth_ppm)
    out: dict = {}
    winners = {mid: rec for mid, rec in measured.items()
               if rec.get('window') and not rec.get('shape')}
    # what every quantified analyte lists, and what it is measured at
    listed: list = []
    for mid, rec in winners.items():
        sig = rec['signal']
        met = (sig.payload or {}).get('met') if isinstance(sig.payload, dict) else None
        pp = per_proton_areas.get(mid) or {}
        vals = {int(k): float(v) for k, v in pp.items()
                if v is not None and np.isfinite(v) and v > 0}
        if not met or not vals:
            continue
        ev = sig.evidence or {}
        f = field_mhz or ev.get('pattern_field_mhz')
        if not f:
            continue
        drift = float(ev.get('study_drift_ppm') or 0.0)
        own = round(float(ev.get('db_ppm') or 0.0), 3)
        for dp in met.get('db_peaks') or ():
            c0 = dp.get('ppm_center', dp.get('ppm'))
            if c0 is None or round(float(c0), 3) == own:
                continue
            e2 = listed_signal_evidence(dp, centre_ppm=float(c0) + drift,
                                        field_mhz=float(f))
            if e2 is None:
                continue
            nh = dp.get('protons')
            try:
                nh = float(nh) if nh else None
            except (TypeError, ValueError):
                nh = None
            if not nh:
                continue
            listed.append((mid, e2, nh, vals, float(np.median(list(vals.values())))))
    if not listed:
        return out
    for mid, rec in winners.items():
        lo, hi = rec['window']
        d = rec.get('deconv')
        already = {c['met_id'] for c in (d or {}).get('components', ())}
        extra = []
        for om, e2, nh, vals, hint in listed:
            if om == mid or om in already:
                continue
            lines = [float(v) for v in e2.get('predicted_ppms') or ()]
            if not lines or not any(lo <= v <= hi for v in lines):
                continue
            label = f"{names.get(om, om)} {e2.get('multiplicity') or ''} (listed)"
            comp = component_from_evidence(om, e2, label=label, linewidth_ppm=lw,
                                           nh=nh, owner=False,
                                           per_proton_hint=hint,
                                           per_proton_by_sample=vals,
                                           pinned=True)
            if comp is not None:
                extra.append(comp)
        if not extra:
            continue
        if d:
            base = [component_from_record(c) for c in d['components']]
            base = [c for c in base if c is not None]
        else:
            ev = dict(rec['signal'].evidence or {})
            found = list(where(rec['signal']) or ())
            if found and not ev.get('shape_fit'):
                ev['predicted_ppms'] = found
                if ev.get('line_weights') and len(ev['line_weights']) != len(found):
                    ev.pop('line_weights')
            owner = component_from_evidence(
                mid, ev, label=f"{names.get(mid, mid)} "
                               f"{str(ev.get('multiplicity') or '')}",
                linewidth_ppm=lw, nh=float(rec['nh']), owner=True)
            if owner is None:
                continue
            base = [owner]
        before = resolve_window(base, ppm, intensities, included=included,
                                noise_sigma=noise_sigma, allow_single=True,
                                refine=not d, remainder_window=rec['window'])
        if not before or before.get('refused'):
            continue
        over0 = float(np.median(list(before['overshoot'].values())))
        floor = float(np.median(list(before['remainder_floor'].values())))
        kept = []
        for comp in extra:
            trial = [component_from_record(_record(b)) for b in base] + \
                    [component_from_record(_record(k)) for k in kept] + [comp]
            # every shape refines again with the listed one present: the
            # owner refined alone had widened onto the intruder, and a
            # fixed intruder cannot be taken from, so the owner may move
            res = resolve_window(trial, ppm, intensities, included=included,
                                 noise_sigma=noise_sigma, allow_single=True,
                                 remainder_window=rec['window'])
            if not res or res.get('refused'):
                continue
            over = float(np.median(list(res['overshoot'].values())))
            if over > over0 + RESIDUAL_FLOOR_MULTIPLE * floor:
                continue               # the listing put it where it is not
            kept.append(comp)
            over0 = over
            final = res
        if not kept:
            continue
        # the owner's share of what the window holds, after the others
        own = float(np.median(list(final['areas'][mid].values())))
        total = sum(float(np.median(list(final['areas'][m].values())))
                    for m in final['areas'])
        if total > 0 and own / total < OWNER_MIN_SHARE:
            rec['taken_over'] = [c.label for c in kept]
            out[mid] = [c.label for c in kept]
            continue
        rec['areas'] = dict(final['areas'][mid])
        rec['area_median'] = (float(np.median(list(rec['areas'].values())))
                              if rec['areas'] else float('nan'))
        rec['captured'] = None
        rec['shared_with'] = sorted(set(list(rec.get('shared_with') or ())
                                        + [c.met_id for c in kept]))
        rec['deconv'] = {'window': final['window'],
                         'remainder': final['remainder'],
                         'components': final['components'],
                         'amplitudes': final['amplitudes'],
                         'skipped': list(final.get('inseparable') or ()),
                         'fixed': list(final.get('fixed') or ()),
                         'residual': list((d or {}).get('residual') or ()),
                         'listed': [c.label for c in kept]}
        per_proton_areas[mid] = {s_i: a / rec['nh']
                                 for s_i, a in rec['areas'].items()}
        out[mid] = [c.label for c in kept]
    return out


def _record(c) -> dict:
    """A component as ``resolve_window`` records it, for building it again."""
    return {'met_id': c.met_id, 'label': c.label, 'owner': c.owner, 'nh': c.nh,
            'lines': list(c.lines), 'ceiling': c.ceiling,
            'separability': c.separability, 'shape_area': c.shape_area,
            'per_proton_hint': c.per_proton_hint, 'evidence': c.evidence,
            'linewidth_ppm': c.linewidth_ppm, 'shift_ppm': c.shift_ppm,
            'width_scale': c.width_scale, 'fixed': c.fixed,
            'pinned': c.pinned, 'pin_apex_ppm': c.pin_apex_ppm,
            'pin_heights': dict(c.pin_heights),
            'per_proton_by_sample': dict(c.per_proton_by_sample)}


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
