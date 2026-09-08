# processing/quant_deconvolution.py
"""What a window holds that belongs to somebody else.

A window drawn around one signal is not a container for that signal alone. A
Lorentzian's tails fall away as the square of the distance and never reach
zero, so every line in the spectrum puts a little of itself into every window,
and a strong neighbour a short way off can put in a great deal.

Measured on a study whose answer was known: L-alanine sits 0.148 ppm from
L-lactic acid, whose line is eight times larger, and after every other
correction alanine still read 6.4% high. That 6.4% is lactate.

Overlap does two things and they need different treatment. The skirt of a
resolved neighbour reaching across a window is one, and the smaller; two
analytes claiming the same peak is the other, and on the reference study it
left 61 analytes with no number at all. Both live here:

    neighbour_leak       how much of a signal's area falls in another's window
    subtract_neighbours  the areas with each other's leaks removed
    anchor_per_proton    what one proton is worth, from windows nobody contests
    resolve_contested    a shared window split between the analytes in it

Nothing is estimated here that is not already measured. Each neighbour's total
area comes from its own window, corrected for what that window left outside, so
the leak is computed from the amount of neighbour there actually is. When the
lineshape model is wrong the correction is wrong in proportion; with no
correction at all the error is the whole leak, which is larger.
"""
from __future__ import annotations

import numpy as np


def _cumulative(x, centre, hwhm):
    """The fraction of one Lorentzian's area lying below ``x``."""
    return 0.5 + np.arctan((float(x) - float(centre)) / float(hwhm)) / np.pi


def _height(x, centre, hwhm):
    """One unit-area Lorentzian's height at ``x``."""
    g = float(hwhm)
    return (g / np.pi) / ((float(x) - float(centre)) ** 2 + g ** 2)


def fraction_in_window(line_ppms, weights, lo, hi, hwhm, *,
                       local_baseline=False) -> float:
    """How much of a signal's area a window of these bounds actually measures.

    The same expression as the capture correction, read the other way round:
    there it says how much of a signal its own window holds, here how much of a
    neighbour a foreign window holds. One model, so a leak subtracted from one
    analyte is the same quantity the other was credited with.

    ``local_baseline`` says the integration subtracts the straight line joining
    the window's endpoints, which is what the qNMR convention does and what
    removes a broad component. That line is drawn through the signal's own
    tails, so it takes away a trapezium of the very area being measured, and
    in a narrow window -- where the feet sit high on the flanks -- it takes a
    great deal. Measured on one line integrated two half-widths either side:
    the window holds 70% of the area and the integral returns 26%.

    The trapezium is as computable as the rest, so it belongs in the model
    rather than in the error. Without it a narrow window's area, divided by the
    fraction it was supposed to hold, comes back far short and the balance
    hands the difference to whoever is left.
    """
    ppms = [float(p) for p in (line_ppms or ()) if p is not None]
    if not ppms or not hwhm or float(hwhm) <= 0:
        return 0.0
    lo, hi = (min(lo, hi), max(lo, hi))
    w = list(weights or ())
    if len(w) != len(ppms) or sum(w) <= 0:
        w = [1.0] * len(ppms)
    total = float(sum(w))
    f = 0.0
    for p, wi in zip(ppms, w):
        share = (_cumulative(hi, p, hwhm) - _cumulative(lo, p, hwhm))
        if local_baseline:
            share -= (hi - lo) * (_height(lo, p, hwhm)
                                  + _height(hi, p, hwhm)) / 2.0
        f += (wi / total) * share
    return float(f) if np.isfinite(f) else 0.0


def neighbour_leak(window, neighbours, hwhm) -> float:
    """The area inside ``window`` that belongs to the signals around it.

    ``neighbours`` are ``(line_ppms, weights, total_area)`` for every other
    signal. Their total areas are what their own windows measured, corrected
    for their own tails, so this asks how much of a known quantity lands here
    rather than inferring an unknown one.
    """
    lo, hi = window
    leak = 0.0
    for ppms, weights, area in neighbours or ():
        if area is None or not np.isfinite(area):
            continue
        leak += float(area) * fraction_in_window(ppms, weights, lo, hi, hwhm)
    return float(leak)


def subtract_neighbours(signals, hwhm, *, max_fraction=0.5):
    """Each signal's area with the other signals' tails taken out.

    ``signals`` maps a key to ``{'window': (lo, hi), 'lines': [...],
    'weights': [...], 'areas': {sample: area}}``. Returns
    ``{key: {sample: corrected_area}}`` alongside ``{key: leak_fraction}`` so
    the caller can say how much was removed and from what.

    One pass, not iterated. The leaks are small — a few percent — so what a
    correction would change on a second pass is a few percent of a few percent,
    below everything else in the measurement, and iterating a model against
    itself invites it to converge on its own assumptions rather than on the
    data.

    A leak larger than ``max_fraction`` of a signal's own area is refused, and
    the area left alone. At that point the window is mostly somebody else and
    subtracting would report the remainder as an analyte; that is the shared
    peak problem, which the election settles and this does not.
    """
    keys = list(signals or {})
    totals = {}
    for k in keys:
        a = [v for v in (signals[k].get('areas') or {}).values()
             if v is not None and np.isfinite(v)]
        totals[k] = float(np.median(a)) if a else float('nan')

    out, removed = {}, {}
    for k in keys:
        sig = signals[k]
        win = sig.get('window')
        areas = dict(sig.get('areas') or {})
        if not win or not areas:
            out[k], removed[k] = areas, 0.0
            continue
        others = [(signals[j].get('lines'), signals[j].get('weights'),
                   totals[j]) for j in keys if j != k]
        leak = neighbour_leak(win, others, hwhm)
        own = totals[k]
        if (not np.isfinite(leak) or leak <= 0 or not np.isfinite(own)
                or own <= 0 or leak > float(max_fraction) * own):
            out[k], removed[k] = areas, 0.0
            continue
        share = leak / own
        out[k] = {s: v * (1.0 - share) for s, v in areas.items()}
        removed[k] = float(share)
    return out, removed


# ── Two analytes on one peak ─────────────────────────────────────────────────
# The other half of what overlap does, and the larger one: on the reference
# study 61 analytes ended with no number because another won the peak they
# shared. The election is right to give a peak to one owner -- two analytes
# integrating the same area is the defect that started this work -- but the
# area itself belongs to both, and the winner is currently credited with all
# of it.
#
# The accounting is linear and needs no new measurement. A window's area is the
# sum over its claimants of concentration times protons. Where a claimant has a
# signal of its own that nobody contests, that signal says what its
# concentration is; subtract what it must be contributing here, and what
# remains belongs to the rest. If exactly one claimant is unaccounted for, the
# remainder is its area. If more than one is, the window has one equation and
# two unknowns and nothing is reported.
#
# This is the shape of targeted profiling, and of the JG-Deconv stage's phases
# 1 to 3, rebuilt where it can be tested.

# A residue smaller than this share of the window is the difference between two
# larger numbers, and its error is the sum of theirs. Reporting it as a
# concentration would give an analyte a number built from other analytes'
# uncertainty.
MIN_RESIDUE_SHARE = 0.10


def anchor_per_proton(windows):
    """``{met_id: {sample: area per proton}}`` from windows nobody contests.

    A signal with one claimant measures that claimant and nothing else, so its
    area over its proton count is what one proton of that metabolite is worth
    in this study. Everything else is solved against these.

    A metabolite with several uncontested signals is averaged over them by the
    median, so one bad window cannot set the scale for the rest.
    """
    per_met: dict = {}
    for w in windows or ():
        claimants = list(w.get('claimants') or ())
        if len(claimants) != 1:
            continue
        mid, nh = claimants[0]
        if not nh or float(nh) <= 0:
            continue
        for s, a in (w.get('areas') or {}).items():
            if a is None or not np.isfinite(a):
                continue
            per_met.setdefault(str(mid), {}).setdefault(int(s), []).append(
                float(a) / float(nh))
    return {mid: {s: float(np.median(v)) for s, v in by_s.items()}
            for mid, by_s in per_met.items()}


def resolve_contested(windows, *, max_passes=5,
                      min_residue_share=MIN_RESIDUE_SHARE):
    """Split each contested window between the analytes that claim it.

    Returns ``(shares, reasons)``. ``shares`` maps each solved window's key to
    ``{met_id: {sample: area}}`` — every claimant's own part of it, the
    unaccounted one's being the remainder and each other's what its own clean
    signal says it must be contributing here.

    The whole window is returned and not only the remainder, because the
    metabolite that won it is currently credited with all of it. Splitting the
    area is as much a correction to the winner as it is a number for the
    analyte that had none.

    ``reasons`` says why a window resolved nothing.

    Cascading: a metabolite solved on one window becomes an anchor for the
    next, so a chain of overlaps unwinds from whatever clean signal it touches.
    Bounded, because a cascade that does not settle is a cascade feeding on its
    own output.

    Nothing is invented. Every number here is a measured area minus the
    contributions of metabolites whose concentrations another window measured.
    A window with two unaccounted claimants has one equation and two unknowns
    and reports nothing at all.
    """
    anchors = anchor_per_proton(windows)
    out: dict = {}
    solved: set = set()
    reasons: dict = {}
    contested = [w for w in (windows or ())
                 if len(w.get('claimants') or ()) > 1]

    for _ in range(int(max_passes)):
        progress = False
        for w in contested:
            key = w.get('key')
            if key in solved:
                continue
            claimants = list(w.get('claimants') or ())
            unknown = [(m, nh) for m, nh in claimants
                       if str(m) not in anchors]
            if len(unknown) > 1:
                reasons[key] = ('%d claimants with no signal of their own — '
                                'one equation, %d unknowns'
                                % (len(unknown), len(unknown)))
                continue
            if not unknown:
                reasons[key] = ('every claimant already measured elsewhere; '
                                'the window adds nothing')
                continue
            mid_u, nh_u = unknown[0]
            if not nh_u or float(nh_u) <= 0:
                reasons[key] = 'the unaccounted claimant has no proton count'
                continue
            split: dict = {}
            ok = True
            for s, total in (w.get('areas') or {}).items():
                if total is None or not np.isfinite(total):
                    continue
                parts = {}
                for m, nh in claimants:
                    if str(m) in anchors:
                        parts[str(m)] = (anchors[str(m)].get(int(s),
                                                             float('nan'))
                                         * float(nh))
                known = sum(parts.values())
                if not np.isfinite(known):
                    continue
                residue = float(total) - float(known)
                if residue <= 0 or residue < min_residue_share * float(total):
                    ok = False
                    break
                parts[str(mid_u)] = residue
                for m, a in parts.items():
                    split.setdefault(m, {})[int(s)] = float(a)
            if not ok or str(mid_u) not in split:
                reasons[key] = ('what the known claimants account for leaves '
                                'no measurable remainder')
                continue
            out[key] = split
            anchors[str(mid_u)] = {s: a / float(nh_u)
                                   for s, a in split[str(mid_u)].items()}
            solved.add(key)
            reasons.pop(key, None)
            progress = True
        if not progress:
            break

    return out, reasons


# A contributor below this share of a window is inside the measurement's own
# noise. Counting it adds an unknown -- and an unknown is what stops a window
# being solved at all -- in exchange for a term smaller than the error on the
# terms that matter.
MIN_CLAIM_FRACTION = 0.05


def claimants_in_window(contributions, *, min_claim=MIN_CLAIM_FRACTION):
    """Who is in this window, and how many protons' worth of each.

    ``contributions`` are ``(met_id, nh, fraction)`` -- one per candidate
    signal whose lines reach the window, the fraction being how much of that
    signal the window holds. A metabolite with two signals reaching in
    contributes both: two of its lines inside one window is twice the signal.

    Returns ``[(met_id, nh * fraction)]``, ordered by met_id so two runs over
    the same data give the same answer. That product is what the balance
    needs -- the window measures ``concentration * protons * fraction`` of each
    claimant, so passing it as the proton count lets one linear solver serve
    windows whose claimants are each cut off at a different place.
    """
    per_met: dict = {}
    for mid, nh, f in contributions or ():
        try:
            eff = float(nh) * float(f)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(eff) or float(f) < float(min_claim) or eff <= 0:
            continue
        per_met[str(mid)] = per_met.get(str(mid), 0.0) + eff
    return [(m, per_met[m]) for m in sorted(per_met)]
