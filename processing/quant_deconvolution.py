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


# ── Two analytes on one peak ─────────────────────────────────────────────────
# The other half of what overlap does, and the larger one: on the reference
# study 61 analytes ended with no number because another won the peak they
# shared. The election is right to give a peak to one owner -- two analytes
# integrating the same area is the defect that started this work -- but the
# area itself belongs to both, and the winner was credited with all of it.
#
# Every window is one equation. Its area is the sum over its claimants of
# concentration times protons times the fraction of that claimant the window
# holds, and the fraction is computed above from where the lines are. Nothing
# in it is fitted to the trace: the shape of the equation comes from the
# database and from detection, and only the concentrations are unknown.
#
# Solved one window at a time this reaches almost nothing. A window with two
# claimants and no independent measurement of either has one equation and two
# unknowns, and on the reference study 215 of 242 contested windows were of
# that kind. But the windows are not independent problems -- a metabolite
# appears in as many of them as it has signals, with a different fraction in
# each -- so together they are 291 equations in about 130 unknowns, and
# overdetermined. Solved together, a metabolite with no clean signal anywhere
# is still determined by the several windows it appears in, provided those
# windows distinguish it from everything else in them.
#
# "Provided" is the whole difficulty, and it is what the old JG-Deconv stage
# had no answer for: it fitted everything and reported everything, including
# analytes whose column the data could not separate from another's. Two
# metabolites that appear in exactly the same windows in exactly the same
# proportions cannot be told apart by any amount of arithmetic, and a solver
# will still return a number for each. So every column is tested for whether
# the data determines it, and the ones that are not determined get no
# concentration and a reason naming what they could not be separated from.
#
#     Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares Problems.
#     Prentice-Hall. Chapter 23: the non-negative least squares algorithm.
#
#     Belsley, D.A., Kuh, E. & Welsch, R.E. (1980). Regression Diagnostics:
#     Identifying Influential Data and Sources of Collinearity. Wiley.
#
#     O'Brien, R.M. (2007). A caution regarding rules of thumb for variance
#     inflation factors. Quality & Quantity 41:673-690.
#
#     Weljie, A.M. et al. (2006). Targeted profiling: quantitative analysis of
#     1H NMR metabolomics data. Analytical Chemistry 78:4430-4442.

# A column whose variance is inflated more than this by the others is one the
# data does not separate. Ten is the classical rule of thumb, and O'Brien 2007
# is right that it is only that -- but the alternative here is not a better
# threshold, it is reporting numbers nobody can attribute.
MAX_VIF = 10.0

# And a concentration whose own standard error is more than half of it says
# the same thing in the other language: the system admits it, but it does not
# pin it down.
MAX_RSD = 0.5


def _vif(design):
    """How much each column's variance is inflated by the other columns.

    ``1 / (1 - R2)`` for each column regressed on the rest, which is the
    definition. A column the others reproduce exactly has infinite VIF and is
    not identifiable at all; one they cannot reproduce has VIF 1.

    Returns ``(vif, partner)`` -- the partner being the other column that
    contributes most to reproducing it, so a refusal can name what an analyte
    could not be separated from rather than only that it could not.
    """
    m = np.asarray(design, dtype=float)
    n_col = m.shape[1]
    vif = np.ones(n_col, dtype=float)
    partner = [None] * n_col
    if n_col < 2:
        return vif, partner
    for i in range(n_col):
        y = m[:, i]
        ss = float(y @ y)
        if ss <= 0:
            vif[i] = np.inf
            continue
        rest = np.delete(m, i, axis=1)
        coef, *_ = np.linalg.lstsq(rest, y, rcond=None)
        resid = y - rest @ coef
        r2 = 1.0 - float(resid @ resid) / ss
        vif[i] = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        if coef.size:
            k = int(np.argmax(np.abs(coef)))
            partner[i] = k if k < i else k + 1
    return vif, partner


def solve_windows(windows, *, max_vif=MAX_VIF, max_rsd=MAX_RSD):
    """Every window at once: what one proton of each metabolite is worth.

    ``windows`` are ``{'key', 'claimants': [(met_id, nh * fraction)],
    'areas': {sample: area}, 'weight': w}``. The claimant's number is already
    protons times the fraction of it the window holds, because that product is
    what multiplies the concentration in the balance.

    Returns ``(per_proton, reasons)``. ``per_proton[met][sample]`` is the area
    one proton of that metabolite's whole signal is worth -- the same quantity
    an uncontested window would have divided out, so it travels onward by the
    same path. ``reasons[met]`` says why a metabolite got none.

    Non-negative, because a concentration is not negative and letting one go
    negative lets it pay for another's excess. ``weight`` is one over the
    noise on that window's area, so a wide noisy window does not outvote a
    narrow clean one; equal weights when it is not given.

    The columns are the same in every sample, so identifiability is decided
    once, and the standard errors then come per sample from that sample's own
    residual.

    The windows overlap, so their residuals are correlated and the standard
    errors below are optimistic. They are used as a floor -- a column already
    too uncertain under a generous assumption is certainly too uncertain --
    never as the reported precision.
    """
    from scipy.optimize import nnls

    wins = [w for w in (windows or ()) if w.get('claimants') and w.get('areas')]
    if not wins:
        return {}, {}
    mets = sorted({str(m) for w in wins for m, _ in w['claimants']})
    col = {m: i for i, m in enumerate(mets)}
    samples = sorted({int(s) for w in wins for s in w['areas']})

    weight = np.array([float(w.get('weight') or 1.0) for w in wins])
    weight[~np.isfinite(weight) | (weight <= 0)] = 1.0

    design = np.zeros((len(wins), len(mets)), dtype=float)
    for r, w in enumerate(wins):
        for m, nh_eff in w['claimants']:
            design[r, col[str(m)]] += float(nh_eff)
    design *= weight[:, None]

    reasons: dict = {}
    norms = np.linalg.norm(design, axis=0)
    live = [i for i in range(len(mets)) if norms[i] > 0]
    for i in range(len(mets)):
        if i not in live:
            reasons[mets[i]] = 'it contributes nothing to any window measured'
    if not live:
        return {}, reasons

    vif, partner = _vif(design[:, live])
    keep = []
    for k, i in enumerate(live):
        if vif[k] <= max_vif:
            keep.append(i)
            continue
        j = partner[k]
        who = mets[live[j]] if j is not None and j < len(live) else None
        reasons[mets[i]] = (
            'the windows do not separate it from %s -- they appear together '
            'in the same proportions, so no arithmetic can say which of them '
            'the area belongs to' % (who or 'the other claimants'))
    if not keep:
        return {}, reasons

    # A window left with no column of its own -- every claimant in it refused
    # -- is no longer an equation about anything kept. Leaving it in the system
    # makes its whole area a residual, which inflates the standard error of
    # every column that IS kept and can refuse an analyte the rest of the
    # windows determine exactly.
    m_keep = design[:, keep]
    has_col = m_keep.any(axis=1)
    per_proton: dict = {}
    rsd_max = {i: 0.0 for i in keep}
    for s in samples:
        b = np.array([float(w['areas'].get(s, np.nan)) for w in wins])
        ok = np.isfinite(b) & has_col
        if ok.sum() < 1:
            continue
        a_s, b_s = m_keep[ok] * 1.0, b[ok] * weight[ok]
        try:
            x, _ = nnls(a_s, b_s)
        except Exception:
            continue
        resid = b_s - a_s @ x
        dof = max(int(ok.sum()) - len(keep), 1)
        sigma = float(np.sqrt(float(resid @ resid) / dof))
        for k, i in enumerate(keep):
            n_i = float(np.linalg.norm(a_s[:, k]))
            if n_i <= 0:
                continue
            se = sigma * float(np.sqrt(max(vif[live.index(i)], 1.0))) / n_i
            if x[k] > 0:
                rsd_max[i] = max(rsd_max[i], se / float(x[k]))
            per_proton.setdefault(mets[i], {})[int(s)] = float(x[k])

    for i in keep:
        m = mets[i]
        vals = [v for v in (per_proton.get(m) or {}).values() if v > 0]
        if not vals:
            per_proton.pop(m, None)
            reasons[m] = ('the balance leaves it nothing: what the other '
                          'claimants account for fills the windows it is in')
        elif rsd_max[i] > max_rsd:
            per_proton.pop(m, None)
            reasons[m] = ('its own standard error reaches %.0f%% of it -- the '
                          'windows admit this analyte but do not pin it down'
                          % (100.0 * rsd_max[i]))
    return per_proton, reasons
