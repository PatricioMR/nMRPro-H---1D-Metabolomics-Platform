# processing/quant_deconvolution.py
"""What a window holds that belongs to somebody else.

A window drawn around one signal is not a container for that signal alone. A
Lorentzian's tails fall away as the square of the distance and never reach
zero, so every line in the spectrum puts a little of itself into every window,
and a strong neighbour a short way off can put in a great deal.

Measured on a study whose answer was known: L-alanine sits 0.148 ppm from
L-lactic acid, whose line is eight times larger, and after every other
correction alanine still read 6.4% high. That 6.4% is lactate.

This is the smaller half of what overlap does. It is not two analytes sharing a
peak — the election forbids that, and no two winners hold overlapping windows —
but the skirt of a resolved neighbour reaching across. The two need different
treatment and this module does only the first:

    neighbour_leak     how much of a signal's area falls in another's window
    subtract_neighbours the areas with each other's leaks removed

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


def fraction_in_window(line_ppms, weights, lo, hi, hwhm) -> float:
    """How much of a signal's area falls inside ``[lo, hi]``.

    The same expression as the capture correction, read the other way round:
    there it says how much of a signal its own window holds, here how much of a
    neighbour a foreign window holds. One model, so a leak subtracted from one
    analyte is the same quantity the other was credited with.
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
        f += (wi / total) * (_cumulative(hi, p, hwhm)
                             - _cumulative(lo, p, hwhm))
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
