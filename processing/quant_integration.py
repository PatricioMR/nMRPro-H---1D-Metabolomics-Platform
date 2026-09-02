# processing/quant_integration.py
"""The area of an elected signal, measured in each sample on its own.

One integration mode: the whole multiplet, foot to foot. The old engine carried
five — multiplet, area, Lorentz fit, MCR component and a database gap-fill —
and every one of them was a second answer to the same question, reached by a
different path and reconciled nowhere.

Foot to foot means the left foot of the leftmost line to the right foot of the
rightmost, taken from detection's own boundaries. Not a window centred on the
database position and not a fixed width: the analyte's lines are where
detection found them, and the area that belongs to the analyte is the area
between its own feet.

Each sample is integrated separately, on its own trace. A mean spectrum has no
concentration, and a study's job is to tell the samples apart.
"""
from __future__ import annotations

import numpy as np


def multiplet_window(line_peaks) -> tuple[float, float] | None:
    """Foot to foot over every line of the signal, as (lo_ppm, hi_ppm).

    Uses each line's own integration boundaries — the feet detection found —
    rather than a margin around the centre, so the window ends where the
    signal ends. Falls back to a line's apex when it carries no boundary,
    which yields a zero-width window if that is all there is; the caller sees
    the width and can refuse it.

    Returns None when there are no lines to bound.
    """
    los, his = [], []
    for pk in line_peaks or ():
        apex = float(pk.get('ppm', 0.0))
        a = float(pk.get('start_ppm', apex))
        b = float(pk.get('end_ppm', apex))
        los.append(min(a, b))
        his.append(max(a, b))
    if not los:
        return None
    return float(min(los)), float(max(his))


def _window_mask(ppm, lo, hi):
    ppm = np.asarray(ppm, dtype=float)
    if lo > hi:
        lo, hi = hi, lo
    return (ppm >= lo) & (ppm <= hi)


def integrate_sample(ppm, y, lo, hi, *, local_baseline=True):
    """Area of one sample between ``lo`` and ``hi`` ppm.

    ``local_baseline`` subtracts the straight line joining the window's two
    endpoints before integrating — the qNMR convention, and what removes the
    broad component a multiplet happens to be sitting on. The endpoints are
    clipped at zero first, so a negative excursion at one foot cannot tilt the
    baseline up into the signal.

    A negative area is returned as it comes out. A window over noise integrates
    to something near zero and sometimes below it, and that is the honest
    reading: clipping it to zero would report an absent analyte as merely very
    dilute.

    Returns None when the window holds too few points to integrate.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    m = _window_mask(ppm, lo, hi)
    if m.sum() < 2:
        return None
    x, seg = ppm[m], y[m]
    order = np.argsort(x)
    x, seg = x[order], seg[order]
    if local_baseline and seg.size >= 2:
        base = np.linspace(max(float(seg[0]), 0.0), max(float(seg[-1]), 0.0),
                           seg.size)
        seg = seg - base
    try:
        return float(np.trapezoid(seg, x=x))
    except AttributeError:                       # numpy < 2
        return float(np.trapz(seg, x=x))


def integrate_across_samples(ppm, intensities, lo, hi, *, included=None,
                             local_baseline=True) -> dict:
    """The signal's area in every sample the caller allows.

    ``included`` is the list of sample indices to measure — blanks and excluded
    samples belong outside it, the same selection detection and annotation use.

    Returns ``{sample_idx: area}``. A sample whose window holds too few points
    is absent from the mapping rather than present with a zero: no measurement
    is not a measurement of nothing.
    """
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_s = arr.shape[1]
    idx = range(n_s) if included is None else [i for i in included
                                              if 0 <= i < n_s]
    out = {}
    for s in idx:
        a = integrate_sample(ppm, arr[:, s], lo, hi,
                             local_baseline=local_baseline)
        if a is not None:
            out[int(s)] = a
    return out


def window_width_ppm(lo, hi) -> float:
    return abs(float(hi) - float(lo))


def integrate_signal(line_peaks, ppm, intensities, *, included=None,
                     local_baseline=True) -> dict:
    """Everything the caller needs about one elected signal's area.

    Returns ``{'window': (lo, hi), 'width_ppm': w, 'areas': {sample: area}}``,
    or None when the signal has no lines to bound a window with. The window
    travels with the areas so the report can show which region produced the
    number — a concentration whose window nobody can see is not auditable.
    """
    win = multiplet_window(line_peaks)
    if win is None:
        return None
    lo, hi = win
    return {
        'window': (lo, hi),
        'width_ppm': window_width_ppm(lo, hi),
        'areas': integrate_across_samples(ppm, intensities, lo, hi,
                                          included=included,
                                          local_baseline=local_baseline),
    }
