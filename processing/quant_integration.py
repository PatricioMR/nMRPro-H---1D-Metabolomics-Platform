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


def window_shifts(line_peaks, *, max_shift_ppm=None) -> dict:
    """How far each sample's window moves to follow the signal it is measuring.

    A resonance drifts between samples with pH, ionic strength and temperature.
    One window fixed at the mean position then measures the analyte in some
    samples and its shoulder in others, and the difference enters the result as
    a concentration difference that is really a drift.

    The per-sample apex positions detection already stored (``ppm_per_sample``)
    are the source, so the window follows the same centres the Signal List and
    the overlays draw. A sample detection marked as not aligned keeps the
    unshifted window rather than being moved on a centre nobody trusted.

    Shifts are clamped to ``max_shift_ppm`` — a third of the window by default.
    Beyond that the window would leave the multiplet, and a drift that large is
    a detection problem, not something integration should quietly absorb.

    Returns ``{sample: delta_ppm}``, empty when no peak carries per-sample
    centres.
    """
    anchor = None
    for pk in line_peaks or ():
        if pk.get('ppm_per_sample'):
            anchor = pk
            break
    if anchor is None:
        return {}
    cen = anchor.get('ppm_per_sample') or []
    ok = anchor.get('aligned_per_sample')
    ref = float(anchor.get('ppm', 0.0))
    win = multiplet_window(line_peaks)
    if max_shift_ppm is None:
        span = window_width_ppm(*win) if win else 0.0
        max_shift_ppm = max(span / 3.0, 0.003)
    out = {}
    for s, c in enumerate(cen):
        if c is None:
            continue
        if ok is not None and s < len(ok) and not ok[s]:
            continue
        d = float(c) - ref
        out[int(s)] = max(-max_shift_ppm, min(max_shift_ppm, d))
    return out


def subtract_blank(areas: dict, blank_areas: dict) -> dict:
    """Remove the mean blank area from every sample's area.

    What the blanks integrate in this window is what the window collects
    without the analyte — buffer, residual solvent, a neighbouring skirt. It is
    not the analyte and it is in every sample.

    The result may go negative, and it is returned negative. An analyte whose
    window integrates less than the blanks did is absent, and reporting it as
    zero would make absence indistinguishable from a very small presence.
    """
    vals = [float(v) for v in (blank_areas or {}).values()
            if v is not None and np.isfinite(v)]
    if not vals:
        return dict(areas or {})
    m = float(np.mean(vals))
    return {s: float(a) - m for s, a in (areas or {}).items()}


def integrate_across_samples(ppm, intensities, lo, hi, *, included=None,
                             local_baseline=True, shifts=None) -> dict:
    """The signal's area in every sample the caller allows.

    ``included`` is the list of sample indices to measure — blanks and excluded
    samples belong outside it, the same selection detection and annotation use.

    ``shifts`` moves each sample's window by its own drift, as
    ``window_shifts`` computes it. A sample with no shift keeps the window
    where it is.

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
    sh = dict(shifts or {})
    out = {}
    for s in idx:
        d = float(sh.get(int(s), 0.0))
        a = integrate_sample(ppm, arr[:, s], lo + d, hi + d,
                             local_baseline=local_baseline)
        if a is not None:
            out[int(s)] = a
    return out


def window_width_ppm(lo, hi) -> float:
    return abs(float(hi) - float(lo))


def integrate_signal(line_peaks, ppm, intensities, *, included=None,
                     local_baseline=True, align=False, blank_indices=None):
    """Everything the caller needs about one elected signal's area.

    Returns ``{'window': (lo, hi), 'width_ppm': w, 'areas': {sample: area},
    'shifts': {sample: delta}, 'blank_area': mean}``, or None when the signal
    has no lines to bound a window with. The window and the shifts travel with
    the areas so the report can show which region produced the number — a
    concentration whose window nobody can see is not auditable.

    ``blank_indices`` names the blank samples. They are integrated with the
    same window as everything else and their mean is subtracted, so the
    correction is measured in the region it corrects rather than assumed.
    """
    win = multiplet_window(line_peaks)
    if win is None:
        return None
    lo, hi = win
    shifts = window_shifts(line_peaks) if align else {}
    areas = integrate_across_samples(ppm, intensities, lo, hi,
                                     included=included,
                                     local_baseline=local_baseline,
                                     shifts=shifts)
    blank_area = float('nan')
    if blank_indices:
        blanks = integrate_across_samples(ppm, intensities, lo, hi,
                                          included=list(blank_indices),
                                          local_baseline=local_baseline,
                                          shifts=shifts)
        if blanks:
            blank_area = float(np.mean(list(blanks.values())))
            areas = subtract_blank(areas, blanks)
    return {
        'window': (lo, hi),
        'width_ppm': window_width_ppm(lo, hi),
        'areas': areas,
        'shifts': shifts,
        'blank_area': blank_area,
    }


# Below this, the window is not describing the signal and the correction stops
# being one. A window holding 60% of its lines is not a tight window -- it is a
# window in the wrong place or around lines that were split, and dividing by
# 0.60 turns a broken measurement into a broken measurement 1.7 times larger.
# Measured: at 1.7 Hz of per-sample drift, where the union splits lines, the
# correction pushed the mean absolute error from 11.8% to 15.7%; refusing it
# there keeps the gain where detection is clean and does no harm where it
# is not.
MIN_TRUSTED_CAPTURE = 0.75


def lorentzian_captured_fraction(line_ppms, weights, lo, hi, hwhm):
    """How much of a signal's true area a window of these bounds contains.

    A Lorentzian's tails never end. Integrating between finite limits always
    leaves some out, and how much depends on the window's width measured in
    half-widths -- so two signals integrated foot to foot lose different
    fractions, and the ratio between them carries the difference.

    That is what the absolute concentrations were wrong by. Measured on a
    synthetic study whose answer was known, every analyte came back about 7%
    high, uniformly, against an internal standard whose own window held a
    different share of its line.

    For one Lorentzian centred at p with half width at half maximum g, the
    fraction inside [lo, hi] is the difference of the cumulative:

        F = (1/pi) * (arctan((hi-p)/g) - arctan((lo-p)/g))

    and for a multiplet it is the weighted sum over its lines, the weights
    being the multiplicity's own intensity pattern. Inner lines lose almost
    nothing, outer lines lose a tail each, so the fraction depends on the
    multiplicity as well as on the width -- which is why correcting it needs
    the line positions and not just the window.

    The lineshape is the assumption. A real line is Lorentzian broadened by
    apodisation, so the correction is exact only to the extent the model holds;
    with no correction at all the error is the whole missing tail, which is
    larger. Returns None when it cannot be computed, and the caller then leaves
    the area alone rather than scaling it by a guess.

        Bharti, S.K. & Roy, R. (2012). Quantitative 1H NMR spectroscopy.
        TrAC Trends in Analytical Chemistry 35:5-26.
    """
    ppms = [float(p) for p in (line_ppms or ()) if p is not None]
    if not ppms or not hwhm or float(hwhm) <= 0:
        return None
    g = float(hwhm)
    lo, hi = (min(lo, hi), max(lo, hi))
    if hi <= lo:
        return None
    w = list(weights or ())
    if len(w) != len(ppms) or sum(w) <= 0:
        w = [1.0] * len(ppms)
    total = float(sum(w))
    f = 0.0
    for p, wi in zip(ppms, w):
        f += (wi / total) * (np.arctan((hi - p) / g)
                             - np.arctan((lo - p) / g)) / np.pi
    if not np.isfinite(f) or f <= 0:
        return None
    if f < MIN_TRUSTED_CAPTURE:
        return None
    return float(min(f, 1.0))


def pascal_weights(n_lines):
    """The intensity pattern of a first-order multiplet with ``n_lines``."""
    row = [1.0]
    for _ in range(max(int(n_lines), 1) - 1):
        row = [a + b for a, b in zip([0.0] + row, row + [0.0])]
    return row
