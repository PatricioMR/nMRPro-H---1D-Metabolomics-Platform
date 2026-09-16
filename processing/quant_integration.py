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


# How far past its outer lines a signal's window reaches, in linewidths
# (FWHM): three holds 90% of a Lorentzian's area on each side, and the
# capture correction then scales by a number close to one. Detection's own
# feet stop where the line meets the noise -- about 1.7 linewidths out for
# a line of SNR 30 -- and a window that narrow held 67% of betaine's
# singlet on a study whose answer was known, with the correction refused
# for it, so the number came out 45% low.
WINDOW_REACH_LINEWIDTHS = 3.0


def signal_window(line_peaks, *, linewidth_ppm, other_peaks=()):
    """Foot to foot over the signal's lines, let out to WINDOW_REACH_LINEWIDTHS
    past the outer lines, and never into a neighbour: a foreign detected
    peak bounds the window at the midpoint between it and the nearest
    line, less half a linewidth.

    Returns ``(lo, hi)`` or None without lines. ``other_peaks`` are
    detected peaks that are not the signal's own lines.
    """
    base = multiplet_window(line_peaks)
    if base is None:
        return None
    lo, hi = base
    apexes = sorted(float(p.get('ppm', 0.0)) for p in line_peaks)
    lw = abs(float(linewidth_ppm or 0.0))
    if lw > 0:
        lo = min(lo, apexes[0] - WINDOW_REACH_LINEWIDTHS * lw)
        hi = max(hi, apexes[-1] + WINDOW_REACH_LINEWIDTHS * lw)
        own = {id(p) for p in line_peaks}
        own_ranks = {p.get('rank') for p in line_peaks if p.get('rank') is not None}
        # A neighbour bounds the window whatever detection's feet said:
        # the feet of a line that runs into its neighbour's are the
        # neighbour's, and valine's doublet read 45% high with isoleucine's
        # line inside its window that way. Never narrower than half a
        # linewidth past the outer apex.
        for q in other_peaks or ():
            if id(q) in own or (q.get('rank') is not None
                                and q.get('rank') in own_ranks):
                continue
            qp = float(q.get('ppm', 0.0))
            if qp < apexes[0]:
                lo = max(lo, min(apexes[0] - lw / 2.0,
                                 (qp + apexes[0]) / 2.0 - lw / 2.0))
            elif qp > apexes[-1]:
                hi = min(hi, max(apexes[-1] + lw / 2.0,
                                 (qp + apexes[-1]) / 2.0 + lw / 2.0))
    return float(lo), float(hi)


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
                     local_baseline=True, align=False, blank_indices=None,
                     linewidth_ppm=None, other_peaks=()):
    """Everything the caller needs about one elected signal's area.

    Returns ``{'window': (lo, hi), 'width_ppm': w, 'areas': {sample: area},
    'shifts': {sample: delta}, 'blank_area': mean}``, or None when the signal
    has no lines to bound a window with. The window and the shifts travel with
    the areas so the report can show which region produced the number — a
    concentration whose window nobody can see is not auditable.

    ``blank_indices`` names the blank samples. They are integrated with the
    same window as everything else and their mean is subtracted, so the
    correction is measured in the region it corrects rather than assumed.

    Given ``linewidth_ppm`` the window is ``signal_window``'s -- let out
    past the outer lines, bounded by the neighbours in ``other_peaks`` --
    and the result says so (``'geometry': 'lines'``), so the caller can
    trust the capture correction on a window whose contents it knows.
    """
    if linewidth_ppm:
        win = signal_window(line_peaks, linewidth_ppm=linewidth_ppm,
                            other_peaks=other_peaks)
        geometry = 'lines'
    else:
        win = multiplet_window(line_peaks)
        geometry = 'feet'
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
        'geometry': geometry,
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


def lorentzian_captured_fraction(line_ppms, weights, lo, hi, hwhm,
                                 *, min_trusted=MIN_TRUSTED_CAPTURE):
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
    the area alone rather than scaling it by a guess. ``min_trusted`` is the
    fraction under which the window is not believed to describe the signal
    (MIN_TRUSTED_CAPTURE); a caller that built the window itself, from the
    lines and the neighbours, knows what it holds and may pass 0.

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
    if f < float(min_trusted):
        return None
    return float(min(f, 1.0))


def pascal_weights(n_lines):
    """The intensity pattern of a first-order multiplet with ``n_lines``."""
    row = [1.0]
    for _ in range(max(int(n_lines), 1) - 1):
        row = [a + b for a, b in zip([0.0] + row, row + [0.0])]
    return row


def refine_apex_ppm(ppm, trace, apex_ppm, *, search=2):
    """Where the line really peaks, between the two points either side of it.

    Detection reports an apex as the grid point holding the largest value, so
    the position it gives is out by up to half a point. That is harmless while
    the position is only used to place a window, and it stops being harmless
    the moment a model is evaluated AT the window's edges: the local baseline's
    trapezium is computed from the lineshape's height there, and a signal
    shifted by a fraction of its half width moves that height a long way.

    Measured on a study whose answer was known: two doublets, one integrated
    with a window whose edges fell on the grid and one whose did not, came back
    with a concentration ratio of 0.52 against a true 0.60. With the apices
    refined the same arithmetic gives 0.600.

    A parabola through the apex and its two neighbours, which is the standard
    estimator for a sampled peak and exact for a parabolic top. Returns the
    original position when there is nothing to fit or the fit does not put the
    vertex between the neighbours -- a vertex outside them means the three
    points are not describing a maximum.
    """
    x = np.asarray(ppm, dtype=float)
    y = np.asarray(trace, dtype=float)
    if x.size < 3 or y.size != x.size:
        return float(apex_ppm)
    i = int(np.argmin(np.abs(x - float(apex_ppm))))
    lo = max(1, i - int(search))
    hi = min(x.size - 2, i + int(search))
    if hi < lo:
        return float(apex_ppm)
    i = int(lo + np.argmax(y[lo:hi + 1]))
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    # On the reciprocal, because 1/L is exactly a parabola for a Lorentzian:
    #     1/L(x) = ((x - p)^2 + g^2) / (A g)
    # so three samples of it give the apex exactly, where three samples of L
    # itself only approximate the top. The same expression falls back to the
    # plain parabola when a sample is not positive, which is where the
    # reciprocal has no meaning.
    if a > 0 and b > 0 and c > 0:
        a, b, c = 1.0 / a, 1.0 / b, 1.0 / c
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(apex_ppm)
    delta = 0.5 * (a - c) / denom
    if not np.isfinite(delta) or abs(delta) > 0.5:
        return float(apex_ppm)
    step = float(x[i + 1] - x[i - 1]) / 2.0
    return float(x[i] + delta * step)
