# processing/quant_integration.py
"""The area of an elected signal, measured in each sample on its own.

One integration mode: the multiplet's own lines, each between its own feet.
The old engine carried five — multiplet, area, Lorentz fit, MCR component and
a database gap-fill — and every one of them was a second answer to the same
question, reached by a different path and reconciled nowhere.

Each line is a window of its own: its feet from detection, let out past the
apex, and never past a foreign detected peak. Lines whose windows touch make
one segment; a foreign peak sitting BETWEEN two lines keeps them apart, and
the gap it sits in is not integrated. A triplet with somebody's doublet in
its middle is measured on its three lines and not on the doublet -- the whole
multiplet, foot to foot, took the doublet in as if it were the analyte's.
What the segments leave outside (the tails between them and beyond the outer
lines) is the same known quantity the capture correction already puts back.

This is the qNMR practice of integrating the resolved part of a multiplet and
scaling by the intensity pattern the multiplicity dictates (Bharti & Roy
2012; Pauli et al. 2012), with the scale computed from the Lorentzian
lineshape rather than read off the binomial row: the lineshape says how much
of every line each segment holds.

    Bharti, S.K. & Roy, R. (2012). Quantitative 1H NMR spectroscopy. TrAC
    Trends in Analytical Chemistry 35:5-26.
    Pauli, G.F., Gödecke, T., Jaki, B.U., Lankin, D.C. (2012). Quantitative
    1H NMR: development and potential of an analytical method -- an update.
    J. Nat. Prod. 75:834-851.

Each sample is integrated separately, on its own trace. A mean spectrum has no
concentration, and a study's job is to tell the samples apart.
"""
from __future__ import annotations

import numpy as np

from processing.quant_deconvolution import baseline_under_line


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


def _bound_by_neighbours(apex, lo, hi, other_ppms, lw):
    """``(lo, hi)`` pulled in to the midpoint between ``apex`` and each
    foreign peak, less half a linewidth, and never nearer than half a
    linewidth to the apex: the feet of a line that runs into its neighbour's
    are the neighbour's, and valine's doublet read 45% high with
    isoleucine's line inside its window that way."""
    for qp in other_ppms:
        if qp < apex:
            lo = max(lo, min(apex - lw / 2.0, (qp + apex) / 2.0 - lw / 2.0))
        elif qp > apex:
            hi = min(hi, max(apex + lw / 2.0, (qp + apex) / 2.0 + lw / 2.0))
    return lo, hi


def _valley(ppm, y, inner, outer):
    """The ppm where the trace is lowest between ``inner`` and ``outer``,
    or ``outer`` when the stretch holds no points."""
    ppm = np.asarray(ppm, dtype=float)
    lo, hi = (inner, outer) if inner <= outer else (outer, inner)
    m = (ppm >= lo) & (ppm <= hi)
    if m.sum() < 1:
        return float(outer)
    idx = np.flatnonzero(m)
    return float(ppm[idx[int(np.argmin(np.asarray(y, dtype=float)[idx]))]])


def foot_at_valley(apex, lo, hi, free_lo, free_hi, lw, trace):
    """A foot a neighbour pulled in moves to the valley between the two
    signals: the lowest point of ``trace`` between half a linewidth from
    the apex (as near as a bound ever comes) and where the neighbour's bound
    put it.

    The bound is the midpoint less half a linewidth, and against a large
    neighbour that is on the neighbour's flank. A local baseline drawn
    through two flanks runs above the signal between them, and a weak
    signal between two strong ones was integrated negative that way (the
    fill in the quantification stage showed the whole window below the
    baseline). The valley is where each signal contributes least, and the
    baseline through the valleys is the floor the signal stands on. A foot
    nobody pulled in stands where it was; a trace is needed to find a valley.
    """
    if trace is None or lw <= 0:
        return lo, hi
    ppm, y = trace
    if lo > free_lo:
        lo = _valley(ppm, y, apex - lw / 2.0, lo)
    if hi < free_hi:
        hi = _valley(ppm, y, apex + lw / 2.0, hi)
    return lo, hi


def _foreign_ppms(line_peaks, other_peaks):
    own = {id(p) for p in line_peaks}
    own_ranks = {p.get('rank') for p in line_peaks if p.get('rank') is not None}
    return [float(q.get('ppm', 0.0)) for q in other_peaks or ()
            if id(q) not in own and not (q.get('rank') is not None
                                         and q.get('rank') in own_ranks)]


def signal_segments(line_peaks, *, linewidth_ppm, other_peaks=(),
                    trace=None):
    """One window per line of the signal, merged where they touch.

    A line's window is its own feet let out to WINDOW_REACH_LINEWIDTHS past
    its apex, and bounded by every foreign detected peak the way
    ``signal_window`` bounds the outer lines: at the midpoint less half a
    linewidth, never nearer than half a linewidth to the apex. Windows that
    overlap are one segment. A foreign peak between two lines bounds them
    both, and where the bounds leave a gap the gap is out.

    Given ``trace`` (``(ppm, y)``, the spectrum the areas are read on), a
    foot a neighbour bounded moves to the valley between the two signals
    (``foot_at_valley``).

    Returns ``[(lo, hi), ...]`` in ascending ppm, or an empty list without
    lines. Without a linewidth the feet stand and the segments are the
    lines' own feet. ``other_peaks`` are detected peaks that are not the
    signal's own lines.
    """
    lw = abs(float(linewidth_ppm or 0.0))
    foreign = _foreign_ppms(line_peaks or (), other_peaks) if lw > 0 else []
    wins = []
    for pk in line_peaks or ():
        apex = float(pk.get('ppm', 0.0))
        a = float(pk.get('start_ppm', apex))
        b = float(pk.get('end_ppm', apex))
        lo, hi = min(a, b), max(a, b)
        if lw > 0:
            lo = min(lo, apex - WINDOW_REACH_LINEWIDTHS * lw)
            hi = max(hi, apex + WINDOW_REACH_LINEWIDTHS * lw)
            free_lo, free_hi = lo, hi
            lo, hi = _bound_by_neighbours(apex, lo, hi, foreign, lw)
            lo, hi = foot_at_valley(apex, lo, hi, free_lo, free_hi, lw, trace)
        wins.append((float(lo), float(hi)))
    wins.sort()
    merged: list = []
    for lo, hi in wins:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def foreign_between_lines(line_peaks, other_peaks) -> list:
    """The foreign detected peaks that lie between the signal's outer apices,
    in ppm: the ones that cut its window, said on the row."""
    apexes = [float(p.get('ppm', 0.0)) for p in line_peaks or ()]
    if len(apexes) < 2:
        return []
    lo, hi = min(apexes), max(apexes)
    return sorted(q for q in _foreign_ppms(line_peaks, other_peaks)
                  if lo < q < hi)


def signal_window(line_peaks, *, linewidth_ppm, other_peaks=(), trace=None):
    """The span of ``signal_segments``: foot to foot over the signal's
    lines, let out to WINDOW_REACH_LINEWIDTHS past the outer lines, and
    never into a neighbour. Returns ``(lo, hi)`` or None without lines.
    """
    segs = signal_segments(line_peaks, linewidth_ppm=linewidth_ppm,
                           other_peaks=other_peaks, trace=trace)
    if not segs:
        return None
    return float(segs[0][0]), float(segs[-1][1])


def _window_mask(ppm, lo, hi):
    ppm = np.asarray(ppm, dtype=float)
    if lo > hi:
        lo, hi = hi, lo
    return (ppm >= lo) & (ppm <= hi)


# The local baseline is read on a stretch this many linewidths wide, its
# median the level: one point is one noise draw, the lowest point of a
# stretch is a noise trough (ADR-020 measured the bias that puts on a clean
# window), the median of a stretch is neither.
BASELINE_STRETCH_LINEWIDTHS = 2.0
# How far past a window's foot the walk to the baseline may go before the
# region is called crowded and the lowest level on the way is taken instead.
ANCHOR_MAX_REACH_LINEWIDTHS = 20.0


def _running_median(y, n):
    y = np.asarray(y, dtype=float)
    n = max(int(n) | 1, 1)
    if n == 1 or y.size < n:
        return y.copy()
    from scipy.ndimage import median_filter
    return median_filter(y, size=n, mode='nearest')


def _cluster_edge(foot, peaks, lw, side, trace):
    """Where the baseline is read beyond ``foot`` on ``side`` (-1 down in
    ppm, +1 up): the centre of the first stretch no detected peak reaches.
    Every peak within WINDOW_REACH_LINEWIDTHS of the walk pushes it on, so
    it crosses the whole cluster of overlapping signals the window sits in
    and stops where the trace has come back to the floor they all stand on.
    Past ANCHOR_MAX_REACH_LINEWIDTHS the region is crowded all the way; the
    stretch with the lowest median on the way is the nearest thing to a
    floor it has. None when that is needed and there is no trace."""
    reach = WINDOW_REACH_LINEWIDTHS * lw
    stretch = BASELINE_STRETCH_LINEWIDTHS * lw
    limit = ANCHOR_MAX_REACH_LINEWIDTHS * lw
    edge = float(foot)
    while True:
        if side < 0:
            blockers = [q for q in peaks
                        if edge - stretch - reach < q < edge + reach]
        else:
            blockers = [q for q in peaks
                        if edge - reach < q < edge + stretch + reach]
        nxt = ((min(blockers) - reach) if side < 0
               else (max(blockers) + reach)) if blockers else edge
        # A free foot stands exactly one reach from its apex, and rounding
        # can put the apex a hair inside it: a blocker that moves nothing
        # is not one.
        if side * (nxt - edge) <= 1e-9 * lw:
            return edge + side * stretch / 2.0
        edge = nxt
        if abs(edge - foot) > limit:
            break
    if trace is None:
        return None
    ppm, y = np.asarray(trace[0], dtype=float), np.asarray(trace[1], dtype=float)
    a, b = sorted((float(foot), float(foot) + side * limit))
    m = (ppm >= a) & (ppm <= b)
    if m.sum() < 1:
        return None
    step = float(np.median(np.abs(np.diff(ppm)))) if ppm.size > 1 else 0.0
    n = int(round(stretch / step)) if step > 0 else 1
    idx = np.flatnonzero(m)
    lo_i, hi_i = max(idx[0] - n, 0), min(idx[-1] + n + 1, ppm.size)
    med = _running_median(y[lo_i:hi_i], n)[idx - lo_i]
    return float(ppm[idx[int(np.argmin(med))]])


def baseline_anchors(lo, hi, peak_ppms, linewidth_ppm, *, trace=None):
    """``(lo_centre, hi_centre, half_width)``: where the local baseline of
    the window ``[lo, hi]`` is read, or None without a linewidth.

    Not at the window's own ends. A window between two signals has its ends
    on their tails -- on the neighbour's flank, or in the valley between
    them, which in an overlap is the sum of both tails and not the floor --
    and a line through them passed over the signal and took it away:
    measured on a line two widths beside a neighbour five times its size,
    76% of it (the whole of it at twenty times). The baseline is read where
    the cluster of overlapping signals ends (``_cluster_edge``), and what a
    neighbour puts inside the window is the model's to take out
    (``fraction_in_window``, the joint fit), not a straight line's.

    ``peak_ppms`` are every detected peak's apex, the window's own lines
    among them. A window with no peak within reach of its ends keeps its
    ends, as before, but reads them on a stretch rather than a point.
    """
    lw = abs(float(linewidth_ppm or 0.0))
    if lw <= 0:
        return None
    peaks = [float(p) for p in peak_ppms or () if p is not None]
    lo, hi = min(lo, hi), max(lo, hi)
    a = _cluster_edge(lo, peaks, lw, -1, trace)
    b = _cluster_edge(hi, peaks, lw, +1, trace)
    if a is None:
        a = lo - BASELINE_STRETCH_LINEWIDTHS * lw / 2.0
    if b is None:
        b = hi + BASELINE_STRETCH_LINEWIDTHS * lw / 2.0
    return float(a), float(b), float(BASELINE_STRETCH_LINEWIDTHS * lw / 2.0)


def cluster_baseline(ppm, X, peak_ppms, linewidth_ppm):
    """``(n_points, n_samples)``: the local baseline under the whole
    spectrum, for the measurements that fit shapes rather than integrate a
    window (the cascade, the joint fit).

    The same rule as ``baseline_anchors``, drawn once for everything: the
    detected peaks are gathered into clusters -- peaks with no free stretch
    between them -- and the baseline is read on the free stretch just past
    each cluster's ends (median of the stretch, clipped at zero), joined by
    straight lines. A cluster longer than two ANCHOR_MAX_REACH_LINEWIDTHS
    is crowded all the way; it is read again at the lowest stretch of every
    such length inside it, the floor it has. A shape fitted on the trace
    less this line is fitted on its signal, not on the pedestal the
    overlapping tails and a broad envelope build under a crowded region:
    on a cell extract that pedestal stood at 20-80% of the lines' height
    in the sugar region and was fitted as analyte.

    None without a linewidth or a peak.
    """
    lw = abs(float(linewidth_ppm or 0.0))
    peaks = sorted(float(p) for p in peak_ppms or ()
                   if p is not None and np.isfinite(float(p)))
    if lw <= 0 or not peaks:
        return None
    ppm = np.asarray(ppm, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    reach = WINDOW_REACH_LINEWIDTHS * lw
    stretch = BASELINE_STRETCH_LINEWIDTHS * lw
    half = stretch / 2.0
    limit = ANCHOR_MAX_REACH_LINEWIDTHS * lw
    clusters, start, prev = [], peaks[0], peaks[0]
    for q in peaks[1:]:
        if q - prev < 2.0 * reach + stretch:
            prev = q
        else:
            clusters.append((start, prev))
            start = prev = q
    clusters.append((start, prev))
    order = np.argsort(ppm)
    xs, mean = ppm[order], X[order].mean(axis=1)
    step = float(np.median(np.diff(xs))) if xs.size > 1 else 0.0
    n_med = int(round(stretch / step)) if step > 0 else 1
    centres = []
    for a, b in clusters:
        centres += [a - reach - half, b + reach + half]
        lo, hi = a - reach, b + reach
        if hi - lo > 2.0 * limit:
            for s0 in np.arange(lo + limit / 2.0, hi - limit, limit):
                m = (xs >= s0) & (xs <= s0 + limit)
                if m.sum() > n_med:
                    idx = np.flatnonzero(m)
                    med = _running_median(mean[idx], n_med)
                    centres.append(float(xs[idx[int(np.argmin(med))]]))
    centres = sorted(c for c in set(centres) if xs[0] <= c <= xs[-1])
    if not centres:
        return None
    levels = np.empty((len(centres), X.shape[1]))
    for k, c in enumerate(centres):
        m = np.abs(ppm - c) <= half
        if not m.any():
            m = np.zeros(ppm.shape, dtype=bool)
            m[int(np.argmin(np.abs(ppm - c)))] = True
        levels[k] = np.maximum(np.median(X[m], axis=0), 0.0)
    B = np.empty_like(X)
    for s in range(X.shape[1]):
        B[:, s] = np.interp(ppm, centres, levels[:, s])
    return B


def anchor_levels(ppm, y, anchors):
    """``((lo_ppm, level), (hi_ppm, level))``: the trace's median on each
    anchor's stretch, clipped at zero as the window ends always were -- a
    negative level would tilt the line up into the signal. None when a
    stretch falls off the spectrum."""
    if anchors is None:
        return None
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    c_lo, c_hi, half = (float(v) for v in anchors)
    out = []
    for c in (c_lo, c_hi):
        m = (ppm >= c - half) & (ppm <= c + half)
        if m.sum() < 1:
            if c < ppm.min() or c > ppm.max():
                return None
            m = np.zeros(ppm.shape, dtype=bool)
            m[int(np.argmin(np.abs(ppm - c)))] = True
        out.append((c, max(float(np.median(y[m])), 0.0)))
    return tuple(out)


def shift_anchors(anchors, d):
    if anchors is None or not d:
        return anchors
    c_lo, c_hi, half = (float(v) for v in anchors)
    return c_lo + d, c_hi + d, half


def local_baseline_line(seg, x=None, levels=None):
    """The local baseline of a window. Given ``levels`` (``anchor_levels``)
    and the window's ``x``, the straight line through the two anchors;
    otherwise the line joining the window's two ends, each clipped at zero
    so a negative foot cannot tilt it up into the signal. ``seg`` is the
    trace inside the window in ascending ppm."""
    seg = np.asarray(seg, dtype=float)
    if seg.size < 2:
        return np.zeros_like(seg)
    if levels is not None and x is not None:
        (c0, v0), (c1, v1) = levels
        x = np.asarray(x, dtype=float)
        if c1 == c0:
            return np.full(seg.shape, (v0 + v1) / 2.0)
        return v0 + (v1 - v0) * (x - c0) / (c1 - c0)
    return np.linspace(max(float(seg[0]), 0.0), max(float(seg[-1]), 0.0),
                       seg.size)


def baseline_and_signal(seg, *, local_baseline=True, clip=True, x=None,
                        levels=None):
    """``(baseline, signal)`` for a window's trace: the baseline the
    integration subtracts (the local line, or zero) and what is left above
    it, clipped at zero -- the part that is area. ``clip=False`` keeps the
    signed difference, for the one reader that needs noise to stay
    symmetric (the detection limit). ``x``/``levels`` draw the line through
    the anchors (``local_baseline_line``)."""
    seg = np.asarray(seg, dtype=float)
    base = (local_baseline_line(seg, x=x, levels=levels) if local_baseline
            else np.zeros_like(seg))
    diff = seg - base
    return base, (np.maximum(diff, 0.0) if clip else diff)


def baseline_under(x, ppm, y, anchors, *, local_baseline=True):
    """The baseline the integration subtracted, at ``x`` (ascending or not),
    for a trace ``y`` on ``ppm``: what the stage plot and the PDF draw, so
    they draw the line the number was measured against."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    seg = np.interp(x[order], np.asarray(ppm, float)[np.argsort(ppm)],
                    np.asarray(y, float)[np.argsort(ppm)])
    lv = anchor_levels(ppm, y, anchors) if local_baseline else None
    base = np.empty_like(x)
    base[order], _ = baseline_and_signal(seg, local_baseline=local_baseline,
                                         x=x[order], levels=lv)
    return base


def below_baseline_share(ppm, y, lo, hi, *, local_baseline=True,
                         segments=None, anchors=None) -> float | None:
    """The fraction of the integrated points where the trace lies under the
    baseline -- what the clip in ``integrate_sample`` set to zero. Near zero
    for a window on a signal; large for a window whose feet stand on
    somebody else's flanks. None when the window holds too few points."""
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    m = _window_mask(ppm, lo, hi)
    if m.sum() < 2:
        return None
    x, seg = ppm[m], y[m]
    order = np.argsort(x)
    x, seg = x[order], seg[order]
    base, _ = baseline_and_signal(
        seg, local_baseline=local_baseline, x=x,
        levels=anchor_levels(ppm, y, anchors) if local_baseline else None)
    pieces = ([(min(a, b), max(a, b)) for a, b in segments]
              if segments else [(min(lo, hi), max(lo, hi))])
    k = np.zeros(x.shape, dtype=bool)
    for a, b in pieces:
        k |= (x >= a) & (x <= b)
    if k.sum() < 2:
        return None
    return float(np.mean(seg[k] < base[k]))


def integrate_sample(ppm, y, lo, hi, *, local_baseline=True, segments=None,
                     clip=True, anchors=None):
    """Area of one sample between ``lo`` and ``hi`` ppm.

    ``local_baseline`` subtracts a straight line before integrating — the
    qNMR convention, and what removes the broad component a multiplet
    happens to be sitting on. Given ``anchors`` (``baseline_anchors``) the
    line runs through the trace's level where the cluster of signals the
    window sits in ends; without them, through the window's two endpoints.
    Either level is clipped at zero first, so a negative excursion cannot
    tilt the baseline up into the signal.

    ``segments`` sums only the points inside those pieces of the window, one
    baseline still drawn under the whole window: a foreign peak
    between two lines cuts the window around it, and a baseline drawn through
    the cut would be drawn through that peak's flank -- measured, it took the
    area negative. The window's feet are where the signal is not; the cut is
    where somebody else is.

    Only what stands ABOVE the baseline is area: where the trace runs
    below it -- a foot on a neighbour's flank, a dip between two signals,
    noise going negative -- the point counts zero (``baseline_and_signal``).
    The analyst's rule, stated after a weak signal between two strong ones
    came out negative with the window shaded wholly under the baseline:
    nothing above the spectrum is signal. The cost is known: a window over
    pure noise integrates to a small positive number (about 0.4 sigma per
    point) instead of about zero, and ``below_baseline_share`` says how
    much of the window that rule silenced, so the row can show it. That
    cost is why the detection limit reads the signed integral
    (``clip=False``): Currie's limits are built from noise that goes both
    ways, and a clipped noise window would stand above them every time.

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
    _base, seg = baseline_and_signal(
        seg, local_baseline=local_baseline, clip=clip, x=x,
        levels=anchor_levels(ppm, y, anchors) if local_baseline else None)
    pieces = ([(min(a, b), max(a, b)) for a, b in segments]
              if segments else [(min(lo, hi), max(lo, hi))])
    total, any_piece = 0.0, False
    for a, b in pieces:
        k = (x >= a) & (x <= b)
        if k.sum() < 2:
            continue
        any_piece = True
        try:
            total += float(np.trapezoid(seg[k], x=x[k]))
        except AttributeError:                       # numpy < 2
            total += float(np.trapz(seg[k], x=x[k]))
    return total if any_piece else None


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
                             local_baseline=True, shifts=None,
                             segments=None, anchors=None) -> dict:
    """The signal's area in every sample the caller allows.

    ``included`` is the list of sample indices to measure — blanks and excluded
    samples belong outside it, the same selection detection and annotation use.

    ``shifts`` moves each sample's window by its own drift, as
    ``window_shifts`` computes it. A sample with no shift keeps the window
    where it is.

    ``segments`` are the pieces of the window the area is summed over (what
    ``signal_segments`` cut around the lines), one baseline under all of
    them; ``lo``/``hi`` are then the span of the segments. ``anchors``
    (``baseline_anchors``) are where that baseline is read, moved with each
    sample's window.

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
    segs = [(float(a), float(b)) for a, b in (segments or ())]
    if segs:
        lo, hi = min(a for a, _ in segs), max(b for _, b in segs)
    out = {}
    for s in idx:
        d = float(sh.get(int(s), 0.0))
        a = integrate_sample(ppm, arr[:, s], lo + d, hi + d,
                             local_baseline=local_baseline,
                             segments=[(a_ + d, b_ + d) for a_, b_ in segs]
                             or None, anchors=shift_anchors(anchors, d))
        if a is not None:
            out[int(s)] = a
    return out


def window_width_ppm(lo, hi) -> float:
    return abs(float(hi) - float(lo))


def _mean_trace(ppm, intensities, included):
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = (list(range(arr.shape[1])) if included is None
            else [i for i in included if 0 <= i < arr.shape[1]])
    if not cols:
        return None
    return np.asarray(ppm, dtype=float), arr[:, cols].mean(axis=1)


def integrate_signal(line_peaks, ppm, intensities, *, included=None,
                     local_baseline=True, align=False, blank_indices=None,
                     linewidth_ppm=None, other_peaks=()):
    """Everything the caller needs about one elected signal's area.

    Returns ``{'window': (lo, hi), 'segments': [(lo, hi), ...],
    'width_ppm': w, 'areas': {sample: area}, 'shifts': {sample: delta},
    'blank_area': mean, 'foreign_inside': [ppm, ...],
    'below_baseline_share': f}``, or None when the
    signal has no lines to bound a window with. The window and the shifts
    travel with the areas so the report can show which region produced the
    number — a concentration whose window nobody can see is not auditable.

    ``blank_indices`` names the blank samples. They are integrated with the
    same window as everything else and their mean is subtracted, so the
    correction is measured in the region it corrects rather than assumed.

    Given ``linewidth_ppm`` the segments are ``signal_segments``' -- one
    per line, let out past the apex, bounded by the neighbours in
    ``other_peaks``, merged where they touch -- and the result says so
    (``'geometry': 'lines'``), so the caller can trust the capture
    correction on windows whose contents it knows. ``foreign_inside`` names
    the neighbours found between the signal's own lines: the ones that kept
    the segments apart. ``window`` is the segments' span.
    ``below_baseline_share`` is the fraction of the window (on the mean of
    the measured samples) where the trace ran under the baseline and
    counted nothing. ``baseline_anchors`` are where the local baseline was
    read (``baseline_anchors``), None without one or without a linewidth.
    """
    mean_trace = _mean_trace(ppm, intensities, included)
    if linewidth_ppm:
        # The feet are placed on the spectrum the areas are read on: the
        # mean of the samples measured, blanks and excluded outside it.
        segs = signal_segments(line_peaks, linewidth_ppm=linewidth_ppm,
                               other_peaks=other_peaks, trace=mean_trace)
        geometry = 'lines'
        foreign = foreign_between_lines(line_peaks, other_peaks)
    else:
        win = multiplet_window(line_peaks)
        segs = [win] if win else []
        geometry = 'feet'
        foreign = []
    if not segs:
        return None
    lo, hi = float(segs[0][0]), float(segs[-1][1])
    anchors = None
    if local_baseline and linewidth_ppm:
        anchors = baseline_anchors(
            lo, hi, [pk.get('ppm') for pk in list(line_peaks or ())
                     + list(other_peaks or ())],
            linewidth_ppm, trace=mean_trace)
    shifts = window_shifts(line_peaks) if align else {}
    areas = integrate_across_samples(ppm, intensities, lo, hi,
                                     included=included,
                                     local_baseline=local_baseline,
                                     shifts=shifts, segments=segs,
                                     anchors=anchors)
    blank_area = float('nan')
    if blank_indices:
        blanks = integrate_across_samples(ppm, intensities, lo, hi,
                                          included=list(blank_indices),
                                          local_baseline=local_baseline,
                                          shifts=shifts, segments=segs,
                                          anchors=anchors)
        if blanks:
            blank_area = float(np.mean(list(blanks.values())))
            areas = subtract_blank(areas, blanks)
    return {
        'window': (lo, hi),
        'segments': [(float(a), float(b)) for a, b in segs],
        'width_ppm': window_width_ppm(lo, hi),
        'areas': areas,
        'shifts': shifts,
        'blank_area': blank_area,
        'geometry': geometry,
        'foreign_inside': foreign,
        'baseline_anchors': anchors,
        'below_baseline_share': (
            below_baseline_share(mean_trace[0], mean_trace[1], lo, hi,
                                 local_baseline=local_baseline, segments=segs,
                                 anchors=anchors)
            if mean_trace is not None else None),
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
                                 *, min_trusted=MIN_TRUSTED_CAPTURE,
                                 segments=None, anchors=None):
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

    ``segments`` replaces the one window with several disjoint ones, the
    fraction being the sum over them: a triplet integrated on its outer lines
    alone holds half of itself, and dividing by that half is what puts the
    middle line back.

    ``anchors`` (``(lo_ppm, hi_ppm)``, where a local baseline was read)
    take out what that line removes of the signal's own tails: the model
    describes what was integrated. Left out, a clean line read 0.92 of
    itself (ADR-016) with the baseline at the window's ends; read beyond
    the cluster the trapezium is small, but it is not zero.

        Bharti, S.K. & Roy, R. (2012). Quantitative 1H NMR spectroscopy.
        TrAC Trends in Analytical Chemistry 35:5-26.
    """
    ppms = [float(p) for p in (line_ppms or ()) if p is not None]
    if not ppms or not hwhm or float(hwhm) <= 0:
        return None
    g = float(hwhm)
    segs = [(min(float(a), float(b)), max(float(a), float(b)))
            for a, b in (segments or ())] or [(min(lo, hi), max(lo, hi))]
    segs = [(a, b) for a, b in segs if b > a]
    if not segs:
        return None
    w = list(weights or ())
    if len(w) != len(ppms) or sum(w) <= 0:
        w = [1.0] * len(ppms)
    total = float(sum(w))
    f = 0.0
    for p, wi in zip(ppms, w):
        for a, b in segs:
            f += (wi / total) * (np.arctan((b - p) / g)
                                 - np.arctan((a - p) / g)) / np.pi
        if anchors is not None:
            f -= (wi / total) * baseline_under_line(p, segs, anchors[:2], g)
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


def pattern_lines(predicted_ppms, found_ppms, *, weights=None, tolerance=None):
    """The multiplet's whole pattern, with the lines detection found at their
    apices and the rest where the annotation predicted them.

    Returns ``(ppms, weights, found_mask, share)``: one entry per pattern
    line, ``found_mask`` saying which carry a detected line, ``share`` the
    fraction of the pattern's intensity those hold. A triplet found on its
    middle line alone has ``share`` 0.5 -- and its area, divided by the
    fraction the window holds of all THREE lines, is the triplet's. That is
    the qNMR practice of integrating the resolved part of a multiplet and
    scaling by the intensity pattern (Pauli et al. 2012), with the scale
    computed from the lineshape.

    Each predicted line takes the nearest found line within ``tolerance``
    (half the pattern's smallest spacing by default, the same rule the
    adapter matches lines with), each found line at most once. Predicted
    lines fewer than the found ones -- nothing to scale by -- give the found
    lines as they are, with ``share`` 1.
    """
    pred = [float(v) for v in (predicted_ppms or ()) if v is not None]
    found = sorted(float(v) for v in (found_ppms or ()) if v is not None)
    if len(pred) <= len(found) or not found:
        return found, list(pascal_weights(len(found))), [True] * len(found), 1.0
    w = list(weights or ())
    if len(w) != len(pred) or sum(w) <= 0:
        w = pascal_weights(len(pred))
    if tolerance is None:
        gaps = sorted(abs(b - a) for a, b in zip(pred, pred[1:]) if b != a)
        tolerance = (gaps[0] / 2.0) if gaps else 0.002
    order = sorted(range(len(pred)), key=lambda i: pred[i])
    pred = [pred[i] for i in order]
    w = [w[i] for i in order]
    ppms, mask, used = [], [], set()
    for p in pred:
        best, bd = None, float(tolerance)
        for j, f in enumerate(found):
            if j in used:
                continue
            d = abs(f - p)
            if d < bd:
                bd, best = d, j
        if best is None:
            ppms.append(p)
            mask.append(False)
        else:
            ppms.append(found[best])
            mask.append(True)
            used.add(best)
    total = float(sum(w))
    share = float(sum(wi for wi, m in zip(w, mask) if m)) / total
    return ppms, w, mask, share
