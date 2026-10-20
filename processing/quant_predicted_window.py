# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/quant_predicted_window.py
"""Where the database predicts a signal, and what the spectrum holds there.

A winner of the election with no detected peak is measured on the window its
reference lines predict: placed at the study's drift, integrated against a
local baseline, and reported at one dot with the reason said
(``quant_engine._window_integral_row``). This module holds the pieces of that
measurement -- the window, the positions it slides over, whether it falls in
an excluded region, the noise on its integral -- and the measure of how
crowded a window's surroundings are (interference), which every quantified
row reports.
"""
from __future__ import annotations

import numpy as np


def integration_noise(sigma, dppm, n_points) -> float:
    """The uncertainty on an integral of ``n_points`` samples of white noise.

    Each point carries σ and the trapezoid sums them over a spacing Δppm, so
    the noise on the area grows as √N rather than N: random error partially
    cancels across the window. Using σ·Δppm·N instead would overstate it in
    proportion to how wide the window is, which would make a broad multiplet
    look noisier than it is for no reason but its width.
    """
    n = max(int(n_points), 0)
    if n <= 0 or not sigma or float(sigma) <= 0:
        return float('nan')
    return float(sigma) * abs(float(dppm)) * float(np.sqrt(n))


def predicted_window(predicted_ppms, *, linewidth_ppm, drift_ppm=0.0):
    """The window a signal would occupy, from the positions the search used.

    Its width is the signal's own: the span of the predicted lines plus one
    linewidth on each side, so the outer lines' tails fall inside.

    The drift is NOT added to the width. Where an analyte sits and how wide it
    is are different questions. Padding by the drift made the window 25.7 Hz
    for a 3.2 Hz line on the reference study -- eight times the linewidth,
    180 points instead of 45 -- integrating 25 Hz of noise and neighbours to
    measure a 3 Hz signal.

    The drift belongs to WHERE to look, which is ``drift_offsets``.

    Returns None when there are no predicted positions to build one from.
    """
    ppms = [float(v) for v in (predicted_ppms or ()) if v is not None]
    if not ppms:
        return None
    m = abs(float(linewidth_ppm or 0.0))
    return min(ppms) - m, max(ppms) + m


def drift_offsets(drift_ppm, linewidth_ppm, *, max_steps=8):
    """Positions to try the window at, to cover where the analyte may have gone.

    A resonance is not at its database position in any real sample; it is near
    it. So the window slides over the plausible range instead of growing to
    cover it, and the caller keeps the position with the most signal. Sliding
    asks "is it anywhere here?" while keeping a single window's width;
    growing asks the same question and integrates the whole range.

    Stepped by half a linewidth, which is fine enough that a line cannot fall
    between two positions, and capped so a study with a large drift residual
    does not turn one measurement into hundreds.

    Always includes 0.0, so an analyte that has not drifted is measured exactly
    where the database puts it.
    """
    d = abs(float(drift_ppm or 0.0))
    lw = abs(float(linewidth_ppm or 0.0))
    if d <= 0 or lw <= 0:
        return [0.0]
    step = lw / 2.0
    n = min(int(round(d / step)), int(max_steps))
    if n <= 0:
        return [0.0]
    return [k * step for k in range(-n, n + 1)]


def window_is_excluded(lo, hi, excluded_regions) -> bool:
    """Whether the window overlaps a region detection was told to ignore.

    Nothing measured in an excluded region belongs to the analyte -- in the
    water window it is the suppressed solvent, which is orders of magnitude
    above any metabolite. Integrating it would report the solvent as the
    analyte: on the reference study 4-pyridoxic acid, guanosine, saligenin
    and trans-4-hydroxy-L-proline all had their windows inside 4.50-5.00 ppm.
    """
    for r_lo, r_hi in (excluded_regions or ()):
        if float(lo) < float(r_hi) and float(r_lo) < float(hi):
            return True
    return False


def project_noise(peaks) -> float:
    """The noise level detection estimated, as the median over the peaks."""
    n = [float(p.get('noise')) for p in (peaks or ())
         if p.get('noise') and float(p['noise']) > 0]
    return float(np.median(n)) if n else float('nan')


def project_drift(metabolites) -> float:
    """How far resonances actually sat from their database positions here.

    The median absolute Δppm over the signals the search matched — measured
    drift, not an assumed tolerance. Falls back to zero when nothing was
    matched, which makes the window the narrowest it can be rather than
    inventing a width.
    """
    d = []
    for met in metabolites or ():
        for mp in met.get('matched_peaks', []) or ():
            v = mp.get('ppm_diff')
            if v is not None and np.isfinite(float(v)):
                d.append(abs(float(v)))
    return float(np.median(d)) if d else 0.0


# How far either side of the window to look when estimating the local
# scatter, in linewidths. Wide enough to contain the baseline structure the
# window sits on, narrow enough that it is still the same neighbourhood.
LOCAL_NOISE_BAND_LINEWIDTHS = 8.0

# How far above the blank a window's surroundings may read before the row
# says the region is crowded. Two: the flanking band holding twice the
# scatter of a signal-free region is not noise any more.
INTERFERENCE_NOTE = 2.0

# 1 / Phi^-1(3/4). Turns a median absolute deviation into the standard
# deviation it corresponds to for normally distributed values.
MAD_TO_SIGMA = 1.4826


def local_noise_sigma(ppm, intensity, lo, hi, *, linewidth_ppm,
                      global_sigma=None, band_linewidths=None):
    """How much of what surrounds this window is not noise: INTERFERENCE.

    What a neighbour's skirt does to a measurement belongs to selectivity
    and costs accuracy. Measured on the reference study: taurine, the second
    most concentrated analyte, had this come back at 1,035,234 where the
    spectrum's noise is 5,349 -- 193 times -- because the band flanking its
    window holds a skirt 16.9 million counts tall and no detected peak at
    all. A region that crowded cannot be measured down to the noise floor,
    and the analyst has to see that; it is the ratio to the blank that says
    it.

    Six analytes claiming the histidine imidazole region at 7.0926-7.1025 ppm
    show why it is measured per window: the window width and the global sigma
    are the same for all of them, and only the surroundings tell the region
    is contested.

    Measured here as a robust scale over a band around the window, after
    removing the local linear trend. The median absolute deviation is used
    rather than the standard deviation because a band in a crowded region is
    mostly peak, and a plain standard deviation would report the peaks
    themselves as noise:

        Rousseeuw, P.J. & Croux, C. (1993). Alternatives to the median absolute
        deviation. Journal of the American Statistical Association
        88(424):1273-1283.

    Where more than half the band really is signal the MAD rises, and it should.

    The band flanks the window and excludes it, so what is being measured
    cannot raise the reading against itself.

    Never returns less than ``global_sigma``: a region cannot be quieter
    than the instrument, so the ratio to the blank starts at one.
    """
    x = np.asarray(ppm, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if y.ndim > 1:
        y = y.mean(axis=1)
    floor = (float(global_sigma)
             if global_sigma and np.isfinite(global_sigma) else None)
    if x.size != y.size or x.size < 4:
        return floor
    lw = abs(float(linewidth_ppm or 0.0))
    if lw <= 0:
        return floor
    k = float(band_linewidths if band_linewidths is not None
              else LOCAL_NOISE_BAND_LINEWIDTHS)
    lo, hi = (min(lo, hi), max(lo, hi))
    # The band flanks the window and excludes it: an estimate that contained
    # what is being measured would read the residual water, orders of
    # magnitude above any metabolite, as its own surroundings.
    m = (((x >= lo - k * lw) & (x < lo))
         | ((x > hi) & (x <= hi + k * lw)))
    if m.sum() < 8:
        return floor
    xb, yb = x[m], y[m]
    order = np.argsort(xb)
    xb, yb = xb[order], yb[order]
    # The trend is the baseline the window's own integration will subtract;
    # what is left is what that subtraction cannot account for.
    slope, intercept = np.polyfit(xb, yb, 1)
    resid = yb - (slope * xb + intercept)
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(resid - np.median(resid))))
    if not np.isfinite(sigma) or sigma <= 0:
        return floor
    return max(sigma, floor) if floor is not None else sigma
