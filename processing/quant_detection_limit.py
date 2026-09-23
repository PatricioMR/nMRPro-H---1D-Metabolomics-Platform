# processing/quant_detection_limit.py
"""What to report where the database predicts a signal and detection found none.

"No detected peak" is not "concentration unknown". It is "concentration below
what this experiment can see in this region", which is a measurement — an upper
bound — and reporting it as one is standard analytical practice rather than a
concession.

The engine this replaces did the opposite: it integrated a window positioned by
the database and reported whatever came out as a concentration. On the
reference study 23 of 73 placed analytes had no detected apex, 15 produced a
number that way, and 8 of the 23 were refused only because their window
collided with a region another analyte was already integrating — meaning the
window had been about to report a neighbour's area as this analyte's
concentration. Substituting a fabricated value for a non-detect is a known way
to ruin a dataset:

    Helsel, D.R. (2006). Fabricating data: How substituting values for
    nondetects can ruin results, and what can be done about it.
    Chemosphere 65(11):2434-2439.

The framework used here instead:

    Currie, L.A. (1968). Limits for qualitative detection and quantitative
    determination. Application to radiochemistry.
    Analytical Chemistry 40(3):586-593.

    Currie, L.A. (1995). Nomenclature in evaluation of analytical methods
    including detection and quantification capabilities (IUPAC Recommendations
    1995). Pure and Applied Chemistry 67(10):1699-1723.

and its conventional application in quantitative NMR, where the detection and
quantification limits are taken at signal-to-noise 3 and 10:

    Bharti, S.K. & Roy, R. (2012). Quantitative 1H NMR spectroscopy.
    TrAC Trends in Analytical Chemistry 35:5-26.

Three outcomes, not one. The third is the useful one: a window with no detected
apex whose area is nonetheless well above the quantification limit does not
hold nothing — it holds something detection could not resolve, most often an
overlapped shoulder or a neighbour's skirt. That is exactly the case the old
gap-fill reported as a clean concentration, and here it identifies itself.

Downstream, censored values stay censored. Turning them into numbers for
statistics is a separate, explicit decision with its own literature — QRILC
(Lazar et al. 2016, J Proteome Res 15:1116-1125) and GSimp (Wei et al. 2018,
PLoS Comput Biol 14:e1005973) — and must never happen silently here.
"""
from __future__ import annotations

import numpy as np

# Signal-to-noise at which a signal is detectable, and at which it can be
# quantified. The qNMR convention, and the same ratio the peak detector uses to
# decide a peak exists — so an analyte reported below the limit is below the
# limit that would have detected it, not below some second, unrelated bar.
K_LOD = 3.0
K_LOQ = 10.0

BELOW_LOD = 'below_lod'
BELOW_LOQ = 'below_loq'
UNRESOLVED = 'unresolved'
EXCLUDED_REGION = 'excluded_region'

CENSOR_TEXT = {
    BELOW_LOD:
        'Below the limit of detection: the region the database predicts holds '
        'no signal above the noise. The analyte is absent, or below {lod}.',
    BELOW_LOQ:
        'Below the limit of quantification: there is signal in the predicted '
        'region but not enough of it to measure. The concentration lies '
        'between {lod} and {loq}.',
    UNRESOLVED:
        'Signal present, not resolved: the predicted region holds area well '
        'above the quantification limit, yet detection found no peak there. '
        'Likely overlap with a neighbour — the area may not be this '
        "analyte's. Check the region before using it.",
    EXCLUDED_REGION:
        'Not assessable: the predicted signal falls in a region excluded from '
        'the spectrum. Detection does not look there, and nothing measured '
        'there belongs to the analyte — in the water window it is the residual '
        'solvent. No limit can be given.',
}


def integration_noise(sigma, dppm, n_points) -> float:
    """The uncertainty on an integral of ``n_points`` samples of white noise.

    Each point carries σ and the trapezoid sums them over a spacing Δppm, so
    the noise on the area grows as √N rather than N: random error partially
    cancels across the window. Using σ·Δppm·N instead would overstate the limit
    in proportion to how wide the window is, which would make a broad multiplet
    look undetectable for no reason but its width.
    """
    n = max(int(n_points), 0)
    if n <= 0 or not sigma or float(sigma) <= 0:
        return float('nan')
    return float(sigma) * abs(float(dppm)) * float(np.sqrt(n))


def detection_limits(sigma, dppm, n_points, *, k_lod=K_LOD, k_loq=K_LOQ):
    """``(lod_area, loq_area)`` for one window, in the units of the areas.

    ``sigma`` is the BLANK's: the scatter of a signal-free region of this
    study. That is what Currie's framework and the qNMR convention cited in
    this module's header both define a limit from, and it is what makes the
    limit a property of the method rather than of whichever neighbours an
    analyte happens to have. What the neighbours do is interference --
    ``local_noise_sigma`` measures it and the row reports it, separately.

    The caller converts them to concentration through the same chain any other
    area travels — divided by nH, referenced to the standard, scaled by the
    dilution. A limit expressed in area units cannot be compared with the
    concentrations in the table, and a limit nobody can compare is not a limit.
    """
    s = integration_noise(sigma, dppm, n_points)
    if not np.isfinite(s):
        return float('nan'), float('nan')
    return float(k_lod) * s, float(k_loq) * s


def predicted_window(predicted_ppms, *, linewidth_ppm, drift_ppm=0.0):
    """The window a signal would occupy, from the positions the search used.

    Its width is the signal's own: the span of the predicted lines plus one
    linewidth on each side, so the outer lines' tails fall inside.

    The drift is NOT added to the width. Where an analyte sits and how wide it
    is are different questions, and only the second belongs in a detection
    limit. Padding by the drift made the window 25.7 Hz for a 3.2 Hz line on
    the reference study — eight times the linewidth, 180 points instead of 45 —
    and since the noise on an integral grows as the root of the point count,
    every limit came back twice what the spectrum supports. Integrating 25 Hz
    of noise to bound a 3 Hz signal adds noise without adding signal.

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
    asks "is it anywhere here?" while keeping the noise of a single window;
    growing asks the same question and pays for the whole range in noise.

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

    Nothing measured in an excluded region belongs to the analyte — in the
    water window it is the suppressed solvent, which is orders of magnitude
    above any metabolite. Giving it a detection limit reports the solvent as
    the analyte's bound; classifying it as unresolved signal reports the
    solvent as a buried analyte. On the reference study both happened:
    4-pyridoxic acid and guanosine came back with limits, saligenin and
    trans-4-hydroxy-L-proline came back marked as overlaps, and all four
    windows sat inside 4.50-5.00 ppm.
    """
    for r_lo, r_hi in (excluded_regions or ()):
        if float(lo) < float(r_hi) and float(r_lo) < float(hi):
            return True
    return False


def censor(area, lod_area, loq_area) -> str:
    """Which of the three this window is.

    An area at or below the detection limit is indistinguishable from noise.
    Between the two limits there is signal but not enough of it to put a number
    on. Above the quantification limit, with no peak detected anywhere in the
    window, something is there that detection could not resolve — and calling
    that a concentration is the mistake this module exists to stop.
    """
    if not np.isfinite(area) or not np.isfinite(loq_area):
        return BELOW_LOD
    if area <= lod_area:
        return BELOW_LOD
    if area < loq_area:
        return BELOW_LOQ
    return UNRESOLVED


def censor_text(verdict, lod_mm, loq_mm) -> str:
    """The reason, with the limits filled in.

    The numbers are the point. "Below the limit of detection" without saying
    what the limit was leaves the analyst unable to tell an informative absence
    from an insensitive measurement, and those mean opposite things about the
    sample.
    """
    def _fmt(v):
        return f'{v:.4g} mM' if np.isfinite(v) else 'um valor indeterminado'
    return CENSOR_TEXT[verdict].format(lod=_fmt(lod_mm), loq=_fmt(loq_mm))


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


# How far either side of the window to look when estimating the local noise,
# in linewidths. Wide enough to contain the baseline structure the window sits
# on, narrow enough that it is still the same neighbourhood.
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

    NOT a detection limit, and no longer used as one. A limit of detection
    is a property of the method and the blank -- Currie's framework and the
    qNMR convention this module's header cites both define it from the
    scatter of a signal-free region -- while what a neighbour's skirt does
    to a measurement is interference, which belongs to selectivity and
    costs accuracy, not detectability. Feeding this into ``detection_limits``
    conflated the two and reported an overlap statement wearing a detection
    limit's name.

    Measured on the reference study: taurine, the second most concentrated
    analyte, had this come back at 1,035,234 where the spectrum's noise is
    5,349 -- 193 times -- because the band flanking its window holds a skirt
    16.9 million counts tall and no detected peak at all. Its quantification
    limit then read 1.566 mM against a measured 2.051, marking the analyte
    weak for having neighbours. Four of the sixteen analytes really in the
    study were marked weak that way; on the blank's sigma, sixteen of
    sixteen read strong and not one stranger joined them.

    Kept, and reported, because the thing it measures is real: a region this
    crowded cannot be measured down to the noise floor, and the analyst has
    to see that. It is the ratio to the blank that says it, not a limit.

    The symptom on the reference study: six analytes claiming the histidine
    imidazole region at 7.0926-7.1025 ppm were given identical limits, differing
    only by their proton counts, because the window width and the global sigma
    were the same for all of them. The limit had no way to know that region is
    contested.

    Measured here as a robust scale over a band around the window, after
    removing the local linear trend. The median absolute deviation is used
    rather than the standard deviation because a band in a crowded region is
    mostly peak, and a plain standard deviation would report the peaks
    themselves as noise:

        Rousseeuw, P.J. & Croux, C. (1993). Alternatives to the median absolute
        deviation. Journal of the American Statistical Association
        88(424):1273-1283.

    Where more than half the band really is signal the MAD rises, and it should:
    a region that crowded cannot be measured down to the noise floor, and
    saying so is the point.

    The band flanks the window and excludes it, so what is being measured
    cannot raise the bar it has to clear.

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
    # The band flanks the window and excludes it. A noise estimate that
    # contained what is being measured would let a real signal raise its own
    # detection limit until it could never exceed it — the residual water,
    # which is orders of magnitude above any metabolite, came back reported as
    # below the limit of detection.
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


# How far above the quantification limit a signal must sit before its strength
# stops limiting what the result is worth. Three times, chosen by the analyst:
# at the limit itself a number is defensible but not strong, and a factor of
# three is where the counting statistics stop dominating.
STRONG_ABOVE_LOQ = 3.0

# The floor a weak signal falls to, never below. A signal that was identified,
# won its peak and then measured near or under the limit is not bad evidence
# about identity -- it is good evidence about a small quantity, and the
# distinction belongs in the number's worth rather than in a verdict on whether
# the analyte is there.
WEAK_STRENGTH = 2
STRONG_STRENGTH = 3


def strength_band(area, loq_area, *, strong_multiple=STRONG_ABOVE_LOQ) -> int:
    """How much the measured area is worth on its own, 3 (strong) or 2.

    Only two values, and 2 is the floor. An analyte whose peak was identified
    and whose contest it won is not made doubtful by being scarce: being under
    the quantification limit says the quantity is small and the number
    imprecise, not that the assignment was wrong. What it must not do is read
    as strong.

    Unknown limits leave the strength unjudged and return the strong band, so
    a missing noise estimate cannot silently mark every analyte weak — the
    caller's other evidence then decides alone.
    """
    if (area is None or loq_area is None
            or not np.isfinite(area) or not np.isfinite(loq_area)
            or loq_area <= 0):
        return STRONG_STRENGTH
    return (STRONG_STRENGTH if float(area) >= float(strong_multiple) * float(loq_area)
            else WEAK_STRENGTH)
