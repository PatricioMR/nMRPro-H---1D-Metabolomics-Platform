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

CENSOR_TEXT = {
    BELOW_LOD:
        'Abaixo do limite de detecção: a região prevista pela base de dados não '
        'contém sinal acima do ruído. O analito está ausente ou abaixo de {lod}.',
    BELOW_LOQ:
        'Abaixo do limite de quantificação: há sinal na região prevista, mas '
        'não o suficiente para medir. A concentração está entre {lod} e {loq}.',
    UNRESOLVED:
        'Sinal presente, não resolvido: a região prevista contém área bem acima '
        'do limite de quantificação, mas a detecção não encontrou pico algum '
        'ali. Provável sobreposição com um vizinho — a área pode não ser deste '
        'analito. Verifique a região antes de usar.',
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
    """Where the signal would be, from the database positions the search used.

    Padded by the project's own measured linewidth, so the outer lines' tails
    fall inside, and by its measured drift, because the resonance is not at the
    database position in any real sample — it is near it. Both come from the
    study rather than from a constant.

    Padding widens the window, which raises the noise on its integral and so
    raises the reported limit. That direction is deliberate: a wider window can
    only make the bound more conservative, and a bound that is too tight would
    call an analyte absent that simply drifted.

    Returns None when there are no predicted positions to build one from.
    """
    ppms = [float(v) for v in (predicted_ppms or ()) if v is not None]
    if not ppms:
        return None
    m = abs(float(linewidth_ppm or 0.0)) + abs(float(drift_ppm or 0.0))
    return min(ppms) - m, max(ppms) + m


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


def project_linewidth(peaks) -> float:
    """The study's typical linewidth, as the median of what detection measured.

    Median over the detected peaks rather than a constant: linewidth depends on
    shimming and on the sample, and a padding derived from one study's peaks is
    the one padding that describes that study.
    """
    w = [float(p.get('width_ppm')) for p in (peaks or ())
         if p.get('width_ppm') and float(p['width_ppm']) > 0]
    return float(np.median(w)) if w else 0.008


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
