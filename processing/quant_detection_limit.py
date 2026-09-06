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
    EXCLUDED_REGION:
        'Não avaliável: o sinal previsto cai numa região excluída do espectro. '
        'A detecção não procura lá e nada medido ali é do analito — na região '
        'da água é o resíduo do solvente. Nenhum limite pode ser dado.',
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
