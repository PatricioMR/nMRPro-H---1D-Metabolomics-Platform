# processing/spectral_regions.py
"""Regions of the spectrum that carry no usable analyte signal.

The residual water resonance is the reason this module exists. In an aqueous
sample it is four to six orders of magnitude larger than any metabolite, and
the presaturation that suppresses it also destroys whatever sits under it. What
survives is a broad, variable artefact — and the reference databases have
signals there: 82 of BMRB_Metabolomics_v01's fall between 4.5 and 4.9 ppm, so
any of them can match the residue and be quantified as an analyte.

That is not hypothetical. On the reference study ATP's H37 triplet (4.639 ppm,
1 H) matched the residue and came back as the most abundant metabolite in the
sample at 288 mM, with an SNR of 31520 — a hundred times the next row.

Excluding the suppressed region before anything is measured is standard
practice in NMR metabolomics, not a choice this project is making on its own:

    Beckonert, O. et al. (2007). Metabolic profiling, metabonomic and
    metabolomic procedures for NMR spectroscopy of urine, plasma, serum and
    tissue extracts. Nature Protocols 2(11):2692-2703.

    Dieterle, F., Ross, A., Schlotterbeck, G. & Senn, H. (2006). Probabilistic
    quotient normalization as robust method to account for dilution of complex
    biological mixtures. Analytical Chemistry 78(13):4281-4290.

The width is a property of the acquisition, not of the analyte, so it is a
project setting with a default rather than a constant.
"""
from __future__ import annotations

# The presaturated window, as a (lo_ppm, hi_ppm) pair. 4.50-5.00 covers the
# residue and the shoulder that presaturation flattens on either side of it at
# the offsets these protocols use. A narrower window leaves the wings, and the
# wings are what a database signal matches.
WATER_REGION = (4.50, 5.00)

DEFAULT_EXCLUDED_REGIONS = [WATER_REGION]

# Below this fraction of the measured linewidth, two apices are more likely two
# readings of one line than two lines. Measured, not assumed: on the study in
# unresolvable_split_warning the flank artefacts were still there at 0.75 of the
# linewidth and gone at 0.93, so the boundary lies between. 0.85 sits inside
# that bracket, and it agrees with the physics -- two Lorentzians separate into
# two maxima at roughly one full width at half maximum.
SPLIT_WARNING_FRAC = 0.85


def parse_regions(text):
    """Read regions from the string the settings field holds.

    Accepts ``4.50-5.00, 0.00-0.10`` and tolerates spaces and semicolons. A
    malformed entry raises rather than being skipped: a region the analyst
    believes is excluded and is not would let the artefact back in silently,
    which is the failure this module exists to prevent.
    """
    out = []
    for chunk in str(text or '').replace(';', ',').split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split('-')
        if len(parts) != 2:
            raise ValueError(f'Malformed region {chunk!r}; expected "lo-hi"')
        lo, hi = (float(p) for p in parts)
        out.append((min(lo, hi), max(lo, hi)))
    return out


def format_regions(regions) -> str:
    """The inverse of ``parse_regions``, for the settings field and the log."""
    return ', '.join(f'{lo:.2f}-{hi:.2f}' for lo, hi in (regions or ()))


def in_excluded_region(ppm, regions) -> bool:
    """Whether one chemical shift falls inside any excluded region."""
    try:
        v = float(ppm)
    except (TypeError, ValueError):
        return False
    return any(float(lo) <= v <= float(hi) for lo, hi in (regions or ()))


def split_excluded(peaks, regions):
    """``(kept, dropped)`` — the peaks outside the regions, and those inside.

    Both halves are returned because the dropped ones have to be reported. A
    peak that disappears with no count attached leaves the analyst unable to
    tell an exclusion from a detection failure, and those call for opposite
    responses.
    """
    if not regions:
        return list(peaks or ()), []
    kept, dropped = [], []
    for pk in peaks or ():
        (dropped if in_excluded_region(pk.get('ppm'), regions)
         else kept).append(pk)
    return kept, dropped


def _interp_crossing(x0, y0, x1, y1, level):
    """Where a straight line between two samples crosses ``level``."""
    if y1 == y0:
        return float(x0)
    return float(x0) + (float(level) - float(y0)) * (
        float(x1) - float(x0)) / (float(y1) - float(y0))


def measure_linewidth_ppm(ppm, intensity, peaks, n_top=30):
    """The study's linewidth, measured on the trace rather than on the peak list.

    Every width the detector records is contaminated by the setting being
    judged: ``width_ppm`` is valley-to-valley and grows when neighbours merge,
    and even ``fwhm_pts`` shrinks when one line is split into fragments.
    Measured on a 21-sample study across min_distance from 1.80 to 4.14 Hz, the
    median ``width_ppm`` ran 4.40 -> 7.81 Hz and the median FWHM 1.70 -> 2.77 Hz
    on the same data. A threshold built on either would advise a value that,
    once applied, moves the measurement again -- advice that never converges.

    So the apex position is taken from the peak list, and nothing else: the
    walk to half maximum runs over the intensity trace and does not consult the
    peak's recorded boundaries. Only the tallest peaks are used, since a weak
    line's half-maximum is reached inside a neighbour's skirt.

    Returns the median FWHM in ppm, or None when nothing can be measured.
    """
    import numpy as _np
    y = _np.asarray(intensity, dtype=float)
    x = _np.asarray(ppm, dtype=float)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if x.size != y.size or x.size < 3:
        return None
    order = _np.argsort(x)
    x, y = x[order], y[order]

    tall = sorted((p for p in (peaks or ()) if p.get('ppm') is not None),
                  key=lambda p: -float(p.get('height') or p.get('snr') or 0))
    widths = []
    for pk in tall[:int(n_top)]:
        i = int(_np.argmin(_np.abs(x - float(pk['ppm']))))
        half = y[i] / 2.0
        if not _np.isfinite(half) or half <= 0:
            continue
        li = i
        while li > 0 and y[li] > half:
            li -= 1
        ri = i
        while ri < len(y) - 1 and y[ri] > half:
            ri += 1
        # A walk that ran to the edge, or one that never rose above half in the
        # first place, measured the neighbourhood rather than the line.
        if li == 0 or ri == len(y) - 1 or ri <= li:
            continue
        # Interpolated to the crossing, not rounded to the grid. Stopping at
        # the first point below half overshoots by up to one grid step on each
        # side, which on a line only ten points wide is an 8% inflation -- and
        # it inflates in the direction that makes the setting look safer.
        left = _interp_crossing(x[li], y[li], x[li + 1], y[li + 1], half)
        right = _interp_crossing(x[ri - 1], y[ri - 1], x[ri], y[ri], half)
        widths.append(abs(float(right - left)))
    return float(_np.median(widths)) if widths else None


def unresolvable_split_warning(linewidth_ppm, min_distance_ppm, mhz=600.0):
    """A warning when detection may split one line into several.

    Two Lorentzians are resolved when their separation approaches their width.
    Below that they merge into one contour, and the maxima a detector finds
    inside that contour are noise, not lines: the same resonance yields an
    apex here in one sample and there in another, and a per-sample union turns
    those into separate master peaks.

    Linewidth is a property of the shim and of T2, so it is near enough uniform
    across a spectrum -- which makes the study's own median linewidth the right
    scale to compare against, rather than a constant. It must come from
    ``measure_linewidth_ppm``, which reads the trace: a width taken from the
    peak list moves with the setting this warning is about.

    Measured on a 21-sample mouse serum study, median linewidth 4.12 Hz, with
    min_distance at 1.80 Hz: L-alanine's methyl doublet had two artefacts
    detected on its flanks, at 1.4758 (on the rising edge) and 1.4877 (in the
    valley between the real lines), each with about 1% of a real line's area.
    Annotation matched alanine to those two rather than to the doublet, and the
    integration window built from their feet clipped one real line out --
    reporting alanine at 0.045 mM, roughly a tenth of what the region holds.

        min_distance   artefacts   real lines   flank artefacts, whole study
          1.80 Hz          2           2            66 of 533  (12.4%)
          2.40 Hz          2           2            53 of 441  (12.0%)
          3.00 Hz          0           2            37 of 361  (10.2%)
          4.14 Hz          0           2            22 of 301  ( 7.3%)

    The study's linewidth, measured on the trace, is 3.22 Hz. So they are still
    there at 0.75 of a linewidth and gone at 0.93 -- the boundary lies between,
    and it lands where the physics says it should: two Lorentzians separate into
    two maxima at roughly one full width at half maximum. Returns None when the setting is at or above
    ``SPLIT_WARNING_FRAC`` of the measured linewidth, and a message naming both
    numbers when it is below -- a warning, never a change: which splittings a
    study can resolve is the analyst's call, and this only makes sure the
    number they are working against is visible.
    """
    if not linewidth_ppm or not min_distance_ppm:
        return None
    lw = float(linewidth_ppm)
    if float(min_distance_ppm) >= lw * SPLIT_WARNING_FRAC:
        return None
    return (f'Min Distance is {min_distance_ppm:.4f} ppm '
            f'({min_distance_ppm * mhz:.2f} Hz) but this spectrum\'s median '
            f'linewidth is {lw:.4f} ppm ({lw * mhz:.2f} Hz). A splitting '
            f'narrower than the linewidth cannot be resolved, so apices that '
            f'close are more likely two readings of one line than two lines. '
            f'Consider at least {lw * SPLIT_WARNING_FRAC:.4f} ppm '
            f'({lw * SPLIT_WARNING_FRAC * mhz:.2f} Hz).')
