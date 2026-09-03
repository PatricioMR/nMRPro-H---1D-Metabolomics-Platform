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
