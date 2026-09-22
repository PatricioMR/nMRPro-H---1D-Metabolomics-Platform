"""Which peak within tolerance is a signal's, when the metabolite's other
signals say how tall it should be.

The search anchors a database signal on the nearest annotatable peak within
tolerance. Position alone chose, and on the skin-extract study creatine's
methyl singlet (3 H) went to a peak of SNR 9 two millippm from the listed
position while the peak of SNR 268 twenty millippm away -- the one its
methylene singlet (2 H, SNR 189) said it must have -- went to creatinine.
Glycine did the same: a noise peak at 3.547, and its singlet of SNR 121 at
3.561 to somebody else.

A metabolite's signals stand in the proportion of their protons. Given the
tallest peak, per proton, that any of the metabolite's signals could be
anchored on, every signal of it has an expected height, and a peak under a
fifth of that -- the ratio ADR-012's contradiction gate already uses -- is
not that signal, whatever its position: overlap makes a peak taller, never
shorter. The nearest peak is chosen among the consistent ones; when none
is, among all, as before -- a reference inflated by a foreign peak under
one of the metabolite's signals can cost nothing that was not already so.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

# A peak under this fraction of the height the metabolite's tallest signal
# predicts for it cannot be that signal. The same ratio the fingerprint
# contradiction gate uses (annotation_detectability).
CONSISTENCY_RATIO = 5.0


def _snr(pk) -> float | None:
    try:
        v = float(pk.get('snr'))
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _nh(entry) -> float | None:
    try:
        v = float(entry.get('protons') or 0)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def reference_per_proton(entries, peaks, *, tolerance, met_key) -> dict:
    """Per metabolite, how tall a peak of one of its protons stands:
    ``{met_id: snr_per_proton}``, the median over its signals.

    ``entries`` are database signals with ``ppm_center`` (or ``ppm``) and
    ``protons``; ``peaks`` are detected peaks with ``ppm`` and ``snr``;
    ``met_key(entry)`` names the metabolite. Signals without a proton
    count take no part.

    The median, over the tallest peak each signal could anchor on, and
    not the tallest of those: the tallest is one number decided by one
    peak, and a single foreign peak that happens to lie within the
    tolerance of any one signal then speaks for the whole molecule.
    ``consistent_peaks`` reads this as the height a peak must reach to be
    this analyte's, so a reference set too high makes the analyte prefer
    tall peaks -- which is how it came to be set by one in the first
    place.

    Measured on the reference bench, annotated the application's own way:
    hydroxyproline's six signals are one proton each and stand at 24, 31,
    39, 46 and 78 sigma, with a sixth at 320 on a peak that is not its.
    The maximum said 320 for all six, so five of its own signals read as
    contradicted; the median says 42. Over the study: betaine's
    concentration went from 111% high to 23%, taurine's from 37% low to
    11%, the ninetieth percentile of the error over every analyte from
    37% to 23% -- and isoleucine, which the maximum measured to within
    8%, lost its peak to a rival and is no longer quantified.
    """
    per: dict = defaultdict(list)
    pk = [(float(p.get('ppm', 0.0)), _snr(p)) for p in peaks or ()
          if p.get('ppm') is not None]
    for e in entries or ():
        n = _nh(e)
        if n is None:
            continue
        c = float(e.get('ppm_center') or e.get('ppm') or 0.0)
        tallest = max((s for p, s in pk
                       if s is not None and abs(p - c) <= tolerance),
                      default=None)
        if tallest is None:
            continue
        per[met_key(e)].append(tallest / n)
    return {m: float(np.median(v)) for m, v in per.items() if v}


def consistent_peaks(candidates, entry, reference_pp,
                     *, ratio=CONSISTENCY_RATIO) -> list:
    """The peaks among ``candidates`` a signal with ``entry['protons']``
    could be, given the metabolite's reference height per proton -- all of
    them when the metabolite has no reference, the signal no proton count,
    or none of them is consistent."""
    n = _nh(entry)
    if reference_pp is None or n is None or not candidates:
        return list(candidates)
    expected = float(reference_pp) * n
    keep = [p for p in candidates
            if _snr(p) is None or _snr(p) * ratio >= expected]
    return keep or list(candidates)
