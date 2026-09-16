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
    """Per metabolite, the tallest peak per proton any of its signals could
    anchor on: ``{met_id: snr_per_proton}``.

    ``entries`` are database signals with ``ppm_center`` (or ``ppm``) and
    ``protons``; ``peaks`` are detected peaks with ``ppm`` and ``snr``;
    ``met_key(entry)`` names the metabolite. Signals without a proton count
    take no part.
    """
    best: dict = defaultdict(float)
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
        mid = met_key(e)
        best[mid] = max(best[mid], tallest / n)
    return dict(best)


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
