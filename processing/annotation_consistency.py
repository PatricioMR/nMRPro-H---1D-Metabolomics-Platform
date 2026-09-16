"""Which of several peaks within tolerance is a signal's, when the metabolite's
other signals say how tall it should be.

The search builds a candidate for every peak within tolerance of a database
signal, and kept one per signal by score. Score is mostly position, so the
nearest peak won -- and on the skin-extract study creatine's methyl singlet
(3 H) went to a peak of SNR 8 two millippm from the listed position while
the peak of SNR 268 twenty millippm away, the one creatine's methylene
singlet (2 H, SNR 189) said it must have, went to creatinine. Glycine did
the same: a noise peak at 3.547, and the singlet of SNR 121 at 3.561 to
somebody else.

A metabolite's signals stand in the proportion of their protons. Given the
tallest candidate peak the metabolite holds, every other signal of it has
an expected height, and a candidate whose peak is under a fifth of that --
the ratio ADR-012's contradiction gate already uses -- is not that signal,
whatever its position: overlap makes a peak taller, never shorter. Among
the candidates for one signal the consistent ones are kept and the rest
dropped, and only then does the score choose.

The reference is the tallest peak, per proton, among the metabolite's
verified multiplets -- structure confirmed, so the height is its own -- or,
without one, the tallest candidate anywhere. It is not judged against
itself, and a metabolite with one signal is left as it was.
"""
from __future__ import annotations

from collections import defaultdict

# A candidate peak under this fraction of the height the metabolite's
# tallest signal predicts for it cannot be that signal. The same ratio the
# fingerprint contradiction gate uses (annotation_detectability).
CONSISTENCY_RATIO = 5.0


def _snr(r, peaks_by_rank) -> float | None:
    rank = r.get('peak_rank')
    pk = peaks_by_rank.get(rank) if rank is not None else None
    if pk is None:
        return None
    try:
        v = float(pk.get('snr'))
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _nh(r) -> float | None:
    try:
        v = float(r.get('protons') or 0)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _verified(r) -> bool:
    n_f, n_e = r.get('_n_found_lines'), r.get('_n_exp_lines')
    try:
        return (n_f is not None and n_e is not None and int(n_e) >= 2
                and int(n_f) >= int(n_e))
    except (TypeError, ValueError):
        return False


def _met_key(r):
    return r.get('hmdb_id') or r.get('bmrb_id', '')


def _sig_key(r):
    return round(float(r.get('ppm_center') or r.get('ppm') or 0), 3)


def consistent_candidates(results, peaks_by_rank, *,
                          ratio=CONSISTENCY_RATIO) -> list:
    """The candidates that the metabolite's own proportions do not rule out.

    Returns a new list, in the input order, with the candidates dropped
    whose peak is under ``1/ratio`` of the height the metabolite's tallest
    signal (per proton) predicts for them -- but never the last candidate
    of a signal: a signal all of whose candidates are inconsistent keeps
    its best-scored one, so nothing disappears that the score would have
    kept, and the contradiction is the fingerprint gate's to report.
    """
    by_met: dict = defaultdict(list)
    for r in results or ():
        by_met[_met_key(r)].append(r)
    dropped: set = set()
    for mid, rs in by_met.items():
        scored = [(r, _snr(r, peaks_by_rank), _nh(r)) for r in rs]
        per_proton = [(s / n, r) for r, s, n in scored
                      if s is not None and n is not None]
        if not per_proton:
            continue
        # The reference: a verified multiplet's peak where the metabolite
        # has one -- its structure is confirmed, so its height is the
        # metabolite's -- and the tallest per proton among those; only
        # without one, the tallest candidate anywhere, which a foreign peak
        # under a singlet could be.
        verified = [(pp, r) for pp, r in per_proton if _verified(r)]
        ref_pp, ref = max(verified or per_proton, key=lambda t: t[0])
        by_sig: dict = defaultdict(list)
        for r, s, n in scored:
            by_sig[_sig_key(r)].append((r, s, n))
        for sig, cands in by_sig.items():
            if len(cands) < 2:
                continue
            keep = []
            for r, s, n in cands:
                if r is ref or s is None or n is None:
                    keep.append(r)
                    continue
                expected = ref_pp * n
                if s * ratio >= expected:
                    keep.append(r)
            if keep:
                for r, _s, _n in cands:
                    if r not in keep:
                        dropped.add(id(r))
    return [r for r in results if id(r) not in dropped]
