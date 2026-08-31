# processing/multiplet_line_fit.py
"""Which apices of ONE sample are the lines of a given multiplet.

The per-sample apex search that came before this located every line on its own:
for each expected line, the strongest point within a radius of its master
position. Two lines of the same multiplet sit closer than that radius often
enough that the taller leg won both searches and the pattern was recorded twice
at one position — 1661 of 2592 (multiplet, sample) pairs on the reference set.
Bounding the radius by the nearest neighbour stopped it, but the search was
still blind: nothing in it knew that these N positions describe ONE pattern and
must therefore be N DIFFERENT apices.

Here the multiplet is fitted as a unit. Every apex the sample really has inside
the window is a candidate, and the lines are assigned to them injectively — one
apex, one line. Folding is then not detected after the fact, it is impossible
to express.

Among the injective assignments the winner is the one whose spacings agree best
with the coupling the database gives, which is the question an analyst asks of
a window: are there N signals in here spaced like this analyte's multiplet. The
positions are the sample's own; nothing is moved to make the answer nicer, so
the J that comes out is still a measurement and its distance from the database
is still evidence.

Thresholds are the annotation's own — ``dj_bad_hz`` decides what spacing
disagreement is too much to call a fit, ``dj_gold_hz`` decides which rival fits
are close enough to count as ties. No tolerance is invented here.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from processing.peak_alignment import parabolic_apex

# A window can hold more apices than a multiplet has lines, and the assignment
# enumerates subsets. The tallest candidates are the ones a multiplet's lines
# can plausibly be, so the field is capped rather than letting a noisy window
# cost seconds. 12 leaves every DB multiplicity (at most 8 lines) room to pick.
MAX_CANDIDATES = 12


def candidate_apices(ppm, y, lo, hi, noise_floor=0.0):
    """Every local maximum of ``y`` between ``lo`` and ``hi`` ppm.

    Returns ``[(ppm, height), ...]`` sorted by ppm ascending, sub-pixel
    refined. Points at or below ``noise_floor`` are not apices — a sample
    without this analyte must yield an empty list, not a list of noise.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if ppm.size < 3 or ppm.size != y.size:
        return []
    lo, hi = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
    idx = np.flatnonzero((ppm >= lo) & (ppm <= hi))
    if idx.size < 3:
        return []
    out = []
    for i in idx:
        if i == 0 or i == ppm.size - 1:
            continue
        yi = float(y[i])
        if yi <= noise_floor:
            continue
        if yi >= float(y[i - 1]) and yi > float(y[i + 1]):
            out.append((parabolic_apex(ppm, y, int(i)), yi))
    out.sort(key=lambda t: t[0])
    return out


def _cap(cands):
    """Keep the tallest ``MAX_CANDIDATES``, back in ppm order."""
    if len(cands) <= MAX_CANDIDATES:
        return cands
    kept = sorted(cands, key=lambda t: -t[1])[:MAX_CANDIDATES]
    return sorted(kept, key=lambda t: t[0])


def _spacing_residual_hz(expected, chosen, field_mhz):
    """How far a candidate assignment's spacings sit from the expected ones.

    Compared pair by consecutive pair, in Hz, so the number means the same
    thing as ΔJ and can be judged by the same threshold. Translation is free:
    only the gaps BETWEEN lines enter, never their absolute position, so a
    multiplet that drifted with pH is not penalised for having drifted.

    The worst pair decides, not the average — one leg landing on a foreign
    resonance must not be hidden by the others agreeing.
    """
    if len(chosen) < 2:
        return None
    res = [abs((chosen[k + 1] - chosen[k]) - (expected[k + 1] - expected[k]))
           for k in range(len(chosen) - 1)]
    return float(max(res) * field_mhz)


def fit_lines(expected_ppms, candidates, *, field_mhz, dj_bad_hz,
              dj_gold_hz, max_drift_ppm=None):
    """Assign this multiplet's lines to the apices one sample actually has.

    ``expected_ppms`` are the predicted line positions, ``candidates`` the
    ``(ppm, height)`` apices found in the window. Both are used in ascending
    ppm order, and the assignment preserves that order: a multiplet's lines
    cannot cross.

    The choice is hierarchical, not a weighted blend — the shape the candidate
    ranking already uses elsewhere in the pipeline:

      1. only assignments whose spacing residual is within ``dj_bad_hz``
      2. among those, the one covering the MOST lines
      3. ties broken by the smaller spacing residual
      4. still tied, by the smaller drift from the expected position

    Returns a dict with ``ppms`` (assigned positions, ascending), ``ranks``
    (which expected line each one is), ``residual_hz``, ``drift_ppm``,
    ``n_alternatives`` and ``fitted``. When nothing fits within ``dj_bad_hz``
    the best attempt is still returned with ``fitted=False``, so the caller can
    say the window held no such pattern instead of showing a forced reading.
    """
    exp = sorted(float(p) for p in (expected_ppms or ()))
    cands = _cap([(float(p), float(h)) for p, h in (candidates or ())])
    blank = dict(ppms=[], ranks=[], residual_hz=None, drift_ppm=None,
                 n_alternatives=0, fitted=False, n_candidates=len(cands))
    if not exp or not cands or field_mhz <= 0:
        return blank

    if len(exp) == 1:
        # A singlet has no spacing to judge, so the nearest apex to where it is
        # expected is the reading — and how many other apices sit just as close
        # is the whole of its uncertainty.
        within = [(abs(p - exp[0]), p) for p, _ in cands
                  if max_drift_ppm is None or abs(p - exp[0]) <= float(max_drift_ppm)]
        if not within:
            return blank
        within.sort()
        return dict(ppms=[within[0][1]], ranks=[0], residual_hz=None,
                    drift_ppm=float(within[0][0]), fitted=True,
                    n_alternatives=len(within) - 1, n_candidates=len(cands))

    if len(cands) < 2:
        return blank

    cand_ppms = [p for p, _ in cands]
    scored = []
    for k in range(min(len(exp), len(cand_ppms)), 1, -1):
        for ranks in combinations(range(len(exp)), k):
            sub_exp = [exp[r] for r in ranks]
            for pick in combinations(range(len(cand_ppms)), k):
                chosen = [cand_ppms[i] for i in pick]
                res = _spacing_residual_hz(sub_exp, chosen, field_mhz)
                if res is None:
                    continue
                drift = abs(float(np.mean(chosen)) - float(np.mean(sub_exp)))
                if max_drift_ppm is not None and drift > float(max_drift_ppm):
                    continue
                scored.append((k, res, drift, list(ranks), chosen))
    if not scored:
        return blank

    ok = [s for s in scored if s[1] <= float(dj_bad_hz)]
    pool = ok or scored
    best = min(pool, key=lambda s: (-s[0], s[1], s[2]))
    k_b, res_b = best[0], best[1]
    # A rival that covers the same number of lines and agrees on the spacing to
    # within dj_gold_hz is not distinguishable from the winner by this
    # evidence. How many there are is the reading's uncertainty, and it travels
    # with the reading rather than being rounded away.
    alts = sum(1 for s in pool
               if s[0] == k_b and s[4] != best[4]
               and abs(s[1] - res_b) <= float(dj_gold_hz))
    return dict(ppms=list(best[4]), ranks=list(best[3]),
                residual_hz=float(res_b), drift_ppm=float(best[2]),
                n_alternatives=int(alts), fitted=bool(ok),
                n_candidates=len(cands))


def observed_j_hz(ranks, ppms, field_mhz):
    """J from the assigned lines: total span over the couplings it covers.

    Lines 0 and 2 of a triplet span two couplings, so the span is divided by
    the index span and not by the number of lines. Identical to the rule the
    per-sample reader already used, kept here so both take the ranks from the
    same assignment.
    """
    if len(ppms) < 2 or field_mhz <= 0:
        return None
    span_idx = max(ranks) - min(ranks)
    if span_idx <= 0:
        return None
    return float(abs(max(ppms) - min(ppms)) * field_mhz / span_idx)


def fit_across_samples(expected_ppms, ppm, intensities, *, included=None,
                       field_mhz, dj_bad_hz, dj_gold_hz, max_drift_ppm,
                       noise_floor=0.0):
    """Fit this multiplet in every sample the caller allows.

    ``intensities`` is the (points, samples) matrix. Returns
    ``{sample_idx: fit}``, one entry per included sample — including the ones
    where nothing fitted, so the caller can tell "not measured here" from "not
    looked at".
    """
    win = fit_window(expected_ppms, max_drift_ppm)
    if win is None:
        return {}
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_s = arr.shape[1]
    idx = range(n_s) if included is None else [i for i in included
                                              if 0 <= i < n_s]
    out = {}
    for s in idx:
        cands = candidate_apices(ppm, arr[:, s], win[0], win[1], noise_floor)
        out[int(s)] = fit_lines(expected_ppms, cands, field_mhz=field_mhz,
                                dj_bad_hz=dj_bad_hz, dj_gold_hz=dj_gold_hz,
                                max_drift_ppm=max_drift_ppm)
    return out


def fit_window(expected_ppms, max_drift_ppm):
    """The ppm range to look in: the pattern's own span plus the drift allowed.

    Nothing outside this can be a line of this multiplet, so the candidate
    search never sees a neighbouring analyte's apex it might have chosen.
    """
    exp = [float(p) for p in (expected_ppms or ())]
    if not exp:
        return None
    pad = abs(float(max_drift_ppm or 0.0))
    return min(exp) - pad, max(exp) + pad
