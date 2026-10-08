"""Signals the analyst splits or adds by hand in the Signal List.

Detection sometimes leaves two resonances in one window (a shoulder, two lines
closer than the valley rule separates). Choosing the window of one or the
other, or both together, is then all the analyst can do. Here the window is
cut in two at a valley, or a new window is drawn where detection found
nothing, and each piece is measured with detection's own arithmetic
(``peak_measures``) on the baseline-corrected matrix.

Every edit is reversible and says what it was:

* a split keeps the parent's rank and dict on the piece that holds its apex
  (whatever downstream refers to it by rank still finds it); the other piece
  gets a new rank, ``manual_origin = 'split'``, ``split_from_rank`` and a copy
  of the parent as it was before, so ``undo_split`` restores it exactly;
* an added signal carries ``manual_origin = 'added'`` and is removed by
  ``remove_added_signal``.

A hand-made window has no automatic grade: the grade of the merged window does
not describe either piece, and inventing one would be a number nobody measured.
It is graded Average (annotated by default) with no score until the analyst
marks it.
"""
import copy

import numpy as np

from processing.peak_alignment import _refine_one_center
from processing.peak_detection import (
    as_app_peak, broad_removed_fractions, peak_measures)
from processing.signal_quality import GATE_MIN_SNR

ORIGIN_SPLIT = 'split'
ORIGIN_ADDED = 'added'

MIN_WINDOW_POINTS = 3
VALLEY_SNAP_POINTS = 3

_SPLIT_SNAPSHOT = '_split_parent_state'


def window_of(peak):
    """``(lo, hi)`` indices of a signal's window, low index first."""
    return tuple(sorted((int(peak['start_idx']), int(peak['end_idx']))))


def next_rank(peaks):
    """One past the largest integer rank, so no existing rank is renumbered."""
    ranks = [int(p['rank']) for p in peaks
             if isinstance(p.get('rank'), (int, np.integer))]
    return max(ranks, default=0) + 1


def overlapping(peaks, lo, hi, ignore=()):
    """Signals whose window shares more than one boundary point with
    ``[lo, hi]`` -- two windows meeting at a valley share that point, as
    detection's own neighbours do."""
    skip = {id(p) for p in ignore}
    out = []
    for p in peaks:
        if id(p) in skip or 'start_idx' not in p:
            continue
        s, e = window_of(p)
        if min(hi, e) - max(lo, s) > 0:
            out.append(p)
    return out


def valley_near(mean, idx, lo, hi, radius=VALLEY_SNAP_POINTS):
    """Lowest point of the mean trace within ``radius`` points of ``idx`` and
    at least ``MIN_WINDOW_POINTS - 1`` points inside ``[lo, hi]``, so both
    pieces keep a window. A click is a few pixels off the valley; the valley
    is where detection itself puts a shared boundary."""
    a = max(idx - radius, lo + MIN_WINDOW_POINTS - 1)
    b = min(idx + radius, hi - MIN_WINDOW_POINTS + 1)
    if a > b:
        return None
    mean = np.asarray(mean, dtype=float)
    return a + int(np.argmin(mean[a:b + 1]))


def measure_window(ppm, raw_2d, corrected_2d, start_idx, end_idx, noise_std,
                   index=None):
    """Everything the Signal List and quantification read about one window:
    ``peak_measures``, the bounds, the per-sample apex (searched inside the
    window only, so it cannot land on the neighbour the window was cut from)
    and the share of the raw area the baseline model removed."""
    ppm = np.asarray(ppm, dtype=float)
    Y = np.asarray(corrected_2d, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    R = np.asarray(raw_2d, dtype=float) if raw_2d is not None else Y
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    lo, hi = sorted((int(start_idx), int(end_idx)))
    m = peak_measures(ppm, Y, lo, hi, noise_std, index=index)
    x = ppm[lo:hi + 1]
    span = float(abs(x[-1] - x[0]))
    floor = GATE_MIN_SNR * float(noise_std) if noise_std else 0.0
    centers, ok = [], []
    for s in range(Y.shape[1]):
        c, a = _refine_one_center(x, Y[lo:hi + 1, s], m['ppm'], span, floor)
        centers.append(c)
        ok.append(a)
    fracs = broad_removed_fractions(x, (R - Y)[lo:hi + 1],
                                    m['area_per_sample'])
    m.update(
        start_idx=lo, end_idx=hi,
        start_ppm=float(ppm[lo]), end_ppm=float(ppm[hi]),
        width_ppm=span,
        ppm_per_sample=centers,
        shift_ppm_per_sample=[c - m['ppm'] for c in centers],
        aligned_per_sample=ok,
        broad_removed_frac_per_sample=fracs,
        broad_removed_frac=float(np.mean(fracs)) if len(fracs) else 0.0,
    )
    return m


def _hand_made(measured, rank, origin):
    p = as_app_peak(dict(measured, rank=rank))
    p['mean_intensity'] = measured['mean_intensity']
    p['manual_origin'] = origin
    return p


def split_signal(peaks, peak, split_idx, ppm, raw_2d, corrected_2d,
                 noise_std):
    """Cut ``peak``'s window in two at ``split_idx`` (shared by both pieces).

    ``peak`` is changed in place into the piece that holds its apex and keeps
    its rank; the other piece is appended to ``peaks`` and returned.
    Raises ``ValueError`` when a piece would have fewer than
    ``MIN_WINDOW_POINTS`` points.
    """
    lo, hi = window_of(peak)
    split_idx = int(split_idx)
    if not (lo + MIN_WINDOW_POINTS - 1 <= split_idx
            <= hi - MIN_WINDOW_POINTS + 1):
        raise ValueError(
            f"Signal #{peak['rank']}: the cut must leave at least "
            f"{MIN_WINDOW_POINTS} points on each side of it.")
    before = copy.deepcopy(peak)
    apex = int(peak.get('index', lo))
    left, right = (lo, split_idx), (split_idx, hi)
    keep, other = (left, right) if apex <= split_idx else (right, left)

    peak.update(measure_window(ppm, raw_2d, corrected_2d, *keep, noise_std,
                               index=apex))
    peak['boundaries_edited'] = True

    piece = _hand_made(
        measure_window(ppm, raw_2d, corrected_2d, *other, noise_std),
        next_rank(peaks), ORIGIN_SPLIT)
    piece['split_from_rank'] = peak['rank']
    piece[_SPLIT_SNAPSHOT] = before
    for p in (peak, piece):
        p['n_cluster_peaks'] = int(before.get('n_cluster_peaks', 1) or 1) + 1
        for k in ('cluster_lo', 'cluster_hi', 'cluster_lo_ppm',
                  'cluster_hi_ppm'):
            if k in before:
                p[k] = before[k]
    peaks.append(piece)
    return piece


def undo_split(peaks, piece):
    """Merge a split piece back: its parent returns exactly to what it was
    before the cut and the piece leaves ``peaks``. Returns the parent.

    Raises ``ValueError`` when the parent no longer exists or when restoring
    its window would cover another signal (a later split of the same window
    must be undone first)."""
    if piece.get('manual_origin') != ORIGIN_SPLIT or _SPLIT_SNAPSHOT not in piece:
        raise ValueError(f"Signal #{piece.get('rank')} did not come from a split.")
    rank = piece['split_from_rank']
    parent = next((p for p in peaks if p is not piece and p.get('rank') == rank),
                  None)
    if parent is None:
        raise ValueError(f"Signal #{rank}, which #{piece['rank']} was split "
                         "from, no longer exists.")
    before = piece[_SPLIT_SNAPSHOT]
    lo, hi = window_of(before)
    clash = overlapping(peaks, lo, hi, ignore=(piece, parent))
    if clash:
        raise ValueError(
            f"Merging #{piece['rank']} back into #{rank} would cover signal(s) "
            f"{', '.join('#' + str(p['rank']) for p in clash)}: undo the later "
            "split(s) of this window first.")
    peaks[:] = [p for p in peaks if p is not piece]
    parent.clear()
    parent.update(copy.deepcopy(before))
    return parent


def add_signal(peaks, start_idx, end_idx, ppm, raw_2d, corrected_2d,
               noise_std):
    """A new signal on the window ``[start_idx, end_idx]``, appended to
    ``peaks`` and returned. Raises ``ValueError`` when the window is too
    short or covers an existing signal (two windows over the same points
    would count that area twice)."""
    lo, hi = sorted((int(start_idx), int(end_idx)))
    if hi - lo + 1 < MIN_WINDOW_POINTS:
        raise ValueError(
            f"A signal needs at least {MIN_WINDOW_POINTS} points.")
    clash = overlapping(peaks, lo, hi)
    if clash:
        raise ValueError(
            "The window covers signal(s) "
            f"{', '.join('#' + str(p['rank']) for p in clash)}. To separate "
            "two signals inside one window use Split; to make room, narrow "
            "the existing signal first (right-drag on its plot).")
    m = measure_window(ppm, raw_2d, corrected_2d, lo, hi, noise_std)
    p = _hand_made(m, next_rank(peaks), ORIGIN_ADDED)
    p.update(cluster_lo=lo, cluster_hi=hi,
             cluster_lo_ppm=m['start_ppm'], cluster_hi_ppm=m['end_ppm'])
    peaks.append(p)
    return p


def remove_added_signal(peaks, peak):
    """Remove a signal the analyst added. Detected signals are not removed
    here: dropping one is a grading decision (mark it Poor)."""
    if peak.get('manual_origin') != ORIGIN_ADDED:
        raise ValueError(f"Signal #{peak.get('rank')} was not added by hand.")
    peaks[:] = [p for p in peaks if p is not peak]
