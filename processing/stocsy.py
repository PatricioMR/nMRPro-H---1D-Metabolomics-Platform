# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/stocsy.py
"""
Graph STOCSY — pairwise peak correlation matrix from area_per_sample vectors.

Two separate functions, kept independent:

  compute_peak_correlation_matrix  — Graph STOCSY  (peak × peak Pearson r)
                                     used for annotation scoring

  compute_stocsy_spectrum          — Seed STOCSY   (seed → full spectrum r)
                                     reserved for future pseudo-2D visualisation

Neither function imports from GUI layers.
"""
import numpy as np


def compute_peak_correlation_matrix(peaks, min_samples=3, included=None):
    """Pairwise Pearson r between every pair of detected peaks.

    Uses each peak's ``area_per_sample`` vector (integrated area per sample)
    as the signal to correlate.  Only peaks with at least *min_samples* valid
    area values are included.

    ``included`` must name the analytical samples. Blanks belong outside it,
    and leaving them in does not merely add noise -- it manufactures
    correlation. Every peak falls to nothing in a blank, so every pair shares
    that descent and reads as co-varying whether or not the molecules have
    anything to do with each other. Measured on 32 samples at a 20% CV with
    five blanks appended: two INDEPENDENT analytes read r = +0.74 to +0.80,
    against |r| < 0.25 over the samples alone. At a corroboration threshold of
    0.80 that would make almost everything look corroborated and the criterion
    would separate nothing.

    Parameters
    ----------
    peaks : list of dict
        Each dict must have ``'rank'`` (int) and ``'area_per_sample'``
        (array-like of length n_samples).
    min_samples : int
        Minimum number of samples required to compute a meaningful correlation.
        Pairs where either vector has fewer points are skipped.
    included : sequence of int, optional
        Sample indices to correlate over. None uses every column, which is
        correct only for a study with no blanks and nothing excluded.

    Returns
    -------
    dict
        ``{(rank_i, rank_j): float}`` — rank_i < rank_j always (upper triangle,
        symmetric by convention).  Only pairs where both peaks have enough data
        and non-zero variance are included.
    """
    eligible = []
    for p in peaks:
        aps = p.get('area_per_sample')
        if aps is None:
            continue
        arr = np.asarray(aps, dtype=float)
        if arr.ndim != 1:
            continue
        if included is not None:
            idx = [i for i in included if 0 <= i < len(arr)]
            if not idx:
                continue
            arr = arr[idx]
        if len(arr) < min_samples:
            continue
        eligible.append((int(p['rank']), arr))

    if len(eligible) < 2:
        return {}

    # One correlation over the whole matrix rather than a Python loop over
    # every pair. The pair count is quadratic -- a 1047-peak study is 548k
    # pairs, which took ~25 s one np.corrcoef call at a time and blocked the
    # detection overlay with no progress to show for it.
    ranks = [r for r, _ in eligible]
    M = np.vstack([a for _, a in eligible])
    keep = np.std(M, axis=1) > 0        # a flat vector has no correlation
    if keep.sum() < 2:
        return {}
    ranks = [r for r, k in zip(ranks, keep) if k]
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.corrcoef(M[keep])

    corr_dict = {}
    n = len(ranks)
    for i in range(n):
        ri = ranks[i]
        for j in range(i + 1, n):
            r = float(R[i, j])
            if not np.isfinite(r):
                continue
            rj = ranks[j]
            lo, hi = (ri, rj) if ri < rj else (rj, ri)
            corr_dict[(lo, hi)] = r

    return corr_dict


def compute_stocsy_spectrum(ppm, all_int, seed_ppm, seed_window_ppm=0.02):
    """Seed-based STOCSY: correlate a seed region against every spectral point.

    For each spectral point j, returns Pearson r between the seed vector
    (mean intensity across the seed window, per sample) and the intensity
    vector at point j (per sample).

    This function is **not used in annotation scoring**.  It is provided for
    future pseudo-2D visualisation where the user selects a seed peak and the
    output is displayed as a correlation-coloured 1D spectrum.

    Parameters
    ----------
    ppm : ndarray, shape (n_pts,)
    all_int : ndarray, shape (n_pts, n_samples)
    seed_ppm : float
    seed_window_ppm : float
        Half-width of the seed integration window in ppm.

    Returns
    -------
    ndarray, shape (n_pts,)
        Pearson r at each spectral point.  NaN where std == 0.
    """
    ppm = np.asarray(ppm, dtype=float)
    all_int = np.asarray(all_int, dtype=float)
    if all_int.ndim == 1:
        all_int = all_int[:, np.newaxis]

    mask = np.abs(ppm - seed_ppm) <= seed_window_ppm
    if not mask.any():
        return np.full(len(ppm), np.nan)

    seed_vec = all_int[mask, :].mean(axis=0)   # shape (n_samples,)
    if np.std(seed_vec) == 0.0:
        return np.full(len(ppm), np.nan)

    r_arr = np.full(len(ppm), np.nan)
    for j in range(len(ppm)):
        col = all_int[j, :]
        if np.std(col) == 0.0:
            continue
        r_arr[j] = float(np.corrcoef(seed_vec, col)[0, 1])

    return r_arr
