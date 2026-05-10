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


def compute_peak_correlation_matrix(peaks, min_samples=3):
    """Pairwise Pearson r between every pair of detected peaks.

    Uses each peak's ``area_per_sample`` vector (integrated area per sample)
    as the signal to correlate.  Only peaks with at least *min_samples* valid
    area values are included.

    Parameters
    ----------
    peaks : list of dict
        Each dict must have ``'rank'`` (int) and ``'area_per_sample'``
        (array-like of length n_samples).
    min_samples : int
        Minimum number of samples required to compute a meaningful correlation.
        Pairs where either vector has fewer points are skipped.

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
        if arr.ndim != 1 or len(arr) < min_samples:
            continue
        eligible.append((int(p['rank']), arr))

    corr_dict = {}
    n = len(eligible)
    for i in range(n):
        ri, ai = eligible[i]
        if np.std(ai) == 0.0:
            continue
        for j in range(i + 1, n):
            rj, aj = eligible[j]
            if np.std(aj) == 0.0:
                continue
            r = float(np.corrcoef(ai, aj)[0, 1])
            if not np.isfinite(r):
                continue
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
