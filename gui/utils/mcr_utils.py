# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/utils/mcr_utils.py
"""
Shared MCR-ALS implementation used by AnnotationStage, QuantificationStage,
and FullSpectrumMCRStage.

Reference: Tauler, R. "Multivariate curve resolution applied to second order
data." Chemometrics and Intelligent Laboratory Systems, 30(1), 133-146 (1995).
DOI: 10.1016/0169-7439(95)00047-X
See also: de Juan, A.; Tauler, R. "Multivariate Curve Resolution (MCR) from
2000: Progress in Concepts and Applications." Critical Reviews in Analytical
Chemistry, 36(3-4), 163-176 (2006). DOI: 10.1080/10408340600970005
"""
import numpy as np

# ── When a resolved component may replace the raw measurement ─────────────────
# Two conditions decide whether MCR resolved something worth trusting over the
# detected peak itself, for scoring and for display alike:
#
#   K >= MCR_MIN_OVERLAP_K — MCR settled on more than one component, so there
#       was a genuine overlap to separate. With a single component the "resolved
#       component" is the peak itself and nothing was deconvolved.
#   pearson_r >= MCR_MIN_SHAPE_R — the component's shape matches this
#       candidate's database profile, so the separated signal is that compound's
#       and not a neighbour's.
#
# Kept here, next to the algorithm, so the scoring path and the fingerprint view
# cannot drift into two different definitions of "trustworthy component".
MCR_MIN_OVERLAP_K = 2
MCR_MIN_SHAPE_R   = 0.70

# Alignment freedom is searched wide (shift_tol_ppm) to absorb genuine pH and
# temperature drift. What matters is not how far the profile moved, but WHERE it
# landed: the annotation already matched this database peak to a detected signal
# at a known offset, so a component is consistent evidence when MCR's own shift
# reproduces that same offset — it worked on the same signal. When the two
# disagree by more than this window, MCR aligned onto something else and its
# metrics describe that other signal, not this match.
#
# An absolute cap on the shift was tried first and was wrong: it rejected the
# cases where MCR agreed exactly with the annotation, because a large but
# CORRECT shift simply mirrors a peak that sits far from its database position.
MCR_SHIFT_AGREEMENT_PPM = 0.01


def _is_singlet_multiplicity(mult_raw):
    """Return True when DB multiplicity denotes a singlet."""
    mult = str(mult_raw or '').lower().strip()
    return mult in ('s', 'singlet')


def mcr_als(X, n_components=1, max_iter=200, tol=1e-6, S_init=None,
            fixed_s_cols=None):
    """
    Simplified ALS-MCR for a region matrix X (n_pts × n_samples).

    Parameters
    ----------
    X            : ndarray (n_pts, n_samples)
    n_components : int — number of pure components
    max_iter     : int — maximum ALS iterations
    tol          : float — convergence tolerance (relative error change)
    S_init       : ndarray (n_pts, n_components) optional initial S matrix.
                   If None, uses the sample column with highest variance.
    fixed_s_cols : list of int — column indices in S that are held fixed
                   (not updated) throughout ALS iterations.  Used for
                   baseline and other constrained components.

    Returns
    -------
    C       : ndarray (n_samples, n_components) — concentrations / amplitudes
    S       : ndarray (n_pts, n_components)     — pure spectral profiles
    metrics : dict with 'lof' (lack-of-fit %), 'r2', 'n_iter'
    """
    n_pts, n_samples = X.shape
    n_components = max(1, n_components)
    if fixed_s_cols is None:
        fixed_s_cols = []

    if S_init is not None:
        S = np.clip(S_init.copy().astype(float), 0, None)
        if S.ndim == 1:
            S = S[:, np.newaxis]
        if S.shape[1] < n_components:
            var_per_sample = np.var(X, axis=0)
            order = np.argsort(var_per_sample)[::-1]
            for i in range(n_components - S.shape[1]):
                col = X[:, order[i % len(order)]].copy()
                col = np.clip(col, 0, None)
                S = np.hstack([S, col[:, np.newaxis]])
        elif S.shape[1] > n_components:
            S = S[:, :n_components]
        norms = np.linalg.norm(S, axis=0) + 1e-12
        S = S / norms
    else:
        var_per_sample = np.var(X, axis=0)
        order = np.argsort(var_per_sample)[::-1]
        cols = []
        for i in range(n_components):
            col = X[:, order[i % n_samples]].copy().astype(float)
            col = np.clip(col, 0, None)
            n = np.linalg.norm(col) + 1e-12
            cols.append(col / n)
        S = np.column_stack(cols)

    # Snapshot of fixed columns so they are never overwritten
    fixed_snapshots = {c: S[:, c].copy() for c in fixed_s_cols}

    prev_err = np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        C = np.linalg.lstsq(S, X, rcond=None)[0].T   # (n_samples, n_comp)
        C = np.clip(C, 0, None)
        S_new = np.linalg.lstsq(C, X.T, rcond=None)[0].T  # (n_pts, n_comp)
        S_new = np.clip(S_new, 0, None)
        # Restore fixed columns before normalisation
        for c, snap in fixed_snapshots.items():
            S_new[:, c] = snap
        norms = np.linalg.norm(S_new, axis=0) + 1e-12
        S = S_new / norms
        C = C * norms
        # Re-apply fixed column snapshots after normalisation
        for c, snap in fixed_snapshots.items():
            norm_c = np.linalg.norm(snap) + 1e-12
            S[:, c] = snap / norm_c
            C[:, c] = C[:, c] * norm_c
        X_hat = S @ C.T
        err = np.linalg.norm(X - X_hat)
        if np.isfinite(prev_err) and abs(prev_err - err) / (prev_err + 1e-12) < tol:
            break
        prev_err = err

    X_hat  = S @ C.T
    ss_res = np.sum((X - X_hat) ** 2)
    ss_tot = np.sum((X - X.mean()) ** 2)
    lof    = 100.0 * np.sqrt(ss_res / (np.sum(X ** 2) + 1e-30))
    # If ss_tot is negligible the region is flat — R² is undefined, not meaningful.
    if ss_tot < 1e-20:
        r2 = float('nan')
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    # Pooled 'r2' above uses a single grand mean (X.mean()) across every point
    # AND every sample — a compound that fits 2/10 samples well and 8/10 poorly
    # can still look "confirmed". r2_per_sample fits each sample column against
    # its own mean instead, so per-sample quality is not diluted by the pool.
    r2_per_sample = _per_sample_r2(X, S, C)
    _finite = r2_per_sample[np.isfinite(r2_per_sample)]
    r2_median = float(np.median(_finite)) if _finite.size else float('nan')

    metrics = {'lof': float(lof), 'r2': r2, 'n_iter': n_iter, 'flat': ss_tot < 1e-20,
               'r2_per_sample': r2_per_sample.tolist(), 'r2_median': r2_median}
    return C, S, metrics


def _per_sample_r2(X, S, C):
    """R² of the MCR reconstruction for each sample column independently.

    Each sample is scored against its own mean (unlike the pooled 'r2' in
    mcr_als, which uses a single grand mean over the whole X matrix).
    """
    X_hat = S @ C.T
    n_samples = X.shape[1]
    r2s = np.full(n_samples, np.nan)
    for j in range(n_samples):
        xj, xhj = X[:, j], X_hat[:, j]
        ss_tot_j = float(np.sum((xj - xj.mean()) ** 2))
        if ss_tot_j > 1e-20:
            ss_res_j = float(np.sum((xj - xhj) ** 2))
            r2s[j] = 1.0 - ss_res_j / ss_tot_j
    return r2s


def simulate_lorentzian_profile(sticks, ppm_arr, hwhm=0.003):
    """
    Build a simulated spectral profile on ppm_arr from a list of
    (ppm, rel_intensity) sticks.  Uses Lorentzian line shape.
    Returns ndarray (n_pts,) normalised to max=1.
    """
    y = np.zeros(len(ppm_arr))
    for sp, si in sticks:
        y += si * hwhm**2 / ((ppm_arr - sp)**2 + hwhm**2)
    m = y.max()
    if m > 0:
        y /= m
    return y


def _baseline_profile(X):
    """
    Estimate a per-point baseline as the per-point minimum across all
    samples, then smooth with a running minimum over a wide window.
    Returns ndarray (n_pts,) normalised to max=1 (or zeros if flat).
    """
    n_pts = X.shape[0]
    # Per-point minimum across samples
    base = np.min(X, axis=1).astype(float)
    base = np.clip(base, 0, None)
    # Running minimum (wide window = 10 % of region length)
    w = max(3, n_pts // 10)
    smoothed = np.array([
        base[max(0, i - w): min(n_pts, i + w + 1)].min()
        for i in range(n_pts)
    ], dtype=float)
    m = smoothed.max()
    if m > 1e-12:
        smoothed /= m
    return smoothed


def compute_j_score(s_component, ppm_arr, db_entry, field_mhz=600.0):
    """
    Verify J-coupling of an MCR component against the DB expected value.

    Finds local maxima in s_component, measures their pairwise spacings in Hz,
    and computes how well they match the expected J from db_entry.

    Parameters
    ----------
    s_component : ndarray (n_pts,) — MCR pure component profile
    ppm_arr     : ndarray (n_pts,) — ppm axis
    db_entry    : dict with keys 'type'/'multiplicity', 'j_coupling', 'protons'
    field_mhz   : float — spectrometer frequency in MHz (ppm → Hz conversion)

    Returns
    -------
    j_score     : float 0–1  (1 = perfect J match, 0 = no match / not applicable)
    measured_j  : float or None — measured J in Hz (mean of observed spacings)
    expected_j  : float or None — J from DB in Hz
    """
    mult = (db_entry.get('type') or db_entry.get('multiplicity') or 's')
    if _is_singlet_multiplicity(mult):
        # For singlets, J is not applicable; expose as 0.0 (requested behavior).
        return 0.0, None, None

    j_raw = db_entry.get('j_coupling')
    expected_j = None
    if j_raw:
        try:
            expected_j = float(str(j_raw).split(',')[0].strip())
        except Exception:
            pass
    if expected_j is None or expected_j <= 0:
        return 0.5, None, None   # J unknown (non-singlet) → neutral score

    # Find local maxima in the component
    n = len(s_component)
    if n < 5:
        return 0.5, None, expected_j

    # Smooth slightly to avoid noise peaks
    w = max(3, n // 20)
    kernel = np.ones(w) / w
    smooth = np.convolve(s_component, kernel, mode='same')

    threshold = smooth.max() * 0.15
    peaks = []
    for i in range(1, n - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > threshold:
            peaks.append(i)

    if len(peaks) < 2:
        return 0.5, None, expected_j  # can't measure spacing

    # Convert peak indices to ppm, then to Hz spacings
    peak_ppms = ppm_arr[peaks]
    spacings_hz = np.abs(np.diff(np.sort(peak_ppms))) * field_mhz
    measured_j = float(np.median(spacings_hz))

    # Score: Gaussian decay from expected_j, tolerance = 20% of expected
    tol_hz = max(1.0, expected_j * 0.20)
    j_score = float(np.exp(-0.5 * ((measured_j - expected_j) / tol_hz) ** 2))

    return j_score, measured_j, expected_j


def align_profile_to_spectrum(profile, ppm_arr, exp_mean, shift_tol_ppm=0.02):
    """
    Find the ppm shift in [-shift_tol_ppm, +shift_tol_ppm] that maximises
    Pearson correlation between a shifted copy of *profile* and *exp_mean*.

    A positive shift moves the profile peaks to higher ppm (downfield).

    Parameters
    ----------
    profile       : ndarray (n_pts,) — DB-simulated profile on ppm_arr grid
    ppm_arr       : ndarray (n_pts,) — ppm axis (ascending or descending)
    exp_mean      : ndarray (n_pts,) — mean experimental spectrum on same grid
    shift_tol_ppm : float — maximum shift magnitude (ppm) to search

    Returns
    -------
    shifted_profile : ndarray (n_pts,) — best-shifted profile (norm max=1)
    best_shift      : float — applied shift in ppm (+= downfield)
    best_corr       : float — Pearson r at best shift
    """
    if profile.std() < 1e-10 or exp_mean.std() < 1e-10:
        return profile.copy(), 0.0, 0.0

    ppm_step = abs(float(ppm_arr[1] - ppm_arr[0])) if len(ppm_arr) > 1 else 0.001
    n_steps  = max(1, int(round(shift_tol_ppm / ppm_step)))
    shifts   = np.arange(-n_steps, n_steps + 1) * ppm_step

    # Work in ascending ppm order for np.interp (requires monotone xp)
    ascending = ppm_arr[0] < ppm_arr[-1]
    if ascending:
        ppm_asc  = ppm_arr
        prof_asc = profile
    else:
        ppm_asc  = ppm_arr[::-1]
        prof_asc = profile[::-1]

    best_shift = 0.0
    best_corr  = -2.0
    best_prof  = profile.copy()

    for sh in shifts:
        # Shift the DB profile's ppm axis by sh (move peaks by +sh ppm)
        shifted_ppm_asc = ppm_asc + sh
        # Interpolate shifted profile back onto the original ppm grid
        p_asc = np.interp(ppm_asc, shifted_ppm_asc, prof_asc,
                          left=0.0, right=0.0)
        # Restore original axis order
        p_shifted = p_asc if ascending else p_asc[::-1]

        if p_shifted.std() < 1e-10:
            continue
        r = float(np.corrcoef(p_shifted, exp_mean)[0, 1])
        if r > best_corr:
            best_corr  = r
            best_shift = sh
            best_prof  = p_shifted

    m = best_prof.max()
    if m > 1e-12:
        best_prof = best_prof / m
    return best_prof, best_shift, best_corr


def mcr_annotation_region(X, db_profiles, r2_threshold=0.95,
                           ppm_arr=None, shift_tol_ppm=0.0,
                           use_baseline=True, db_entries=None,
                           field_mhz=600.0, max_k_override=None):
    """
    Run MCR-ALS on a spectral region with automatic K selection.

    Increases K from 1 up to len(db_profiles) until R² >= r2_threshold.
    Each component is initialised from the corresponding DB profile.

    Optionally:
    - Aligns each DB profile to the experimental mean before MCR
      (shift_tol_ppm > 0 enables this).
    - Adds a fixed baseline component so baseline drift does not contaminate
      metabolite components (use_baseline=True).
    - Verifies J-coupling spacing of each extracted component against the DB
      expected J value (db_entries required).

    Parameters
    ----------
    X              : ndarray (n_pts, n_samples)
    db_profiles    : list of ndarray (n_pts,) — one simulated profile per
                     DB candidate
    r2_threshold   : float — target R²
    ppm_arr        : ndarray (n_pts,) — ppm axis, required for shift alignment
    shift_tol_ppm  : float — ppm search window for shift alignment (0 = disabled)
    use_baseline   : bool — add a fixed baseline component to MCR
    db_entries     : list of dicts — DB candidate dicts with J/multiplicity info
    field_mhz      : float — spectrometer frequency for Hz↔ppm conversion

    Returns
    -------
    S           : ndarray (n_pts, K)  — metabolite components only (no baseline)
    C           : ndarray (n_samples, K)
    metrics     : dict with 'lof', 'r2', 'n_iter'
    K           : int — final number of metabolite components
    assignments : list of (candidate_idx, pearson_r, shift_ppm,
                            j_score, measured_j_hz, expected_j_hz,
                            combined_conf) per component
    aligned_profiles : list of ndarray — DB profiles after shift alignment
    """
    max_k    = max(1, min(len(db_profiles),
                         max_k_override if max_k_override else len(db_profiles)))
    exp_mean = X.mean(axis=1) if X.ndim == 2 else X.copy()

    # ── PPM shift alignment ───────────────────────────────────────────────────
    aligned_profiles = []
    shift_applied    = []   # (best_shift, best_corr) per candidate
    for prof in db_profiles:
        if shift_tol_ppm > 0 and ppm_arr is not None and len(ppm_arr) == len(prof):
            ap, sh, cr = align_profile_to_spectrum(
                prof, ppm_arr, exp_mean, shift_tol_ppm)
        else:
            ap = prof.copy()
            sh = 0.0
            m  = ap.max()
            cr = (float(np.corrcoef(ap / m if m > 0 else ap, exp_mean)[0, 1])
                  if ap.std() > 0 and exp_mean.std() > 0 else 0.0)
        aligned_profiles.append(ap)
        shift_applied.append((sh, cr))

    # ── Baseline profile (fixed during ALS) ───────────────────────────────────
    if use_baseline and X.shape[0] >= 4:
        base_prof = _baseline_profile(X)
    else:
        base_prof = None

    # Pre-subtract baseline from X so MCR only fits peak signals.
    # This preserves metabolite component amplitudes (unlike a fixed ALS column
    # which competes with metabolite components for concentration).
    if base_prof is not None:
        # Estimate baseline amplitude per sample via least-squares
        base_col = base_prof[:, np.newaxis]  # (n_pts, 1)
        base_amps = np.linalg.lstsq(base_col, X, rcond=None)[0]  # (1, n_samples)
        base_amps = np.clip(base_amps, 0, None)
        X_fit = np.clip(X - base_col * base_amps, 0, None)
    else:
        X_fit = X

    # Pre-rank candidates by correlation with the mean spectrum so that
    # MCR is always initialized with the K profiles most similar to what
    # is actually in the data.  Without this, the first K by ppm order
    # would be used, causing candidates farther down the list (e.g. Alanine
    # when Lipoic acid appears first) to never influence the MCR fit.
    exp_norm = exp_mean / (exp_mean.max() + 1e-12)
    pre_corr = []
    for prof in aligned_profiles:
        if prof.std() > 0 and exp_norm.std() > 0:
            pre_corr.append(float(np.corrcoef(prof / (prof.max() + 1e-12),
                                              exp_norm)[0, 1]))
        else:
            pre_corr.append(0.0)
    rank_order = np.argsort(pre_corr)[::-1].tolist()  # best-first

    best_result = None

    for K in range(1, max_k + 1):
        # Use the top-K candidates (by pre-correlation) as S initialisation
        top_k_idx = rank_order[:K]
        init_profiles = [aligned_profiles[i] for i in top_k_idx]
        S_init = np.column_stack(init_profiles)

        C_met, S_met, metrics = mcr_als(
            X_fit, n_components=K, S_init=S_init)

        best_result = (S_met, C_met, metrics, K)
        if metrics['r2'] >= r2_threshold:
            break

    S, C, metrics, K = best_result

    # ── Match each component to best DB candidate (unique assignment) ────────
    # Build a combined score matrix: rows=components, cols=candidates
    # Score = 0.6 * Pearson(S_k, prof_i) + 0.4 * J_score(S_k, entry_i)
    # Use the Hungarian-style greedy unique assignment so two components
    # cannot both claim the same candidate.
    n_cands = len(aligned_profiles)
    score_matrix = np.zeros((K, n_cands))
    corr_matrix  = np.zeros((K, n_cands))
    j_matrix     = np.full((K, n_cands), 0.5)
    meas_j_mat   = [[None] * n_cands for _ in range(K)]
    exp_j_mat    = [[None] * n_cands for _ in range(K)]

    for k in range(K):
        s_k = S[:, k]
        for i, prof in enumerate(aligned_profiles):
            if len(prof) == len(s_k) and s_k.std() > 0 and prof.std() > 0:
                r = float(np.corrcoef(s_k, prof)[0, 1])
            else:
                r = 0.0
            corr_matrix[k, i] = r
            entry = (db_entries[i] if db_entries and i < len(db_entries) else {})
            if ppm_arr is not None and len(ppm_arr) == len(s_k):
                js, mj, ej = compute_j_score(s_k, ppm_arr, entry, field_mhz)
            else:
                js, mj, ej = 0.5, None, None
            j_matrix[k, i]    = js
            meas_j_mat[k][i]  = mj
            exp_j_mat[k][i]   = ej
            # For singlets, J is non-applicable and shown as 0.0; avoid
            # penalising rank by using correlation-only score in this case.
            mult = (entry.get('type') or entry.get('multiplicity') or 's')
            if _is_singlet_multiplicity(mult):
                score_matrix[k, i] = max(0.0, r)
            else:
                score_matrix[k, i] = 0.60 * max(0.0, r) + 0.40 * js

    # Greedy unique assignment: pick highest score, mark both row+col used
    assignments  = [None] * K
    used_cands   = set()
    order = sorted(range(K), key=lambda k: score_matrix[k].max(), reverse=True)
    for k in order:
        scores = score_matrix[k].copy()
        scores[list(used_cands)] = -1.0   # forbid already-claimed candidates
        best_i = int(np.argmax(scores))
        used_cands.add(best_i)
        sh, _     = shift_applied[best_i]
        pearson_r = corr_matrix[k, best_i]
        j_score   = j_matrix[k, best_i]
        meas_j    = meas_j_mat[k][best_i]
        exp_j     = exp_j_mat[k][best_i]
        combined  = score_matrix[k, best_i]
        assignments[k] = (best_i, pearson_r, sh,
                          j_score, meas_j, exp_j, combined)

    # Expose per-candidate scores for the "other candidates" panel
    cand_scores = []
    for ci in range(n_cands):
        # Best matching component for this candidate
        best_k = int(np.argmax(corr_matrix[:, ci])) if K > 0 else 0
        sh_ci, _ = shift_applied[ci]
        cand_scores.append({
            'corr':     float(corr_matrix[best_k, ci]),
            'shift':    float(sh_ci),
            'j_score':  float(j_matrix[best_k, ci]),
            'meas_j':   meas_j_mat[best_k][ci],
            'exp_j':    exp_j_mat[best_k][ci],
            'combined': float(score_matrix[best_k, ci]),
        })

    return S, C, metrics, K, assignments, aligned_profiles, cand_scores
