# processing/peak_alignment.py
"""
NMR Peak Alignment — Reference Signal Method.

align_to_reference_signal
    Aligns all samples to 0 ppm (or any reference ppm) using a known
    internal reference compound (DSS, TSP, TMSP).

    Strategy:
    1. Find the reference peak in the BLK samples (cleanest signal).
    2. Compute the mean shift needed to bring the BLK peak to ref_ppm.
    3. Apply that same global integer shift to ALL samples.

    This is the standard approach in 1H NMR metabolomics (Wishart 2022,
    Emwas 2019).  It is deterministic, reversible, and does not distort
    peak shapes or create inter-segment discontinuities.
"""
import numpy as np


def _shift_1d(y, shift_pts):
    """Integer-shift a 1-D array; fill edges with edge value."""
    if shift_pts == 0:
        return y.copy()
    n   = len(y)
    out = np.empty_like(y)
    if shift_pts > 0:
        out[:shift_pts] = y[0]
        out[shift_pts:] = y[:n - shift_pts]
    else:
        s = -shift_pts
        out[n - s:] = y[-1]
        out[:n - s] = y[s:]
    return out


def _normalized_overlap_score(ref, sample, shift_pts):
    """Correlation score for one integer shift using only overlapping points."""
    if shift_pts > 0:
        ref_seg = ref[shift_pts:]
        samp_seg = sample[:-shift_pts]
    elif shift_pts < 0:
        offset = -shift_pts
        ref_seg = ref[:-offset]
        samp_seg = sample[offset:]
    else:
        ref_seg = ref
        samp_seg = sample

    if ref_seg.size < 8 or samp_seg.size < 8:
        return -np.inf

    ref_seg = ref_seg - np.mean(ref_seg)
    samp_seg = samp_seg - np.mean(samp_seg)
    denom = np.linalg.norm(ref_seg) * np.linalg.norm(samp_seg)
    if denom <= 1e-12:
        return -np.inf
    return float(np.dot(ref_seg, samp_seg) / denom)


def _best_global_shift(ref, sample, max_shift_pts, use_derivative=True):
    """Find the best rigid shift that maximizes global peak agreement."""
    if max_shift_pts <= 0:
        return 0

    ref_work = np.asarray(ref, dtype=float)
    sample_work = np.asarray(sample, dtype=float)
    if use_derivative:
        ref_work = np.gradient(ref_work)
        sample_work = np.gradient(sample_work)

    best_shift = 0
    best_score = -np.inf
    for shift_pts in range(-max_shift_pts, max_shift_pts + 1):
        score = _normalized_overlap_score(ref_work, sample_work, shift_pts)
        if score > best_score:
            best_score = score
            best_shift = shift_pts
    return int(best_shift)


def global_peak_rigid_align(ppm, all_intensities, samples_info=None,
                            ref_idx=None, max_shift_ppm=0.03,
                            iterations=2, use_derivative=True):
    """
    Rigid global alignment driven by overall peak agreement.

    Each sample receives a single integer shift chosen to maximize its global
    correlation against the current reference spectrum, which is recomputed on
    the aligned matrix for a small number of iterations.

    Parameters
    ----------
    ppm : 1-D array
        Chemical shift axis.
    all_intensities : 2-D array
        Spectrum matrix with shape (n_points, n_samples).
    samples_info : list[dict] or None
        Sample metadata. BLK and excluded samples are ignored when building the
        reference spectrum, but still receive the final rigid shift.
    ref_idx : int or None
        Optional explicit reference sample index.
    max_shift_ppm : float
        Maximum rigid shift allowed for each sample.
    iterations : int
        Number of align-reference-refine passes.
    use_derivative : bool
        If True, align using the first derivative to reduce baseline bias.

    Returns
    -------
    aligned : ndarray
        Shifted copy of all_intensities.
    shifts_pts : ndarray
        Integer shift per sample.
    shifts_ppm : ndarray
        Shift per sample in ppm.
    reference : ndarray
        Final aligned reference spectrum.
    """
    arr = np.asarray(all_intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < 4 or arr.shape[1] < 2:
        zeros = np.zeros(arr.shape[1], dtype=int)
        return arr.copy(), zeros, zeros.astype(float), np.mean(arr, axis=1)

    ppm = np.asarray(ppm, dtype=float)
    ppm_step = abs(float(ppm[1] - ppm[0]))
    max_shift_pts = max(0, int(round(max_shift_ppm / max(ppm_step, 1e-12))))
    n_samples = arr.shape[1]

    if samples_info:
        valid_cols = [
            i for i, sample in enumerate(samples_info)
            if i < n_samples
            and str(sample.get('type', sample.get('label', 'Sample'))).upper() != 'BLK'
            and not sample.get('excluded', False)
        ]
    else:
        valid_cols = list(range(n_samples))
    if not valid_cols:
        valid_cols = list(range(n_samples))

    aligned = arr.copy()
    shifts_pts = np.zeros(n_samples, dtype=int)
    reference = aligned[:, ref_idx].copy() if ref_idx is not None else np.mean(aligned[:, valid_cols], axis=1)

    for _ in range(max(1, int(iterations))):
        for sample_idx in range(n_samples):
            if ref_idx is not None and sample_idx == ref_idx:
                shifts_pts[sample_idx] = 0
                aligned[:, sample_idx] = arr[:, sample_idx]
                continue
            best_shift = _best_global_shift(
                reference,
                arr[:, sample_idx],
                max_shift_pts=max_shift_pts,
                use_derivative=use_derivative,
            )
            shifts_pts[sample_idx] = best_shift
            aligned[:, sample_idx] = _shift_1d(arr[:, sample_idx], best_shift)

        if ref_idx is not None:
            reference = aligned[:, ref_idx].copy()
        else:
            reference = np.mean(aligned[:, valid_cols], axis=1)

    shifts_ppm = shifts_pts.astype(float) * ppm_step
    return aligned, shifts_pts, shifts_ppm, reference


def align_to_reference_signal(ppm, all_intensities, samples_info,
                               ref_ppm=0.0, window_ppm=0.05):
    """
    Align all samples so the internal reference signal sits at ref_ppm.

    Parameters
    ----------
    ppm             : 1-D array (n_pts,)
    all_intensities : 2-D array (n_pts, n_samples)
    samples_info    : list of dicts with keys 'type' and 'label'
    ref_ppm         : target ppm for the reference signal (default 0.0)
    window_ppm      : search window around ref_ppm (default ±0.05 ppm)

    Returns
    -------
    aligned  : (n_pts, n_samples) — shifted copy of all_intensities
    shift_ppm: float — shift applied (positive = spectrum moved right)
    n_pts_shift: int — shift in data points

    How it works
    ------------
    * Find all BLK-type sample indices.
    * Average the BLK spectra in [ref_ppm ± window_ppm].
    * Locate the intensity maximum in that window → current ref position.
    * Compute integer shift = target_idx − current_idx.
    * Apply the same shift to every sample.
    """
    if all_intensities.ndim == 1:
        all_intensities = all_intensities.reshape(-1, 1)

    n_pts, n_samples = all_intensities.shape
    ppm_step = abs(float(ppm[1] - ppm[0]))

    # ── Find BLK sample indices ───────────────────────────────────────────────
    blk_idxs = [i for i, s in enumerate(samples_info or [])
                if str(s.get('type', s.get('label', ''))).upper() == 'BLK'
                and i < n_samples]

    if not blk_idxs:
        # No BLK — use all samples as fallback
        print('   align_to_reference: no BLK samples found, using all samples')
        blk_idxs = list(range(n_samples))

    # ── Mean BLK spectrum in search window ───────────────────────────────────
    blk_mean   = np.mean(all_intensities[:, blk_idxs], axis=1)
    win_mask   = (ppm >= ref_ppm - window_ppm) & (ppm <= ref_ppm + window_ppm)
    if not np.any(win_mask):
        print(f'   align_to_reference: no data in {ref_ppm}±{window_ppm} ppm')
        return all_intensities.copy(), 0.0, 0

    # Current position of the reference peak in BLK
    win_indices  = np.where(win_mask)[0]
    local_max    = int(np.argmax(blk_mean[win_mask]))
    current_idx  = win_indices[local_max]
    current_ppm  = float(ppm[current_idx])

    # Target index (closest ppm point to ref_ppm)
    target_idx   = int(np.argmin(np.abs(ppm - ref_ppm)))
    shift_pts    = target_idx - current_idx
    shift_ppm    = float(shift_pts * ppm_step)

    print(f'   align_to_reference: BLK peak at {current_ppm:.5f} ppm → '
          f'{ref_ppm:.4f} ppm  (shift {shift_pts:+d} pts = '
          f'{shift_ppm*1000:+.2f} mppm)')

    if shift_pts == 0:
        return all_intensities.copy(), 0.0, 0

    # ── Apply shift to all samples ────────────────────────────────────────────
    aligned = np.empty_like(all_intensities)
    for s in range(n_samples):
        aligned[:, s] = _shift_1d(all_intensities[:, s], shift_pts)

    return aligned, shift_ppm, shift_pts


# ── Segment-wise (icoshift) alignment ────────────────────────────────────────

def _seg_xcorr_shift(ref_seg, samp_seg, max_shift_pts):
    """
    Circular cross-correlation shift to align samp_seg to ref_seg.
    Returns integer shift (points); positive = shift right.
    """
    n = len(ref_seg)
    if max_shift_pts <= 0 or n < 4:
        return 0
    max_s = min(max_shift_pts, n // 2)
    xcorr = np.fft.irfft(
        np.fft.rfft(ref_seg, n=n) * np.conj(np.fft.rfft(samp_seg, n=n)),
        n=n)
    # Candidate lags: 0, +1..+max_s, and n-max_s..n-1 (= -max_s..-1)
    pos_idx = np.arange(0, max_s + 1)
    neg_idx = np.arange(n - max_s, n)
    cands = np.concatenate([pos_idx, neg_idx])
    best = int(cands[int(np.argmax(xcorr[cands]))])
    return best if best <= n // 2 else best - n


def icoshift_align(ppm, all_intensities, ref_idx=None,
                   n_segments=50, max_shift_ppm=0.03):
    """
    Segment-wise (icoshift) spectral alignment.

    Divides the spectrum into `n_segments` equal-width segments and aligns
    each segment independently using circular cross-correlation.  The
    per-segment integer shifts are interpolated across the full spectrum and
    applied via linear resampling, giving a smooth piecewise warp without
    hard discontinuities at segment boundaries.

    Parameters
    ----------
    ppm              : 1-D array (n_pts,)
    all_intensities  : 2-D array (n_pts, n_samples)
    ref_idx          : int or None — reference sample (None = mean spectrum)
    n_segments       : int — number of non-overlapping segments
    max_shift_ppm    : float — maximum per-segment shift (ppm)

    Returns
    -------
    aligned          : (n_pts, n_samples) float array
    shifts_pts       : (n_segs, n_samples) int array — shift per segment
    seg_centers_ppm  : (n_segs,) float array — ppm at segment centres
    """
    if all_intensities.ndim == 1:
        all_intensities = all_intensities.reshape(-1, 1)
    n_pts, n_samples = all_intensities.shape
    ppm_step = abs(float(ppm[1] - ppm[0]))
    max_shift_pts = max(1, int(round(max_shift_ppm / ppm_step)))

    ref = (np.mean(all_intensities, axis=1)
           if ref_idx is None else all_intensities[:, ref_idx].copy())

    # Segment boundaries — enforce minimum 20 pts per segment so that
    # cross-correlation has enough data to produce a meaningful shift.
    # Silently reduce n_segments if the region is too narrow.
    min_seg_pts = 20
    n_segments  = min(n_segments, max(1, n_pts // min_seg_pts))
    seg_size = max(min_seg_pts, n_pts // n_segments)
    seg_starts = list(range(0, n_pts - seg_size + 1, seg_size))
    n_seg = len(seg_starts)
    seg_ends = [min(s + seg_size, n_pts) for s in seg_starts]
    seg_centers_idx = np.array([(s + e) / 2.0
                                for s, e in zip(seg_starts, seg_ends)])
    seg_centers_ppm = np.interp(seg_centers_idx, np.arange(n_pts), ppm)

    # Per-segment shifts
    shifts_pts = np.zeros((n_seg, n_samples), dtype=int)
    for s_idx in range(n_samples):
        if ref_idx is not None and s_idx == ref_idx:
            continue
        samp = all_intensities[:, s_idx]
        for seg_i, (start, end) in enumerate(zip(seg_starts, seg_ends)):
            shifts_pts[seg_i, s_idx] = _seg_xcorr_shift(
                ref[start:end], samp[start:end], max_shift_pts)

    # Apply smooth warp: interpolate shift at every data point, resample
    idx_axis = np.arange(n_pts, dtype=float)
    aligned = np.empty((n_pts, n_samples), dtype=float)
    for s_idx in range(n_samples):
        if ref_idx is not None and s_idx == ref_idx:
            aligned[:, s_idx] = all_intensities[:, s_idx].astype(float)
            continue
        shift_at_pt = np.interp(idx_axis, seg_centers_idx,
                                shifts_pts[:, s_idx].astype(float))
        src_idx = np.clip(idx_axis - shift_at_pt, 0, n_pts - 1)
        aligned[:, s_idx] = np.interp(
            src_idx, idx_axis, all_intensities[:, s_idx].astype(float))

    return aligned, shifts_pts, seg_centers_ppm