# processing/peak_detection.py
"""
Peak detection for NMR spectroscopy with baseline correction.

Features:
  - Baseline correction (arPLS, SNIP) before detection
  - Peak separation by valley points (foot-head-foot)
  - SNR-based thresholding
  - Individual peak integration
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import traceback

from .baseline_correction import correct_baseline, arPLS, normalize_method
from .peak_alignment import global_peak_rigid_align
from .sample_selection import included_sample_indices
from .signal_quality import (
    classify_signal_quality, measure_fwhm_points, corrected_profile)
from .spectral_regions import format_regions, split_excluded


def diagnose_array(arr, name):
    """Diagnostic helper to print array details."""
    print(f"\n[DIAG] {name}")
    print(f"  Type: {type(arr)}")

    if hasattr(arr, 'shape'):
        print(f"  Shape: {arr.shape}")
        print(f"  Dimensions: {arr.ndim}")
        print(f"  Total size: {arr.size}")
        print(f"  dtype: {arr.dtype}")
        if arr.size > 0:
            print(f"  Min: {np.min(arr):.4f}")
            print(f"  Max: {np.max(arr):.4f}")
            print(f"  First 5: {arr.flatten()[:5]}")
    else:
        print(f"  [WARN] Not a NumPy array: {type(arr)}")
        try:
            arr_np = np.array(arr)
            print(f"  Converted to NumPy: shape {arr_np.shape}")
        except:
            print(f"  [ERROR] Could not convert to NumPy")





def foot_line_area(x, y):
    """``(area, line)`` of one trace over a signal's window: the area above the
    straight line from the window's first point to its last, only where the
    trace stands above it (what is below the line is not signal), by the
    trapezoid rule. The line is returned so a figure can draw the very region
    the number came from.

    This is the area the Signal List reports (per sample, then its mean) and
    what its viewer shades; one function, so they cannot differ.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 or len(x) != n:
        return 0.0, y.copy()
    line = y[0] + (y[-1] - y[0]) / (n - 1) * np.arange(n)
    above = np.maximum(y - line, 0.0)
    return abs(float(np.trapezoid(above, x))), line


def peak_measures(ppm, corrected_2d, start_idx, end_idx, noise_std, index=None):
    """What the Signal List reports about a signal on the window
    ``[start_idx, end_idx]`` of the baseline-corrected matrix (points x
    samples): the per-sample areas and their mean, the height on the mean
    spectrum above the line between the window's ends, the mean intensity at
    the apex and its SNR. The same arithmetic as detection's last step, so a
    signal whose boundaries the analyst moved can be re-measured without
    running detection again. ``index`` is kept when inside the window, else
    the apex of the mean spectrum.
    """
    ppm = np.asarray(ppm, dtype=float)
    Y = np.asarray(corrected_2d, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    lo, hi = sorted((int(start_idx), int(end_idx)))
    x = ppm[lo:hi + 1]
    areas = np.array([foot_line_area(x, Y[lo:hi + 1, s])[0]
                      for s in range(Y.shape[1])])
    mean = Y.mean(axis=1)
    apex = (int(index) if index is not None and lo <= int(index) <= hi
            else lo + int(np.argmax(mean[lo:hi + 1])))
    height = float(max(np.max(mean[lo:hi + 1]) - 0.5 * (mean[lo] + mean[hi]), 0.0))
    mean_int = float(np.mean(Y[apex, :]))
    return {'area_per_sample': areas, 'area': float(np.mean(areas)),
            'height': height, 'mean_intensity': mean_int,
            'snr': mean_int / noise_std if noise_std and noise_std > 0 else 0.0,
            'index': apex, 'ppm': float(ppm[apex])}


def broad_removed_fractions(x, baseline_window, areas):
    """Per sample, the share of the raw area inside a window that the baseline
    model removed: ``|∫baseline| / (|∫baseline| + signal area)``. 0 = nothing
    removed, →1 = the window was almost entirely broad component.
    ``baseline_window`` is points x samples over the window ``x``."""
    B = np.asarray(baseline_window, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    fracs = []
    for s in range(B.shape[1]):
        a_base = abs(float(np.trapezoid(B[:, s], x)))
        denom = a_base + float(areas[s])
        fracs.append(a_base / denom if denom > 1e-30 else 0.0)
    return np.array(fracs)


def separate_overlapping_peaks(intensity, peak_indices, ppm, noise_std=None,
                               method="auto", baseline_level=0.0):
    """
    Two-phase foot-head-foot boundary algorithm — one record per apex.

    Phase 1 — Cluster detection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each detected apex, walk outward (left and right) until the
    baseline-corrected signal drops to ≤ 3σ.  All apices whose outer
    envelopes overlap are merged into one *cluster* with shared outer feet
    [cluster_lo, cluster_hi].

    Phase 2 — Per-apex boundaries inside each cluster
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *  The leftmost peak in a cluster gets the cluster's left baseline foot
       as its left boundary.
    *  The rightmost peak gets the cluster's right baseline foot as its right
       boundary.
    *  Every adjacent pair of peaks within a cluster shares the global
       minimum between them (the valley) as their mutual inner boundary.

    This gives one record per apex with meaningful individual boundaries that
    annotation can use to identify multiplet lines (doublets, triplets …).

    References
    ----------
    Cobas et al., "A new general-purpose fully automatic baseline-correction
    procedure for 1D and 2D NMR data", JMRB 183 (2006).
    Günther et al., "AUREMOL: automatic structure elucidation of proteins
    from NMR spectra", Prog. NMR Spectrosc. 44 (2004).
    Farès et al., "Automated NMR peak-picking using a rule-based algorithm",
    Magn. Reson. Chem. 49 (2011).

    Parameters
    ----------
    intensity    : ndarray (n,) — baseline-corrected mean spectrum
    peak_indices : list of int  — detected apex indices
    ppm          : ndarray (n,) — chemical shift axis (may be decreasing)
    noise_std    : float or None — noise σ; estimated from edges if None
    method       : kept for API compatibility; ignored

    Returns
    -------
    separated_peaks : list of dict  (one entry per apex, ordered by index)
        Standard keys : index, ppm, height, area, start_idx, end_idx,
                        start_ppm, end_ppm, width_ppm,
                        left_valley_idx, right_valley_idx,
                        left_valley_ppm, right_valley_ppm
        Quality keys  : gaussian_r2, gaussian_quality ('good'/'poor'/'overlap'),
                        signal_quality ('Good'/'Average'/'Poor'),
                        signal_quality_score, signal_quality_detail, fwhm_pts,
                        left_grounded, right_grounded, baseline_grounded
        Cluster keys  : n_cluster_peaks, cluster_rank,
                        cluster_lo, cluster_hi,
                        cluster_lo_ppm, cluster_hi_ppm
    """
    n = len(intensity)
    if len(peak_indices) == 0:
        return []

    sorted_indices = sorted(peak_indices)

    # Effective noise — shared baseline threshold
    eff_noise = (float(noise_std)
                 if (noise_std is not None and float(noise_std) > 0)
                 else estimate_noise_region(intensity))
    # Foot = baseline level + 3σ. On a per-sample-max reference the quiet
    # baseline sits above zero (upward bias of the max of N draws); ignoring it
    # would never ground the feet there. baseline_level defaults to 0 (mean path).
    thr = float(baseline_level) + 3.0 * eff_noise      # intensity ≤ thr → baseline

    # Where the feet are is decided on a copy smoothed over half a linewidth;
    # heights, areas and grades are read on the trace itself. On a per-sample
    # maximum the top of a real line follows ONE sample's noise, point to
    # point, while ``noise_std`` is the quiet region's spread of the maximum
    # -- half as large. The walk below took each such wiggle for a valley:
    # on the reference bench filled to 100 analytes, 29 of 264 peaks (SNR
    # 52 in median) came out 3 to 11 points wide against 42, their height a
    # wiggle's, graded Poor and barred from annotation -- valine's doublet
    # among them. Half a linewidth leaves a line's shape and its valleys.
    _lw = _linewidth_points(intensity, sorted_indices)
    _win = int(_lw // 2) | 1 if _lw else 0
    walk_ref = (savgol_filter(np.asarray(intensity, float), _win, 2)
                if _win >= 5 and _win < n else np.asarray(intensity, float))

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1a  ·  outer baseline feet for every apex
    # ──────────────────────────────────────────────────────────────────────────
    outer_left  = {}   # apex → left  baseline-crossing index
    outer_right = {}   # apex → right baseline-crossing index
    lg_map      = {}   # apex → bool  left  grounded
    rg_map      = {}   # apex → bool  right grounded

    # Foot = first of (baseline return ≤ thr) OR (valley: signal turns back up).
    # The rise margin keeps noise wiggles from being mistaken for a valley.
    rise_margin = 2.0 * eff_noise

    # A valley is also a dip the line's own height notices: the top of a
    # tall line on a per-sample maximum carries structure far above the
    # noise (the internal standard, 5.4e6 high, a 2e5 dip where samples
    # sit a fraction of a point apart), and at twice the noise alone the
    # walk stopped on it. The dips between apices are decided below, on
    # the trace, by the minimum between them -- not by this walk.
    VALLEY_FRACTION = 0.05

    def _walk(apex, step):
        """Walk outward from apex (step ±1). Returns (foot_idx, grounded)."""
        margin = max(rise_margin, VALLEY_FRACTION * (float(walk_ref[apex])
                                                      - float(baseline_level)))
        run_min_idx, run_min_val = apex, float(walk_ref[apex])
        j = apex + step
        while 0 <= j < n:
            v = float(walk_ref[j])
            if v <= thr:                       # reached baseline
                return j, True
            if v < run_min_val:
                run_min_val, run_min_idx = v, j
            elif v > run_min_val + margin:     # signal rising again → valley
                return run_min_idx, bool(run_min_val <= thr)
            j += step
        # Hit the spectrum edge without grounding → lowest point seen.
        return run_min_idx, bool(run_min_val <= thr)

    for apex in sorted_indices:
        # from the smoothed copy's own top: one point off it on a tall line
        # is uphill by more than the margin, and the walk would call the
        # apex its own valley (DSS, SNR 870, went to height 0 and was lost)
        _lo, _hi = max(0, apex - _win), min(n, apex + _win + 1)
        top = _lo + int(np.argmax(walk_ref[_lo:_hi]))
        lf, lg = _walk(top, -1)
        rf, rg = _walk(top, +1)
        lf, rf = min(lf, apex), max(rf, apex)
        outer_left[apex]  = lf
        outer_right[apex] = rf
        lg_map[apex]      = lg
        rg_map[apex]      = rg

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1b  ·  group apices into clusters by the saddle between them
    # ──────────────────────────────────────────────────────────────────────────
    # Two adjacent apices belong to the same (overlapping) cluster when the
    # minimum between them does NOT return to baseline (saddle > thr). If the
    # valley grounds (≤ thr) the peaks are resolved → separate clusters. This is
    # robust with the valley-stopping foot walk above (where an isolated peak
    # stops at its own saddle instead of extending across a neighbour).
    clusters = []     # list of {'lo', 'hi', 'apices': [...]}
    cur = [sorted_indices[0]]
    for a, b in zip(sorted_indices, sorted_indices[1:]):
        saddle = float(np.min(intensity[a:b + 1])) if b > a else float(intensity[a])
        if saddle > thr:
            cur.append(b)          # overlapping → same cluster
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)
    clusters = [{'lo': outer_left[ap[0]], 'hi': outer_right[ap[-1]], 'apices': ap}
                for ap in clusters]

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2  ·  per-apex boundaries
    # ──────────────────────────────────────────────────────────────────────────
    # Outer peak  → baseline foot as outer boundary.
    # Adjacent pair inside a cluster → valley (global min between them) as
    #   their shared inner boundary.  The valley is the foot of the right-hand
    #   peak on its left side and of the left-hand peak on its right side.
    separated_peaks = []

    # Outer (free-flank) boundary rule — user-defined: the lateral limit is the
    # point where the flank, descending from the apex, drops to 5 % of the peak
    # height ABOVE the local baseline (the value at the cluster's baseline foot
    # on that side). If a valley/baseline foot is reached before the 5 % level
    # (overlap), the foot wins — the valley is the boundary. This is a relative,
    # scale-aware window that behaves consistently across tall/short peaks and a
    # sloped baseline, unlike an absolute σ threshold.
    FOOT_FRACTION = 0.05

    def _foot_5pct(apex, foot_idx, step):
        """Index where the flank from ``apex`` toward ``foot_idx`` first reaches
        baseline + 5 % of the apex height; capped at ``foot_idx`` (valley)."""
        base = float(intensity[foot_idx])
        thr5 = base + FOOT_FRACTION * (float(intensity[apex]) - base)
        j = apex
        while j != foot_idx:
            if float(intensity[j]) <= thr5:
                return j
            j += step
        return foot_idx

    for cl in clusters:
        apices   = sorted(cl['apices'])
        c_lo     = cl['lo']
        c_hi     = cl['hi']
        n_ap     = len(apices)

        for rank, apex in enumerate(apices):

            # ── Left boundary ─────────────────────────────────────────────────
            # Outer (free) flank → 5 %-height crossing; inner → valley.
            if rank == 0:
                left_idx      = _foot_5pct(apex, c_lo, -1)
                left_grounded = lg_map[apex]
            else:
                prev      = apices[rank - 1]
                seg       = intensity[prev:apex + 1]
                left_idx  = prev + int(np.argmin(seg))
                left_grounded = float(intensity[left_idx]) <= thr

            # ── Right boundary ────────────────────────────────────────────────
            if rank == n_ap - 1:
                right_idx      = _foot_5pct(apex, c_hi, +1)
                right_grounded = rg_map[apex]
            else:
                nxt        = apices[rank + 1]
                seg        = intensity[apex:nxt + 1]
                right_idx  = apex + int(np.argmin(seg))
                right_grounded = float(intensity[right_idx]) <= thr

            if n_ap == 1:
                # ── Isolated peak → both feet at the 5 %-height level ──────────
                # Free flanks on both sides → the two crossings land at the same
                # height, giving a horizontal integration baseline naturally.
                left_val  = float(intensity[left_idx])
                right_val = float(intensity[right_idx])
                foot_level = max(left_val, right_val)
                start_idx = min(left_idx, apex)
                end_idx   = max(right_idx, apex)
                height = float(intensity[apex]) - foot_level
                peak_reg = intensity[start_idx:end_idx + 1]
                x_reg    = ppm[start_idx:end_idx + 1]
                corr_reg = np.maximum(peak_reg - foot_level, 0.0)
                area = (abs(float(np.trapezoid(corr_reg, x_reg)))
                        if len(x_reg) > 1 else 0.0)
            else:
                # ── Cluster member (multiplet line / shoulder) ────────────────
                # The valley IS the boundary; do NOT raise to the neighbour-side
                # foot (that would zero an overlapped line). Linear baseline
                # between the two feet (one of which is the shared valley).
                start_idx = min(left_idx, apex)
                end_idx   = max(right_idx, apex)
                lv = float(intensity[left_idx])
                rv = float(intensity[right_idx])
                # baseline height under the apex (linear between feet)
                if right_idx != left_idx:
                    t = (apex - left_idx) / (right_idx - left_idx)
                    foot_bl_apex = lv + t * (rv - lv)
                else:
                    foot_bl_apex = lv
                height = float(intensity[apex]) - foot_bl_apex
                peak_reg = intensity[start_idx:end_idx + 1]
                x_reg    = ppm[start_idx:end_idx + 1]
                if end_idx > start_idx and right_idx != left_idx:
                    slope = (rv - lv) / (right_idx - left_idx)
                    bl_reg = lv + slope * np.arange(end_idx - start_idx + 1)
                    corr_reg = np.maximum(peak_reg - bl_reg, 0.0)
                else:
                    corr_reg = peak_reg
                area = (abs(float(np.trapezoid(corr_reg, x_reg)))
                        if len(x_reg) > 1 else 0.0)

            # ── Gaussian quality ──────────────────────────────────────────────
            # For isolated peaks (n_ap == 1) run the Gaussian fit to assess
            # shape quality.  For peaks inside a cluster the boundaries are
            # already at valleys — the Gaussian fit on each individual line
            # is still meaningful and helps distinguish clean multiplet lines
            # (good R²) from badly overlapping regions (poor R²).
            ppm_reg = ppm[start_idx:end_idx + 1]
            int_reg = intensity[start_idx:end_idx + 1]
            # 5 points is the minimum a 3-parameter line-shape fit needs
            # (fit_peak_quality guards n<5). Requiring 8 wrongly forced narrow,
            # sharp singlets to gaussian_r2=0 → 'overlap'/Poor.
            if end_idx - start_idx + 1 >= 5:
                gauss_r2, _fit_model, sym_score = fit_peak_quality(ppm_reg, int_reg)
            else:
                gauss_r2, sym_score = 0.0, 0.5
            # Both shape fit and symmetry must qualify for Good/Average.
            if gauss_r2 >= 0.90 and sym_score >= 0.75:
                gauss_qual = 'good'
            elif gauss_r2 >= 0.70 and sym_score >= 0.50:
                gauss_qual = 'poor'
            else:
                gauss_qual = 'overlap'

            separated_peaks.append({
                # Core
                'index':    int(apex),
                'ppm':      float(ppm[apex]),
                'height':   float(max(0.0, height)),
                'area':     float(area),
                'start_idx': int(start_idx),
                'end_idx':   int(end_idx),
                'start_ppm': float(ppm[start_idx]),
                'end_ppm':   float(ppm[end_idx]),
                'width_ppm': float(abs(ppm[end_idx] - ppm[start_idx])),
                # Foot indices (kept as "valley" for backward compat)
                'left_valley_idx':  int(left_idx),
                'right_valley_idx': int(right_idx),
                'left_valley_ppm':  float(ppm[left_idx]),
                'right_valley_ppm': float(ppm[right_idx]),
                # Quality
                'gaussian_r2':       float(gauss_r2),
                'symmetry_score':    float(sym_score),
                'gaussian_quality':  gauss_qual,
                'left_grounded':     bool(left_grounded),
                'right_grounded':    bool(right_grounded),
                'baseline_grounded': bool(left_grounded and right_grounded),
                # Cluster metadata  (used by annotation for multiplicity)
                'n_cluster_peaks':  int(n_ap),
                'cluster_rank':     int(rank),          # 0 = leftmost apex
                'cluster_lo':       int(c_lo),
                'cluster_hi':       int(c_hi),
                'cluster_lo_ppm':   float(ppm[c_lo]),
                'cluster_hi_ppm':   float(ppm[c_hi]),
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3  ·  signal quality (Good / Average / Poor)
    # ──────────────────────────────────────────────────────────────────────────
    # Second pass: the linewidth component compares each signal against the
    # spectrum-wide median FWHM, which only exists once every signal is measured.
    _fwhms = []
    for p in separated_peaks:
        y_c = corrected_profile(intensity, p['start_idx'], p['end_idx'],
                                 p['left_valley_idx'], p['right_valley_idx'])
        p['_fwhm_pts'] = measure_fwhm_points(y_c)
        if p['_fwhm_pts'] > 0:
            _fwhms.append(p['_fwhm_pts'])
    median_fwhm_pts = float(np.median(_fwhms)) if _fwhms else None

    for p in separated_peaks:
        tier, q_score, q_detail = classify_signal_quality(
            intensity,
            start_idx=p['start_idx'], end_idx=p['end_idx'],
            left_idx=p['left_valley_idx'], right_idx=p['right_valley_idx'],
            cluster_lo_idx=p['cluster_lo'], cluster_hi_idx=p['cluster_hi'],
            height=p['height'], shape_r2=p['gaussian_r2'],
            noise_std=eff_noise, baseline_level=float(baseline_level),
            median_fwhm_pts=median_fwhm_pts)
        p['signal_quality']        = tier          # 'Good'/'Average'/'Poor'
        p['signal_quality_score']  = float(q_score)
        p['signal_quality_detail'] = q_detail
        p['fwhm_pts']              = p.pop('_fwhm_pts')

    # Return in apex-index order (same order as sorted_indices input)
    separated_peaks.sort(key=lambda p: p['index'])
    return separated_peaks


def estimate_noise_region(intensity, region_fraction=0.1):
    """
    Estimate noise standard deviation from spectrum edges.

    Assumes peaks are concentrated in the center of the spectrum
    and edges contain mostly noise/baseline.

    Parameters
    ----------
    intensity : ndarray (n,)
        Intensity values
    region_fraction : float
        Fraction of spectrum to use from each edge

    Returns
    -------
    noise_std : float
        Estimated noise standard deviation
    """
    n = len(intensity)
    n_edge = max(5, int(n * region_fraction))

    # Use both edges
    edge_values = np.concatenate([intensity[:n_edge], intensity[-n_edge:]])

    # MAD (Median Absolute Deviation) - robust to outliers
    median = np.median(edge_values)
    mad = np.median(np.abs(edge_values - median))

    # Convert MAD to std estimate (for Gaussian noise)
    noise_std = mad * 1.4826

    return max(noise_std, 1e-10)


def _quiet_segment_stats(intensity, n_segments=32):
    """Baseline level and noise σ of the quietest region of the spectrum.

    Splits the spectrum into ``n_segments`` and selects the lowest-MAD segments
    (a genuinely signal-free region, wherever it sits) rather than trusting the
    blind edges — which in 1H spectra carry TSP/DSS and downfield signals and a
    rolling baseline that inflates an edge estimate. Returns ``(median, sigma)``
    of that quiet region; ``median`` is the local baseline level (≈0 on a mean
    spectrum, but positive on a per-sample-max reference) and ``sigma`` its MAD-
    based noise spread.
    """
    intensity = np.asarray(intensity, dtype=float)
    n = len(intensity)
    if n < 2 * n_segments:
        med = float(np.median(intensity))
        mad = float(np.median(np.abs(intensity - med))) * 1.4826
        return med, max(mad, 1e-10)
    seg_len = n // n_segments
    stats = []   # (mad, median)
    for s in range(n_segments):
        seg = intensity[s * seg_len:(s + 1) * seg_len]
        if seg.size < 3:
            continue
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med))) * 1.4826
        if mad > 0:
            stats.append((mad, med))
    if not stats:
        med = float(np.median(intensity))
        mad = float(np.median(np.abs(intensity - med))) * 1.4826
        return med, max(mad, 1e-10)
    mads = np.array([a for a, _ in stats])
    order = np.argsort(mads)
    k = max(1, len(order) // 10)        # lowest ~10% of segments = quiet region
    sel = order[:k]
    sigma = float(np.median(mads[sel]))
    med   = float(np.median([stats[i][1] for i in sel]))
    return med, max(sigma, 1e-10)


def estimate_noise_minimum_segment(intensity, n_segments=32):
    """Noise σ from the quietest segment (see ``_quiet_segment_stats``)."""
    _, sigma = _quiet_segment_stats(intensity, n_segments)
    return sigma


def fit_peak_quality(ppm_region, intensity_region):
    """
    Assess peak quality from shape fit and symmetry.

    Fits Lorentzian and Gaussian; keeps best R².
    Computes symmetry as HWHM ratio (left vs right of apex).
    Both metrics are needed to classify a peak as good.

    Parameters
    ----------
    ppm_region       : 1-D array — chemical shift axis (may decrease)
    intensity_region : 1-D array — intensities (same length)

    Returns
    -------
    r2       : float — best shape R² (0–1)
    model    : str   — 'lorentzian', 'gaussian', or 'none'
    sym      : float — symmetry score (0–1); 1.0 = perfectly symmetric
    """
    from scipy.optimize import curve_fit

    n = len(ppm_region)
    if n < 5:
        return 0.0, 'none', 0.5

    ppm_region = np.asarray(ppm_region, dtype=float)
    y = np.asarray(intensity_region, dtype=float).copy()

    # Subtract linear baseline between the two feet
    bl = np.linspace(y[0], y[-1], n)
    y_raw_max = float(y.max())
    y -= bl
    y = np.maximum(y, 0.0)

    y_max = float(y.max())
    if y_max <= 1e-15:
        return 0.0, 'none', 0.5

    # Peak amplitude after baseline removal is < 5% of raw amplitude →
    # signal dominated by underlying broad peak/baseline, not a real peak.
    if y_raw_max > 1e-15 and y_max / y_raw_max < 0.05:
        return 0.0, 'none', 0.5

    apex_idx = int(np.argmax(y))
    cen0     = float(ppm_region[apex_idx])

    # ── Symmetry via HWHM on each side of apex ───────────────────────────────
    # ppm_region may decrease (NMR convention); use abs() for widths.
    half = y_max * 0.5

    left_y    = y[:apex_idx]
    left_cross = np.where(left_y >= half)[0]
    lhw = (abs(float(ppm_region[left_cross[0]]) - cen0)
           if len(left_cross) > 0
           else abs(float(ppm_region[0]) - cen0))

    right_y    = y[apex_idx:]
    right_cross = np.where(right_y >= half)[0]
    rhw = (abs(cen0 - float(ppm_region[apex_idx + right_cross[-1]]))
           if len(right_cross) > 0
           else abs(cen0 - float(ppm_region[-1])))

    if lhw > 1e-9 and rhw > 1e-9:
        sym = float(min(lhw, rhw) / max(lhw, rhw))
    elif lhw < 1e-9 and rhw < 1e-9:
        sym = 1.0
    else:
        sym = 0.3  # one side collapsed → very asymmetric

    # ── Lineshape fit ─────────────────────────────────────────────────────────
    above = np.where(y >= half)[0]
    fwhm  = (abs(float(ppm_region[above[-1]] - ppm_region[above[0]]))
             if len(above) >= 2
             else abs(float(ppm_region[-1] - ppm_region[0])) / 3.0)
    fwhm = max(fwhm, 1e-5)

    ppm_lo = float(min(ppm_region[0], ppm_region[-1]))
    ppm_hi = float(max(ppm_region[0], ppm_region[-1]))
    span   = max(ppm_hi - ppm_lo, 1e-5)
    amp0   = y_max

    def _lorentz(x, amp, cen, hwhm):
        return amp * hwhm ** 2 / ((x - cen) ** 2 + hwhm ** 2)

    def _gauss(x, amp, cen, sig):
        return amp * np.exp(-(x - cen) ** 2 / (2.0 * sig ** 2))

    best_r2, best_model = 0.0, 'none'
    for fn, name, wid_init in [
        (_lorentz, 'lorentzian', fwhm / 2.0),
        (_gauss,   'gaussian',   fwhm / 2.355),
    ]:
        try:
            popt, _ = curve_fit(
                fn, ppm_region, y,
                p0=[amp0, cen0, max(wid_init, 1e-5)],
                bounds=([0.0, ppm_lo, 1e-5], [amp0 * 3.0, ppm_hi, span]),
                maxfev=3000,
            )
            y_fit = fn(ppm_region, *popt)
            tss = float(np.sum((y - np.mean(y)) ** 2))
            rss = float(np.sum((y - y_fit) ** 2))
            r2  = max(0.0, 1.0 - rss / tss) if tss > 1e-30 else 0.0
            if r2 > best_r2:
                best_r2    = r2
                best_model = name
        except Exception:
            pass

    return float(best_r2), best_model, float(sym)






def _second_derivative_shoulders(spectrum, distance_in_points, min_height,
                                 curvature_k=3.0):
    """Curvature-based candidates that recover overlapping shoulders.

    A shoulder on the flank of a larger peak has NO local maximum in the
    intensity, so ``find_peaks`` (apex search) misses it — yet it is a local
    maximum of the negative second derivative (a concave-down inflection).
    Picking maxima of −d²I/dδ² therefore surfaces the overlapping lines/shoulders
    that the apex search drops, which are real peaks the annotation needs.

    The raw 2nd derivative amplifies noise, so the spectrum is smoothed with a
    Savitzky–Golay filter (window ≈ peak width) and candidates are gated by a
    curvature signal-to-noise threshold (k·σ of the curvature, robust MAD) plus
    the same intensity floor used for apices — noise blips never pass both.

    Returns a sorted list of int candidate indices.
    """
    n = len(spectrum)
    spectrum = np.asarray(spectrum, dtype=float)
    # SG window ≈ peak width (odd, ≥ polyorder+2), capped so genuinely broad
    # features are not over-smoothed into nothing.
    win = int(distance_in_points) | 1
    win = max(5, min(win, 31))
    if win >= n:
        return []
    try:
        d2 = savgol_filter(spectrum, win, polyorder=3, deriv=2)
    except Exception:
        return []
    neg = -d2                       # concave-down (peak tops / shoulders) → > 0
    sigma = float(np.median(np.abs(neg - np.median(neg)))) * 1.4826
    if sigma <= 0:
        return []
    cand, _ = find_peaks(
        neg,
        height=curvature_k * sigma,
        distance=max(2, int(distance_in_points) // 2),   # finer than apex spacing
    )
    return [int(i) for i in cand if spectrum[i] >= min_height]


def _detect_single(spectrum, eff_noise, min_height, min_snr, distance_in_points,
                   detect_shoulders=True, shoulder_curvature_k=3.0):
    """Apex indices in ONE spectrum via find_peaks, augmented with shoulders.

    Shared by the consolidated path and per-sample detection. Returns a sorted
    list of int apex indices (boundaries/clusters are computed later by
    separate_overlapping_peaks on the chosen reference).

    When ``detect_shoulders`` is set, overlapping lines that have no intensity
    maximum (shoulders) are recovered from the second derivative and merged in,
    so crowded multiplets/clusters do not silently drop analytes.
    """
    idx_primary, _ = find_peaks(
        spectrum,
        height=min_height,
        distance=distance_in_points,
        prominence=eff_noise * min_snr * 0.5,
        width=2,                                 # reject 1-point noise spikes
    )
    apices = [int(i) for i in idx_primary]
    if detect_shoulders:
        # Merge shoulder candidates, but never duplicate an apex already found:
        # a curvature maximum coincident with an intensity maximum is the same
        # line. The dedupe gap is finer than the apex spacing so genuinely close
        # shoulders are still added.
        dedupe_gap = max(2, int(distance_in_points) // 3)
        existing = sorted(apices)
        for c in _second_derivative_shoulders(
                spectrum, distance_in_points, min_height, shoulder_curvature_k):
            if all(abs(c - a) > dedupe_gap for a in existing):
                apices.append(c)
                existing = sorted(apices)
    return sorted(set(apices))


# A resonance seen on the MEAN of the samples and missed in each of them:
# 20 sigma on the mean of 18 samples is ~5 sigma in any one, under the
# per-sample gate. On the virtual bench's tube of 100, 24 analytes had a
# clean signal of 14-22 sigma (glucose 5.23, xylose 5.19, hypoxanthine,
# cis-aconitate) and no peak on it; 21 of them were lost. Taken only where
# nothing detected is within MEAN_ISOLATION_LINEWIDTHS: the version that
# detected on the group mean everywhere (ADR-047) made fragments on big
# peaks' tails -- the DSS lost 27% of its area to one -- and more strangers.
MEAN_MIN_SNR = 10.0
MEAN_ISOLATION_LINEWIDTHS = 3.0


def _isolated_mean_apices(mean_ref, existing, lw_pts, distance_in_points):
    """Apices standing MEAN_MIN_SNR above the mean trace's own floor (height
    and prominence) that no detected apex is within MEAN_ISOLATION_LINEWIDTHS
    of. ``existing`` are the apices per-sample detection found."""
    y = np.asarray(mean_ref, dtype=float)
    med, sig = _quiet_segment_stats(y)
    if not np.isfinite(sig) or sig <= 0 or not lw_pts:
        return []
    idx, _ = find_peaks(y, height=med + MEAN_MIN_SNR * sig,
                        prominence=MEAN_MIN_SNR * sig,
                        distance=max(1, int(distance_in_points)), width=2)
    reach = MEAN_ISOLATION_LINEWIDTHS * float(lw_pts)
    have = np.asarray(sorted(int(e) for e in existing), dtype=int)
    out = []
    for i in idx:
        if have.size and np.min(np.abs(have - int(i))) <= reach:
            continue
        out.append(int(i))
    return out


def _linewidth_points(reference, apex_indices, n_top=30):
    """The reference spectrum's linewidth in points, at half maximum.

    Read off the trace at the tallest apices, walking out to half height. Not
    from the peaks' recorded extents: those are valley to valley and move with
    the very setting this is used to choose. A walk that runs to the edge, or
    one whose apex is not above the half level it is compared against, is
    discarded rather than counted as a very wide line.

    Returns None when nothing can be measured.
    """
    y = np.asarray(reference, dtype=float)
    idx = sorted((int(i) for i in (apex_indices or ())
                  if 0 <= int(i) < y.size), key=lambda i: -y[i])[:int(n_top)]
    widths = []
    for i in idx:
        half = y[i] / 2.0
        if not np.isfinite(half) or half <= 0:
            continue
        li = i
        while li > 0 and y[li] > half:
            li -= 1
        ri = i
        while ri < y.size - 1 and y[ri] > half:
            ri += 1
        if li == 0 or ri == y.size - 1 or ri <= li:
            continue
        widths.append(ri - li)
    return int(round(float(np.median(widths)))) if widths else None


def _consensus_apices(per_sample_idx, n_points, merge_tol_pts):
    """Merge per-sample apex indices into a consensus master set (union).

    Apices from different samples that fall within ``merge_tol_pts`` of each
    other are one analyte line. Greedy single-link clustering over the pooled,
    sorted apices.

    Parameters
    ----------
    per_sample_idx : list[list[int]]  — apex indices found in each sample
    n_points       : int              — spectrum length (unused guard)
    merge_tol_pts  : int              — max index gap to treat apices as the same line

    Returns
    -------
    consensus_idx : list[int]          — one master index per cluster (median of members)
    presence      : list[list[bool]]   — presence[k][s] True if sample s contributed to cluster k
    """
    n_s = len(per_sample_idx)
    pooled = []  # (idx, sample)
    for s, idxs in enumerate(per_sample_idx):
        for i in idxs:
            pooled.append((int(i), s))
    if not pooled:
        return [], []
    pooled.sort(key=lambda t: t[0])

    # Width-bounded clustering (NOT single-link): a cluster spans at most
    # merge_tol_pts from its first member. Single-link would chain distinct
    # lines in dense regions (a→b→c each within tol) into one giant blob;
    # bounding the width keeps lines that are ≥ detection-distance apart separate.
    clusters = []  # list of dict(members=[idx...], samples=set())
    cur = {'members': [pooled[0][0]], 'samples': {pooled[0][1]}}
    cur_start = pooled[0][0]
    for idx, s in pooled[1:]:
        if idx - cur_start <= merge_tol_pts:
            cur['members'].append(idx)
            cur['samples'].add(s)
        else:
            clusters.append(cur)
            cur = {'members': [idx], 'samples': {s}}
            cur_start = idx
    clusters.append(cur)

    consensus_idx, presence = [], []
    for cl in clusters:
        consensus_idx.append(int(np.median(cl['members'])))
        presence.append([s in cl['samples'] for s in range(n_s)])
    return consensus_idx, presence


def detect_peaks(ppm, intensities, noise_std=None, min_snr=3.0,
                 min_distance=0.01, baseline_correct=True,
                 baseline_method='snip', baseline_lam=1e5,
                 baseline_niter=40,
                 global_align=True, global_align_max_shift_ppm=0.03,
                 samples_info=None, progress_callback=None,
                 return_details=False, per_peak_align=True,
                 per_peak_max_shift_ppm=0.02,
                 per_sample_detection=False,
                 detect_shoulders=True, shoulder_curvature_k=3.0,
                 min_group_presence=1, baseline_2d=None,
                 excluded_regions=None):
    """
    Detect peaks with SNR > min_snr - WITH BASELINE CORRECTION.

    Each detected peak is a separate record with:
    - Individual integration boundaries (valley-to-valley)
    - Height above local baseline
    - Area (trapezoid with baseline subtraction)

    Parameters
    ----------
    ppm : ndarray (n,)
        Chemical shift axis
    intensities : ndarray (n,) or (n, m)
        Intensity values (1D for single spectrum, 2D for multiple samples)
    noise_std : float, optional
        Noise standard deviation. If None, estimated from spectrum edges
    min_snr : float
        Minimum SNR threshold for peak detection
    min_distance : float
        Minimum SEPARATION between peak apices, in ppm — goes straight to
        find_peaks(distance=...), so despite the "Min Distance" label in the UI
        it is not a linewidth. Shoulder detection uses half of it, so apices
        closer than this can still be reported. The application passes the
        project's own value; see the default in peak_detection_stage for the
        measurement behind it.
    baseline_correct : bool
        If True, apply baseline correction before detection
    baseline_method : str
        'snip' (default), 'arPLS', 'rolling', or 'constant'
    baseline_lam : float
        Smoothness of the arPLS baseline. Larger = stiffer = follows only very
        broad features. At 1e5 arPLS eats the skirts of sharp lines and biases
        their area low by ~20%, which is why 'snip' is the default.
    baseline_niter : int
        SNIP clipping iterations. More = deeper clipping = broader features are
        treated as baseline.
    global_align : bool
        If True, rigidly align all samples before building the reference
        spectrum used for peak detection.
    global_align_max_shift_ppm : float
        Maximum rigid alignment shift allowed per sample.
    samples_info : list of dict, optional
        Sample metadata. BLK samples are ignored when building the alignment
        reference, but still receive the final rigid shift.

    Returns
    -------
    peak_list : list of dict
        Each peak contains all fields from separate_overlapping_peaks plus:
        - 'snr': Signal-to-noise ratio
        - 'rank': Peak rank by SNR (1 = highest SNR)
        - 'mean_intensity': Mean intensity across samples at apex
        - 'area_per_sample': Array of areas for each sample
    """
    print("\n" + "=" * 70)
    print("PEAK DETECTION WITH BASELINE CORRECTION")
    print("=" * 70)

    def _report(progress, message):
        if progress_callback is None:
            return
        try:
            progress_callback(float(progress), str(message))
        except Exception:
            pass

    try:
        _report(0.02, "Validating inputs")
        # STEP 1: Diagnose inputs
        diagnose_array(ppm, "ppm")
        diagnose_array(intensities, "intensities")

        # STEP 2: Ensure numpy arrays
        ppm = np.asarray(ppm, dtype=float)
        if not isinstance(intensities, np.ndarray):
            intensities = np.array(intensities)

        # STEP 3: Handle dimensionality
        if intensities.ndim == 1:
            intensities_2d = intensities.reshape(-1, 1)
        elif intensities.ndim == 2:
            intensities_2d = intensities
        else:
            raise ValueError(f"intensities has {intensities.ndim} dimensions")

        n_points = len(ppm)
        n_samples = intensities_2d.shape[1]

        # STEP 4: Compute mean spectrum
        _report(0.10, "Building reference spectrum")
        if n_samples > 1:
            mean_intensity = np.mean(intensities_2d, axis=1)
        else:
            mean_intensity = intensities_2d.flatten()

        print(f"\n[INFO] Spectrum: {n_points} points, {n_samples} samples")

        # STEP 5: Baseline correction
        shift_pts = np.zeros(n_samples, dtype=int)
        shift_ppm = np.zeros(n_samples, dtype=float)

        # The baseline this step removes is not only instrumental distortion: in
        # plasma and serum it is mostly the broad macromolecule envelope, which
        # is genuine NMR signal. Removing it is what lets a small metabolite
        # resting on that envelope be detected and integrated honestly, but it
        # IS a subtraction of real signal — so the per-sample baselines are kept
        # and returned, and each peak records how much of its raw area came from
        # the broad component (see 'broad_removed_frac' below).
        #
        # Every method runs per sample. arPLS used to be estimated once on the
        # mean spectrum and reused with only a per-sample constant offset, for
        # speed. A constant cannot absorb an envelope that differs in shape and
        # amplitude between samples — exactly the case in plasma, where protein
        # content is the biological variable — and it injected variance into
        # areas that should have been identical (measured: 22.6% CV on a
        # constant analyte across samples with differing humps).
        if baseline_2d is not None:
            # The caller already applied the project's baseline model, so the
            # correction is not repeated here — only carried through, together
            # with the baselines it removed, so the audit trail below stays
            # exact. This is the path the GUI uses: one correction, shared by
            # detection, quantification, statistics and MCR alike.
            _report(0.20, "Using the project baseline model")
            baseline_2d = np.asarray(baseline_2d, dtype=float)
            corrected_2d = intensities_2d.astype(float)
            corrected_mean = np.mean(corrected_2d, axis=1)
            baseline = np.mean(baseline_2d, axis=1)
            baseline_correct = True
        elif baseline_correct:
            _report(0.20, f"Applying baseline correction ({baseline_method})")
            print(f"\n[INFO] Applying baseline correction ({baseline_method}, "
                  f"per sample)...")
            # Each method takes its own width-cutoff parameter; passing the
            # wrong one silently falls back to that method's own default.
            baseline_method = normalize_method(baseline_method)
            if baseline_method == 'arPLS':
                _bl_kwargs = {'lam': baseline_lam}
            elif baseline_method == 'snip':
                _bl_kwargs = {'niter': int(baseline_niter)}
            else:
                _bl_kwargs = {}
            baseline_2d = np.empty_like(intensities_2d, dtype=float)
            corrected_samples = []
            for s in range(n_samples):
                if n_samples > 1:
                    _report(0.20 + 0.22 * ((s + 1) / n_samples),
                            f"Baseline correction per sample ({s + 1}/{n_samples})")
                corr_s, base_s = correct_baseline(
                    ppm, intensities_2d[:, s],
                    method=baseline_method, **_bl_kwargs)
                corrected_samples.append(corr_s)
                baseline_2d[:, s] = base_s
            corrected_2d = np.column_stack(corrected_samples)
            corrected_mean = np.mean(corrected_2d, axis=1)
            baseline = np.mean(baseline_2d, axis=1)
            print(f"   Baseline estimated. Mean baseline level: {np.mean(baseline):.4f}")
        else:
            corrected_mean = mean_intensity
            corrected_2d = intensities_2d
            baseline = np.zeros(n_points)
            baseline_2d = np.zeros_like(intensities_2d, dtype=float)

        if global_align and n_samples > 1:
            _report(0.48, "Applying global rigid alignment")
            print(f"\n[INFO] Applying rigid global alignment (+/-{global_align_max_shift_ppm:.3f} ppm)...")
            corrected_2d, shift_pts, shift_ppm, corrected_mean = global_peak_rigid_align(
                ppm,
                corrected_2d,
                samples_info=samples_info,
                max_shift_ppm=global_align_max_shift_ppm,
                excluded_regions=excluded_regions,
            )
            print(
                "   Alignment shifts (mppm): min="
                f"{np.min(shift_ppm) * 1000:.2f}  max={np.max(shift_ppm) * 1000:.2f}"
            )
            # Carry the baselines through the same rigid shift, so the removed
            # broad component still lines up with the spectrum it came from.
            for s in range(n_samples):
                if shift_pts[s]:
                    baseline_2d[:, s] = np.roll(baseline_2d[:, s], int(shift_pts[s]))
        else:
            corrected_mean = np.mean(corrected_2d, axis=1) if n_samples > 1 else corrected_2d.flatten()

        # Detection reference: pointwise max across the individual samples so that
        # a resonance present in only ONE sample (or a few) is preserved at full
        # height instead of being diluted by the cross-sample mean — otherwise an
        # analyte present in sample X but absent in Y would never enter the peak
        # list and would be silently lost. A peak real in ANY (non-BLK, included)
        # sample therefore becomes a candidate. The mean spectrum is still used for
        # the displayed reference and the per-peak (mean) SNR below; noise spikes
        # in the max spectrum are rejected by the prominence requirement and the
        # positive >= min_height gate.
        if n_samples > 1:
            _incl = included_sample_indices(samples_info, n_samples)
            detection_ref = np.max(corrected_2d[:, _incl], axis=1).astype(float)
            print(f"   Detection reference: pointwise-max of {len(_incl)} "
                  f"included sample(s) (single-sample analytes preserved)")
        else:
            detection_ref = np.asarray(corrected_mean, dtype=float)

        # STEP 6: Estimate or validate noise
        _report(0.60, "Estimating detection threshold")
        _noise_was_provided = noise_std is not None
        if not _noise_was_provided:
            # Robust estimate from the quietest segment (not blind edges, which
            # in 1H spectra carry TSP/DSS and downfield signals).
            noise_std = estimate_noise_minimum_segment(corrected_mean)
            print(f"\n[INFO] Noise estimated (min-segment): sigma = {noise_std:.6f}")
        else:
            print(f"\n[INFO] Using provided noise_std: {noise_std:.6f}")

        # The detection reference is the per-sample MAX over N samples. The
        # maximum of N noise draws is biased upward with a wider upper tail
        # (≈σ·√(2·ln N)); gating with the plain mean sigma would flood the result
        # with noise maxima. So: take the quiet-region baseline + spread of the
        # detection spectrum itself and inflate the spread by √(2·ln N) to keep
        # the false-positive rate at the requested min_snr. The returned
        # noise_std (and per-peak SNR in STEP 10) stays mean-based.
        _n_det = len(_incl) if n_samples > 1 else 1
        _infl = float(np.sqrt(2.0 * np.log(_n_det))) if _n_det >= 2 else 1.0
        # The floor and spread of the trace the peaks are FOUND on, always.
        # A provided noise_std is the mean trace's (what the SNR column and
        # every limit downstream read) and says nothing about the per-sample
        # maximum this gates. Using it here -- with the maximum's positive
        # floor taken as zero -- was a second code path: the interface
        # estimates sigma on the first Detect, stores it, and passes it on
        # the second, so the same number found 81 peaks on the first press
        # and 90 on the second on the reference study, with the feet 16,048
        # counts up instead of 110,264. What detection finds cannot depend on
        # how many times the button was pressed.
        det_med, det_noise = _quiet_segment_stats(detection_ref)
        print(f"[INFO] Detection floor: baseline={det_med:.6f} sigma={det_noise:.6f} "
              f"inflation(N={_n_det})={_infl:.2f}")

        # STEP 7: Calculate detection parameters
        _report(0.68, "Computing peak detection parameters")
        ppm_range = np.max(ppm) - np.min(ppm)
        ppm_step = ppm_range / n_points
        distance_in_points = max(1, int(min_distance / ppm_step))
        _eff_noise = det_noise * _infl
        min_height = det_med + _eff_noise * min_snr

        print(f"\n[INFO] Detection parameters:")
        print(f"   ppm_range: {ppm_range:.4f}")
        print(f"   distance: {min_distance} ppm = {distance_in_points} points")
        print(f"   min_height: {min_height:.6f} (baseline + SNR={min_snr} * eff_sigma={_eff_noise:.6f})")

        # STEP 8+9: find apexes (find_peaks) then separate. Wrapped so it can be
        # re-run on a peak-aligned reference.
        def _detect_and_separate(ref):
            idx = np.asarray(_detect_single(
                ref, _eff_noise, min_height, min_snr, distance_in_points,
                detect_shoulders=detect_shoulders,
                shoulder_curvature_k=shoulder_curvature_k),
                dtype=int)
            if len(idx) == 0:
                return []
            # Boundary geometry uses the REAL noise spread (det_noise), not the
            # detection-inflated eff_noise — inflation is for the false-positive
            # gate only; using it here puts the 3σ baseline foot too high and
            # clips peak tails (narrow widths / under-integrated areas).
            seps = separate_overlapping_peaks(
                ref, idx, ppm, noise_std=det_noise, method='auto',
                baseline_level=det_med)
            # Positive gate on the detection reference. Cluster members
            # (shoulders / multiplet lines) can have ~0 height above the local
            # linear baseline yet be real signal — gate those on apex intensity.
            return [pk for pk in seps
                    if ref[pk['index']] >= min_height
                    and (pk['height'] > 0 or pk.get('n_cluster_peaks', 1) > 1)]

        # STEP 8: Find peaks
        _report(0.76, "Locating candidate apexes")
        _presence_full = None   # per-sample mode: master apex idx -> presence vector
        if per_sample_detection and n_samples > 1:
            # Genuine per-sample detection, then UNION merge. A peak found in any
            # single sample (above that sample's own SNR gate) is kept; samples
            # where it is absent are recorded so downstream can mark them n.d.
            print("\n[INFO] Per-sample detection (union over samples)...")
            per_sample_idx = []
            for s in _incl:
                col = corrected_2d[:, s].astype(float)
                med_s, sig_s = _quiet_segment_stats(col)
                mh_s = med_s + sig_s * min_snr
                per_sample_idx.append(_detect_single(
                    col, sig_s, mh_s, min_snr, distance_in_points,
                    detect_shoulders=detect_shoulders,
                    shoulder_curvature_k=shoulder_curvature_k))
            # Merge tolerance absorbs the drift of the SAME line between
            # samples. Half the minimum spacing was not enough: at
            # min_distance 0.004-0.005 ppm that is 1.2-1.5 Hz, while a
            # resonance drifts 1.7 Hz between the groups of the reference
            # study. A line then lands in two clusters and becomes two master
            # peaks, each present only in the samples on its side of the drift
            # — which also makes them look group-specific, and manufactures a
            # difference where there is one molecule.
            #
            # The physical bound is the linewidth: two apices closer than that
            # are not two resolved lines, whatever put them there. Measured
            # here on the detection reference from a first pass's apices, at
            # 0.7 of it so a genuine splitting the detector is allowed to
            # resolve is still kept apart, and never below the old value so
            # this cannot merge less than before.
            merge_tol_pts = max(1, distance_in_points // 2)
            _first, _ = _consensus_apices(per_sample_idx, n_points,
                                          merge_tol_pts)
            _lw_pts = _linewidth_points(detection_ref, _first)
            if _lw_pts:
                merge_tol_pts = max(merge_tol_pts,
                                    min(int(round(0.7 * _lw_pts)),
                                        max(1, distance_in_points - 1)))
                print(f"   Linewidth on the reference: {_lw_pts} pts; "
                      f"merge tolerance {merge_tol_pts} pts")
            cons_idx, presence = _consensus_apices(
                per_sample_idx, n_points, merge_tol_pts)
            print(f"   Per-sample union: {sum(len(x) for x in per_sample_idx)} raw "
                  f"apex(es) -> {len(cons_idx)} master peak(s)")
            # Snap each consensus index (median of member apices) to the nearest
            # LOCAL MAX on the detection reference. The median can land off-apex
            # on a flank; separate_overlapping_peaks then mistakes the adjacent
            # rise for a valley and collapses the peak.
            _snap_w = max(1, distance_in_points // 2)
            def _snap(ci):
                lo = max(0, ci - _snap_w); hi = min(n_points, ci + _snap_w + 1)
                return lo + int(np.argmax(detection_ref[lo:hi]))
            # Expand presence (over included cols) to all samples (BLK/excluded → False).
            idx_to_presence = {}
            snapped = []
            for k, ci in enumerate(cons_idx):
                sc = _snap(int(ci))
                full = [False] * n_samples
                for pos, s in enumerate(_incl):
                    full[s] = presence[k][pos]
                # Same master apex can appear twice after snapping → union presence.
                if sc in idx_to_presence:
                    idx_to_presence[sc] = [a or b for a, b in
                                           zip(idx_to_presence[sc], full)]
                else:
                    idx_to_presence[sc] = full
                    snapped.append(sc)

            # Enforce the GLOBAL min peak distance on the master apices (same as
            # the consolidated path keeps via find_peaks(distance=...)). Without
            # this, the union over-splits broad/cross-sample-shifted features that
            # the global distance would merge — inflating the peak count. Keep the
            # stronger apex (on the max reference) and UNION the presence vectors.
            snapped.sort()
            deduped = []
            for sc in snapped:
                if deduped and (sc - deduped[-1]) < distance_in_points:
                    prev = deduped[-1]
                    merged = [a or b for a, b in
                              zip(idx_to_presence[prev], idx_to_presence[sc])]
                    if detection_ref[sc] > detection_ref[prev]:
                        deduped[-1] = sc
                        idx_to_presence[sc] = merged
                    else:
                        idx_to_presence[prev] = merged
                else:
                    deduped.append(sc)
            cons_idx = deduped
            # What the mean shows and no single sample could: isolated
            # resonances too weak per sample (MEAN_MIN_SNR). Present in the
            # included samples -- the mean they were seen on.
            # Isolated from what detection KEEPS: the union also holds the
            # same resonance seen weakly in one or two samples, which the
            # gate below drops -- measured against those, glucose's apex
            # was never isolated from itself. Such a weak apex at the same
            # place is taken up rather than doubled.
            _strong = [c for c in cons_idx if detection_ref[c] >= min_height]
            _mean_only = set()
            for _mi in _isolated_mean_apices(
                    np.mean(corrected_2d[:, _incl], axis=1), _strong,
                    _lw_pts, distance_in_points):
                _same = [c for c in cons_idx
                         if abs(c - _mi) <= distance_in_points]
                if not _same:
                    cons_idx.append(_mi)
                    _same = [_mi]
                for _c in _same:
                    _mean_only.add(_c)
                    idx_to_presence[_c] = [s in _incl
                                           for s in range(n_samples)]
            if _mean_only:
                print(f"   Mean-trace apices (isolated, >= {MEAN_MIN_SNR:g} sigma): "
                      f"{len(_mean_only)}")
            if cons_idx:
                idx_arr = np.asarray(sorted(cons_idx), dtype=int)
                seps = separate_overlapping_peaks(
                    detection_ref, idx_arr, ppm, noise_std=det_noise, method='auto',
                    baseline_level=det_med)
                separated_peaks = [
                    pk for pk in seps
                    if (detection_ref[pk['index']] >= min_height
                        or pk['index'] in _mean_only)
                    and (pk['height'] > 0 or pk.get('n_cluster_peaks', 1) > 1)]
                _presence_full = idx_to_presence
            else:
                separated_peaks = []
        else:
            print("\n[INFO] Running find_peaks on consolidated reference...")
            separated_peaks = _detect_and_separate(detection_ref)
        print(f"   [OK] Raw peaks detected: {len(separated_peaks)}")

        # Per-peak alignment + detection refinement (one bounded extra pass):
        # find each analyte's apex per sample, warp each sample onto those anchors
        # to sharpen the per-sample-max reference (misaligned peaks stop smearing),
        # then re-detect once on the sharper reference. Deterministic, no loop.
        # Skipped in per-sample mode — the master set is already the result there.
        if (per_peak_align and n_samples > 1 and separated_peaks
                and not per_sample_detection):
            from .peak_alignment import (refine_peak_centers_per_sample,
                                         warp_from_peak_shifts)
            _report(0.80, "Per-peak alignment refinement")
            refine_peak_centers_per_sample(
                separated_peaks, ppm, corrected_2d, noise_std=det_noise,
                min_snr=min_snr, max_shift_ppm=per_peak_max_shift_ppm)
            _warped = warp_from_peak_shifts(
                ppm, corrected_2d[:, _incl], separated_peaks,
                max_shift_ppm=per_peak_max_shift_ppm)
            _warped_ref = np.max(_warped, axis=1).astype(float)
            _refined = _detect_and_separate(_warped_ref)
            # Aligning samples should SHARPEN the max reference (de-smear), giving
            # the same or fewer apexes — never substantially more. If the warp
            # would inflate the peak count (over-splitting that breaks multiplet
            # patterns the DB matches), reject it and keep the initial peaks.
            if _refined and len(_refined) <= max(3, int(round(len(separated_peaks) * 1.1))):
                detection_ref = _warped_ref
                separated_peaks = _refined
                print(f"   [OK] After alignment refinement: {len(separated_peaks)} peaks")
            else:
                print(f"   Alignment refinement rejected "
                      f"(initial={len(separated_peaks)}, refined={len(_refined)}); "
                      f"keeping initial peaks")

        if len(separated_peaks) == 0:
            print("\n[WARN] No peaks detected. Try lowering min_snr or adjusting baseline.")
            empty_result = []
            if return_details:
                return {
                    'peaks': empty_result,
                    'reference_intensity': corrected_mean,
                    'aligned_intensities': corrected_2d,
                    'baseline': baseline,
                    'baseline_per_sample': baseline_2d,
                    'baseline_method': baseline_method if baseline_correct else None,
                    'baseline_lam': float(baseline_lam) if baseline_correct else None,
                    'baseline_niter': int(baseline_niter) if baseline_correct else None,
                    'noise_std': float(noise_std),
                    'shift_points': shift_pts,
                    'shift_ppm': shift_ppm,
                    'alignment_applied': bool(global_align and n_samples > 1),
                }
            return empty_result

        # Final per-sample apex positions on the ORIGINAL (un-warped) spectra, so
        # quant and overlays use each analyte's true position in each sample.
        if n_samples > 1:
            from .peak_alignment import refine_peak_centers_per_sample
            refine_peak_centers_per_sample(
                separated_peaks, ppm, corrected_2d, noise_std=det_noise,
                min_snr=min_snr, max_shift_ppm=per_peak_max_shift_ppm)

        # Per-sample mode: aligned_per_sample = GENUINE per-sample detection
        # presence (which samples actually had this apex), overriding the
        # signal-gated inference from refine. Drives true n.d. downstream.
        # (Keys on the current peak['index']; must run BEFORE the apex snap.)
        if _presence_full is not None:
            for peak in separated_peaks:
                pres = _presence_full.get(peak['index'])
                if pres is not None:
                    peak['aligned_per_sample'] = list(pres)

        # ── Group-presence filter ─────────────────────────────────────────────
        # A real analyte peak should be detected in several replicates of a
        # group, not in a single sample. Drop peaks not present in at least
        # ``min_group_presence`` samples of ANY one group — this removes the
        # spurious peaks that per-sample detection picks up in a lone sample
        # (e.g. a noise apex next to a real multiplet that then gets annotated).
        # A group smaller than the threshold needs all its samples (min(thr,n)).
        if (min_group_presence and min_group_presence > 1 and n_samples > 1
                and samples_info):
            from collections import Counter
            _grp_of = {}
            for i in range(n_samples):
                _info = samples_info[i] if i < len(samples_info) else {}
                if (str(_info.get('type', 'Sample')).upper() == 'BLK'
                        or _info.get('excluded', False)):
                    continue
                _grp_of[i] = _info.get('label', _info.get('group', 'Unknown'))
            _grp_size = Counter(_grp_of.values())
            # Per-sample presence threshold from each sample's own quiet region —
            # signal-based and deterministic (independent of how the peak was
            # detected/merged), so it is reliable for the filter.
            _thr_s = {}
            for i in _grp_of:
                _m_i, _s_i = _quiet_segment_stats(corrected_2d[:, i])
                _thr_s[i] = _m_i + _s_i * min_snr

            def _detected_in(peak):
                """Which samples this peak was actually DETECTED in.

                Per-sample detection knows the answer exactly — it is the
                union's presence vector, already on the peak. Asking the
                spectrum instead ("is the intensity here above the noise?")
                answers a different question, and on the flank of a strong
                line the answer is yes in every sample. A fragment detected as
                an apex in one sample of sixteen then passed a filter written
                to remove exactly that, at any threshold.

                Measured on a 21-sample study: L-alanine's methyl doublet had
                two such fragments beside it, detected in 1 and 2 samples, and
                the annotation matched alanine to them instead of to the two
                real lines detected in all 16 — they sat 0.003 ppm closer to
                the database position. 129 of 394 peaks in that project were
                detected in fewer than three samples of any group.

                Falls back to the intensity test only when there is no
                presence vector, which is the single-spectrum path where no
                per-sample detection ran.
                """
                pres = peak.get('aligned_per_sample')
                if pres:
                    return {_i for _i in _grp_of
                            if _i < len(pres) and pres[_i]}
                idx = int(peak.get('index', 0))
                if idx < 0 or idx >= corrected_2d.shape[0]:
                    return set(_grp_of)
                return {_i for _i in _grp_of
                        if float(corrected_2d[idx, _i]) >= _thr_s[_i]}

            def _passes_presence(peak):
                _in = _detected_in(peak)
                cnt = {}
                for _i in _in:
                    _g = _grp_of[_i]
                    cnt[_g] = cnt.get(_g, 0) + 1
                for _g, _sz in _grp_size.items():
                    if cnt.get(_g, 0) >= min(int(min_group_presence), _sz):
                        return True
                return False

            _n_before = len(separated_peaks)
            separated_peaks = [p for p in separated_peaks if _passes_presence(p)]
            print(f"   [presence filter] kept {len(separated_peaks)}/{_n_before} "
                  f"peaks (>={min_group_presence} samples in some group)")

        # Mark the apex at the top of the DISPLAYED (mean) spectrum, not the
        # per-sample-MAX reference used for detection — the max apex can sit a
        # few points off the visible mean peak (per-sample shifts), leaving the
        # overlay marker beside the top. Fall back to the detection apex when the
        # mean is flat there (single-sample analyte, no mean peak to land on).
        # The first-press behaviour, whatever the press: the measured one.
        # It holds a mean-trace apex to the maximum trace's floor, which is
        # conservative -- an apex it does not move stays where detection put
        # it -- and is kept because it is what was measured over three seeds.
        _mean_gate = det_med + det_noise * min_snr
        for peak in separated_peaks:
            s, e = peak['start_idx'], peak['end_idx']
            if e > s:
                ma = s + int(np.argmax(corrected_mean[s:e + 1]))
                if float(corrected_mean[ma]) >= _mean_gate:
                    peak['index'] = int(ma)
                    peak['ppm'] = float(ppm[ma])

        print(f"   [OK] Separated peaks: {len(separated_peaks)}")

        # STEP 10: Calculate per-sample areas and final SNR
        _report(0.92, "Calculating per-sample areas")
        for peak in separated_peaks:
            start_idx = peak['start_idx']
            end_idx = peak['end_idx']
            x_region = ppm[start_idx:end_idx + 1]

            areas = [foot_line_area(x_region, corrected_2d[start_idx:end_idx + 1, s])[0]
                     for s in range(n_samples)]

            peak['area_per_sample'] = np.array(areas)
            peak['mean_intensity'] = float(np.mean([corrected_2d[peak['index'], s]
                                                     for s in range(n_samples)]))

            # ── Audit trail for the removed broad component ────────────────────
            # How much of the raw area inside this window was the macromolecule
            # envelope rather than the analyte. A signal sitting on a large
            # envelope has most of its raw integral removed here; the analyst
            # must be able to see that, because the subtraction is of real NMR
            # signal and it sets how much the reported area depends on the
            # baseline model. 0 = nothing removed, →1 = the window was almost
            # entirely broad component.
            if baseline_correct and len(x_region) > 1:
                fracs = broad_removed_fractions(
                    x_region, baseline_2d[start_idx:end_idx + 1], areas)
                peak['broad_removed_frac_per_sample'] = fracs
                peak['broad_removed_frac'] = float(np.mean(fracs))
            else:
                peak['broad_removed_frac_per_sample'] = np.zeros(n_samples)
                peak['broad_removed_frac'] = 0.0

            # height/area come from separate_overlapping_peaks on the per-sample
            # MAX reference (used for detection). Re-express them on the MEAN
            # spectrum so they match the displayed spectrum (overlay markers) and
            # the mean-scale SNR/quant — otherwise markers float far above the
            # plotted mean line.
            peak['height'] = float(max(
                np.max(corrected_mean[start_idx:end_idx + 1]) -
                0.5 * (corrected_mean[start_idx] + corrected_mean[end_idx]),
                0.0))
            peak['area'] = float(np.mean(areas)) if len(areas) else float(peak.get('area', 0.0))

            # SNR based on mean intensity
            peak['snr'] = peak['mean_intensity'] / noise_std if noise_std > 0 else 0.0
            peak['alignment_applied'] = bool(global_align and n_samples > 1)

        # STEP 11: Drop the suppressed regions, then sort and rank.
        #
        # Before ranking, so the ranks stay contiguous, and here rather than in
        # the callers so annotation, quantification and the Signal List cannot
        # disagree about which peaks the study contains. The residual water
        # resonance is the reason — see processing/spectral_regions.py.
        excluded_peaks = []
        if excluded_regions:
            separated_peaks, excluded_peaks = split_excluded(
                separated_peaks, excluded_regions)
            if excluded_peaks:
                print(f"   Excluded {format_regions(excluded_regions)} ppm: "
                      f"{len(excluded_peaks)} peaks dropped")

        _report(0.97, "Ranking peaks")
        separated_peaks.sort(key=lambda x: x['snr'], reverse=True)
        for i, peak in enumerate(separated_peaks, 1):
            peak['rank'] = i

        # STEP 12: Summary
        print(f"\n[OK] DETECTION COMPLETE")
        print(f"   Total peaks: {len(separated_peaks)}")
        if separated_peaks:
            print(f"   Highest SNR: {separated_peaks[0]['ppm']:.4f} ppm (SNR={separated_peaks[0]['snr']:.1f})")
            print(f"   Lowest SNR: {separated_peaks[-1]['ppm']:.4f} ppm (SNR={separated_peaks[-1]['snr']:.1f})")

        _report(1.00, "Detection complete")

        if return_details:
            return {
                'peaks': separated_peaks,
                'reference_intensity': corrected_mean,
                'detection_reference': detection_ref,
                'aligned_intensities': corrected_2d,
                'baseline': baseline,
                'baseline_per_sample': baseline_2d,
                'baseline_method': baseline_method if baseline_correct else None,
                'baseline_lam': float(baseline_lam) if baseline_correct else None,
                'baseline_niter': int(baseline_niter) if baseline_correct else None,
                'noise_std': float(noise_std),
                'shift_points': shift_pts,
                'shift_ppm': shift_ppm,
                'alignment_applied': bool(global_align and n_samples > 1),
                'excluded_regions': list(excluded_regions or ()),
                'excluded_peaks': excluded_peaks,
            }
        return separated_peaks

    except Exception as e:
        print(f"\n[ERROR] PEAK DETECTION ERROR:")
        print(f"   {str(e)}")
        print("\n[TRACEBACK]")
        traceback.print_exc()
        print("=" * 70)
        raise

def canonical_peak_order(peaks) -> list:
    """Detected peaks in the one order annotation reads them: by detection
    rank, then by ppm (high to low) for any without one.

    The search and the scoring settle ties by which peak comes first. Fed
    the same 99 peaks of one study reversed, annotation returned different
    matches; a saved project, a re-detection or an edit of the signal list
    may hand them over in another order. One order defined by the peaks
    themselves makes the annotation a function of the data alone."""
    def key(p):
        r = p.get('rank')
        try:
            v = float(p.get('ppm'))
        except (TypeError, ValueError):
            v = 0.0
        return (r is None, r if r is not None else 0, -v)
    return sorted(peaks or (), key=key)


def as_app_peak(p) -> dict:
    """One detected peak in the form the rest of the application reads.

    The detection stage built this inline, which left anything that wanted
    to replay an analysis without the interface -- the validation bench
    first of all -- to copy the conversion and drift from it. One function,
    read by both.
    """
    from processing.signal_quality import SIGNAL_AVERAGE
    return {
        'rank': p.get('rank', 0),
        'ppm': p.get('ppm', 0),
        'height': p.get('height', 0),
        'snr': p.get('snr', 0),
        'start_ppm': p.get('start_ppm', 0),
        'end_ppm': p.get('end_ppm', 0),
        'width_ppm': p.get('width_ppm', 0),
        'start_idx': p.get('start_idx', 0),
        'end_idx': p.get('end_idx', 0),
        'index': p.get('index', 0),
        'area': p.get('area', 0),
        'area_per_sample': p.get('area_per_sample', np.array([])),
        # Per-sample apex alignment (same analyte across samples)
        'ppm_per_sample':       p.get('ppm_per_sample'),
        'shift_ppm_per_sample': p.get('shift_ppm_per_sample'),
        'aligned_per_sample':   p.get('aligned_per_sample'),
        'overlapping': False,  # Will be set by overlap search
        # Quality fields
        'gaussian_r2':       p.get('gaussian_r2',      float('nan')),
        'symmetry_score':    p.get('symmetry_score',    float('nan')),
        'gaussian_quality':  p.get('gaussian_quality',  'unknown'),
        # Automatic signal quality: Good / Average / Poor, plus the
        # score and the per-component breakdown that justifies it.
        'signal_quality':        p.get('signal_quality', SIGNAL_AVERAGE),
        'signal_quality_score':  p.get('signal_quality_score', float('nan')),
        'signal_quality_detail': p.get('signal_quality_detail', {}),
        'fwhm_pts':              p.get('fwhm_pts', float('nan')),
        # Fraction of this window's raw area that was broad
        # component (macromolecule envelope) removed by the
        # baseline model — real NMR signal, so its removal is
        # recorded rather than assumed.
        'broad_removed_frac':    p.get('broad_removed_frac', 0.0),
        'broad_removed_frac_per_sample':
            p.get('broad_removed_frac_per_sample', np.array([])),
        # Manual annotation status starts unset — the user marks
        # Good/Average/Poor; it is NOT derived from the signal quality.
        'status': None,
        'left_grounded':     p.get('left_grounded',     False),
        'right_grounded':    p.get('right_grounded',    False),
        'baseline_grounded': p.get('baseline_grounded', False),
        # Cluster fields (used by annotation for multiplicity)
        'n_cluster_peaks':  p.get('n_cluster_peaks',  1),
        'cluster_rank':     p.get('cluster_rank',     0),
        'cluster_lo':       p.get('cluster_lo',       p.get('start_idx', 0)),
        'cluster_hi':       p.get('cluster_hi',       p.get('end_idx',   0)),
        'cluster_lo_ppm':   p.get('cluster_lo_ppm',   p.get('start_ppm', 0.0)),
        'cluster_hi_ppm':   p.get('cluster_hi_ppm',   p.get('end_ppm',   0.0)),
    }
