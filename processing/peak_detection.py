# processing/peak_detection.py
"""
Peak detection for NMR spectroscopy with baseline correction.

Features:
  - Baseline correction (airPLS, SNIP) before detection
  - Peak separation by valley points (foot-head-foot)
  - SNR-based thresholding
  - Individual peak integration
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import traceback

from .baseline_correction import correct_baseline, airPLS
from .peak_alignment import global_peak_rigid_align


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


def find_peak_boundaries_derivative(intensity, peak_idx, left_bound, right_bound, noise_std=None, min_drop=1.5):
    """
    Encontra limites de pico usando a primeira derivada e limiar de ruído.
    - Limite: ponto onde derivada cruza zero (mudança de sinal) ou intensidade cai abaixo de baseline + min_drop*ruído.
    """
    n = len(intensity)
    if noise_std is None:
        noise_std = estimate_noise_region(intensity)
    baseline = np.min(intensity)
    threshold = baseline + min_drop * noise_std

    # Derivada
    deriv = np.gradient(intensity)

    # Esquerda do ápice
    left = peak_idx
    for i in range(peak_idx, left_bound, -1):
        if intensity[i] < threshold:
            left = i
            break
        if deriv[i-1] <= 0 < deriv[i]:  # zero crossing (mínimo)
            left = i
            break

    # Direita do ápice
    right = peak_idx
    for i in range(peak_idx, right_bound):
        if intensity[i] < threshold:
            right = i
            break
        if deriv[i] >= 0 > deriv[i+1]:  # zero crossing (mínimo)
            right = i
            break
        if i+1 >= n-1:
            break

    return left, right

def find_valley_points(intensity, peak_idx, left_bound, right_bound):
    """
    Find valley (local minimum) points that define peak boundaries.

    Searches for the lowest points between the current peak and
    adjacent peaks or baseline.

    Parameters
    ----------
    intensity : ndarray (n,)
        Baseline-corrected intensity
    peak_idx : int
        Index of the peak apex
    left_bound : int
        Left search boundary (previous peak or spectrum start)
    right_bound : int
        Right search boundary (next peak or spectrum end)

    Returns
    -------
    left_valley : int
        Index of left valley point (foot)
    right_valley : int
        Index of right valley point (foot)
    """
    n = len(intensity)

    # Search left valley: find minimum between previous peak and current peak
    left_search_start = max(0, left_bound)
    left_search_end = peak_idx

    if left_search_end <= left_search_start:
        left_valley = left_search_start
    else:
        # Find the local minimum in this region
        left_region = intensity[left_search_start:left_search_end + 1]

        # Check for local minima using gradient sign changes
        gradient = np.diff(left_region)
        sign_changes = np.where(np.diff(np.sign(gradient)) > 0)[0]

        if len(sign_changes) > 0:
            # Pick the minimum closest to the peak
            left_valley = left_search_start + sign_changes[-1] + 1
        else:
            # No clear minimum - use the lowest point in region
            left_valley = left_search_start + np.argmin(left_region)

    # Search right valley: find minimum between current peak and next peak
    right_search_start = peak_idx
    right_search_end = min(n - 1, right_bound)

    if right_search_end <= right_search_start:
        right_valley = right_search_end
    else:
        right_region = intensity[right_search_start:right_search_end + 1]

        # Check for local minima
        gradient = np.diff(right_region)
        sign_changes = np.where(np.diff(np.sign(gradient)) > 0)[0]

        if len(sign_changes) > 0:
            # Pick the minimum closest to the peak
            right_valley = right_search_start + sign_changes[0] + 1
        else:
            # No clear minimum - use the lowest point in region
            right_valley = right_search_start + np.argmin(right_region)

    return left_valley, right_valley


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

    def _walk(apex, step):
        """Walk outward from apex (step ±1). Returns (foot_idx, grounded)."""
        run_min_idx, run_min_val = apex, float(intensity[apex])
        j = apex + step
        while 0 <= j < n:
            v = float(intensity[j])
            if v <= thr:                       # reached baseline
                return j, True
            if v < run_min_val:
                run_min_val, run_min_idx = v, j
            elif v > run_min_val + rise_margin:  # signal rising again → valley
                return run_min_idx, bool(run_min_val <= thr)
            j += step
        # Hit the spectrum edge without grounding → lowest point seen.
        return run_min_idx, bool(run_min_val <= thr)

    for apex in sorted_indices:
        lf, lg = _walk(apex, -1)
        rf, rg = _walk(apex, +1)
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

    for cl in clusters:
        apices   = sorted(cl['apices'])
        c_lo     = cl['lo']
        c_hi     = cl['hi']
        n_ap     = len(apices)

        for rank, apex in enumerate(apices):

            # ── Left boundary ─────────────────────────────────────────────────
            if rank == 0:
                left_idx      = c_lo
                left_grounded = lg_map[apex]
            else:
                prev      = apices[rank - 1]
                seg       = intensity[prev:apex + 1]
                left_idx  = prev + int(np.argmin(seg))
                left_grounded = float(intensity[left_idx]) <= thr

            # ── Right boundary ────────────────────────────────────────────────
            if rank == n_ap - 1:
                right_idx      = c_hi
                right_grounded = rg_map[apex]
            else:
                nxt        = apices[rank + 1]
                seg        = intensity[apex:nxt + 1]
                right_idx  = apex + int(np.argmin(seg))
                right_grounded = float(intensity[right_idx]) <= thr

            if n_ap == 1:
                # ── Isolated peak → symmetric feet ────────────────────────────
                # Make start/end land at the same intensity (Y_start ≈ Y_end):
                # raise the LOWER foot to the HIGHER one and trim that side inward
                # to where the flank crosses that level. Horizontal integration
                # baseline so a neighbour on one side cannot skew the area.
                left_val  = float(intensity[left_idx])
                right_val = float(intensity[right_idx])
                foot_level = max(left_val, right_val)
                if left_val < foot_level:
                    j = left_idx
                    while j < apex and float(intensity[j]) < foot_level:
                        j += 1
                    left_idx = j
                elif right_val < foot_level:
                    j = right_idx
                    while j > apex and float(intensity[j]) < foot_level:
                        j -= 1
                    right_idx = j
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


def _second_derivative_candidates(spectrum, min_height, smooth_win):
    """Apex indices for peaks AND shoulders, via the second derivative.

    ``find_peaks`` only locates local maxima, so a shoulder — or a partially
    overlapped multiplet line with no maximum of its own — is invisible. A peak
    is a minimum (most-negative curvature) of d²/dx², so detecting minima of the
    second derivative recovers those hidden apexes. Candidates are gated by:
      • spectrum value ≥ ``min_height`` (must sit on real signal), and
      • curvature prominence ≥ 3·MAD(d²) (significant inflection, not noise).

    Returns a list of integer indices (may be empty).
    """
    from scipy.signal import savgol_filter, find_peaks as _fp
    n = len(spectrum)
    win = int(smooth_win)
    if win % 2 == 0:
        win += 1
    win = max(5, win)
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < 5:
        return []
    try:
        d2 = savgol_filter(np.asarray(spectrum, dtype=float),
                           window_length=win, polyorder=2, deriv=2)
    except Exception:
        return []
    d2_mad = float(np.median(np.abs(d2 - np.median(d2)))) * 1.4826
    prom = max(3.0 * d2_mad, 1e-12)
    idx, _ = _fp(-d2, prominence=prom)        # peaks/shoulders = maxima of -d²
    spectrum = np.asarray(spectrum, dtype=float)
    out = []
    for i in idx:
        if spectrum[i] < min_height:
            continue
        # Reject single-point spikes: a real apex/shoulder spans ≥2 points above
        # half of its own height; a lone spike has only itself.
        half = 0.5 * spectrum[i]
        w = 1
        j = i - 1
        while j >= 0 and spectrum[j] >= half:
            w += 1; j -= 1
        j = i + 1
        while j < n and spectrum[j] >= half:
            w += 1; j += 1
        if w >= 2:
            out.append(int(i))
    return out


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


def fit_gaussian_quality(ppm_region, intensity_region):
    """Backward-compatible wrapper — delegates to fit_peak_quality."""
    r2, _model, _sym = fit_peak_quality(ppm_region, intensity_region)
    return r2, None


def check_foot_at_baseline(intensity, foot_idx, noise_std,
                            threshold_factor=3.0):
    """
    Return True if the intensity at foot_idx is at or near baseline.

    A peak foot (valley) is considered "baseline-grounded" when its
    intensity is ≤ threshold_factor × noise_std.  Peaks where BOTH
    feet are grounded are isolated singlets; peaks with an ungrounded
    foot share a valley with an adjacent peak and are part of an
    overlapping cluster that should go to the deconvolution tool.

    Parameters
    ----------
    intensity        : 1-D array — baseline-corrected spectrum
    foot_idx         : int       — index of the left or right valley
    noise_std        : float     — estimated noise standard deviation
    threshold_factor : float     — multiplier on noise_std (default 3.0)
    """
    if noise_std is None or float(noise_std) <= 0:
        return True   # cannot assess → assume grounded
    return float(intensity[foot_idx]) <= threshold_factor * float(noise_std)


def _detect_single(spectrum, eff_noise, min_height, min_snr, distance_in_points,
                   detect_shoulders=False, smooth_win=None):
    """Apex indices in ONE spectrum (find_peaks + optional hidden-shoulder pass).

    Shared by the consolidated path and per-sample detection. Returns a sorted
    list of int apex indices (boundaries/clusters are computed later by
    separate_overlapping_peaks on the chosen reference).
    """
    if smooth_win is None:
        smooth_win = max(5, 2 * distance_in_points + 1)
    idx_primary, _ = find_peaks(
        spectrum,
        height=min_height,
        distance=distance_in_points,
        prominence=eff_noise * min_snr * 0.5,
        width=2,                                 # reject 1-point noise spikes
    )
    idx_primary = sorted(int(i) for i in idx_primary)
    kept_shoulders = []
    if detect_shoulders:
        prim = np.asarray(idx_primary, dtype=int)
        for si in _second_derivative_candidates(spectrum, min_height, smooth_win):
            if prim.size and np.min(np.abs(prim - si)) <= distance_in_points:
                continue                                 # near a real apex
            if kept_shoulders and min(abs(si - k) for k in kept_shoulders) \
                    <= distance_in_points:
                continue                                 # near another shoulder
            kept_shoulders.append(int(si))
    return sorted(idx_primary + kept_shoulders)


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
                 baseline_method='airPLS', baseline_lam=1e5,
                 global_align=True, global_align_max_shift_ppm=0.03,
                 samples_info=None, progress_callback=None,
                 return_details=False, per_peak_align=True,
                 per_peak_max_shift_ppm=0.02, detect_shoulders=False,
                 per_sample_detection=False):
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
        Minimum distance between peaks in ppm
    baseline_correct : bool
        If True, apply baseline correction before detection
    baseline_method : str
        'airPLS', 'snip', 'rolling', or 'constant'
    baseline_lam : float
        Baseline smoothing parameter (for airPLS)
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

        if baseline_correct:
            _report(0.20, f"Applying baseline correction ({baseline_method})")
            print(f"\n[INFO] Applying baseline correction ({baseline_method})...")
            corrected_mean, baseline = correct_baseline(
                ppm, mean_intensity,
                method=baseline_method,
                lam=baseline_lam if baseline_method == 'airPLS' else None
            )
            print(f"   Baseline estimated. Mean baseline level: {np.mean(baseline):.4f}")

            # Correct each sample: for airPLS reuse mean baseline shape + per-sample
            # constant offset (running airPLS per sample is O(n_samples * n * niter)).
            corrected_samples = []
            edge_pts = max(5, n_points // 20)
            edge_mask = np.concatenate([np.arange(edge_pts),
                                        np.arange(n_points - edge_pts, n_points)])
            for s in range(n_samples):
                if n_samples > 1:
                    _report(0.20 + 0.22 * ((s + 1) / n_samples),
                            f"Baseline correction per sample ({s + 1}/{n_samples})")
                if baseline_method == 'airPLS':
                    offset = np.median(intensities_2d[edge_mask, s] - baseline[edge_mask])
                    corr_s = intensities_2d[:, s] - (baseline + offset)
                else:
                    corr_s, _ = correct_baseline(
                        ppm, intensities_2d[:, s],
                        method=baseline_method,
                        lam=baseline_lam
                    )
                corrected_samples.append(corr_s)
            corrected_2d = np.column_stack(corrected_samples)
        else:
            corrected_mean = mean_intensity
            corrected_2d = intensities_2d
            baseline = np.zeros(n_points)

        if global_align and n_samples > 1:
            _report(0.48, "Applying global rigid alignment")
            print(f"\n[INFO] Applying rigid global alignment (+/-{global_align_max_shift_ppm:.3f} ppm)...")
            corrected_2d, shift_pts, shift_ppm, corrected_mean = global_peak_rigid_align(
                ppm,
                corrected_2d,
                samples_info=samples_info,
                max_shift_ppm=global_align_max_shift_ppm,
            )
            print(
                "   Alignment shifts (mppm): min="
                f"{np.min(shift_ppm) * 1000:.2f}  max={np.max(shift_ppm) * 1000:.2f}"
            )
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
            _incl = []
            for _s in range(n_samples):
                _info = (samples_info[_s]
                         if (samples_info and _s < len(samples_info)) else {})
                if (str(_info.get('type', 'Sample')).upper() == 'BLK'
                        or _info.get('excluded', False)):
                    continue
                _incl.append(_s)
            if not _incl:
                _incl = list(range(n_samples))
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
        if _noise_was_provided:
            det_med, det_noise = 0.0, float(noise_std)
        else:
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

        # STEP 8+9: find apexes (find_peaks + conservative second-derivative)
        # then separate. Wrapped so it can be re-run on a peak-aligned reference.
        _smooth_win = max(5, 2 * distance_in_points + 1)

        def _detect_and_separate(ref):
            idx = np.asarray(_detect_single(
                ref, _eff_noise, min_height, min_snr, distance_in_points,
                detect_shoulders, _smooth_win), dtype=int)
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
                    detect_shoulders, _smooth_win))
            # Merge tolerance only absorbs cross-sample alignment jitter of the
            # SAME line — it must stay well below distance_in_points (the minimum
            # spacing between distinct apices), or separate multiplet lines /
            # neighbouring peaks collapse into one blob.
            merge_tol_pts = max(1, distance_in_points // 2)
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
            if cons_idx:
                idx_arr = np.asarray(sorted(cons_idx), dtype=int)
                seps = separate_overlapping_peaks(
                    detection_ref, idx_arr, ppm, noise_std=det_noise, method='auto',
                    baseline_level=det_med)
                separated_peaks = [
                    pk for pk in seps
                    if detection_ref[pk['index']] >= min_height
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

        # Mark the apex at the top of the DISPLAYED (mean) spectrum, not the
        # per-sample-MAX reference used for detection — the max apex can sit a
        # few points off the visible mean peak (per-sample shifts), leaving the
        # overlay marker beside the top. Fall back to the detection apex when the
        # mean is flat there (single-sample analyte, no mean peak to land on).
        _mean_gate = (det_med if not _noise_was_provided else 0.0) + det_noise * min_snr
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

            areas = []
            for s in range(n_samples):
                sample_region = corrected_2d[start_idx:end_idx + 1, s]

                # Baseline correction for integration (linear between valleys)
                left_val = corrected_2d[start_idx, s]
                right_val = corrected_2d[end_idx, s]

                if end_idx > start_idx:
                    slope = (right_val - left_val) / (end_idx - start_idx)
                    baseline_s = left_val + slope * np.arange(end_idx - start_idx + 1)
                    corrected_s = sample_region - baseline_s
                    corrected_s = np.maximum(corrected_s, 0)
                else:
                    corrected_s = sample_region

                if len(x_region) > 1 and len(corrected_s) > 1:
                    area = abs(np.trapezoid(corrected_s, x_region))
                else:
                    area = 0.0

                areas.append(area)

            peak['area_per_sample'] = np.array(areas)
            peak['mean_intensity'] = float(np.mean([corrected_2d[peak['index'], s]
                                                     for s in range(n_samples)]))

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

        # STEP 11: Sort by SNR and assign rank
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
                'noise_std': float(noise_std),
                'shift_points': shift_pts,
                'shift_ppm': shift_ppm,
                'alignment_applied': bool(global_align and n_samples > 1),
            }
        return separated_peaks

    except Exception as e:
        print(f"\n[ERROR] PEAK DETECTION ERROR:")
        print(f"   {str(e)}")
        print("\n[TRACEBACK]")
        traceback.print_exc()
        print("=" * 70)
        raise
