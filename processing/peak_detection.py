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


def separate_overlapping_peaks(intensity, peak_indices, ppm, noise_std=None, method="auto"):
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
    thr = 3.0 * eff_noise      # intensity ≤ thr  →  at baseline

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1a  ·  outer baseline feet for every apex
    # ──────────────────────────────────────────────────────────────────────────
    outer_left  = {}   # apex → left  baseline-crossing index
    outer_right = {}   # apex → right baseline-crossing index
    lg_map      = {}   # apex → bool  left  grounded
    rg_map      = {}   # apex → bool  right grounded

    for apex in sorted_indices:
        # Walk left
        lf, lg = 0, False
        for j in range(apex, -1, -1):
            if float(intensity[j]) <= thr:
                lf, lg = j, True
                break
        if not lg:
            lf = int(np.argmin(intensity[:apex + 1])) if apex > 0 else 0

        # Walk right
        rf, rg = n - 1, False
        for j in range(apex, n):
            if float(intensity[j]) <= thr:
                rf, rg = j, True
                break
        if not rg:
            rf = apex + int(np.argmin(intensity[apex:])) if apex < n - 1 else n - 1

        outer_left[apex]  = lf
        outer_right[apex] = rf
        lg_map[apex]      = lg
        rg_map[apex]      = rg

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1b  ·  merge overlapping envelopes into clusters
    # ──────────────────────────────────────────────────────────────────────────
    # A cluster is a contiguous spectral segment bounded by two baseline
    # crossings that contains one or more detected apices.
    clusters = []     # list of {'lo', 'hi', 'apices': [...]}
    for apex in sorted_indices:
        lf = outer_left[apex]
        rf = outer_right[apex]
        merged = False
        for cl in clusters:
            # Envelopes overlap ↔ [lf,rf] and [cl['lo'],cl['hi']] intersect
            if lf <= cl['hi'] and rf >= cl['lo']:
                cl['lo']    = min(cl['lo'], lf)
                cl['hi']    = max(cl['hi'], rf)
                cl['apices'].append(apex)
                merged = True
                break
        if not merged:
            clusters.append({'lo': lf, 'hi': rf, 'apices': [apex]})

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

            start_idx = min(left_idx,  apex)
            end_idx   = max(right_idx, apex)

            # ── Height ────────────────────────────────────────────────────────
            foot_bl = (float(intensity[left_idx]) + float(intensity[right_idx])) / 2.0
            height  = float(intensity[apex]) - foot_bl

            # ── Area (trapezoid, linear baseline between the two feet) ─────────
            peak_reg  = intensity[start_idx:end_idx + 1]
            x_reg     = ppm[start_idx:end_idx + 1]
            if end_idx > start_idx and right_idx != left_idx:
                denom = right_idx - left_idx
                slope = (float(intensity[right_idx]) - float(intensity[left_idx])) / denom
                bl_reg = (float(intensity[left_idx])
                          + slope * np.arange(end_idx - start_idx + 1))
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
            if end_idx - start_idx + 1 >= 8:
                gauss_r2, _fit_model = fit_peak_quality(ppm_reg, int_reg)
            else:
                gauss_r2 = 0.0
            if gauss_r2 >= 0.90:
                gauss_qual = 'good'
            elif gauss_r2 >= 0.70:
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


def fit_peak_quality(ppm_region, intensity_region):
    """
    Fit both a Lorentzian and a Gaussian to a peak region; return the best R².

    NMR solution-state peaks follow a Lorentzian lineshape (T2 broadening),
    so trying both models and keeping the better fit gives a reliable quality
    metric even for peaks that are not perfectly symmetric.

    Parameters
    ----------
    ppm_region       : 1-D array — chemical shift axis for the peak region
    intensity_region : 1-D array — intensities (same length)

    Returns
    -------
    r2    : float — best coefficient of determination (0–1)
    model : str   — 'lorentzian', 'gaussian', or 'none'
    """
    from scipy.optimize import curve_fit

    n = len(ppm_region)
    if n < 5:
        return 0.0, 'none'

    ppm_region = np.asarray(ppm_region, dtype=float)
    y = np.asarray(intensity_region, dtype=float).copy()

    # Subtract linear baseline between the two feet
    bl = np.linspace(y[0], y[-1], n)
    y -= bl
    y = np.maximum(y, 0.0)

    y_max = float(y.max())
    if y_max <= 1e-15:
        return 0.0, 'none'

    amp0 = y_max
    cen0 = float(ppm_region[int(np.argmax(y))])
    above = np.where(y >= y_max * 0.5)[0]
    fwhm = abs(float(ppm_region[above[-1]] - ppm_region[above[0]])) if len(above) >= 2 \
           else abs(float(ppm_region[-1] - ppm_region[0])) / 3.0
    fwhm = max(fwhm, 1e-5)

    ppm_lo = float(min(ppm_region[0], ppm_region[-1]))
    ppm_hi = float(max(ppm_region[0], ppm_region[-1]))
    span   = max(ppm_hi - ppm_lo, 1e-5)

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
                best_r2   = r2
                best_model = name
        except Exception:
            pass

    return float(best_r2), best_model


def fit_gaussian_quality(ppm_region, intensity_region):
    """Backward-compatible wrapper — delegates to fit_peak_quality."""
    r2, _model = fit_peak_quality(ppm_region, intensity_region)
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


def detect_peaks(ppm, intensities, noise_std=None, min_snr=3.0,
                 min_distance=0.01, baseline_correct=True,
                 baseline_method='airPLS', baseline_lam=1e5,
                 global_align=True, global_align_max_shift_ppm=0.03,
                 samples_info=None, progress_callback=None,
                 return_details=False):
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

            # Also correct each sample individually for integration
            corrected_samples = []
            for s in range(n_samples):
                if n_samples > 1:
                    _report(0.20 + 0.22 * ((s + 1) / n_samples),
                            f"Baseline correction per sample ({s + 1}/{n_samples})")
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

        # STEP 6: Estimate or validate noise
        _report(0.60, "Estimating detection threshold")
        if noise_std is None:
            # Estimate from baseline-corrected spectrum edges
            noise_std = estimate_noise_region(corrected_mean)
            print(f"\n[INFO] Noise estimated from edges: sigma = {noise_std:.6f}")
        else:
            print(f"\n[INFO] Using provided noise_std: {noise_std:.6f}")

        # STEP 7: Calculate detection parameters
        _report(0.68, "Computing peak detection parameters")
        ppm_range = np.max(ppm) - np.min(ppm)
        ppm_step = ppm_range / n_points
        distance_in_points = max(1, int(min_distance / ppm_step))
        min_height = noise_std * min_snr

        print(f"\n[INFO] Detection parameters:")
        print(f"   ppm_range: {ppm_range:.4f}")
        print(f"   distance: {min_distance} ppm = {distance_in_points} points")
        print(f"   min_height: {min_height:.6f} (SNR={min_snr} * sigma={noise_std:.6f})")

        # STEP 8: Find peaks in baseline-corrected spectrum
        _report(0.76, "Locating candidate apexes")
        print("\n[INFO] Running find_peaks on baseline-corrected spectrum...")
        peak_indices, properties = find_peaks(
            corrected_mean,
            height=min_height,
            distance=distance_in_points,
            prominence=noise_std * min_snr * 0.5  # Add prominence requirement
        )

        print(f"   [OK] Raw peaks detected: {len(peak_indices)}")

        if len(peak_indices) == 0:
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

        # STEP 9: Separate peaks by valleys
        _report(0.84, "Estimating peak boundaries")
        print("\n[INFO] Separating peaks by valley points...")
        separated_peaks = separate_overlapping_peaks(
            corrected_mean, peak_indices, ppm, noise_std=noise_std, method='auto')

        # Safety gate: this workflow only accepts positive resonances.
        # Valleys / negative deflections must never be promoted to peaks.
        positive_peaks = [
            peak for peak in separated_peaks
            if peak['height'] > 0 and corrected_mean[peak['index']] >= min_height
        ]
        if len(positive_peaks) != len(separated_peaks):
            print(f"   [INFO] Discarded {len(separated_peaks) - len(positive_peaks)} non-positive candidate(s)")
        separated_peaks = positive_peaks

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
