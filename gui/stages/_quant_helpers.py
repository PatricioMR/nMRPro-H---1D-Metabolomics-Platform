# gui/stages/_quant_helpers.py
"""
Standalone helper functions used by the Quantification stage.

Extracted from quantification_stage.py to keep each file focused:
  - _quant_helpers.py  → pure functions + constants (this file)
  - _quant_visualization.py → _QuantVisMixin  (detail plots / tooltips)
  - _quant_computation.py   → _QuantCompMixin (_run, VIP, update_sample_area, export)
"""
import numpy as np


def _trapz(y, x):
    try:
        return np.trapezoid(y, x=x)
    except AttributeError:
        return np.trapz(y, x=x)


# ── Robust / non-parametric statistics for J-Resolve ─────────────────────────
# qNMR fundamental: every peak of a compound yields the same concentration
# (area ∝ nH·conc).  Per-peak concentration probes should therefore agree.  CV
# (std/mean) assumes symmetric dispersion and is fragile with 2–3 probes — poor
# for skewed NMR data.  fold_range (max/min) is scale-free, distribution-free,
# robust at small n, and its tolerance is physically motivated by T1 differential
# saturation (≈ the 0.35–2.50 anchor_ratio window → ~2.5× fold).

def fold_range(values):
    """Max/min ratio of strictly-positive values.

    Returns 1.0 for a single positive value (perfectly consistent by
    definition) and inf if any non-positive/empty (cannot form a ratio).
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0],
                   dtype=float)
    if v.size == 0:
        return float('inf')
    if v.size == 1:
        return 1.0
    lo = float(v.min())
    hi = float(v.max())
    return hi / lo if lo > 0 else float('inf')


def bootstrap_ci_median(values, n_boot=1000, ci=0.95, min_n=4, seed=0):
    """Non-parametric percentile bootstrap CI for the median of `values`.

    Resamples `values` with replacement n_boot times, takes the median of each
    resample, and returns the percentile CI.

    Returns (point_median, lo, hi).  When fewer than `min_n` finite-positive
    values are available the bootstrap is unreliable, so (median, nan, nan) is
    returned — the caller must treat the CI as unavailable, never fabricate it.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0],
                   dtype=float)
    if v.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    point = float(np.median(v))
    if v.size < min_n:
        return (point, float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    meds = np.median(v[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(meds, alpha * 100.0))
    hi = float(np.percentile(meds, (1.0 - alpha) * 100.0))
    return (point, lo, hi)


def bootstrap_fraction_ci(values, n_boot=1000, ci=0.95, min_n=6, seed=0):
    """Non-parametric percentile bootstrap CI for the MEAN of a sample cohort.

    Used for cohort-level uncertainty (J-Resolve fractions / concentrations per
    group), resampling sample columns with replacement.  Distribution-free, so
    valid for skewed/log-normal metabolomics data.

    Returns (point_mean, lo, hi); (mean, nan, nan) when n < min_n (too few
    samples to bootstrap — report 'unavailable', do not invent an interval).
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    point = float(np.mean(v))
    if v.size < min_n:
        return (point, float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = np.mean(v[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, alpha * 100.0))
    hi = float(np.percentile(means, (1.0 - alpha) * 100.0))
    return (point, lo, hi)


def _pls_vip(X, y, n_components=2):
    """Compute PLS-DA VIP scores (no sklearn required).

    X : (n_samples, n_features)  — concentrations, NaN columns are skipped
    y : (n_samples,)             — group labels encoded as integers
    Returns array of shape (n_features,) with VIP scores (NaN for skipped cols).
    """
    n, p = X.shape
    n_comp = min(n_components, min(n - 1, p))
    if n_comp < 1 or n < 3:
        return np.full(p, float('nan'))

    col_means = np.nanmean(X, axis=0)
    nan_cols   = np.all(np.isnan(X), axis=0)
    Xc = np.where(np.isnan(X), col_means, X)
    x_std = Xc.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    Xc = (Xc - col_means) / x_std

    yc = y.astype(float)
    yc -= yc.mean()
    y_std = yc.std()
    if y_std < 1e-12:
        return np.full(p, float('nan'))
    yc /= y_std

    W = np.zeros((p, n_comp))
    T = np.zeros((n, n_comp))
    SS = np.zeros(n_comp)

    Xr, yr = Xc.copy(), yc.copy()
    for k in range(n_comp):
        w = Xr.T @ yr
        wn = np.linalg.norm(w)
        if wn < 1e-12:
            break
        w /= wn
        t = Xr @ w
        tt = float(t @ t)
        if tt < 1e-12:
            break
        q = float(yr @ t) / tt
        p_k = Xr.T @ t / tt
        W[:, k] = w
        T[:, k] = t
        SS[k] = q * q * tt
        Xr -= np.outer(t, p_k)
        yr -= q * t

    SS_total = SS.sum()
    if SS_total < 1e-12:
        return np.full(p, float('nan'))

    vip = np.sqrt(p * (W ** 2 @ SS) / SS_total)
    vip[nan_cols] = float('nan')
    return vip


def _cv(vals):
    """CV% of a 1-D array (sample std, ddof=1). NaN if < 2 finite positive values."""
    v = [x for x in vals if np.isfinite(x) and x > 0]
    if len(v) < 2:
        return float('nan')
    m = float(np.mean(v))
    return float(np.std(v, ddof=1) / m * 100) if m > 0 else float('nan')


def _pqn(areas_dict, ref_dict):
    quotients = [v / ref_dict[k]
                 for k, v in areas_dict.items()
                 if k in ref_dict and ref_dict[k] > 0 and v > 0]
    if len(quotients) < 3:
        return dict(areas_dict), 1.0
    f = float(np.median(quotients))
    if f <= 0:
        return dict(areas_dict), 1.0
    return {k: v / f for k, v in areas_dict.items()}, f


def _effective_nh(hit, detected_ppm, spectrometer_mhz=600.0):
    """Return the effective nH for the specific detected line of a multiplet."""
    import re as _re
    nH = max(1.0, float(hit.get('protons', 1) or 1))
    # Result entries store multiplicity as 'type' (from DB), not 'pattern'.
    # 'type' is often a non-multiplicity string (e.g. 'HMDB', 'lipid_region') —
    # only use 'pattern' or 'multiplicity' to avoid misidentifying the multiplet.
    pattern = (hit.get('pattern') or hit.get('multiplicity') or 's').lower().strip()

    _pascal = {
        's':   [1],
        'd':   [1, 1],
        't':   [1, 2, 1],
        'q':   [1, 3, 3, 1],
        'p':   [1, 4, 6, 4, 1],
        'h':   [1, 5, 10, 10, 5, 1],
        'dd':  [1, 1, 1, 1],
        # dt = doublet-of-triplets: [1,1]⊗[1,2,1] → large-J doublet, small-J triplet
        'dt':  [1, 2, 1, 1, 2, 1],
        # td = triplet-of-doublets: [1,2,1]⊗[1,1] → large-J triplet, small-J doublet
        'td':  [1, 1, 2, 2, 1, 1],
        # ddd = [1,1]⊗[1,1]⊗[1,1] → 8 equal lines
        'ddd': [1, 1, 1, 1, 1, 1, 1, 1],
        # dq = [1,1]⊗[1,3,3,1]
        'dq':  [1, 3, 3, 1, 1, 3, 3, 1],
        # tt = [1,2,1]⊗[1,2,1]
        'tt':  [1, 2, 1, 2, 4, 2, 1, 2, 1],
    }
    anchor_idx = hit.get('anchor_line_idx')
    pred_lines = hit.get('predicted_lines', [])
    coeffs     = _pascal.get(pattern)

    # Priority 0: line_idx stored at annotation time (ppm-matched to predicted_lines).
    # This is the most direct mapping: annotation search compared detected ppm to
    # each predicted line and recorded which index matched — no estimation needed.
    _line_idx = hit.get('line_idx')
    if (_line_idx is not None and len(pred_lines) > 1
            and 0 <= _line_idx < len(pred_lines)):
        rel_int = float(pred_lines[_line_idx].get('rel_intensity', 0))
        total   = sum(float(pl.get('rel_intensity', 0)) for pl in pred_lines)
        if total > 0 and rel_int > 0:
            return nH * rel_int / total

    # Priority 1: Pascal coefficients with known anchor index.
    # More reliable than measured pred_ri (which uses observed heights,
    # not theoretical areas, and can be asymmetric due to overlap/noise).
    if (coeffs and anchor_idx is not None
            and 0 <= anchor_idx < len(coeffs)):
        return nH * coeffs[anchor_idx] / sum(coeffs)

    # Priority 2: pred_lines with known anchor (complex multiplets dd/dt/etc.
    # where Pascal is not directly applicable).
    if anchor_idx is not None and 0 <= anchor_idx < len(pred_lines):
        rel_int = float(pred_lines[anchor_idx].get('rel_intensity', 0))
        total   = sum(float(pl.get('rel_intensity', 0)) for pl in pred_lines)
        if total > 0 and rel_int > 0:
            return nH * rel_int / total

    # Priority 3: Pascal + J-coupling positional matching (anchor unknown).
    if coeffs and len(coeffs) > 1:
        j_str  = str(hit.get('j_coupling') or '').strip()
        j_nums = [float(x) for x in _re.findall(r'\d+\.?\d*', j_str)
                  if float(x) > 0]
        if j_nums:
            j_hz        = j_nums[0]
            spacing_ppm = j_hz / spectrometer_mhz
            n           = len(coeffs)
            centre      = float(hit.get('ppm_center', hit.get('ppm', detected_ppm)))
            positions   = [centre + (i - (n - 1) / 2.0) * spacing_ppm
                           for i in range(n)]
            dists   = [abs(detected_ppm - pos) for pos in positions]
            closest = dists.index(min(dists))
            return nH * coeffs[closest] / sum(coeffs)

    # Priority 4: pred_lines positional matching (anchor unknown, complex mult).
    if len(pred_lines) > 1:
        ppms = [float(pl.get('ppm', float('inf'))) for pl in pred_lines]
        if any(p != float('inf') for p in ppms):
            dists   = [abs(detected_ppm - p) for p in ppms]
            closest = dists.index(min(dists))
            rel_int = float(pred_lines[closest].get('rel_intensity', 0))
            total   = sum(float(pl.get('rel_intensity', 0)) for pl in pred_lines)
            if total > 0 and rel_int > 0:
                return nH * rel_int / total

    # Priority 5: uniform distribution — pred_lines known but no rel_intensity.
    # Area/Lorentz integrate ONE detected line; nH/n_lines is the unbiased estimate.
    if len(pred_lines) > 1:
        return nH / len(pred_lines)

    return nH


def _effective_nh_source(hit, detected_ppm, spectrometer_mhz=600.0):
    """Same logic as _effective_nh but returns (nH, source_description)."""
    import re as _re
    nH = max(1.0, float(hit.get('protons', 1) or 1))
    pattern = (hit.get('pattern') or hit.get('multiplicity') or 's').lower().strip()
    _pascal = {
        's': [1], 'd': [1, 1], 't': [1, 2, 1], 'q': [1, 3, 3, 1],
        'p': [1, 4, 6, 4, 1], 'h': [1, 5, 10, 10, 5, 1],
        'dd': [1, 1, 1, 1], 'dt': [1, 2, 1, 1, 2, 1], 'td': [1, 1, 2, 2, 1, 1],
        'ddd': [1, 1, 1, 1, 1, 1, 1, 1], 'dq': [1, 3, 3, 1, 1, 3, 3, 1],
        'tt': [1, 2, 1, 2, 4, 2, 1, 2, 1],
    }
    anchor_idx = hit.get('anchor_line_idx')
    pred_lines = hit.get('predicted_lines', [])
    coeffs     = _pascal.get(pattern)
    n_lines    = len(pred_lines)

    # Priority 0: line_idx from annotation-time ppm matching.
    _line_idx = hit.get('line_idx')
    if (_line_idx is not None and n_lines > 1
            and 0 <= _line_idx < n_lines):
        ri  = float(pred_lines[_line_idx].get('rel_intensity', 0))
        tot = sum(float(pl.get('rel_intensity', 0)) for pl in pred_lines)
        if tot > 0 and ri > 0:
            frac = ri / tot
            return nH * frac, f'line_idx={_line_idx}  {nH}×{frac:.4f}'

    if coeffs and anchor_idx is not None and 0 <= anchor_idx < len(coeffs):
        frac = coeffs[anchor_idx] / sum(coeffs)
        return nH * frac, f'Pascal+anchor  [{pattern} line {anchor_idx}]  {nH}×{frac:.4f}'

    if anchor_idx is not None and 0 <= anchor_idx < n_lines:
        ri  = float(pred_lines[anchor_idx].get('rel_intensity', 0))
        tot = sum(float(pl.get('rel_intensity', 0)) for pl in pred_lines)
        if tot > 0 and ri > 0:
            frac = ri / tot
            return nH * frac, f'pred_lines+anchor  [line {anchor_idx}]  {nH}×{frac:.4f}'

    if coeffs and len(coeffs) > 1:
        j_str  = str(hit.get('j_coupling') or '').strip()
        j_nums = [float(x) for x in _re.findall(r'\d+\.?\d*', j_str) if float(x) > 0]
        if j_nums:
            j_hz        = j_nums[0]
            spacing_ppm = j_hz / spectrometer_mhz
            n           = len(coeffs)
            centre      = float(hit.get('ppm_center', hit.get('ppm', detected_ppm)))
            positions   = [centre + (i - (n - 1) / 2.0) * spacing_ppm for i in range(n)]
            dists       = [abs(detected_ppm - pos) for pos in positions]
            closest     = dists.index(min(dists))
            frac        = coeffs[closest] / sum(coeffs)
            dist_ppm    = dists[closest]
            return (nH * frac,
                    f'Pascal+J  [{pattern} J={j_hz}Hz line {closest}/{n-1}]  '
                    f'{nH}×{frac:.4f}  Δppm={dist_ppm:.4f}')

    if n_lines > 1:
        ppms = [float(pl.get('ppm', float('inf'))) for pl in pred_lines]
        if any(p != float('inf') for p in ppms):
            dists   = [abs(detected_ppm - p) for p in ppms]
            closest = dists.index(min(dists))
            ri  = float(pred_lines[closest].get('rel_intensity', 0))
            tot = sum(float(pl.get('rel_intensity', 0)) for pl in pred_lines)
            if tot > 0 and ri > 0:
                frac = ri / tot
                return nH * frac, f'pred_lines+pos  [line {closest}/{n_lines-1}]  {nH}×{frac:.4f}'

    # Priority 5: uniform distribution — pred_lines known but no rel_intensity.
    # Area/Lorentz integrate ONE detected line; nH/n_lines is the unbiased estimate.
    if n_lines > 1:
        frac = 1.0 / n_lines
        return nH * frac, f'uniform  [{pattern} {n_lines} lines]  {nH}×{frac:.4f}'

    return nH, f'FALLBACK full-nH  (no J/anchor/pred_lines for {pattern})'


def _mcr_als(X, n_components=1, max_iter=200, tol=1e-6, S_init=None):
    """Wrapper around shared MCR-ALS implementation."""
    from gui.utils.mcr_utils import mcr_als as _shared_mcr
    return _shared_mcr(X, n_components=n_components, max_iter=max_iter,
                       tol=tol, S_init=S_init)


def _lorentz_shape(x, cen, wid):
    """Unit-height Lorentzian: 1 at centre, HWHM = wid."""
    return wid ** 2 / ((x - cen) ** 2 + wid ** 2)


def _lorentz_func(x, A, cen, wid):
    return A * _lorentz_shape(x, cen, wid)


def _linear_baseline(x, y):
    """2-point linear baseline anchored at window endpoints.

    Returns (y0, slope, baseline_array) where baseline_array has the same
    shape as x.  y0 is the baseline value at x[0]; slope is in units/ppm.
    """
    x0, y0 = float(x[0]), float(y[0])
    x1, y1 = float(x[-1]), float(y[-1])
    slope = (y1 - y0) / (x1 - x0) if abs(x1 - x0) > 1e-10 else 0.0
    return float(y0), float(slope), y0 + slope * (x - x0)


def _fit_lorentz_mean(x, y_mean, with_baseline=True):
    """Fit Lorentzian to mean spectrum.

    with_baseline=True: subtract 2-point linear baseline before fitting.
    Returns (A, cen, wid_hwhm).  Area = A * pi * wid.
    """
    from scipy.optimize import curve_fit
    y_raw = np.clip(y_mean, 0, None)
    if len(x) < 4 or y_raw.max() <= 0:
        i = int(np.argmax(y_raw))
        return float(y_raw[i]), float(x[i]), float(abs(x[-1] - x[0]) / 4)

    if with_baseline:
        _, _, baseline = _linear_baseline(x, y_raw)
        y = np.clip(y_raw - baseline, 0, None)
        if y.max() <= 0:
            y = y_raw
    else:
        y = y_raw

    i_max = int(np.argmax(y))
    A0 = float(y[i_max])
    c0 = float(x[i_max])
    above = np.where(y >= A0 / 2.0)[0]
    w0 = (abs(float(x[above[-1]] - x[above[0]])) / 2.0
          if len(above) >= 2 else abs(float(x[-1] - x[0])) / 4.0)
    dppm = abs(float(x[1] - x[0])) if len(x) > 1 else 1e-4
    w0 = max(w0, dppm)
    x_span = abs(float(x[-1] - x[0]))
    try:
        popt, _ = curve_fit(
            _lorentz_func, x, y,
            p0=[A0, c0, w0],
            bounds=([0, float(x.min()), dppm],
                    [A0 * 5.0, float(x.max()), x_span]),
            maxfev=4000)
        if not all(np.isfinite(v) for v in popt):
            return A0, c0, w0
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return A0, c0, w0


def _fit_lorentz_mean_with_bl(x, y_mean, with_baseline=True):
    """Like _fit_lorentz_mean but also returns baseline parameters.

    Returns (A, cen, wid, bl_y0, bl_slope).
    When with_baseline=False, bl_y0 and bl_slope are 0.
    """
    y_raw = np.clip(y_mean, 0, None)
    if with_baseline:
        bl_y0, bl_slope, _ = _linear_baseline(x, y_raw)
    else:
        bl_y0, bl_slope = 0.0, 0.0
    A, cen, wid = _fit_lorentz_mean(x, y_mean, with_baseline=with_baseline)
    return A, cen, wid, bl_y0, bl_slope


def _fit_lorentz_sample(x, y, A0, cen0, wid0, max_shift_ppm=0.02, with_baseline=True):
    """Per-sample Lorentzian fit.

    with_baseline=True: subtract 2-point linear baseline before fitting.
    Area = A * pi * wid.  Returns (area, A, cen, wid) or None on failure.
    """
    from scipy.optimize import curve_fit
    y_raw = np.clip(y, 0, None)
    if len(x) < 4 or y_raw.max() <= 0:
        return None
    if with_baseline:
        _, _, baseline = _linear_baseline(x, y_raw)
        y_cl = np.clip(y_raw - baseline, 0, None)
        if y_cl.max() <= 0:
            y_cl = y_raw
    else:
        y_cl = y_raw
    dppm = abs(float(x[1] - x[0])) if len(x) > 1 else 1e-4
    A_hi  = max(float(y_cl.max()) * 3.0, A0 * 5.0)
    w_lo  = max(wid0 * 0.3, dppm)
    w_hi  = wid0 * 3.0
    c_lo  = max(cen0 - max_shift_ppm, float(x.min()))
    c_hi  = min(cen0 + max_shift_ppm, float(x.max()))
    if c_lo >= c_hi:
        c_lo = float(x.min())
        c_hi = float(x.max())
    if not (np.isfinite(A0) and np.isfinite(cen0) and np.isfinite(wid0)):
        return None
    try:
        popt, _ = curve_fit(
            _lorentz_func, x, y_cl,
            p0=[A0, cen0, wid0],
            bounds=([0.0, c_lo, w_lo], [A_hi, c_hi, w_hi]),
            maxfev=6000)
        A, cen, wid = float(popt[0]), float(popt[1]), float(popt[2])
        if not (np.isfinite(A) and np.isfinite(cen) and np.isfinite(wid)):
            return None
        return A * np.pi * wid, A, cen, wid
    except Exception:
        return None


def _fit_multi_lorentz(x, y_mean, all_y, peak_ppms, max_shift=0.015):
    """Fit sum of N Lorentzians to a cluster region for deconvolution.

    x         : ppm array (any order — sorted internally)
    y_mean    : mean spectrum intensity over x
    all_y     : (len(x), n_samples) intensity matrix
    peak_ppms : initial ppm position per component (one per cluster peak)
    max_shift : max center shift allowed from initial ppm (ppm)

    Returns {'r2': float, 'components': [(area_mean, cen_ppm, areas_per_sample), ...]}
    or None on failure / insufficient data.

    Per-sample amplitudes are solved via NNLS with shapes fixed from the mean
    fit — all components simultaneously, so no area leaks between neighbours.
    Area = A × π × wid  (analytical integral of Lorentzian).
    """
    from scipy.optimize import curve_fit, nnls

    n = len(peak_ppms)
    if n < 2 or len(x) < 2 * n:
        return None

    # Sort to ascending for curve_fit (ppm_arr may be descending in NMR)
    order   = np.argsort(x)
    x_s     = x[order]
    ym_s    = np.clip(y_mean[order], 0, None)
    all_y_s = all_y[order, :]
    n_samp  = all_y_s.shape[1]

    span  = float(x_s[-1] - x_s[0]) or 1.0
    ym_pk = float(ym_s.max()) if ym_s.max() > 0 else 1e-6

    def _model(xv, *params):
        yv = np.zeros_like(xv, dtype=float)
        for i in range(n):
            A, cen, wid = params[3 * i], params[3 * i + 1], params[3 * i + 2]
            yv += A * wid ** 2 / ((xv - cen) ** 2 + wid ** 2)
        return yv

    p0, lo, hi = [], [], []
    for ppm0 in peak_ppms:
        idx  = int(np.argmin(np.abs(x_s - ppm0)))
        A0   = max(float(ym_s[idx]), 1e-6)
        wid0 = max(span / (4.0 * n), 1e-4)
        p0  += [A0, float(ppm0), wid0]
        lo  += [0.0,
                max(float(x_s[0]),  float(ppm0) - max_shift),
                1e-5]
        hi  += [max(5.0 * ym_pk, A0 * 10.0),
                min(float(x_s[-1]), float(ppm0) + max_shift),
                span]

    try:
        popt, _ = curve_fit(_model, x_s, ym_s, p0=p0,
                            bounds=(lo, hi), maxfev=8000)
    except Exception:
        return None

    y_fit  = _model(x_s, *popt)
    ss_res = float(np.sum((ym_s - y_fit) ** 2))
    ss_tot = float(np.sum((ym_s - ym_s.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    # Build unit-height shape matrix (n_points × n_components) from mean fit.
    # Per-sample amplitudes solved via NNLS — all components simultaneously so
    # no area leaks between neighbours (equivalent to fixed-shape MCR-ALS step).
    shapes = np.zeros((len(x_s), n), dtype=float)
    wids   = np.empty(n)
    for i in range(n):
        _, cen_i, wid_i = float(popt[3 * i]), float(popt[3 * i + 1]), float(popt[3 * i + 2])
        shapes[:, i] = wid_i ** 2 / ((x_s - cen_i) ** 2 + wid_i ** 2)
        wids[i]      = wid_i

    comp_areas_s = np.zeros((n, n_samp))
    for si in range(n_samp):
        y_si     = np.clip(all_y_s[:, si], 0, None)
        amps, _  = nnls(shapes, y_si)          # non-negative amplitudes
        comp_areas_s[:, si] = amps * np.pi * wids   # area = A × π × wid

    comp_results = []
    for i in range(n):
        A_i   = float(popt[3 * i])
        cen_i = float(popt[3 * i + 1])
        wid_i = float(popt[3 * i + 2])
        comp_results.append((float(A_i * np.pi * wid_i), cen_i, comp_areas_s[i]))

    return {'r2': r2, 'components': comp_results}


# kept for back-compat — used by update_sample_area fallback path
def _align_sample_center(x, y, cen_mean, max_shift_ppm=0.02):
    # Delegates to the DSP-layer apex finder (single source of truth, shared with
    # detection-time per-peak alignment). noise_floor=0 preserves prior behaviour
    # (always return the apex, falling back to cen_mean only when no signal).
    from processing.peak_alignment import _refine_one_center
    center, _ = _refine_one_center(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float),
                                   float(cen_mean), max_shift_ppm, 0.0)
    return center


def _lorentz_amplitude(x, y, cen, wid):
    """NNLS amplitude for a Lorentzian with fixed shape (fallback only)."""
    from scipy.optimize import nnls as _nnls
    wid = max(abs(float(wid)), 1e-5)
    L = _lorentz_shape(x, cen, wid).reshape(-1, 1)
    A, _ = _nnls(L, np.clip(y, 0, None))
    return float(A[0])


_STD_PRESETS = {
    'TSP':         {'ppm': '0.0',  'nh': '9'},
    'DSS':         {'ppm': '0.0',  'nh': '9'},
    'TMSP':        {'ppm': '0.0',  'nh': '9'},
    'Maleic acid': {'ppm': '6.0',  'nh': '2'},
    'Formate':     {'ppm': '8.46', 'nh': '1'},
    'Custom':      {'ppm': '',     'nh': ''},
}
