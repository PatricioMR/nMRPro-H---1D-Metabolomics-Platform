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
    """CV% of a 1-D array, returns NaN if < 2 finite positive values."""
    v = [x for x in vals if np.isfinite(x) and x > 0]
    if len(v) < 2:
        return float('nan')
    m = float(np.mean(v))
    return float(np.std(v) / m * 100) if m > 0 else float('nan')


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
    pattern = (hit.get('pattern') or 's').lower().strip()

    _pascal = {
        's':  [1],
        'd':  [1, 1],
        't':  [1, 2, 1],
        'q':  [1, 3, 3, 1],
        'p':  [1, 4, 6, 4, 1],
        'h':  [1, 5, 10, 10, 5, 1],
        'dd': [1, 1, 1, 1],
        'dt': [1, 1, 2, 2, 1, 1],
        'td': [1, 2, 1, 1, 2, 1],
    }
    anchor_idx = hit.get('anchor_line_idx')
    pred_lines = hit.get('predicted_lines', [])
    if anchor_idx is not None and 0 <= anchor_idx < len(pred_lines):
        rel_int = float(pred_lines[anchor_idx].get('rel_intensity', 0))
        total   = sum(float(pl.get('rel_intensity', 0)) for pl in pred_lines)
        if total > 0 and rel_int > 0:
            return nH * rel_int / total

    coeffs = _pascal.get(pattern)
    if coeffs is None or len(coeffs) == 1:
        return nH

    j_str  = str(hit.get('j_coupling') or '').strip()
    j_nums = [float(x) for x in _re.findall(r'\d+\.?\d*', j_str) if float(x) > 0]
    if not j_nums:
        return nH

    j_hz   = j_nums[0]
    spacing_ppm = j_hz / spectrometer_mhz

    n      = len(coeffs)
    centre = float(hit.get('ppm_center', hit.get('ppm', detected_ppm)))
    positions = [centre + (i - (n - 1) / 2.0) * spacing_ppm for i in range(n)]

    dists   = [abs(detected_ppm - pos) for pos in positions]
    closest = dists.index(min(dists))

    fraction = coeffs[closest] / sum(coeffs)
    return nH * fraction


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


def _fit_lorentz_mean(x, y_mean):
    """Fit Lorentzian to mean spectrum after 2-point baseline subtraction.

    Returns (A, cen, wid_hwhm).  A and wid describe the baseline-corrected
    peak; area = A * pi * wid.
    """
    from scipy.optimize import curve_fit
    y_raw = np.clip(y_mean, 0, None)
    if len(x) < 4 or y_raw.max() <= 0:
        i = int(np.argmax(y_raw))
        return float(y_raw[i]), float(x[i]), float(abs(x[-1] - x[0]) / 4)

    _, _, baseline = _linear_baseline(x, y_raw)
    y = np.clip(y_raw - baseline, 0, None)
    if y.max() <= 0:
        y = y_raw   # fallback: no baseline subtraction

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
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return A0, c0, w0


def _fit_lorentz_mean_with_bl(x, y_mean):
    """Like _fit_lorentz_mean but also returns baseline parameters.

    Returns (A, cen, wid, bl_y0, bl_slope).
    bl_y0 is baseline at x[0]; bl_slope in units/ppm.
    """
    y_raw = np.clip(y_mean, 0, None)
    bl_y0, bl_slope, _ = _linear_baseline(x, y_raw)
    A, cen, wid = _fit_lorentz_mean(x, y_mean)
    return A, cen, wid, bl_y0, bl_slope


def _fit_lorentz_sample(x, y, A0, cen0, wid0, max_shift_ppm=0.02):
    """Per-sample Lorentzian fit after 2-point baseline subtraction.

    Fits each sample independently with tight bounds around the mean-spectrum
    parameters.  Area = A * pi * wid (analytical integral of baseline-corrected
    peak).  Returns (area, A, cen, wid) or None on failure.
    """
    from scipy.optimize import curve_fit
    y_raw = np.clip(y, 0, None)
    if len(x) < 4 or y_raw.max() <= 0:
        return None
    _, _, baseline = _linear_baseline(x, y_raw)
    y_cl = np.clip(y_raw - baseline, 0, None)
    if y_cl.max() <= 0:
        y_cl = y_raw   # fallback
    dppm = abs(float(x[1] - x[0])) if len(x) > 1 else 1e-4
    A_hi  = max(float(y_cl.max()) * 3.0, A0 * 5.0)
    w_lo  = max(wid0 * 0.3, dppm)
    w_hi  = wid0 * 3.0
    c_lo  = max(cen0 - max_shift_ppm, float(x.min()))
    c_hi  = min(cen0 + max_shift_ppm, float(x.max()))
    if c_lo >= c_hi:
        c_lo = float(x.min())
        c_hi = float(x.max())
    try:
        popt, _ = curve_fit(
            _lorentz_func, x, y_cl,
            p0=[A0, cen0, wid0],
            bounds=([0.0, c_lo, w_lo], [A_hi, c_hi, w_hi]),
            maxfev=6000)
        A, cen, wid = float(popt[0]), float(popt[1]), float(popt[2])
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
    mask = np.abs(x - cen_mean) <= max_shift_ppm
    if mask.sum() < 2:
        return float(cen_mean)
    x_w, y_w = x[mask], y[mask]
    if y_w.max() <= 0:
        return float(cen_mean)
    return float(x_w[int(np.argmax(y_w))])


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
