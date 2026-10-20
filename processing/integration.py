# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/integration.py
"""
Integration & lineshape DSP for 1H-NMR quantification.

Pure signal-processing primitives — no GUI, no ``self`` state — used by the
Quantification stage to turn spectral regions into areas:

  • trapezoidal region integration            (_trapz)
  • 2-point linear baseline                    (_linear_baseline)
  • Lorentzian lineshape / model               (_lorentz_shape, _lorentz_func)
  • single-line Lorentzian fits (mean/sample)  (_fit_lorentz_mean*, _fit_lorentz_sample)
  • multi-Lorentzian cluster deconvolution     (_fit_multi_lorentz)
  • fixed-shape amplitude (NNLS)               (_lorentz_amplitude)
  • per-line-weighted multiplet integration    (jr_pointwise_multiplet_area)

NMR lines are Lorentzian in the absence of field inhomogeneity; the analytical
integral of a Lorentzian A·w²/((x−c)²+w²) is A·π·w (used throughout for area).
Per-sample amplitudes in clustered regions are solved by non-negative least
squares (Lawson & Hanson, "Solving Least Squares Problems", 1974) with shapes
fixed from the mean fit, so no area leaks between overlapping neighbours.

Extracted from gui/stages/_quant_helpers.py to keep signal processing in the
processing/ layer (MVC); _quant_helpers re-exports these names for callers.
"""
import numpy as np


def _trapz(y, x):
    try:
        return np.trapezoid(y, x=x)
    except AttributeError:
        return np.trapz(y, x=x)


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


def _lorentz_amplitude(x, y, cen, wid):
    """NNLS amplitude for a Lorentzian with fixed shape (fallback only)."""
    from scipy.optimize import nnls as _nnls
    wid = max(abs(float(wid)), 1e-5)
    L = _lorentz_shape(x, cen, wid).reshape(-1, 1)
    A, _ = _nnls(L, np.clip(y, 0, None))
    return float(A[0])


def jr_pointwise_multiplet_area(x_window, seg_matrix, line_ppms, line_fracs):
    """Per-sample multiplet area with per-line J-Resolve fractions.

    A multiplet spans several lines that NNLS (J-Resolve) may resolve
    differently — one leg of a doublet can be clean while the other is shared.
    Integrating the whole window and scaling by a single (anchor) fraction loses
    that per-leg reality. Here every spectral point of the window is weighted by
    the fraction of the predicted line NEAREST to it, so each leg contributes its
    own share, then the weighted curve is integrated.

    Reduces EXACTLY to ``fraction × whole_window_area`` when all line fractions
    are equal (and to the plain area when all are 1.0), so multiplets that were
    not resolved per-line are unchanged.

    Parameters
    ----------
    x_window : (m,) ppm of the integration window.
    seg_matrix : (m,) or (m, n_s) intensities (already baseline-handled).
    line_ppms : (L,) predicted line positions (ppm).
    line_fracs : list length L; each entry a scalar in [0, 1] or an (n_s,) array.

    Returns
    -------
    (n_s,) integrated area per sample (absolute value, like the trapz path).
    """
    x   = np.asarray(x_window, dtype=float)
    seg = np.asarray(seg_matrix, dtype=float)
    if seg.ndim == 1:
        seg = seg[:, None]
    m, n_s = seg.shape
    lp = np.asarray(line_ppms, dtype=float)
    if lp.size == 0 or m < 2:
        return np.array([abs(float(_trapz(seg[:, s], x))) for s in range(n_s)])

    # Normalise each line fraction to an (n_s,) vector, clipped to [0, 1].
    fr = []
    for f in line_fracs:
        fa = np.asarray(f, dtype=float)
        fa = np.full(n_s, float(fa)) if fa.ndim == 0 else fa
        fr.append(np.clip(fa, 0.0, 1.0))

    nearest = np.argmin(np.abs(x[:, None] - lp[None, :]), axis=1)   # (m,)
    weight  = np.empty((m, n_s), dtype=float)
    for k in range(m):
        weight[k, :] = fr[nearest[k]]

    weighted = seg * weight
    return np.array([abs(float(_trapz(weighted[:, s], x))) for s in range(n_s)])
