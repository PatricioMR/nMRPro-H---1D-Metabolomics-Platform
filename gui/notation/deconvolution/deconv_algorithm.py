# gui/notation/deconvolution/deconv_algorithm.py
"""
NMR Deconvolution Algorithm.

Models available
----------------
Lorentzian      — standard for solution-state 1H NMR (T2 relaxation)
Gaussian        — useful for solid-state or inhomogeneity-broadened peaks
Pseudo-Voigt    — linear combination of Lorentzian + Gaussian (most flexible)
Onion (Auto-BIC)— NMR-Onion inspired (Sørensen et al., Heliyon 2024):
                  fits all three shape models + Skewed-Lorentzian, selects
                  the best by BIC, and estimates parameter uncertainty by
                  bootstrap resampling of the residuals (50 replicates).

Reference: Brinks Sørensen M. et al., "NMR-Onion - a transparent multi-model
based 1D NMR deconvolution algorithm", Heliyon 10(17) e36998 (2024).
DOI: 10.1016/j.heliyon.2024.e36998
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize, nnls
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def _trapz(y, x):
    """np.trapz was removed in NumPy 2.0 — use np.trapezoid when available."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x)
    return np.trapz(y, x=x)


# ── Peak shape functions ──────────────────────────────────────────────────────

def lorentzian(x, amp, cen, wid):
    """Pure Lorentzian — standard NMR solution-state line shape."""
    return amp * wid**2 / ((x - cen)**2 + wid**2)


def gaussian(x, amp, cen, wid):
    """Gaussian line shape."""
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))


def pseudo_voigt(x, amp, cen, wid, eta=0.5):
    """Pseudo-Voigt: eta × Lorentzian + (1-eta) × Gaussian."""
    return eta * lorentzian(x, amp, cen, wid) + \
           (1 - eta) * gaussian(x, amp, cen, wid)


def skewed_lorentzian(x, amp, cen, wid, gamma):
    """
    Skewed Lorentzian — NMR-Onion model for asymmetric line shapes.

    Asymmetry arises from magnetic field inhomogeneity, exchange effects,
    or residual dipolar couplings. gamma=0 reduces to a pure Lorentzian.
    gamma > 0: right skew; gamma < 0: left skew.
    gamma is bounded to [-π/2, π/2] as in the NMR-Onion paper.

    Implementation: Lorentzian with a linear phase-modulated amplitude
    (analogous to the imaginary part of a damped sinusoid in time domain).
    """
    lor = wid**2 / ((x - cen)**2 + wid**2)
    # Phase ramp across the peak — creates asymmetry
    skew = 1.0 + np.sin(gamma) * (x - cen) / (wid + 1e-10)
    return amp * lor * np.clip(skew, 0, None)


# ── BIC helper ────────────────────────────────────────────────────────────────

def _bic(rss, n_pts, n_params):
    """Bayesian Information Criterion (lower = better)."""
    if rss <= 0 or n_pts <= n_params:
        return np.inf
    return n_pts * np.log(rss / n_pts) + n_params * np.log(n_pts)


# ── Single-model fit ──────────────────────────────────────────────────────────

def _fit_model(x, y_norm, peaks_idx, model_name, min_width,
               noise_level, max_width=None, progress_callback=None):
    """
    Fit one model to the region.  Returns (opt_params, bic, r2, params_per_peak).
    """
    def _prog(msg, pct):
        if progress_callback:
            try: progress_callback(msg, pct)
            except Exception: pass

    if model_name == 'Lorentzian':
        fn         = lorentzian
        n_pp       = 3    # amp, cen, wid
    elif model_name == 'Gaussian':
        fn         = gaussian
        n_pp       = 3
    elif model_name == 'Pseudo-Voigt':
        fn         = pseudo_voigt
        n_pp       = 4    # amp, cen, wid, eta
    elif model_name == 'Skewed-Lorentzian':
        fn         = skewed_lorentzian
        n_pp       = 4    # amp, cen, wid, gamma
    else:
        fn         = lorentzian
        n_pp       = 3

    x_min   = float(min(x[0], x[-1]))
    x_max   = float(max(x[0], x[-1]))
    x_span  = max(x_max - x_min, min_width * 2)
    # Width upper bound: respect user-supplied max_width; fall back to a
    # conservative 20 % of region span (was 45 %) to prevent wide degenerate fits.
    if max_width is not None:
        wid_max = max(min_width, float(max_width))
    else:
        wid_max = max(min_width * 4, x_span * 0.20)

    bounds = []
    for idx in peaks_idx:
        amp_g = float(y_norm[idx])
        cen_g = float(x[idx])
        # Centre drift limited to ±10 % of region span, clipped to region edges
        drift = min(0.03, x_span * 0.10)
        bounds += [
            (max(noise_level * 0.5, amp_g * 0.1), amp_g * 3.0),
            (max(x_min, cen_g - drift), min(x_max, cen_g + drift)),
            (min_width, wid_max),
        ]
        if n_pp == 4:
            if model_name == 'Pseudo-Voigt':
                bounds += [(0.0, 1.0)]          # eta
            else:
                bounds += [(-np.pi/2, np.pi/2)] # gamma (NMR-Onion)

    def objective(p):
        y_fit = np.zeros_like(x, dtype=float)
        for i in range(len(peaks_idx)):
            b = i * n_pp
            if n_pp == 3:
                y_fit += fn(x, p[b], p[b+1], p[b+2])
            else:
                y_fit += fn(x, p[b], p[b+1], p[b+2], p[b+3])
        return float(np.sum((y_norm - y_fit)**2))

    _iter = [0]
    def _cb(xk, convergence=None):
        _iter[0] += 1
        pct = 30 + min(55, int(_iter[0] / 1500 * 55))
        _prog(f'{model_name} optimising… iter {_iter[0]}', pct)
        return False

    try:
        res = differential_evolution(
            objective, bounds, maxiter=1500,
            popsize=15, tol=1e-6, seed=42, callback=_cb)
        p_opt = res.x
    except Exception:
        p_opt = np.array([b[0] + (b[1]-b[0])*0.5 for b in bounds])

    y_fit = np.zeros_like(x, dtype=float)
    for i in range(len(peaks_idx)):
        b = i * n_pp
        if n_pp == 3:
            y_fit += fn(x, p_opt[b], p_opt[b+1], p_opt[b+2])
        else:
            y_fit += fn(x, p_opt[b], p_opt[b+1], p_opt[b+2], p_opt[b+3])

    rss = float(np.sum((y_norm - y_fit)**2))
    tss = float(np.sum((y_norm - np.mean(y_norm))**2))
    r2  = max(0.0, 1 - rss/tss) if tss > 0 else 0.0
    bic = _bic(rss, len(x), len(peaks_idx) * n_pp)

    return p_opt, bic, r2, n_pp, y_fit


# ── Bootstrap uncertainty (NMR-Onion step 5) ─────────────────────────────────

def _bootstrap_uncertainty(x, y_norm, p_opt, model_name, n_pp,
                            peaks_idx, min_width, n_boot=50):
    """
    Estimate parameter uncertainty by residual bootstrap.
    Resamples residuals 50× and refits with L-BFGS-B from p_opt.
    Returns std of each parameter across replicates.
    """
    if model_name == 'Lorentzian':   fn = lorentzian
    elif model_name == 'Gaussian':   fn = gaussian
    elif model_name == 'Pseudo-Voigt': fn = pseudo_voigt
    else:                             fn = skewed_lorentzian

    def _eval(p):
        y_fit = np.zeros_like(x, dtype=float)
        for i in range(len(peaks_idx)):
            b = i * n_pp
            args = tuple(p[b:b+n_pp])
            y_fit += fn(x, *args)
        return y_fit

    y_fitted  = _eval(p_opt)
    residuals = y_norm - y_fitted
    boot_params = []

    _x_min  = float(min(x[0], x[-1]))
    _x_max  = float(max(x[0], x[-1]))
    _x_span = max(_x_max - _x_min, min_width * 2)
    _wid_max_b = max(min_width * 4, _x_span * 0.45)

    bounds_list = []
    for idx in peaks_idx:
        _b = len(bounds_list)
        _drift_b = min(0.03, _x_span * 0.10)
        _cen_opt = float(p_opt[_b + 1]) if (_b + 1) < len(p_opt) else float(x[idx])
        bounds_list += [
            (max(1e-6, p_opt[_b] * 0.1), p_opt[_b] * 3 + 1e-6),
            (max(_x_min, _cen_opt - _drift_b),
             min(_x_max, _cen_opt + _drift_b)),
            (min_width, _wid_max_b),
        ]
        if n_pp == 4:
            if model_name == 'Pseudo-Voigt':
                bounds_list += [(0.0, 1.0)]
            else:
                bounds_list += [(-np.pi/2, np.pi/2)]

    for _ in range(n_boot):
        r_resample = residuals[
            np.random.randint(0, len(residuals), len(residuals))]
        y_boot = y_fitted + r_resample
        def obj_b(p):
            return float(np.sum((y_boot - _eval(p))**2))
        try:
            res = minimize(obj_b, p_opt, method='L-BFGS-B',
                           bounds=bounds_list,
                           options={'maxiter': 200, 'ftol': 1e-8})
            boot_params.append(res.x)
        except Exception:
            boot_params.append(p_opt.copy())

    if len(boot_params) < 2:
        return np.zeros_like(p_opt)
    return np.std(np.array(boot_params), axis=0)


# ── Baseline estimation ───────────────────────────────────────────────────────

def _estimate_linear_baseline(x, y, edge_frac=0.12, percentile=15):
    """
    Estimate a linear baseline by sampling the low-percentile of the edge regions.

    Works well when the region edges are free of sharp peaks (or the peaks there
    are the ones being fit and the broad background is what we want to remove).

    Parameters
    ----------
    x           : ppm axis (1-D array)
    y           : raw intensity (1-D array, same length)
    edge_frac   : fraction of points at each edge to consider (default 12 %)
    percentile  : percentile of edge values used as baseline anchor (default 15th)
                  Lower → more conservative (less subtraction).

    Returns
    -------
    baseline : 1-D array same shape as y
    """
    n      = len(y)
    n_edge = max(3, int(n * edge_frac))

    # Anchor values: robust low-end of each edge
    bl_left  = float(np.percentile(y[:n_edge],  percentile))
    bl_right = float(np.percentile(y[-n_edge:], percentile))
    x_left   = float(np.mean(x[:n_edge]))
    x_right  = float(np.mean(x[-n_edge:]))

    # Linear interpolation between anchors
    if abs(x_right - x_left) > 1e-9:
        slope    = (bl_right - bl_left) / (x_right - x_left)
        baseline = bl_left + slope * (x.astype(float) - x_left)
    else:
        baseline = np.full(n, (bl_left + bl_right) / 2.0)

    # Baseline cannot exceed the data (avoid subtracting more than present)
    baseline = np.minimum(baseline, y)
    return baseline


def _deconvolve_mcr_constrained(x, y, params, y_samples=None,
                                 progress_callback=None):
    """
    MCR-ALS constrained deconvolution.

    Builds theoretical Lorentzian profiles from each peak's J coupling and
    multiplicity, then uses MCR-ALS to find the best-fit concentrations.
    Enforces peak spacing and area ratios implicitly via fixed spectral shapes.
    """
    def _prog(msg, pct):
        if progress_callback:
            try: progress_callback(msg, pct)
            except Exception: pass

    _prog("MCR constrained: building profiles…", 10)

    try:
        from gui.utils.mcr_utils import mcr_annotation_region, simulate_lorentzian_profile
        from gui.widgets.annotation_results_window import _build_stick_positions
    except ImportError as e:
        print(f"   MCR constrained: import error — {e}. Falling back to Lorentzian.")
        return None

    noise_std = params.get('noise_std') or float(np.median(np.abs(y - np.median(y))) * 1.4826)

    # ── Baseline subtraction (linear, conservative) ───────────────────────────
    baseline = _estimate_linear_baseline(x, y)
    y_bsub   = np.maximum(y - baseline, 0.0)

    # ── Find candidate peak positions in baseline-subtracted data ─────────────
    from scipy.signal import find_peaks, gaussian_filter1d
    y_smooth = gaussian_filter1d(y_bsub / (y_bsub.max() or 1.0), sigma=max(1, len(x) * 0.03))
    noise_norm = noise_std / (y_bsub.max() or 1.0)
    min_snr    = params.get('min_snr', 2.5)
    pk_idx, _  = find_peaks(y_smooth, height=noise_norm * min_snr)

    if len(pk_idx) == 0:
        _prog("MCR constrained: no peaks found", 95)
        return {'peaks': [], 'bic': 0.0, 'r_squared': 0.0,
                'model': 'MCR (Constrained)', 'fit': np.zeros_like(y),
                'residual': y_bsub, 'baseline': baseline}

    # Clamp to max_peaks
    max_peaks = params.get('max_peaks', 8)
    if len(pk_idx) > max_peaks:
        heights   = y_bsub[pk_idx]
        pk_idx    = pk_idx[np.argsort(heights)[-max_peaks:]]

    _prog("MCR constrained: building Lorentzian profiles…", 25)

    # ── Build one theoretical profile per candidate peak ─────────────────────
    db_profiles = []
    db_entries  = []
    min_width = params.get('min_width', 0.003)

    for idx in pk_idx:
        cen_ppm = float(x[idx])
        # Estimate width from the peak shape (FWHM heuristic)
        half_h  = float(y_bsub[idx]) * 0.5
        left    = idx - 1
        while left > 0 and y_bsub[left] > half_h:
            left -= 1
        right   = idx + 1
        while right < len(y_bsub) - 1 and y_bsub[right] > half_h:
            right += 1
        lw_ppm = max(abs(float(x[right]) - float(x[left])) / 2.0, min_width)

        # Use a simple singlet Lorentzian profile (no multiplicity info at this stage)
        sticks  = [(cen_ppm, 1.0)]
        profile = simulate_lorentzian_profile(sticks, x, lw_ppm=lw_ppm)
        db_profiles.append(profile)
        db_entries.append({'ppm': cen_ppm, 'width_ppm': lw_ppm})

    if not db_profiles:
        return None

    _prog("MCR constrained: running MCR-ALS…", 40)

    # ── MCR-ALS ───────────────────────────────────────────────────────────────
    X = y_samples if (y_samples is not None and y_samples.ndim == 2) else y_bsub[:, np.newaxis]
    try:
        S, C, mcr_m, K, assignments, aligned, scores = mcr_annotation_region(
            X, db_profiles,
            r2_threshold=0.70,
            ppm_arr=x,
            shift_tol_ppm=0.02,
            use_baseline=False,
            db_entries=db_entries,
            field_mhz=600,
            max_k_override=len(db_profiles))
    except Exception as e:
        print(f"   MCR constrained: mcr_annotation_region failed — {e}")
        return None

    if S is None or C is None or K == 0:
        return None

    _prog("MCR constrained: building result peaks…", 85)

    y_max = float(y_bsub.max()) or 1.0
    result_peaks = []
    for k in range(K):
        sk     = S[:, k]
        ck_mean = float(C[:, k].mean()) if C.ndim == 2 else float(C[k])
        amp    = float(sk.max() * ck_mean * y_max)
        pk_pos = int(np.argmax(sk))
        cen    = float(x[pk_pos])
        wid    = float(db_entries[k]['width_ppm']) if k < len(db_entries) else min_width
        area   = float(np.trapz(sk * ck_mean * y_max, x))
        result_peaks.append({
            'ppm':         cen,
            'height':      amp,
            'width':       wid,
            'width_ppm':   wid,
            'seed_kind': 'top',
            'area':        abs(area),
            'snr':         amp / max(noise_std, 1e-10),
            'model':       'MCR (Constrained)',
            'deconvolved': True,
            'start_ppm':   cen - 3 * wid,
            'end_ppm':     cen + 3 * wid,
        })

    C_mean = C.mean(axis=1) if C.ndim == 2 else C
    fit_arr = np.dot(S, C_mean) * y_max
    residual_arr = y_bsub - fit_arr
    ss_res = float(np.sum(residual_arr ** 2))
    ss_tot = float(np.sum((y_bsub - y_bsub.mean()) ** 2))
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    _prog("MCR constrained: done", 100)
    print(f"   ✅ MCR constrained: {K} component(s)  R²={r2:.4f}")

    return {
        'peaks':     result_peaks,
        'bic':       None,
        'r_squared': r2,
        'model':     'MCR (Constrained)',
        'fit':       fit_arr,
        'residual':  residual_arr,
        'baseline':  baseline,
    }


def refit_with_selected_peaks(x, y, selected_ppms, model='Lorentzian',
                               min_width=0.002, noise_std=None):
    """
    Re-fit the NMR region y(x) using ONLY the peaks at the given PPM positions.

    Used after the user marks some sub-peaks as Poor: the surviving (confirmed)
    peaks are re-optimised so that their amplitudes/areas reflect the fitted
    spectrum with only those components — not the original N-component fit.

    Parameters
    ----------
    x             : ppm axis (1-D array, same as deconvolve_region was called with)
    y             : mean-spectrum intensities (1-D array)
    selected_ppms : list of float — center ppm of each confirmed (non-Poor) peak
    model         : line-shape model (Lorentzian / Gaussian / Pseudo-Voigt / Skewed-Lorentzian)
    min_width     : minimum half-width in ppm
    noise_std     : calibrated noise σ (from Step-1); used to set noise_level

    Returns
    -------
    (result_peaks, r2, bic) where result_peaks is a list of dicts with updated
    ppm / height / width / width_ppm / area keys, or (None, None, None) on failure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 5 or not selected_ppms:
        return None, None, None

    # Baseline / normalise (same pre-processing as deconvolve_region)
    baseline = _estimate_linear_baseline(x, y)
    y_bsub   = np.maximum(y - baseline, 0.0)
    y_max    = float(np.max(y_bsub))
    if y_max <= 0:
        y_max  = float(np.max(y)) or 1.0
        y_bsub = np.maximum(y - np.min(y), 0.0)
        y_max  = float(np.max(y_bsub)) or 1.0
    y_norm = y_bsub / y_max

    if noise_std and y_max > 0:
        noise_level = noise_std / y_max
    else:
        noise_level = float(np.median(np.abs(y_norm - np.median(y_norm)))) * 1.4826
    noise_level = max(noise_level, 1e-6)

    # Map each confirmed ppm to the nearest array index
    peaks_idx = np.array([int(np.argmin(np.abs(x - ppm))) for ppm in selected_ppms],
                         dtype=int)

    try:
        p_opt, bic, r2, n_pp, _y_fit = _fit_model(
            x, y_norm, peaks_idx, model, min_width, noise_level)
    except Exception as e:
        print(f"   refit_with_selected_peaks failed: {e}")
        return None, None, None

    if model == 'Lorentzian':    fn = lorentzian
    elif model == 'Gaussian':    fn = gaussian
    elif model == 'Pseudo-Voigt': fn = pseudo_voigt
    else:                         fn = skewed_lorentzian

    result_peaks = []
    for i in range(len(peaks_idx)):
        b   = i * n_pp
        amp = float(p_opt[b])   * y_max
        cen = float(p_opt[b+1])
        wid = float(p_opt[b+2])
        wid = max(wid, min_width)

        extra = {}
        if n_pp == 4:
            ev = float(p_opt[b+3])
            extra = {'eta': ev} if model == 'Pseudo-Voigt' else {'gamma': ev}

        # Compute individual area for this component
        args_norm = [float(p_opt[b]), cen, wid] + ([ev] if n_pp == 4 else [])
        y_pk_norm = fn(x, *args_norm)
        area = abs(float(_trapz(y_pk_norm * y_max, x)))

        result_peaks.append({
            'ppm':       cen,
            'height':    amp,
            'width':     wid,
            'width_ppm': wid,
            'area':      area,
            'seed_kind':   'top',
            **extra,
        })

    return result_peaks, r2, bic


def _estimate_whittaker_baseline(y, lam=1e5, p=0.01, niter=15):
    """
    Asymmetric least-squares Whittaker smoother (Eilers 2003).

    lam  : smoothness penalty — higher → flatter baseline (default 1e5)
    p    : asymmetry weight  — fraction treated as above-baseline (default 0.01)
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = len(y)
    if n < 5:
        return _estimate_linear_baseline(np.arange(n, dtype=float), y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format='csc')
    w = np.ones(n)
    z = y.copy().astype(float)
    for _ in range(niter):
        W = sparse.diags(w, 0, format='csc')
        Z = W + lam * D.T.dot(D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return np.minimum(z, y)   # baseline never above raw data


# ── Main entry point ──────────────────────────────────────────────────────────

def deconvolve_region(x, y, params, progress_callback=None, y_samples=None):
    """
    Deconvolve an NMR region.

    params keys
    -----------
    min_snr          : float  — SNR threshold for candidate peaks
    max_peaks        : int    — maximum peaks to fit
    model            : str    — 'Lorentzian' | 'Gaussian' | 'Pseudo-Voigt'
                                | 'Onion (Auto-BIC)'
    min_width        : float  — minimum peak width in ppm
    noise_std        : float  — calibrated noise (from Step 1)

    Returns
    -------
    dict with keys: peaks, bic, r_squared, model, uncertainty (Onion only)
    """
    def _prog(msg, pct):
        if progress_callback:
            try: progress_callback(msg, pct)
            except Exception: pass

    _prog("Initialising…", 5)
    print(f"\n🔧 DECONV — {len(x)} points  model={params.get('model','Lorentzian')}")

    if len(x) < 5:
        return {'peaks': [], 'bic': 0, 'r_squared': 0,
                'model': params.get('model', 'Lorentzian')}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return {
            'peaks': [],
            'bic': 0,
            'r_squared': 0,
            'model': params.get('model', 'Lorentzian'),
            'error': 'Input region contains non-finite values.',
        }

    if bool(params.get('require_baseline_edges', False)) \
            and not bool(params.get('baseline_edges_valid', True)):
        return {
            'peaks': [],
            'bic': 0,
            'r_squared': 0,
            'model': params.get('model', 'Lorentzian'),
            'error': 'Region rejected: baseline is not stable on both PPM edges.',
            'baseline_validation': params.get('baseline_validation', {}),
        }

    d = np.diff(x)
    if d.size > 0:
        step = float(np.median(np.abs(d)))
        if step <= 0:
            return {
                'peaks': [],
                'bic': 0,
                'r_squared': 0,
                'model': params.get('model', 'Lorentzian'),
                'error': 'Invalid PPM axis for deconvolution.',
            }
        max_gap = float(np.max(np.abs(d)))
        if max_gap > (3.0 * step):
            return {
                'peaks': [],
                'bic': 0,
                'r_squared': 0,
                'model': params.get('model', 'Lorentzian'),
                'error': 'Region is not a continuous PPM interval.',
            }

    min_snr              = params.get('min_snr',              2.5)
    max_peaks            = params.get('max_peaks',            8)
    model                = params.get('model',                'Lorentzian')
    min_width            = params.get('min_width',            0.001)
    max_width            = params.get('max_width',            None)
    noise_std            = params.get('noise_std',            None)
    detect_tops          = params.get('detect_tops',          True)
    detect_shoulders     = params.get('detect_shoulders',     True)
    shoulder_snr_factor  = params.get('shoulder_snr_factor',  0.25)
    n_boot               = params.get('n_boot',               50)
    onion_mode = (model == 'Onion (Auto-BIC)')
    mcr_mode   = (model == 'MCR (Constrained)')

    # ── MCR constrained fitting (early exit) ──────────────────────────────────
    if mcr_mode:
        return _deconvolve_mcr_constrained(
            x, y, params,
            y_samples=y_samples,
            progress_callback=progress_callback)

    # ── Baseline subtraction ───────────────────────────────────────────────────
    baseline_method = params.get('baseline_method', 'linear')
    if baseline_method == 'whittaker':
        _prog("Estimating baseline (Whittaker)…", 8)
        baseline = _estimate_whittaker_baseline(
            y,
            lam=params.get('whittaker_lam', 1e5),
            p=params.get('whittaker_p', 0.01))
        print(f"   Baseline (Whittaker): mean={baseline.mean():.4g}/{np.max(y):.4g}")
    else:
        baseline = _estimate_linear_baseline(x, y)
        print(f"   Baseline (linear): left={baseline[0]:.4g}  right={baseline[-1]:.4g}"
              f"  removed={baseline.mean():.4g}/{np.max(y):.4g}")
    y_bsub = np.maximum(y - baseline, 0.0)

    # ── Normalise ─────────────────────────────────────────────────────────────
    y_max = float(np.max(y_bsub))
    if y_max <= 0:                     # flat region after baseline removal
        y_max  = float(np.max(y)) or 1.0
        y_bsub = np.maximum(y - np.min(y), 0.0)
        y_max  = float(np.max(y_bsub)) or 1.0
    y_norm = y_bsub / y_max

    # Smooth sigma in points: ~10% of region width in points, min 1.
    # Keeps peaks/shoulders that are visually resolved from being merged.
    ppm_step      = abs(float(x[1]) - float(x[0])) if len(x) > 1 else 0.001
    min_width_pts = max(2, int(min_width / ppm_step))
    _sigma_main   = max(1.0, len(x) * 0.03)   # 3 % of region length
    y_smooth = gaussian_filter1d(y_norm, sigma=_sigma_main)

    if noise_std and y_max > 0:
        noise_level = noise_std / y_max
    else:
        noise_level = float(np.median(np.abs(y_norm - np.median(y_norm)))) * 1.4826
    noise_level = max(noise_level, 1e-6)

    # ── Candidate peak positions ──────────────────────────────────────────────

    _prog("Finding candidate peaks…", 15)

    # ── Regular peaks (local maxima) ──────────────────────────────────────────
    peaks_idx = np.array([], dtype=int)
    peak_seed_kind = {}
    if detect_tops:
        peaks_idx, _ = find_peaks(
            y_smooth,
            height=noise_level * min_snr,
            distance=min_width_pts,
            prominence=noise_level * 0.5,
        )
        peak_seed_kind.update({int(idx): 'top' for idx in peaks_idx})

    # ── Shoulder detection via second derivative ───────────────────────────────
    # Shoulders appear as inflection points (sign change / local minimum of d²y)
    # where the signal is still above the noise floor.
    # Use a slightly stronger smooth to reduce derivative noise.
    _sigma_d2 = max(1.5, min_width_pts * 0.5)
    y_smooth2 = gaussian_filter1d(y_norm, sigma=_sigma_d2)
    d2y = np.gradient(np.gradient(y_smooth2))
    d2y_smooth = gaussian_filter1d(d2y, sigma=_sigma_d2)

    # Local minima of d²y → most concave points (peaks and shoulders)
    shoulder_cands, _ = find_peaks(
        -d2y_smooth,
        distance=min_width_pts,
        prominence=0.0,
    )
    # Keep only shoulders: must be above noise*snr*shoulder_snr_factor and not
    # already near a detected peak.  Set detect_shoulders=False to skip entirely.
    if detect_shoulders and shoulder_snr_factor > 0:
        new_shoulders = [
            sc for sc in shoulder_cands
            if y_smooth[sc] > noise_level * min_snr * shoulder_snr_factor
            and not any(abs(sc - pi) < min_width_pts for pi in peaks_idx)
        ]
        if new_shoulders:
            new_shoulders = np.array(new_shoulders, dtype=int)
            peaks_idx = np.sort(np.concatenate([peaks_idx, new_shoulders]))
            peak_seed_kind.update({int(idx): 'shoulder' for idx in new_shoulders})
            print(f"   + {len(new_shoulders)} shoulder(s) detected")
    else:
        print("   shoulder detection disabled")

    if not detect_tops:
        print("   local-maxima detection disabled")

    # ── Edge-peak check ────────────────────────────────────────────────────────
    # scipy find_peaks() cannot return index 0 or len-1.  If a peak sits very
    # close to the region boundary (e.g. the region was trimmed to start right
    # at a peak) its y_smooth value may still be ≥ threshold; add it manually.
    if detect_tops:
        _edge_margin = max(1, min_width_pts)
        for ei in [0, len(y_norm) - 1]:
            if y_norm[ei] < noise_level * min_snr:
                continue
            # Must be a local high-point relative to points just inside the edge
            inner = min(ei + _edge_margin, len(y_norm) - 1) if ei == 0 \
                    else max(ei - _edge_margin, 0)
            if y_norm[ei] < y_norm[inner] * 0.80:
                continue                         # just a rising baseline tail
            if any(abs(ei - pi) < min_width_pts for pi in peaks_idx):
                continue                         # already covered
            peaks_idx = np.sort(np.concatenate([peaks_idx, [ei]]))
            peak_seed_kind[int(ei)] = 'top'
            print(f"   + Edge peak at idx={ei} ({x[ei]:.4f} ppm)")

    print(f"   Candidates: {len(peaks_idx)}")

    if len(peaks_idx) == 0:
        return {'peaks': [], 'bic': 0, 'r_squared': 0, 'model': model,
                'error': 'No candidate peaks found with the current seed settings.'}

    if len(peaks_idx) > max_peaks:
        heights   = y_smooth[peaks_idx]
        peaks_idx = peaks_idx[np.argsort(heights)[-max_peaks:]]
        peak_seed_kind = {int(idx): peak_seed_kind.get(int(idx), 'top')
                          for idx in peaks_idx}
        print(f"   Limited to {max_peaks}")

    # ── Fit ───────────────────────────────────────────────────────────────────
    if onion_mode:
        # NMR-Onion: fit all four models, select by BIC
        _prog("Onion — fitting Lorentzian…", 20)
        candidates = ['Lorentzian', 'Gaussian', 'Pseudo-Voigt', 'Skewed-Lorentzian']
        best_bic   = np.inf
        best_model = 'Lorentzian'
        best_popt  = None
        best_npp   = 3
        best_r2    = 0.0
        best_yfit  = None

        offsets = [20, 40, 60, 80]
        for mi, cand in enumerate(candidates):
            _prog(f"Onion — fitting {cand}…", offsets[mi])
            try:
                p_opt, bic, r2, n_pp, y_fit = _fit_model(
                    x, y_norm, peaks_idx, cand, min_width, noise_level,
                    max_width=max_width, progress_callback=progress_callback)
                print(f"   {cand}: BIC={bic:.1f}  R²={r2:.4f}")
                if bic < best_bic:
                    best_bic, best_model = bic, cand
                    best_popt, best_npp   = p_opt, n_pp
                    best_r2, best_yfit    = r2, y_fit
            except Exception as e:
                print(f"   {cand} failed: {e}")

        _prog("Onion — bootstrap uncertainty…", 85)
        try:
            uncertainty = _bootstrap_uncertainty(
                x, y_norm, best_popt, best_model, best_npp,
                peaks_idx, min_width, n_boot=n_boot)
        except Exception:
            uncertainty = np.zeros_like(best_popt)

        p_opt  = best_popt
        n_pp   = best_npp
        model  = best_model
        r2     = best_r2
        bic    = best_bic
        y_fit  = best_yfit
        print(f"   Onion selected: {model}  BIC={bic:.1f}  R²={r2:.4f}")

    else:
        # Single model as before
        _prog(f"Fitting {model}…", 20)
        p_opt, bic, r2, n_pp, y_fit = _fit_model(
            x, y_norm, peaks_idx, model, min_width, noise_level,
            max_width=max_width, progress_callback=progress_callback)
        uncertainty = None

    # ── Build peak list ───────────────────────────────────────────────────────
    _prog("Building results…", 95)

    if model == 'Lorentzian':   fn = lorentzian
    elif model == 'Gaussian':   fn = gaussian
    elif model == 'Pseudo-Voigt': fn = pseudo_voigt
    else:                        fn = skewed_lorentzian

    deconvolved = []
    for i, idx in enumerate(peaks_idx):
        b   = i * n_pp
        amp = float(p_opt[b])   * y_max
        cen = float(p_opt[b+1])
        wid = float(p_opt[b+2])
        if wid < min_width:
            wid = min_width

        # Extra params
        extra = {}
        if n_pp == 4:
            extra_val = float(p_opt[b+3])
            if model == 'Pseudo-Voigt':
                extra = {'eta': extra_val}
            else:
                extra = {'gamma': extra_val}   # skewness

        # Area under peak
        if model == 'Lorentzian':
            area = amp * wid * np.pi
        elif model == 'Gaussian':
            area = amp * wid * np.sqrt(2 * np.pi)
        else:
            area = float(abs(_trapz(
                    fn(x, amp/y_max, cen, wid,
                       *extra.values()) * y_max, x)))

        # Uncertainty for this peak (std of cen estimate from bootstrap)
        cen_std = float(uncertainty[b+1]) if uncertainty is not None else None

        peak_entry = {
            'ppm':         cen,
            'height':      amp,
            'width':       wid,
            'area':        area,
            'snr':         amp / (noise_level * y_max),
            'model':       model,
            'seed_kind':   peak_seed_kind.get(int(idx), 'top'),
            'deconvolved': True,
            'start_ppm':   cen - 3 * wid,
            'end_ppm':     cen + 3 * wid,
            **extra,
        }
        if cen_std is not None:
            peak_entry['cen_uncertainty'] = cen_std

        deconvolved.append(peak_entry)

    # ── Per-Sample Areas via NNLS ─────────────────────────────────────────────
    if y_samples is not None and deconvolved:
        _prog("Computing per-sample areas (NNLS)…", 98)
        n_samples = y_samples.shape[1]
        n_peaks = len(deconvolved)
        # Build design matrix A where each column is the shape of a sub-peak with amp=1
        A = np.zeros((len(x), n_peaks))
        for j, pk in enumerate(deconvolved):
            if model == 'Lorentzian':
                A[:, j] = fn(x, 1.0, pk['ppm'], pk['width'])
            elif model == 'Gaussian':
                A[:, j] = fn(x, 1.0, pk['ppm'], pk['width'])
            elif model == 'Pseudo-Voigt':
                A[:, j] = fn(x, 1.0, pk['ppm'], pk['width'], pk['eta'])
            else:
                A[:, j] = fn(x, 1.0, pk['ppm'], pk['width'], pk['gamma'])

        for j in range(n_peaks):
            deconvolved[j]['area_per_sample']   = np.zeros(n_samples)
            deconvolved[j]['height_per_sample']  = np.zeros(n_samples)

        for k in range(n_samples):
            # Subtract the same linear baseline from each sample column so that
            # NNLS fits the peak shapes rather than the baseline contribution.
            y_k      = y_samples[:, k]
            bl_k     = _estimate_linear_baseline(x, y_k)
            y_k_bsub = np.maximum(y_k - bl_k, 0.0)
            try:
                c, _ = nnls(A, y_k_bsub)
            except Exception:
                c = np.zeros(n_peaks)
            for j, pk in enumerate(deconvolved):
                amp_k = float(c[j])
                # Store real per-sample amplitude directly from NNLS
                deconvolved[j]['height_per_sample'][k] = amp_k
                # Compute area analytically for Lorentzian/Gaussian,
                # numerically for Pseudo-Voigt / Skewed-Lorentzian
                if model == 'Lorentzian':
                    area_k = amp_k * pk['width'] * np.pi
                elif model == 'Gaussian':
                    area_k = amp_k * pk['width'] * np.sqrt(2 * np.pi)
                else:
                    if model == 'Pseudo-Voigt':
                        y_sh = fn(x, amp_k, pk['ppm'], pk['width'],
                                  pk.get('eta', 0.5))
                    else:
                        y_sh = fn(x, amp_k, pk['ppm'], pk['width'],
                                  pk.get('gamma', 0.0))
                    area_k = float(abs(_trapz(y_sh, x)))
                deconvolved[j]['area_per_sample'][k] = area_k


    print(f"   ✅ {len(deconvolved)} peaks  model={model}")

    # Build fit and residual arrays in original (non-normalised) scale
    if model == 'Lorentzian':   _fn = lorentzian
    elif model == 'Gaussian':   _fn = gaussian
    elif model == 'Pseudo-Voigt': _fn = pseudo_voigt
    else:                        _fn = skewed_lorentzian

    fit_arr = np.zeros_like(y_bsub)
    for pk in deconvolved:
        amp = pk['height']
        cen = pk['ppm']
        wid = pk['width']
        if model == 'Pseudo-Voigt':
            fit_arr += _fn(x, amp, cen, wid, pk.get('eta', 0.5))
        elif model in ('Skewed-Lorentzian',):
            fit_arr += _fn(x, amp, cen, wid, pk.get('gamma', 0.0))
        else:
            fit_arr += _fn(x, amp, cen, wid)
    residual_arr = y_bsub - fit_arr   # in baseline-subtracted scale

    # Residual whiteness diagnostics for post-fit quality control.
    if len(residual_arr) > 2:
        r0 = residual_arr[:-1]
        r1 = residual_arr[1:]
        denom = float(np.sum(r0 * r0) * np.sum(r1 * r1))
        lag1_autocorr = float(np.sum(r0 * r1) / np.sqrt(denom)) if denom > 0 else 0.0
        dw_num = float(np.sum((r1 - r0) ** 2))
        dw_den = float(np.sum(residual_arr ** 2)) + 1e-12
        durbin_watson = dw_num / dw_den
    else:
        lag1_autocorr = 0.0
        durbin_watson = 2.0

    result = {
        'peaks':     deconvolved,
        'bic':       float(bic),
        'r_squared': float(r2),
        'model':     model,
        'fit':       fit_arr,
        'residual':  residual_arr,
        'baseline':  baseline,
        'residual_whiteness': {
            'lag1_autocorr': float(lag1_autocorr),
            'durbin_watson': float(durbin_watson),
            'is_white_like': bool(abs(lag1_autocorr) < 0.2 and 1.5 <= durbin_watson <= 2.5),
        },
    }
    if uncertainty is not None:
        result['bootstrap_uncertainty'] = uncertainty.tolist()

    return result