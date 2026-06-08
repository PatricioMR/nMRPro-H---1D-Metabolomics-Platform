# processing/spectral_quality.py
"""
Spectral quality assessment for 1H NMR metabolomics.

Measures FWHM and asymmetry of the reference signal (DSS/TSP/TMSP) at 0 ppm
on a per-sample basis.  All functions are pure (no GUI deps) and never raise —
failures are reported via a 'warning' key in the returned dict.

Scientific basis
----------------
The reference singlet at 0 ppm carries no J-coupling broadening, so its
linewidth (FWHM) directly reflects:
  - Magnetic field homogeneity (shimming quality)
  - Sample viscosity / T2 relaxation
  - Paramagnetic contamination

Thresholds (per Wishart 2022 / Emwas 2019 guidelines):
  When spectrometer MHz is known → grade in Hz:
    FWHM < 2 Hz  → good
    FWHM < 4 Hz  → ok
    FWHM ≥ 4 Hz  → poor
  When MHz unknown → grade in ppm (field-agnostic fallback):
    FWHM < 0.005 ppm → good  (≈ 3 Hz @ 600 MHz)
    FWHM < 0.010 ppm → ok   (≈ 6 Hz @ 600 MHz)
    FWHM ≥ 0.010 ppm → poor

Asymmetry reflects phase distortion or gradient artefacts.
"""
import numpy as np

# ── Quality thresholds ────────────────────────────────────────────────────────
# When spectrometer MHz is known, grade in Hz (field-independent).
# Fallback ppm thresholds when MHz is not available.
FWHM_THRESHOLDS_HZ  = {'good': 2.0,   'ok': 4.0}    # Hz  — primary
FWHM_THRESHOLDS_PPM = {'good': 0.005, 'ok': 0.010}  # ppm — fallback
ASYMMETRY_THRESHOLDS = {'good': 0.050, 'ok': 0.150}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Linear interpolation between two hex colours."""
    t = max(0.0, min(1.0, float(t)))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def grade_color(value, thresholds: dict) -> str:
    """
    Return a hex colour on a green→yellow→red gradient.

    green  (#c8e6c9) → value ≤ thresholds['good']
    yellow (#fff9c4) → value = thresholds['ok']
    red    (#ffcdd2) → value > thresholds['ok']
    grey   (#f0f0f0) → value is None / NaN
    """
    if value is None:
        return '#f0f0f0'
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return '#f0f0f0'
    if np.isnan(v):
        return '#f0f0f0'
    good = thresholds['good']
    ok   = thresholds['ok']
    if v <= good:
        t = v / good if good > 0 else 0.0
        return _lerp_color('#c8e6c9', '#fff9c4', t)
    elif v <= ok:
        t = (v - good) / (ok - good) if ok > good else 1.0
        return _lerp_color('#fff9c4', '#ffcdd2', t)
    else:
        return '#ffcdd2'


def grade_metric(value, thresholds: dict) -> str:
    """Return 'good'/'ok'/'poor'/'unknown'."""
    if value is None:
        return 'unknown'
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return 'unknown'
    if np.isnan(v):
        return 'unknown'
    if v <= thresholds['good']:
        return 'good'
    if v <= thresholds['ok']:
        return 'ok'
    return 'poor'


def _find_reference_apex(ppm: np.ndarray, intensity_1d: np.ndarray,
                          ref_ppm: float, window_ppm: float):
    """
    Locate the intensity maximum inside [ref_ppm ± window_ppm].

    Returns (apex_idx, apex_ppm) or (None, None) if region is empty.
    """
    mask = (ppm >= ref_ppm - window_ppm) & (ppm <= ref_ppm + window_ppm)
    if not np.any(mask):
        return None, None
    win_indices = np.where(mask)[0]
    local_max   = int(np.argmax(intensity_1d[mask]))
    apex_idx    = int(win_indices[local_max])
    apex_ppm    = float(ppm[apex_idx])
    return apex_idx, apex_ppm


# ── Core measurement functions ────────────────────────────────────────────────

def compute_fwhm_ppm(ppm: np.ndarray, intensity_1d: np.ndarray,
                     apex_idx: int, window_ppm: float = 0.15,
                     spectrometer_mhz: float = None) -> dict:
    """
    Measure FWHM of the peak at apex_idx within a local window.

    Reuses the exact half-max crossing idiom from peak_detection.py:456-458
    after applying a linear baseline correction between window endpoints.

    Parameters
    ----------
    ppm              : 1-D chemical shift axis (decreasing OK)
    intensity_1d     : 1-D intensity array (same length)
    apex_idx         : index of the peak apex in ppm/intensity_1d
    window_ppm       : half-width of the analysis window (ppm)
    spectrometer_mhz : if supplied, also computes fwhm_hz

    Returns
    -------
    dict with keys: fwhm_ppm, fwhm_hz, apex_ppm, left_ppm, right_ppm, warning
    All values may be None on failure; 'warning' is None on success.
    """
    result = {
        'fwhm_ppm': None, 'fwhm_hz': None,
        'apex_ppm': None, 'left_ppm': None, 'right_ppm': None,
        'warning': None,
    }
    try:
        n = len(ppm)
        if apex_idx < 0 or apex_idx >= n:
            result['warning'] = f'apex_idx {apex_idx} out of bounds'
            return result

        apex_ppm_val = float(ppm[apex_idx])
        result['apex_ppm'] = apex_ppm_val

        # Extract window (ppm may be decreasing — sort by ppm for indexing)
        lo = apex_ppm_val - window_ppm
        hi = apex_ppm_val + window_ppm
        mask = (ppm >= lo) & (ppm <= hi)
        if mask.sum() < 3:
            result['warning'] = 'insufficient resolution in window'
            return result

        ppm_region = ppm[mask].copy()
        y_region   = intensity_1d[mask].copy()

        # Sort ascending in ppm for area / interpolation consistency
        order      = np.argsort(ppm_region)
        ppm_region = ppm_region[order]
        y_region   = y_region[order]

        # Linear baseline between window endpoints (same as peak_detection.py:446)
        if len(ppm_region) >= 2:
            slope   = (y_region[-1] - y_region[0]) / (ppm_region[-1] - ppm_region[0] + 1e-15)
            baseline = y_region[0] + slope * (ppm_region - ppm_region[0])
            y_region = y_region - baseline

        apex_val = float(np.max(y_region))
        if apex_val <= 0:
            result['warning'] = 'peak apex ≤ 0 after baseline correction'
            return result

        half_max = apex_val / 2.0

        # Half-max crossings — exact idiom from peak_detection.py:456
        above = np.where(y_region >= half_max)[0]
        if len(above) < 2:
            result['warning'] = 'half-maximum not crossed on both sides'
            return result

        left_ppm_val  = float(ppm_region[above[0]])
        right_ppm_val = float(ppm_region[above[-1]])
        fwhm          = abs(right_ppm_val - left_ppm_val)
        fwhm          = max(fwhm, 1e-6)   # guard against zero

        result['fwhm_ppm']  = fwhm
        result['left_ppm']  = left_ppm_val
        result['right_ppm'] = right_ppm_val
        if spectrometer_mhz and spectrometer_mhz > 0:
            result['fwhm_hz'] = fwhm * float(spectrometer_mhz)

    except Exception as exc:
        result['warning'] = f'fwhm computation error: {exc}'

    return result


def compute_asymmetry(ppm: np.ndarray, intensity_1d: np.ndarray,
                      apex_idx: int, window_ppm: float = 0.15) -> dict:
    """
    Measure peak asymmetry as (area_right − area_left) / area_total.

    A perfect Lorentzian/Gaussian = 0.  Positive = trailing edge on high-ppm
    side; negative = trailing edge on low-ppm side.

    Parameters
    ----------
    ppm          : 1-D chemical shift axis
    intensity_1d : 1-D intensity array
    apex_idx     : index of the peak apex
    window_ppm   : half-width of the analysis window (ppm)

    Returns
    -------
    dict with keys: asymmetry, area_left, area_right, warning
    """
    result = {
        'asymmetry': None, 'area_left': None,
        'area_right': None, 'warning': None,
    }
    try:
        n = len(ppm)
        if apex_idx < 0 or apex_idx >= n:
            result['warning'] = f'apex_idx {apex_idx} out of bounds'
            return result

        apex_ppm_val = float(ppm[apex_idx])
        lo = apex_ppm_val - window_ppm
        hi = apex_ppm_val + window_ppm
        mask = (ppm >= lo) & (ppm <= hi)
        if mask.sum() < 3:
            result['warning'] = 'insufficient resolution in window'
            return result

        ppm_region = ppm[mask].copy()
        y_region   = intensity_1d[mask].copy()

        order      = np.argsort(ppm_region)
        ppm_region = ppm_region[order]
        y_region   = y_region[order]

        # Linear baseline correction
        if len(ppm_region) >= 2:
            slope    = (y_region[-1] - y_region[0]) / (ppm_region[-1] - ppm_region[0] + 1e-15)
            baseline = y_region[0] + slope * (ppm_region - ppm_region[0])
            y_region = y_region - baseline

        y_region = np.maximum(y_region, 0.0)

        # Split at apex ppm
        left_mask  = ppm_region <= apex_ppm_val
        right_mask = ppm_region >= apex_ppm_val

        if left_mask.sum() < 2 or right_mask.sum() < 2:
            result['warning'] = 'apex too close to window edge for asymmetry'
            return result

        area_left  = float(abs(np.trapezoid(y_region[left_mask],  ppm_region[left_mask])))
        area_right = float(abs(np.trapezoid(y_region[right_mask], ppm_region[right_mask])))
        total      = area_left + area_right

        if total <= 0:
            result['warning'] = 'total area ≤ 0 after baseline correction'
            return result

        result['asymmetry']  = (area_right - area_left) / total
        result['area_left']  = area_left
        result['area_right'] = area_right

    except Exception as exc:
        result['warning'] = f'asymmetry computation error: {exc}'

    return result


# ── Sample-level orchestration ────────────────────────────────────────────────

def assess_sample_quality(ppm: np.ndarray, intensity_1d: np.ndarray,
                           sample_name: str, sample_group: str,
                           ref_ppm: float = 0.0, window_ppm: float = 0.15,
                           spectrometer_mhz: float = None) -> dict:
    """
    Compute FWHM and asymmetry of the IS/reference signal for one sample.

    Parameters
    ----------
    ppm              : 1-D chemical shift axis
    intensity_1d     : 1-D intensity array for this sample
    sample_name      : sample identifier (for display)
    sample_group     : group label (for display)
    ref_ppm          : expected position of the reference signal
    window_ppm       : search half-width around ref_ppm
    spectrometer_mhz : if known, enables Hz display

    Returns
    -------
    dict with quality metrics and grades.
    """
    result = {
        'sample_name':     sample_name,
        'sample_group':    sample_group,
        'has_reference':   False,
        'apex_ppm':        None,
        'fwhm_ppm':        None,
        'fwhm_hz':         None,
        'fwhm_grade':      'unknown',
        'asymmetry':       None,
        'asymmetry_grade': 'unknown',
        'warnings':        [],
    }

    try:
        ppm_arr = np.asarray(ppm, dtype=float)
        y_arr   = np.asarray(intensity_1d, dtype=float)

        if y_arr.ndim != 1 or len(y_arr) != len(ppm_arr):
            result['warnings'].append('intensity array shape mismatch')
            return result

        apex_idx, apex_ppm_val = _find_reference_apex(ppm_arr, y_arr, ref_ppm, window_ppm)
        if apex_idx is None:
            result['warnings'].append(
                f'no data in [{ref_ppm - window_ppm:.3f}, {ref_ppm + window_ppm:.3f}] ppm')
            return result

        result['has_reference'] = True
        result['apex_ppm']      = apex_ppm_val

        # FWHM
        fwhm_res = compute_fwhm_ppm(ppm_arr, y_arr, apex_idx,
                                     window_ppm=window_ppm,
                                     spectrometer_mhz=spectrometer_mhz)
        result['fwhm_ppm'] = fwhm_res['fwhm_ppm']
        result['fwhm_hz']  = fwhm_res['fwhm_hz']
        # Grade in Hz when spectrometer frequency is known (field-independent);
        # fall back to ppm thresholds when MHz is unavailable.
        if fwhm_res['fwhm_hz'] is not None:
            result['fwhm_grade'] = grade_metric(fwhm_res['fwhm_hz'], FWHM_THRESHOLDS_HZ)
        else:
            result['fwhm_grade'] = grade_metric(fwhm_res['fwhm_ppm'], FWHM_THRESHOLDS_PPM)
        if fwhm_res['warning']:
            result['warnings'].append(f'FWHM: {fwhm_res["warning"]}')

        # Asymmetry
        asym_res = compute_asymmetry(ppm_arr, y_arr, apex_idx, window_ppm=window_ppm)
        result['asymmetry']       = asym_res['asymmetry']
        result['asymmetry_grade'] = grade_metric(asym_res['asymmetry'], ASYMMETRY_THRESHOLDS)
        if asym_res['warning']:
            result['warnings'].append(f'Asymmetry: {asym_res["warning"]}')

    except Exception as exc:
        result['warnings'].append(f'assess_sample_quality error: {exc}')

    return result


def assess_all_samples(ppm: np.ndarray, intensities_2d: np.ndarray,
                        samples_info: list,
                        ref_ppm: float = 0.0, window_ppm: float = 0.15,
                        spectrometer_mhz: float = None) -> list:
    """
    Run quality assessment for every sample in the dataset.

    Parameters
    ----------
    ppm            : 1-D chemical shift axis (n_pts,)
    intensities_2d : 2-D intensity matrix (n_pts, n_samples); 1-D also accepted
    samples_info   : list of sample metadata dicts (keys: 'name', 'label')
    ref_ppm        : expected IS/reference position (default 0.0)
    window_ppm     : search window half-width (default 0.15 ppm)
    spectrometer_mhz : if known, enables Hz output

    Returns
    -------
    list of dicts — one per sample, in the same order as samples_info.
    Each dict has the structure returned by assess_sample_quality().
    """
    ppm_arr = np.asarray(ppm, dtype=float)

    intensities = np.asarray(intensities_2d, dtype=float)
    if intensities.ndim == 1:
        intensities = intensities.reshape(-1, 1)

    n_pts, n_samples = intensities.shape
    results = []

    for s_idx in range(n_samples):
        if s_idx < len(samples_info):
            meta        = samples_info[s_idx]
            sample_name = str(meta.get('name', f'Sample {s_idx + 1}'))
            sample_group = str(meta.get('label', ''))
        else:
            sample_name  = f'Sample {s_idx + 1}'
            sample_group = ''

        y_1d = intensities[:, s_idx]
        res  = assess_sample_quality(
            ppm_arr, y_1d,
            sample_name=sample_name,
            sample_group=sample_group,
            ref_ppm=ref_ppm,
            window_ppm=window_ppm,
            spectrometer_mhz=spectrometer_mhz,
        )
        results.append(res)

    return results
