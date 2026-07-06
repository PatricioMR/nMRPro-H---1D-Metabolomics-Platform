# gui/stages/_quant_helpers.py
"""
Standalone helper functions used by the Quantification stage.

Extracted from quantification_stage.py to keep each file focused:
  - _quant_helpers.py  → pure functions + constants (this file)
  - _quant_visualization.py → _QuantVisMixin  (detail plots / tooltips)
  - _quant_computation.py   → _QuantCompMixin (_run, VIP, update_sample_area, export)
"""
import numpy as np

# Integration & lineshape DSP now lives in the processing/ layer (MVC).
# Re-exported here so existing callers (`from ._quant_helpers import _trapz, …`)
# keep working unchanged.
from processing.integration import (
    _trapz, _lorentz_shape, _lorentz_func, _lorentz_amplitude,
    _linear_baseline, _fit_lorentz_mean, _fit_lorentz_mean_with_bl,
    _fit_lorentz_sample, _fit_multi_lorentz, jr_pointwise_multiplet_area,
)


# ── Não-quantificados ────────────────────────────────────────────────────────
# Um metabolito é "não quantificado" quando não há sinal mensurável que lhe possa
# ser atribuído: ou perdeu a competição por todos os picos atribuídos (todos
# ficaram com metabolitos de maior score), ou a sua concentração média é 0/NaN
# em todas as amostras (integrou apenas ruído). Estes nunca entram em exports
# nem nas estatísticas — apenas aparecem a cinzento na lista, com a razão — para
# não fabricar dados (CLAUDE.md: nunca reportar ruído como sinal).
NQ_REASON_LOST_COMPETITION = (
    'Não quantificado: perdeu a competição de picos — todos os picos atribuídos '
    'foram quantificados para metabolitos de maior score.')
NQ_REASON_ZERO_NAN = (
    'Não quantificado: concentração 0/NaN em todas as amostras — sem sinal acima '
    'do limite de deteção.')


def is_unquantified_conc(conc_mm):
    """True when a metabolite's mean concentration carries no measurable signal.

    NaN (no finite positive sample anywhere) or a non-positive mean both mean
    nothing was quantified — such an analyte must be excluded from exports and
    statistics, never reported as a real value."""
    try:
        c = float(conc_mm)
    except (TypeError, ValueError):
        return True
    return (not np.isfinite(c)) or c <= 0.0


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
    # Preserve fractional per-signal nH (the BMRB DB stores e.g. 0.33 when a
    # proton is split across overlapping assignments; they sum to total_protons).
    # Truncating/flooring to 1.0 biases eff_nH by up to ~3× → wrong concentration.
    nH = float(hit.get('protons', 1) or 1)
    if nH <= 0:
        nH = 1.0
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


_SINGLET_MULT_TYPES = {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs', 'br. s'}

# Competition tier boundary between a multiplet and a singlet. A multiplet only
# outranks a singlet when its J quality s_dj exceeds this value — mirroring the
# "s_dj >= 0.5 = decent J" threshold used elsewhere (annotation post-processing).
SINGLET_TIER_PENALTY = 0.5


def multiplet_tier_penalty(is_singlet: bool, s_dj, singlet_penalty: float = SINGLET_TIER_PENALTY) -> float:
    """Continuous competition penalty replacing the old binary multiplet tier.

    Lower = better (sorts first). A J-confirmed multiplet (s_dj→1) approaches 0;
    a poor-J multiplet (s_dj→0) approaches 1 and falls BELOW a singlet, which sits
    at the fixed boundary ``singlet_penalty``. This removes the cliff where a
    near-perfect multiplet (e.g. s_dj=0.98) was demoted to the singlet tier purely
    because it fell short of the hard s_dj>=1.0 cut. s_dj is monotonic in |ΔJ|
    (1.0 below dj_gold, linear decay), so ordering multiplets by this penalty is
    consistent with the subsequent ΔJ tie-break.
    """
    if is_singlet:
        return float(singlet_penalty)
    try:
        sd = float(s_dj)
    except (TypeError, ValueError):
        sd = 0.0
    return 1.0 - min(max(sd, 0.0), 1.0)


def multiplet_sdj_missing(mult, has_sdj: bool, singlet_mults) -> bool:
    """True when a multiplet hit cannot be quantified because its s_dj is unknown.

    s_dj (J quality) is computed only in the Annotation Results post-processing.
    When it is absent for a multiplet — e.g. a project loaded before scores were
    computed, or the window never post-processed — the multiplet/singlet tier
    collapses silently and the multiplet-over-singlet priority is lost. Rather
    than quantify on a fabricated tier (CLAUDE.md: never silence data warnings),
    such a multiplet is blocked. Singlets need no J confirmation, so they are
    never blocked here.
    """
    m = str(mult or '').strip().lower()
    if m in singlet_mults:
        return False
    return not has_sdj


def _anchor_lines_complete(mp):
    """True if a matched peak may act as a quantification anchor on line count.

    A singlet needs no J pattern. A multiplet (doublet, triplet, sextet, …) is
    only trustworthy when EVERY predicted line was found — a partial multiplet
    has no confirmed J coupling, so it must not anchor quantification (the user
    rule: "não pode ser quantificado se não tem J bom com todas as linhas").
    Returns True when the line counts are unknown (no metadata to gate on).
    """
    mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
    if mult in _SINGLET_MULT_TYPES:
        return True
    nf = mp.get('_n_found_lines')
    ne = mp.get('_n_exp_lines')
    if nf is None or ne is None:
        return True                      # no line metadata → don't gate here
    return int(ne) > 0 and int(nf) >= int(ne)


def jr_analyte_fraction(rank, mid, contribs, contribs_ps, n_samples):
    """Fraction of a JG-deconvolved peak that belongs to one analyte.

    J-Resolve (JG-Deconv) splits an overlapping peak between the analytes that
    share it, storing ``{rank: {mid: frac}}`` (and an optional per-sample
    vector). When integrating a peak for a given analyte the quant pipeline must
    decide how much of that peak's area is really this analyte's:

      • peak ``rank`` not in ``contribs``/``contribs_ps`` → JG never deconvolved
        it; the analyte owns the peak → ``1.0``.
      • peak deconvolved and the analyte has a share → that share (the
        per-sample vector is preferred so group differences survive).
      • peak deconvolved but the analyte is ABSENT → ``0.0``. JG assigned the
        peak to other analytes; returning ``1.0`` here would quantify the same
        ppm twice (fabricated signal — see the integrity rules in CLAUDE.md).

    Returns a scalar in [0, 1] or an ``(n_samples,)`` array (per-sample case).
    """
    contribs    = contribs or {}
    contribs_ps = contribs_ps or {}
    if rank not in contribs_ps and rank not in contribs:
        return 1.0                       # not deconvolved → analyte owns it
    fa_ps = contribs_ps.get(rank, {}).get(mid)
    if fa_ps is not None and len(fa_ps) == n_samples:
        return np.clip(np.asarray(fa_ps, dtype=float), 0.0, 1.0)
    fs = contribs.get(rank, {}).get(mid)
    if fs is not None and 0.0 < float(fs) <= 1.0:
        return float(fs)
    return 0.0                           # deconvolved without this analyte


def _effective_nh_source(hit, detected_ppm, spectrometer_mhz=600.0):
    """Same logic as _effective_nh but returns (nH, source_description)."""
    import re as _re
    # Preserve fractional per-signal nH (the BMRB DB stores e.g. 0.33 when a
    # proton is split across overlapping assignments; they sum to total_protons).
    # Truncating/flooring to 1.0 biases eff_nH by up to ~3× → wrong concentration.
    nH = float(hit.get('protons', 1) or 1)
    if nH <= 0:
        nH = 1.0
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


_STD_PRESETS = {
    'TSP':         {'ppm': '0.0',  'nh': '9'},
    'DSS':         {'ppm': '0.0',  'nh': '9'},
    'TMSP':        {'ppm': '0.0',  'nh': '9'},
    'Maleic acid': {'ppm': '6.0',  'nh': '2'},
    'Formate':     {'ppm': '8.46', 'nh': '1'},
    'Custom':      {'ppm': '',     'nh': ''},
}
