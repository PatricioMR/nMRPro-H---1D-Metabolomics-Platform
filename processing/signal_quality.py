# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Per-signal quality classification for 1D NMR peak detection.

A detected signal is graded Good / Average / Poor from properties that decide
whether its integral can be trusted:

  * lineshape fidelity   — does it fit a Lorentzian/Gaussian at all
  * symmetry             — even/odd decomposition about the apex; the odd part
                           separates a dispersive (phase) residual from a
                           one-sided shoulder
  * baseline horizontality — foot-to-foot tilt under the whole cluster
  * baseline grounding   — feet at the noise level, not riding a broad hump
  * hidden shoulders     — second-derivative minima inside the signal bounds
                           that were not resolved as their own apex
  * linewidth anomaly    — FWHM against the spectrum-wide median

Two hard gates short-circuit to Poor before any score is computed, because a
signal that fails them cannot be quantified regardless of how well shaped it is:

  * SNR below GATE_MIN_SNR       — at or under the detection limit
  * fewer than GATE_MIN_PTS_FWHM points across the FWHM — the integral and the
    lineshape fit are both dominated by digitisation error

Cluster membership is deliberately NOT penalised. A cleanly resolved multiplet
line is a high-quality signal; the baseline metrics are therefore measured on
the cluster's outer feet (the real baseline) rather than on the shared valley,
and overlap is detected explicitly by the shoulder term instead.

References
----------
Malz, F. & Jancke, H., "Validation of quantitative NMR",
    J. Pharm. Biomed. Anal. 38 (2005) 813–823.  — digital resolution and SNR
    requirements for quantitative integration.
Currie, L. A., "Nomenclature in evaluation of analytical methods including
    detection and quantification capabilities", Pure Appl. Chem. 67 (1995)
    1699–1723.  — SNR = 3 as the limit of detection.
Savitzky, A. & Golay, M. J. E., "Smoothing and differentiation of data by
    simplified least squares procedures", Anal. Chem. 36 (1964) 1627–1639.
Vivó-Truyols, G. et al., "Automatic program for peak detection and
    deconvolution of multi-overlapped chromatographic signals", J. Chromatogr. A
    1096 (2005) 133–145.  — second-derivative detection of unresolved shoulders.
"""

import numpy as np

# ── Tiers ─────────────────────────────────────────────────────────────────────
# Values match the manual status labels in the peak list so an automatic grade
# and a user-assigned one share a single vocabulary and colour map.
SIGNAL_GOOD    = 'Good'
SIGNAL_AVERAGE = 'Average'
SIGNAL_POOR    = 'Poor'

SIGNAL_TIERS = (SIGNAL_GOOD, SIGNAL_AVERAGE, SIGNAL_POOR)

# ── Hard gates ────────────────────────────────────────────────────────────────
GATE_MIN_SNR      = 3.0   # IUPAC limit of detection (Currie 1995)
GATE_MIN_PTS_FWHM = 4.0   # below this the integral is digitisation-limited

# ── Score weights (sum = 1) ───────────────────────────────────────────────────
# Lineshape and symmetry answer "is this one clean signal"; the baseline pair
# answers "is its integral resting on nothing"; the shoulder term answers "is
# something else hiding inside it".
W_SHAPE       = 0.30
W_SYM         = 0.20
W_BASE_TILT   = 0.12
W_BASE_GROUND = 0.08
W_SHOULDER    = 0.20
W_WIDTH       = 0.10

_WEIGHTS = {
    'shape':       W_SHAPE,
    'sym':         W_SYM,
    'base_tilt':   W_BASE_TILT,
    'base_ground': W_BASE_GROUND,
    'shoulder':    W_SHOULDER,
    'width':       W_WIDTH,
}

# ── Tier thresholds on the composite score ────────────────────────────────────
THRESH_GOOD    = 0.80
THRESH_AVERAGE = 0.50

# Good means nothing is badly wrong, not merely that the average is high. A
# single failed component (a line three times too broad, a foot far off
# baseline) caps the grade at Average even when the weighted score clears
# THRESH_GOOD, so one severe defect cannot be averaged away by five clean
# components.
COMPONENT_GOOD_FLOOR = 0.35

# ── Linewidth anomaly band (ratio to the spectrum median FWHM) ────────────────
WIDTH_RATIO_OK  = 1.0   # at or below the median → no penalty
WIDTH_RATIO_BAD = 4.0   # score reaches 0 here

# Centroid of the odd (antisymmetric) energy, in half-widths from the apex,
# above which the asymmetry is read as a dispersive phase residual rather than
# a neighbouring component. Measured separation on synthetic lines: shoulders
# land at 1.3–1.8 half-widths, phase errors at 3.3–3.4.
PHASE_ODD_CENTROID_HALFWIDTHS = 2.5

# Second-derivative minima shallower than this fraction of the apex's curvature
# are noise ripples, not unresolved components.
SHOULDER_MIN_DEPTH = 0.15


def _clip01(v):
    return float(np.clip(v, 0.0, 1.0))


def corrected_profile(intensity, start_idx, end_idx, left_idx, right_idx):
    """Region intensities minus the linear baseline joining the two feet.

    Mirrors the baseline used by the integration step, so the quality metrics
    and the reported area describe the same object.
    """
    seg = np.asarray(intensity[start_idx:end_idx + 1], dtype=float)
    n = seg.size
    if n < 2:
        return seg.copy()
    lv = float(intensity[left_idx])
    rv = float(intensity[right_idx])
    if right_idx != left_idx:
        # Baseline sampled at the same positions as the segment.
        pos = np.arange(start_idx, end_idx + 1, dtype=float)
        t = (pos - left_idx) / (right_idx - left_idx)
        base = lv + t * (rv - lv)
    else:
        base = np.full(n, lv, dtype=float)
    return seg - base


def _symmetry_from_profile(y_corr, apex_pos, fwhm_pts):
    """Even/odd decomposition of the signal about its apex.

    Returns (s_sym, asym_frac, asym_kind).

    The profile is split into y_even = (y(+d) + y(-d)) / 2 and
    y_odd = (y(+d) - y(-d)) / 2 over the overlapping half-width.  A perfectly
    symmetric line has no odd part.  ``asym_kind`` names the likely cause:
    'phase' for a dispersive residual (correctable, not a sign of overlap),
    'shoulder' for a neighbouring component, 'none' for a symmetric line.
    """
    n = y_corr.size
    d_max = int(min(apex_pos, n - 1 - apex_pos))
    if d_max < 2:
        return 1.0, 0.0, 'none'

    d = np.arange(1, d_max + 1)
    y_plus  = y_corr[apex_pos + d]
    y_minus = y_corr[apex_pos - d]
    y_even = 0.5 * (y_plus + y_minus)
    y_odd  = 0.5 * (y_plus - y_minus)

    even_sum = float(np.sum(np.abs(y_even)))
    if even_sum <= 1e-30:
        return 1.0, 0.0, 'none'
    asym_frac = float(np.sum(np.abs(y_odd)) / even_sum)

    # What separates a correctable phase error from a real neighbouring
    # component is how far out the odd energy reaches.  The dispersion
    # lineshape falls off as 1/Δ, so a phase residual spreads its odd part far
    # into the wings; a shoulder is another absorption line falling off as
    # 1/Δ², so its odd part stays localised near the neighbour's offset.  The
    # centroid of the odd energy therefore lands several half-widths out for a
    # phase error and one or two for a shoulder.  Amplitude cannot tell them
    # apart: a moderate phase error barely dips below baseline, because
    # absorption still dominates where the two overlap.
    if asym_frac < 0.05:
        kind = 'none'
    else:
        odd_abs = np.abs(y_odd)
        odd_sum = float(odd_abs.sum())
        half_width = max(float(fwhm_pts) / 2.0, 1.0)
        centroid = (float((d * odd_abs).sum() / odd_sum)
                    if odd_sum > 1e-30 else 0.0)
        kind = ('phase'
                if centroid > PHASE_ODD_CENTROID_HALFWIDTHS * half_width
                else 'shoulder')

    return _clip01(1.0 - asym_frac), asym_frac, kind


def _shoulder_score(y_corr, apex_pos, fwhm_pts):
    """Detect unresolved shoulders from second-derivative minima.

    Every peak-like feature produces a local minimum of the second derivative.
    The apex owns the deepest one; any other local minimum inside the signal
    bounds is a component that was not resolved into its own apex.  The score
    is 1 minus the depth of the strongest such minimum relative to the apex's.

    The smoothing window is scaled to the linewidth — a fixed one leaves noise
    ripples on a broad line, and each ripple would be counted as a shoulder.
    A minimum must also reach SHOULDER_MIN_DEPTH of the apex's curvature to
    count, so residual noise does not manufacture components.

    Returns (s_shoulder, n_shoulders) or (None, 0) when the region is too short
    to differentiate reliably — the caller then drops the component and
    renormalises the remaining weights.
    """
    n = y_corr.size
    if n < 9:
        return None, 0

    # Savitzky-Golay smoothing before differentiation; window odd, larger than
    # the polynomial order, and a fraction of the linewidth — a window as wide
    # as the FWHM smooths the shoulder away along with the noise.
    win = int(max(5, min((fwhm_pts or 5) / 3.0, n // 3)))
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    y_s = y_corr
    if win >= 5:
        try:
            from scipy.signal import savgol_filter
            y_s = savgol_filter(y_corr, win, 3)
        except Exception:
            y_s = y_corr

    d2 = np.gradient(np.gradient(y_s))

    apex_d2 = float(np.min(d2))
    if apex_d2 >= 0:
        return 1.0, 0

    # Local minima of d2, excluding the two edge samples where the gradient
    # stencil is one-sided, and the apex's own minimum.
    min_sep = max(2, int(round((fwhm_pts or 2) / 2)))
    shoulders = []
    for i in range(2, n - 2):
        if not (d2[i] < d2[i - 1] and d2[i] <= d2[i + 1] and d2[i] < 0):
            continue
        if abs(i - apex_pos) < min_sep:
            continue
        depth = float(d2[i]) / apex_d2      # both negative → positive ratio
        if depth >= SHOULDER_MIN_DEPTH:
            shoulders.append(depth)

    if not shoulders:
        return 1.0, 0
    return _clip01(1.0 - max(shoulders)), len(shoulders)


def _width_score(fwhm_pts, median_fwhm_pts):
    """Penalise lines that are anomalously broad for this spectrum.

    A line narrower than the median is not penalised — good local shim.  Broad
    lines signal poor local shim, chemical exchange, or two fused signals.
    """
    if not median_fwhm_pts or median_fwhm_pts <= 0 or not fwhm_pts:
        return None
    ratio = float(fwhm_pts) / float(median_fwhm_pts)
    if ratio <= WIDTH_RATIO_OK:
        return 1.0
    if ratio >= WIDTH_RATIO_BAD:
        return 0.0
    return _clip01(1.0 - (ratio - WIDTH_RATIO_OK)
                   / (WIDTH_RATIO_BAD - WIDTH_RATIO_OK))


def _baseline_scores(intensity, left_idx, right_idx,
                     cluster_lo_idx, cluster_hi_idx,
                     height, noise_std, baseline_level):
    """Horizontality and grounding of this signal's integration baseline.

    The two halves answer different questions and are therefore measured on
    different points:

    * Tilt uses the CLUSTER's outer feet, because that is where the spectrum's
      real baseline is readable.  For a multiplet line the inner foot is a
      neighbour's flank, not the baseline; measuring tilt there would condemn
      every well-resolved multiplet.
    * Grounding uses this signal's OWN feet, because those are what the
      integral actually rests on.  A signal riding a broad macromolecule hump
      has feet far above the noise even though the cluster eventually returns
      to baseline — its area is inflated and that must show.  For a multiplet
      line the shared valley is its own foot, so a deep valley (well resolved)
      barely penalises while a shallow one (fused lines) penalises hard.
    """
    amp = max(float(height), 1e-12)

    s_tilt = _clip01(1.0 - abs(float(intensity[cluster_lo_idx])
                               - float(intensity[cluster_hi_idx])) / amp)

    # Continuous rather than a boolean, so a signal on a slight residual hump
    # degrades instead of failing outright.
    thr = float(baseline_level) + 3.0 * float(noise_std)
    excess = [max(0.0, float(intensity[i]) - thr) / amp
              for i in (left_idx, right_idx)]
    s_ground = _clip01(1.0 - float(np.mean(excess)))

    return s_tilt, s_ground


def measure_fwhm_points(y_corr):
    """Number of samples at or above half the corrected apex height."""
    if y_corr.size == 0:
        return 0.0
    y_max = float(y_corr.max())
    if y_max <= 0:
        return 0.0
    return float(np.count_nonzero(y_corr >= 0.5 * y_max))


def classify_signal_quality(intensity, *, start_idx, end_idx,
                            left_idx, right_idx,
                            cluster_lo_idx, cluster_hi_idx,
                            height, shape_r2, noise_std,
                            baseline_level=0.0,
                            median_fwhm_pts=None):
    """Grade one detected signal.

    Parameters
    ----------
    intensity        : ndarray — full (baseline-corrected) reference spectrum
    start_idx/end_idx: integration bounds of this signal
    left_idx/right_idx: its two feet (a foot may be a shared valley)
    cluster_lo_idx/cluster_hi_idx : outer feet of the cluster it belongs to
    height           : apex height above its local baseline
    shape_r2         : R² of the best Lorentzian/Gaussian fit
    noise_std        : noise σ of the spectrum
    baseline_level   : offset of the quiet baseline (0 for a mean spectrum)
    median_fwhm_pts  : spectrum-wide median FWHM in points; None drops the
                       linewidth component and renormalises the weights

    Returns
    -------
    (tier, score, detail) where tier ∈ SIGNAL_TIERS, score ∈ [0, 1] (0.0 when a
    hard gate fired) and detail is a dict of every component and gate value, so
    the grade can be explained to the analyst rather than asserted.
    """
    noise_std = float(noise_std) if noise_std and noise_std > 0 else 0.0
    height = float(height)

    y_corr = corrected_profile(intensity, start_idx, end_idx,
                                left_idx, right_idx)
    fwhm_pts = measure_fwhm_points(y_corr)
    snr = (height / noise_std) if noise_std > 0 else float('inf')

    detail = {
        'snr':          float(snr) if np.isfinite(snr) else None,
        'fwhm_pts':     fwhm_pts,
        'gate_failed':  None,
        'asym_frac':    None,
        'asym_kind':    None,
        'n_shoulders':  0,
        'components':   {},
    }

    # ── Hard gates ────────────────────────────────────────────────────────────
    # A straight ramp needs no dedicated test: its baseline-corrected height is
    # ~0, so it fails the SNR gate here.
    if snr < GATE_MIN_SNR:
        detail['gate_failed'] = 'snr'
        return SIGNAL_POOR, 0.0, detail
    if fwhm_pts < GATE_MIN_PTS_FWHM:
        detail['gate_failed'] = 'digital_resolution'
        return SIGNAL_POOR, 0.0, detail

    # ── Components ────────────────────────────────────────────────────────────
    apex_pos = int(np.argmax(y_corr)) if y_corr.size else 0

    s_sym, asym_frac, asym_kind = _symmetry_from_profile(
        y_corr, apex_pos, fwhm_pts)
    detail['asym_frac'] = asym_frac
    detail['asym_kind'] = asym_kind

    s_shoulder, n_shoulders = _shoulder_score(y_corr, apex_pos, fwhm_pts)
    detail['n_shoulders'] = n_shoulders

    s_tilt, s_ground = _baseline_scores(
        intensity, left_idx, right_idx, cluster_lo_idx, cluster_hi_idx,
        height, noise_std, baseline_level)

    s_width = _width_score(fwhm_pts, median_fwhm_pts)

    scores = {
        'shape':       _clip01(shape_r2),
        'sym':         s_sym,
        'base_tilt':   s_tilt,
        'base_ground': s_ground,
        'shoulder':    s_shoulder,
        'width':       s_width,
    }

    # Unavailable components (None) are excluded and their weight redistributed
    # over the rest, rather than defaulting to a neutral value that would invent
    # evidence the spectrum does not provide.
    total_w = 0.0
    acc = 0.0
    for key, val in scores.items():
        if val is None:
            continue
        w = _WEIGHTS[key]
        total_w += w
        acc += w * val
        detail['components'][key] = float(val)

    score = (acc / total_w) if total_w > 1e-9 else 0.0

    weakest = min(detail['components'].values(), default=1.0)
    detail['weakest_component'] = (
        min(detail['components'], key=detail['components'].get)
        if detail['components'] else None)

    if score >= THRESH_GOOD and weakest >= COMPONENT_GOOD_FLOOR:
        tier = SIGNAL_GOOD
    elif score >= THRESH_AVERAGE:
        tier = SIGNAL_AVERAGE
    else:
        tier = SIGNAL_POOR

    return tier, float(score), detail


# ── Annotation eligibility ────────────────────────────────────────────────────
# A Poor signal failed a hard gate (SNR below the detection limit, or too few
# points across the FWHM) or is badly distorted. Its apex position and its line
# spacings are unreliable, so matching it against a database shift or coupling
# constant produces a chemical-shift agreement that the data does not support.
# Annotation therefore reads Good and Average signals only; Poor signals stay
# visible in the peak list and are reported as skipped, never silently dropped.
ANNOTATABLE_TIERS = frozenset({SIGNAL_GOOD, SIGNAL_AVERAGE})


def normalize_tiers(tiers) -> frozenset:
    """Coerce a user/serialized tier selection into a tier set.

    Accepts any iterable of tier names in any case. An empty or unrecognised
    selection falls back to the default rather than annotating nothing — a
    typo in a saved project must not silently produce zero annotations.
    """
    if not tiers:
        return ANNOTATABLE_TIERS
    wanted = {str(t).strip().lower() for t in tiers}
    picked = frozenset(t for t in SIGNAL_TIERS if t.lower() in wanted)
    return picked or ANNOTATABLE_TIERS


def effective_tier(peak) -> str:
    """Grade that governs a peak's use downstream.

    A manual status set by the analyst in the peak list overrides the automatic
    grade. An unrecognised or missing value falls back to Average so a
    vocabulary mismatch can never silently discard a signal.
    """
    raw = str(peak.get('status') or peak.get('signal_quality') or '').strip()
    for tier in SIGNAL_TIERS:
        if raw.lower() == tier.lower():
            return tier
    return SIGNAL_AVERAGE


def is_annotatable(peak, tiers=None) -> bool:
    """True when the signal's grade is one the analyst allows for annotation."""
    return effective_tier(peak) in normalize_tiers(tiers)


def filter_annotatable(peaks, tiers=None):
    """Split peaks into (usable, rejected) for annotation.

    Returns both halves so the caller can warn the user with the exact count
    instead of losing signals without trace.
    """
    peaks = list(peaks or [])
    usable   = [p for p in peaks if is_annotatable(p, tiers)]
    rejected = [p for p in peaks if not is_annotatable(p, tiers)]
    return usable, rejected
