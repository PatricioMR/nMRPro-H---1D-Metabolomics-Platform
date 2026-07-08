"""
Digital Shim — reference deconvolution (FIDDLE) for 1D NMR.

An imperfect static field / shim distorts every line in the spectrum by the
SAME instrumental lineshape. A clean, isolated reference signal (TSP, DSS,
formate…) carries that distortion. Dividing the time-domain signal of the whole
spectrum by the reference's time-domain signal — and multiplying by an ideal
target lineshape — removes the common distortion from every peak while
conserving integrated areas (the DC/first time-domain point is preserved).

Method (frequency-domain implementation, Wiener-regularised):

    f_exp(t)   = IFT(spectrum)                       # pseudo-FID of full spectrum
    r_exp(t)   = IFT(reference region, isolated)      # experimental ref lineshape
    r_ideal(t) = IFT(target reference lineshape)      # desired (symmetric) shape
    f_corr(t)  = f_exp(t) · r_ideal(t) · conj(r_exp(t)) / (|r_exp(t)|² + ε)
    corrected  = Re( FT(f_corr(t)) )

ε = reg · max(|r_exp(t)|²) damps the division where the reference signal has
decayed into noise (large t), preventing noise blow-up — the standard Wiener
regularisation for an otherwise ill-posed deconvolution.

References
----------
Morris GA. "Compensation of instrumental imperfections by deconvolution using
an internal reference signal." J. Magn. Reson. 80 (1988) 547.
Gibbs A, Morris GA. "Reference deconvolution. A simple and effective method..."
J. Magn. Reson. 91 (1991) 77.
Morris GA, Barjat H, Horne TJ. "Reference deconvolution methods."
Prog. Nucl. Magn. Reson. Spectrosc. 31 (1997) 197.
"""
from __future__ import annotations

import numpy as np


def _region_indices(ppm, lo, hi):
    lo, hi = (min(lo, hi), max(lo, hi))
    idx = np.where((ppm >= lo) & (ppm <= hi))[0]
    return idx


def estimate_fwhm_ppm(ppm, y):
    """Full width at half maximum (in ppm) of a single isolated peak."""
    y = np.asarray(y, dtype=float)
    if y.size < 3 or y.max() <= 0:
        return None
    half = 0.5 * float(y.max())
    above = np.where(y >= half)[0]
    if above.size < 2:
        return None
    return abs(float(ppm[above[-1]] - ppm[above[0]]))


def _lorentzian(ppm, center, fwhm_ppm, area):
    """Lorentzian on the ppm grid with the given integrated area."""
    hwhm = max(fwhm_ppm / 2.0, 1e-9)
    shape = hwhm ** 2 / ((ppm - center) ** 2 + hwhm ** 2)
    # Area of this shape on the grid → scale to requested area.
    grid_area = abs(np.trapezoid(shape, ppm))
    if grid_area <= 0:
        return shape * 0.0
    return shape * (area / grid_area)


def digital_shim(ppm, spectrum, ref_lo, ref_hi,
                 target_fwhm_hz=None, spectrometer_mhz=600.0,
                 reg=1e-3, target='lorentzian'):
    """Digital-shim (reference-deconvolve) ONE spectrum (1-D).

    Parameters
    ----------
    ppm            : (n,) chemical-shift axis (decreasing is fine).
    spectrum       : (n,) real intensities (single spectrum).
    ref_lo, ref_hi : ppm bounds of the isolated reference signal (e.g. TSP).
    target_fwhm_hz : desired reference FWHM in Hz. None → keep the experimental
                     reference width (symmetrise only; conservative, no extra
                     line-narrowing, so noise is not amplified beyond Wiener).
    spectrometer_mhz: spectrometer frequency (to convert Hz↔ppm).
    reg            : Wiener regularisation fraction (of max |r_exp|²).
    target         : ideal reference lineshape ('lorentzian').

    Returns
    -------
    (corrected, info)
        corrected : (n,) real corrected spectrum (same length).
        info      : dict with 'ref_center_ppm', 'ref_fwhm_ppm',
                    'target_fwhm_ppm', 'area_ratio' (post/pre, ≈1.0).
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(spectrum, dtype=float).copy()
    n = y.size
    idx = _region_indices(ppm, ref_lo, ref_hi)
    if idx.size < 5:
        raise ValueError("Reference region too small / empty — widen it.")

    # ── Isolate the reference: linear baseline-subtract, zero outside ──────────
    i0, i1 = int(idx[0]), int(idx[-1])
    seg = y[i0:i1 + 1]
    base = np.linspace(seg[0], seg[-1], seg.size)
    seg_bs = np.maximum(seg - base, 0.0)

    r_exp = np.zeros(n, dtype=float)
    r_exp[i0:i1 + 1] = seg_bs

    if float(r_exp.max()) <= 0:
        raise ValueError("No positive reference signal in the region.")

    ref_center = float(ppm[i0 + int(np.argmax(seg_bs))])
    ref_fwhm   = estimate_fwhm_ppm(ppm[i0:i1 + 1], seg_bs) or \
        abs(float(ppm[i1] - ppm[i0])) / 4.0
    ref_area   = abs(float(np.trapezoid(seg_bs, ppm[i0:i1 + 1])))

    # ── Target (ideal) reference lineshape — same area as experimental ────────
    if target_fwhm_hz is not None and target_fwhm_hz > 0:
        target_fwhm = float(target_fwhm_hz) / float(spectrometer_mhz)
    else:
        target_fwhm = ref_fwhm                       # symmetrise only
    r_ideal = _lorentzian(ppm, ref_center, target_fwhm, ref_area)

    # ── Time domain (pseudo-FID), Wiener division, back to frequency ──────────
    F_exp   = np.fft.ifft(y)
    R_exp   = np.fft.ifft(r_exp)
    R_ideal = np.fft.ifft(r_ideal)

    eps = float(reg) * float(np.max(np.abs(R_exp) ** 2))
    if eps <= 0:
        eps = 1e-30
    H = R_ideal * np.conj(R_exp) / (np.abs(R_exp) ** 2 + eps)

    corrected = np.real(np.fft.fft(F_exp * H))

    # Area bookkeeping (should be ≈ conserved by construction).
    pre  = abs(float(np.trapezoid(y, ppm)))
    post = abs(float(np.trapezoid(corrected, ppm)))
    info = {
        'ref_center_ppm':  ref_center,
        'ref_fwhm_ppm':    float(ref_fwhm),
        'target_fwhm_ppm': float(target_fwhm),
        'area_ratio':      float(post / pre) if pre > 0 else float('nan'),
    }
    return corrected, info


def digital_shim_2d(ppm, intensities, ref_lo, ref_hi,
                    target_fwhm_hz=None, spectrometer_mhz=600.0,
                    reg=1e-3, progress=None):
    """Apply Digital Shim to every column of a (n, m) matrix.

    Each sample is corrected against its OWN reference signal, so per-sample
    shim differences are each removed. Returns (corrected_2d, infos)."""
    intensities = np.asarray(intensities, dtype=float)
    if intensities.ndim == 1:
        corr, info = digital_shim(
            ppm, intensities, ref_lo, ref_hi, target_fwhm_hz,
            spectrometer_mhz, reg)
        return corr, [info]
    n, m = intensities.shape
    out = np.empty_like(intensities)
    infos = []
    for s in range(m):
        if progress is not None:
            try:
                progress((s + 1) / m, f"Digital Shim {s + 1}/{m}")
            except Exception:
                pass
        corr, info = digital_shim(
            ppm, intensities[:, s], ref_lo, ref_hi, target_fwhm_hz,
            spectrometer_mhz, reg)
        out[:, s] = corr
        infos.append(info)
    return out, infos
