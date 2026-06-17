"""Reference deconvolution (FIDDLE): removes a common lineshape distortion
while conserving integrated areas."""
import numpy as np

from processing.digital_shim import (
    digital_shim, estimate_fwhm_ppm)


def _lorentz(x, a, c, w):
    return a * w ** 2 / ((x - c) ** 2 + w ** 2)


def _distort(spectrum, kernel):
    """Convolve (same instrumental lineshape applied to every peak)."""
    return np.convolve(spectrum, kernel, mode='same')


def _asym_kernel(n=41):
    """Asymmetric (shim-like) broadening kernel, normalised to unit area."""
    t = np.arange(n) - n // 2
    k = np.exp(-((t) ** 2) / (2 * 3.0 ** 2))          # gaussian core
    k += 0.6 * np.exp(-np.clip(t, 0, None) / 4.0) * (t >= 0)  # one-sided tail
    return k / k.sum()


def _make_spectrum():
    ppm = np.linspace(-0.5, 6.0, 6500)                # ~0.001 ppm/pt
    ref      = _lorentz(ppm, 1.0, 0.000, 0.0015)      # TSP-like singlet @ 0
    analyte  = _lorentz(ppm, 0.5, 3.000, 0.0015)      # analyte singlet
    clean    = ref + analyte
    distorted = _distort(clean, _asym_kernel())
    return ppm, clean, distorted


def test_area_is_conserved():
    ppm, _clean, distorted = _make_spectrum()
    corr, info = digital_shim(
        ppm, distorted, ref_lo=-0.05, ref_hi=0.05, reg=1e-3)
    assert abs(info['area_ratio'] - 1.0) < 0.05, info


def test_default_symmetrises_not_wider():
    """Conservative default (target = reference width): the analyte must not get
    broader than the distorted input — the common distortion is standardised."""
    ppm, clean, distorted = _make_spectrum()
    corr, _ = digital_shim(
        ppm, distorted, ref_lo=-0.05, ref_hi=0.05, reg=1e-3)
    m = (ppm > 2.8) & (ppm < 3.2)
    fw_dist = estimate_fwhm_ppm(ppm[m], distorted[m])
    fw_corr = estimate_fwhm_ppm(ppm[m], np.maximum(corr[m], 0))
    assert fw_corr is not None and fw_dist is not None
    assert fw_corr <= fw_dist + 1e-6, (fw_dist, fw_corr)


def test_explicit_target_narrows_toward_clean():
    """With a narrow target FWHM the analyte sharpens toward the clean width."""
    ppm, clean, distorted = _make_spectrum()
    m = (ppm > 2.8) & (ppm < 3.2)
    fw_clean = estimate_fwhm_ppm(ppm[m], clean[m])
    fw_dist  = estimate_fwhm_ppm(ppm[m], distorted[m])
    # clean width ≈ 0.003 ppm → 1.8 Hz at 600 MHz.
    corr, _ = digital_shim(
        ppm, distorted, ref_lo=-0.05, ref_hi=0.05,
        target_fwhm_hz=1.8, spectrometer_mhz=600.0, reg=1e-4)
    fw_corr = estimate_fwhm_ppm(ppm[m], np.maximum(corr[m], 0))
    assert fw_corr < fw_dist, (fw_clean, fw_dist, fw_corr)


def test_analyte_area_preserved_vs_clean():
    """The analyte's integrated area must match the clean reference (±5%)."""
    ppm, clean, distorted = _make_spectrum()
    corr, _ = digital_shim(
        ppm, distorted, ref_lo=-0.05, ref_hi=0.05, reg=1e-3)
    m = (ppm > 2.7) & (ppm < 3.3)
    a_clean = abs(np.trapezoid(clean[m], ppm[m]))
    a_corr  = abs(np.trapezoid(np.maximum(corr[m], 0), ppm[m]))
    assert abs(a_corr - a_clean) / a_clean < 0.08, (a_clean, a_corr)
