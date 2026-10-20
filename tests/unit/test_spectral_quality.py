# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

import numpy as np
import pytest

from processing.spectral_quality import (
    compute_fwhm_ppm,
    compute_asymmetry,
    assess_sample_quality,
    assess_all_samples,
    grade_metric,
    grade_color,
    FWHM_THRESHOLDS_PPM,
    FWHM_THRESHOLDS_HZ,
    ASYMMETRY_THRESHOLDS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lorentzian(ppm, center, hwhm, amp=1.0):
    return amp * hwhm**2 / ((ppm - center)**2 + hwhm**2)


def _make_spectrum(hwhm_ppm=0.005, noise=0.0, center=0.0, n=4096,
                   ppm_lo=12.0, ppm_hi=-1.0):
    ppm = np.linspace(ppm_lo, ppm_hi, n)
    y = _lorentzian(ppm, center, hwhm_ppm, amp=1000.0)
    if noise > 0.0:
        rng = np.random.default_rng(42)
        y = y + rng.normal(0, noise, n)
    return ppm, y


def _apex_idx(ppm, center=0.0):
    return int(np.argmin(np.abs(ppm - center)))


# ── grade_metric ──────────────────────────────────────────────────────────────

class TestGradeMetric:
    def test_good(self):
        assert grade_metric(0.003, FWHM_THRESHOLDS_PPM) == 'good'

    def test_ok(self):
        assert grade_metric(0.007, FWHM_THRESHOLDS_PPM) == 'ok'

    def test_poor(self):
        assert grade_metric(0.080, FWHM_THRESHOLDS_PPM) == 'poor'

    def test_none_returns_unknown(self):
        assert grade_metric(None, FWHM_THRESHOLDS_PPM) == 'unknown'

    def test_nan_returns_unknown(self):
        assert grade_metric(float('nan'), FWHM_THRESHOLDS_PPM) == 'unknown'

    def test_negative_uses_abs(self):
        # asymmetry can be negative — abs is taken internally
        assert grade_metric(-0.010, ASYMMETRY_THRESHOLDS) == 'good'


# ── grade_color ───────────────────────────────────────────────────────────────

class TestGradeColor:
    def test_none_returns_grey(self):
        assert grade_color(None, FWHM_THRESHOLDS_PPM) == '#f0f0f0'

    def test_good_value_is_greenish(self):
        # 0.001 << 0.005 (good threshold) → clearly green
        col = grade_color(0.001, FWHM_THRESHOLDS_PPM)
        r = int(col[1:3], 16)
        g = int(col[3:5], 16)
        assert g > r   # green channel dominates

    def test_poor_value_is_reddish(self):
        col = grade_color(0.100, FWHM_THRESHOLDS_PPM)
        r = int(col[1:3], 16)
        g = int(col[3:5], 16)
        assert r > g   # red channel dominates


# ── compute_fwhm_ppm ──────────────────────────────────────────────────────────

class TestComputeFwhmPpm:
    def test_known_lorentzian(self):
        # HWHM=0.020 → FWHM≈0.040 ppm; with step≈0.003 ppm this gives ~12 points
        # above half-max, ensuring < 15% discretisation error
        hwhm = 0.020
        ppm, y = _make_spectrum(hwhm_ppm=hwhm)
        idx = _apex_idx(ppm)
        res = compute_fwhm_ppm(ppm, y, idx)
        assert res['fwhm_ppm'] is not None
        assert res['warning'] is None
        assert abs(res['fwhm_ppm'] - 2 * hwhm) / (2 * hwhm) < 0.15

    def test_flat_spectrum_no_crossing(self):
        ppm = np.linspace(12, -1, 1000)
        y   = np.ones(1000)
        res = compute_fwhm_ppm(ppm, y, 500)
        assert res['fwhm_ppm'] is None
        assert res['warning'] is not None

    def test_all_zeros(self):
        ppm = np.linspace(12, -1, 1000)
        y   = np.zeros(1000)
        res = compute_fwhm_ppm(ppm, y, 500)
        assert res['fwhm_ppm'] is None
        assert res['warning'] is not None

    def test_apex_out_of_bounds(self):
        ppm = np.linspace(12, -1, 1000)
        y   = np.ones(1000)
        res = compute_fwhm_ppm(ppm, y, 9999)
        assert res['fwhm_ppm'] is None
        assert 'out of bounds' in res['warning']

    def test_fwhm_hz_with_mhz(self):
        ppm, y = _make_spectrum(hwhm_ppm=0.005)
        idx = _apex_idx(ppm)
        res = compute_fwhm_ppm(ppm, y, idx, spectrometer_mhz=600.0)
        assert res['fwhm_hz'] is not None
        assert abs(res['fwhm_hz'] - res['fwhm_ppm'] * 600.0) < 1e-6

    def test_fwhm_hz_none_without_mhz(self):
        ppm, y = _make_spectrum()
        idx = _apex_idx(ppm)
        res = compute_fwhm_ppm(ppm, y, idx, spectrometer_mhz=None)
        assert res['fwhm_hz'] is None

    def test_no_error_when_peak_at_edge(self):
        # Peak very close to left edge of ppm axis — should not raise
        ppm = np.linspace(12, -1, 1000)
        y   = _lorentzian(ppm, 11.9, 0.005, 1000.0)
        idx = int(np.argmax(y))
        res = compute_fwhm_ppm(ppm, y, idx, window_ppm=0.15)
        # may or may not find FWHM, but must not raise
        assert isinstance(res, dict)

    def test_with_noise(self):
        ppm, y = _make_spectrum(hwhm_ppm=0.005, noise=10.0)
        idx = _apex_idx(ppm)
        res = compute_fwhm_ppm(ppm, y, idx)
        # Noisy but peak is strong (amp=1000, noise=10) — should still work
        assert res['fwhm_ppm'] is not None


# ── compute_asymmetry ─────────────────────────────────────────────────────────

class TestComputeAsymmetry:
    def test_symmetric_lorentzian(self):
        ppm, y = _make_spectrum(hwhm_ppm=0.010)
        idx = _apex_idx(ppm)
        res = compute_asymmetry(ppm, y, idx)
        assert res['asymmetry'] is not None
        assert abs(res['asymmetry']) < 0.02   # nearly zero for perfect Lorentzian

    def test_skewed_right(self):
        ppm = np.linspace(12, -1, 4096)
        # Wider on the right (higher ppm) side
        y = np.where(ppm > 0,
                     _lorentzian(ppm, 0.0, 0.050, 1000.0),
                     _lorentzian(ppm, 0.0, 0.005, 1000.0))
        idx = _apex_idx(ppm)
        res = compute_asymmetry(ppm, y, idx)
        assert res['asymmetry'] is not None
        # Higher-ppm side is wider → area_right < area_left in ppm convention
        # (ppm decreasing: "right" in ppm coords = higher ppm)
        # The sign depends on convention; just check it is non-zero
        assert abs(res['asymmetry']) > 0.10

    def test_all_zeros(self):
        ppm = np.linspace(12, -1, 1000)
        y   = np.zeros(1000)
        res = compute_asymmetry(ppm, y, 500)
        assert res['asymmetry'] is None

    def test_apex_out_of_bounds(self):
        ppm = np.linspace(12, -1, 500)
        y   = np.ones(500)
        res = compute_asymmetry(ppm, y, 9999)
        assert res['asymmetry'] is None
        assert 'out of bounds' in res['warning']

    def test_no_exception_ever(self):
        rng = np.random.default_rng(0)
        ppm = np.linspace(10, -0.5, 200)
        y   = rng.standard_normal(200)
        res = compute_asymmetry(ppm, y, 100)
        assert isinstance(res, dict)


# ── assess_sample_quality ─────────────────────────────────────────────────────

class TestAssessSampleQuality:
    def test_clean_reference(self):
        ppm, y = _make_spectrum()
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL')
        assert res['has_reference'] is True
        assert res['fwhm_ppm'] is not None
        assert res['fwhm_grade'] in ('good', 'ok', 'poor')
        assert res['asymmetry'] is not None

    def test_no_reference_region(self):
        ppm = np.linspace(12.0, 2.0, 2000)   # does not cover 0 ppm
        y   = np.ones(2000)
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL')
        assert res['has_reference'] is False
        assert len(res['warnings']) > 0

    def test_grade_good_no_mhz(self):
        # Without MHz: grade in ppm; HWHM=0.001 → FWHM≈0.002 ppm < 0.005 → good
        # Narrow ppm range (1 ppm) → step≈0.000244 ppm → peak resolvable
        ppm, y = _make_spectrum(hwhm_ppm=0.001, ppm_lo=0.5, ppm_hi=-0.5)
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL', spectrometer_mhz=None)
        assert res['fwhm_grade'] == 'good'

    def test_grade_poor_no_mhz(self):
        # Without MHz: HWHM=0.040 → FWHM≈0.080 ppm > 0.010 → poor
        ppm, y = _make_spectrum(hwhm_ppm=0.040)
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL', spectrometer_mhz=None)
        assert res['fwhm_grade'] == 'poor'

    def test_grade_good_with_mhz(self):
        # With MHz=600: HWHM=0.001 ppm → FWHM≈0.002 ppm = 1.2 Hz < 2 Hz → good
        # Narrow ppm range → step≈0.000244 ppm → peak resolvable
        ppm, y = _make_spectrum(hwhm_ppm=0.001, ppm_lo=0.5, ppm_hi=-0.5)
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL', spectrometer_mhz=600.0)
        assert res['fwhm_grade'] == 'good'

    def test_grade_poor_with_mhz(self):
        # With MHz=600: HWHM=0.010 ppm → FWHM≈0.020 ppm = 12 Hz > 4 Hz → poor
        ppm, y = _make_spectrum(hwhm_ppm=0.010)
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL', spectrometer_mhz=600.0)
        assert res['fwhm_grade'] == 'poor'

    def test_grade_ok_with_mhz(self):
        # With MHz=600: HWHM=0.003 ppm → FWHM≈0.006 ppm = 3.6 Hz → ok (2–4 Hz)
        # Narrow ppm range (0.4 ppm) → step≈0.0000977 ppm → peak resolvable
        ppm, y = _make_spectrum(hwhm_ppm=0.003, ppm_lo=0.2, ppm_hi=-0.2)
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL', spectrometer_mhz=600.0)
        assert res['fwhm_grade'] == 'ok'

    def test_hz_with_mhz(self):
        ppm, y = _make_spectrum()
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL', spectrometer_mhz=600.0)
        assert res['fwhm_hz'] is not None
        assert abs(res['fwhm_hz'] - res['fwhm_ppm'] * 600.0) < 1e-6

    def test_hz_none_without_mhz(self):
        ppm, y = _make_spectrum()
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL', spectrometer_mhz=None)
        assert res['fwhm_hz'] is None

    def test_zero_spectrum(self):
        ppm = np.linspace(12, -1, 1000)
        y   = np.zeros(1000)
        res = assess_sample_quality(ppm, y, 'S1', 'CTRL')
        # Zero spectrum: apex found but FWHM/asymmetry fail gracefully
        assert isinstance(res, dict)
        # Must not raise; fwhm_ppm may be None

    def test_sample_name_and_group_preserved(self):
        ppm, y = _make_spectrum()
        res = assess_sample_quality(ppm, y, 'MySample', 'GroupA')
        assert res['sample_name'] == 'MySample'
        assert res['sample_group'] == 'GroupA'


# ── assess_all_samples ────────────────────────────────────────────────────────

class TestAssessAllSamples:
    def test_returns_one_per_sample(self):
        ppm, y = _make_spectrum()
        n = 5
        intensities = np.column_stack([y] * n)
        si = [{'name': f'S{i}', 'label': 'CTRL'} for i in range(n)]
        results = assess_all_samples(ppm, intensities, si)
        assert len(results) == n

    def test_1d_intensities_accepted(self):
        ppm, y = _make_spectrum()
        si = [{'name': 'S0', 'label': 'CTRL'}]
        results = assess_all_samples(ppm, y, si)
        assert len(results) == 1

    def test_identical_samples_same_fwhm(self):
        ppm, y = _make_spectrum()
        n = 3
        intensities = np.column_stack([y] * n)
        si = [{'name': f'S{i}', 'label': 'CTRL'} for i in range(n)]
        results = assess_all_samples(ppm, intensities, si)
        fwhms = [r['fwhm_ppm'] for r in results if r['fwhm_ppm'] is not None]
        assert len(fwhms) == n
        assert max(fwhms) - min(fwhms) < 1e-9

    def test_blk_sample_still_assessed(self):
        ppm, y = _make_spectrum()
        intensities = np.column_stack([y, y])
        si = [{'name': 'BLK1', 'label': 'BLK'}, {'name': 'S1', 'label': 'CTRL'}]
        results = assess_all_samples(ppm, intensities, si)
        assert len(results) == 2
        assert results[0]['sample_name'] == 'BLK1'
        assert results[0]['has_reference'] is True

    def test_no_reference_region_all_samples(self):
        ppm = np.linspace(12.0, 2.0, 1000)
        intensities = np.ones((1000, 3))
        si = [{'name': f'S{i}', 'label': 'CTRL'} for i in range(3)]
        results = assess_all_samples(ppm, intensities, si)
        assert all(not r['has_reference'] for r in results)

    def test_missing_samples_info_uses_defaults(self):
        ppm, y = _make_spectrum()
        intensities = np.column_stack([y, y])
        results = assess_all_samples(ppm, intensities, [])   # empty samples_info
        assert len(results) == 2
        assert results[0]['sample_name'] == 'Sample 1'
