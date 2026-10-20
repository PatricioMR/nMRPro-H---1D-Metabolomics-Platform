# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_integration_dsp.py
"""Unit tests for processing/integration.py — the pure integration/lineshape DSP
extracted from the GUI layer. Verifies the analytical Lorentzian area (A·π·w),
baseline subtraction, multi-Lorentzian NNLS deconvolution, and that the
_quant_helpers re-exports point at the exact same objects (no drift)."""
import numpy as np
import pytest

from processing.integration import (
    _trapz, _lorentz_shape, _lorentz_func, _lorentz_amplitude,
    _linear_baseline, _fit_lorentz_mean, _fit_lorentz_mean_with_bl,
    _fit_lorentz_sample, _fit_multi_lorentz, jr_pointwise_multiplet_area,
)


def _lorentz(x, A, c, w):
    return A * w**2 / ((x - c) ** 2 + w**2)


def test_reexport_identity():
    """_quant_helpers must re-export the SAME function objects (single source)."""
    from gui.stages import _quant_helpers as H
    for n in ('_trapz', '_lorentz_shape', '_fit_lorentz_mean',
              '_fit_multi_lorentz', 'jr_pointwise_multiplet_area'):
        import processing.integration as I
        assert getattr(H, n) is getattr(I, n)


def test_trapz_zero_and_area():
    x = np.linspace(0.0, 1.0, 100)
    assert _trapz(np.zeros_like(x), x) == pytest.approx(0.0)
    assert _trapz(np.ones_like(x), x) == pytest.approx(1.0, abs=1e-9)


def test_lorentz_shape_peak_and_hwhm():
    x = np.array([0.0, 0.004, -0.004])
    s = _lorentz_shape(x, 0.0, 0.004)
    assert s[0] == pytest.approx(1.0)          # unit height at centre
    assert s[1] == pytest.approx(0.5)          # half height at HWHM
    assert s[2] == pytest.approx(0.5)


def test_linear_baseline_endpoints():
    x = np.linspace(1.0, 2.0, 11)
    y = 3.0 + 2.0 * (x - x[0])
    y0, slope, bl = _linear_baseline(x, y)
    assert y0 == pytest.approx(3.0)
    assert slope == pytest.approx(2.0)
    assert np.allclose(bl, y)


def test_fit_lorentz_mean_recovers_area():
    x = np.linspace(1.40, 1.55, 400)
    A, c, w = 10.0, 1.475, 0.004
    y = _lorentz(x, A, c, w)
    fa, fc, fw = _fit_lorentz_mean(x, y, with_baseline=False)
    assert fc == pytest.approx(c, abs=1e-3)
    # analytical Lorentzian area = A·π·w
    assert (fa * np.pi * fw) == pytest.approx(A * np.pi * w, rel=0.05)


def test_fit_lorentz_mean_with_bl_returns_baseline():
    x = np.linspace(1.40, 1.55, 300)
    y = _lorentz(x, 5.0, 1.475, 0.004) + (0.5 + 1.0 * (x - x[0]))
    A, c, w, y0, slope = _fit_lorentz_mean_with_bl(x, y, with_baseline=True)
    assert slope == pytest.approx(1.0, abs=0.2)
    assert w > 0


def test_fit_lorentz_sample_area():
    x = np.linspace(1.40, 1.55, 400)
    A, c, w = 8.0, 1.475, 0.004
    y = _lorentz(x, A, c, w)
    out = _fit_lorentz_sample(x, y, A0=A, cen0=c, wid0=w, with_baseline=False)
    assert out is not None
    area, fA, fc, fw = out
    assert area == pytest.approx(A * np.pi * w, rel=0.05)


def test_lorentz_amplitude_fixed_shape():
    x = np.linspace(1.40, 1.55, 300)
    A, c, w = 4.0, 1.475, 0.004
    y = _lorentz(x, A, c, w)
    assert _lorentz_amplitude(x, y, c, w) == pytest.approx(A, rel=1e-3)


def test_fit_multi_lorentz_separates_two_peaks():
    x = np.linspace(3.20, 3.40, 600)
    y = _lorentz(x, 10.0, 3.27, 0.004) + _lorentz(x, 4.0, 3.33, 0.004)
    all_y = y[:, None]
    res = _fit_multi_lorentz(x, y, all_y, peak_ppms=[3.27, 3.33])
    assert res is not None
    assert res['r2'] > 0.98
    centres = sorted(comp[1] for comp in res['components'])
    assert centres[0] == pytest.approx(3.27, abs=2e-3)
    assert centres[1] == pytest.approx(3.33, abs=2e-3)


def test_jr_pointwise_equals_scaled_area_when_uniform():
    x = np.linspace(8.40, 8.50, 400)
    y = _lorentz(x, 1.0, 8.44, 0.004) + _lorentz(x, 1.0, 8.46, 0.004)
    seg = y[:, None]
    whole = abs(float(_trapz(seg[:, 0], x)))
    out = jr_pointwise_multiplet_area(x, seg, [8.44, 8.46], [0.5, 0.5])
    assert out[0] == pytest.approx(0.5 * whole, rel=1e-6)
