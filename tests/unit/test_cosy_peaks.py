# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_cosy_peaks.py
"""Testes do picking de cross-peaks COSY."""
import numpy as np
import pytest

from processing.cosy_peaks import pick_cosy_crosspeaks, estimate_noise


def _make_cosy(n=200, ppm_lo=0.0, ppm_hi=10.0, peaks=(), width=3, amp=100.0,
               noise=1.0, seed=0):
    """Cria um espectro COSY sintético com cross-peaks gaussianos simétricos."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(ppm_hi, ppm_lo, n)   # ppm decrescente (NMR)
    data = rng.normal(0, noise, (n, n))
    def _idx(ppm):
        return int(np.argmin(np.abs(axis - ppm)))
    for (a, b) in peaks:
        for (fa, fb) in ((a, b), (b, a)):     # simétrico
            ci, cj = _idx(fb), _idx(fa)        # i=F1(linha), j=F2(coluna)
            for di in range(-width, width + 1):
                for dj in range(-width, width + 1):
                    ii, jj = ci + di, cj + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        data[ii, jj] += amp * np.exp(-(di*di + dj*dj) / (2*1.0))
    return axis, axis.copy(), data


def test_estimate_noise_positive():
    data = np.random.default_rng(1).normal(0, 2.0, (100, 100))
    assert estimate_noise(data) > 0


def test_picks_offdiagonal_crosspeak():
    f2, f1, data = _make_cosy(peaks=[(1.5, 3.5)], amp=200, noise=1.0)
    cps = pick_cosy_crosspeaks(f2, f1, data, noise_mult=6.0,
                               diag_width=0.1, symmetric=True)
    assert len(cps) >= 1
    # O cross-peak detectado deve estar perto de (1.5,3.5) ou (3.5,1.5)
    found = any(
        (abs(c['f2'] - 1.5) < 0.2 and abs(c['f1'] - 3.5) < 0.2) or
        (abs(c['f2'] - 3.5) < 0.2 and abs(c['f1'] - 1.5) < 0.2)
        for c in cps)
    assert found


def test_excludes_diagonal():
    # Pico apenas na diagonal → não deve ser devolvido
    f2, f1, data = _make_cosy(peaks=[(2.0, 2.0)], amp=200, noise=1.0)
    cps = pick_cosy_crosspeaks(f2, f1, data, noise_mult=6.0,
                               diag_width=0.15, symmetric=False)
    assert all(abs(c['f2'] - c['f1']) >= 0.15 for c in cps)


def test_symmetry_filters_noise():
    # Cross-peak assimétrico (só um lado) → rejeitado quando symmetric=True
    f2, f1, data = _make_cosy(peaks=[], noise=1.0)
    axis = f2
    ci = int(np.argmin(np.abs(axis - 3.5)))
    cj = int(np.argmin(np.abs(axis - 1.5)))
    data[ci, cj] += 300.0   # pico único, sem parceiro simétrico
    cps = pick_cosy_crosspeaks(f2, f1, data, noise_mult=6.0,
                               diag_width=0.1, symmetric=True)
    assert len(cps) == 0


def test_empty_when_below_threshold():
    f2, f1, data = _make_cosy(peaks=[(1.5, 3.5)], amp=5, noise=1.0)
    cps = pick_cosy_crosspeaks(f2, f1, data, noise_mult=50.0)
    assert cps == []
