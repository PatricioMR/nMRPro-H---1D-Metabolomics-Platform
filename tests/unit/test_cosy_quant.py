# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_cosy_quant.py
"""Testes da quantificação relativa por volume de cross-peaks COSY."""
import numpy as np
import pytest

from processing.cosy_quant import integrate_crosspeak, quantify_metabolites


def _grid(n=200, lo=0.0, hi=10.0):
    axis = np.linspace(hi, lo, n)
    return axis, axis.copy()


def _add_peak(data, axis, f2c, f1c, amp, width=2):
    cj = int(np.argmin(np.abs(axis - f2c)))
    ci = int(np.argmin(np.abs(axis - f1c)))
    for di in range(-width, width + 1):
        for dj in range(-width, width + 1):
            ii, jj = ci + di, cj + dj
            if 0 <= ii < data.shape[0] and 0 <= jj < data.shape[1]:
                data[ii, jj] += amp * np.exp(-(di*di + dj*dj) / 2.0)


def test_integrate_returns_positive_volume():
    f2, f1 = _grid()
    data = np.zeros((200, 200))
    _add_peak(data, f2, 1.5, 3.5, amp=100)
    v = integrate_crosspeak(f2, f1, data, 1.5, 3.5, half_f2=0.2, half_f1=0.2)
    assert v > 0


def test_stronger_peak_larger_volume():
    f2, f1 = _grid()
    data = np.zeros((200, 200))
    _add_peak(data, f2, 1.5, 3.5, amp=100)
    _add_peak(data, f2, 2.5, 5.5, amp=300)
    v_small = integrate_crosspeak(f2, f1, data, 1.5, 3.5, 0.2, 0.2)
    v_big   = integrate_crosspeak(f2, f1, data, 2.5, 5.5, 0.2, 0.2)
    assert v_big > v_small


def test_floor_excludes_noise():
    f2, f1 = _grid()
    data = np.full((200, 200), 5.0)   # ruído uniforme
    _add_peak(data, f2, 1.5, 3.5, amp=100)
    v = integrate_crosspeak(f2, f1, data, 1.5, 3.5, 0.2, 0.2, floor=10.0)
    v_nofloor = integrate_crosspeak(f2, f1, data, 1.5, 3.5, 0.2, 0.2, floor=0.0)
    assert v < v_nofloor


def test_quantify_metabolites_relative():
    f2, f1 = _grid()
    data = np.zeros((200, 200))
    _add_peak(data, f2, 1.5, 3.5, amp=100)   # metabolito A
    _add_peak(data, f2, 2.5, 5.5, amp=300)   # metabolito B (3×)
    cosy = {'f2_ppm': f2, 'f1_ppm': f1, 'data': data}
    results = [
        {'name': 'A', 'matches': [{'f2': 1.5, 'f1': 3.5}]},
        {'name': 'B', 'matches': [{'f2': 2.5, 'f1': 5.5}]},
    ]
    q = quantify_metabolites(results, cosy, half_f2=0.2, half_f1=0.2,
                             noise_mult=0.0)
    # B deve vir primeiro (volume maior) e ter abundância relativa maior
    assert q[0]['name'] == 'B'
    assert q[0]['volume'] > q[1]['volume']
    assert abs(sum(o['rel_abundance'] for o in q) - 1.0) < 1e-6


def test_empty_matches_zero_volume():
    f2, f1 = _grid()
    cosy = {'f2_ppm': f2, 'f1_ppm': f1, 'data': np.zeros((200, 200))}
    q = quantify_metabolites([{'name': 'X', 'matches': []}], cosy)
    assert q[0]['volume'] == 0.0
