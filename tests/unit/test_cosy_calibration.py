# tests/unit/test_cosy_calibration.py
"""Testes da calibração (referenciação TSP/DSS) de espectros COSY."""
import numpy as np
import pytest

from processing.cosy_calibration import detect_reference, calibrate_cosy


def _grid(n=200, lo=-1.0, hi=11.0):
    axis = np.linspace(hi, lo, n)   # ppm decrescente
    return axis, axis.copy()


def _put_diag_peak(data, axis, ppm, amp=500, w=2):
    c = int(np.argmin(np.abs(axis - ppm)))
    for di in range(-w, w + 1):
        for dj in range(-w, w + 1):
            ii, jj = c + di, c + dj
            if 0 <= ii < data.shape[0] and 0 <= jj < data.shape[1]:
                data[ii, jj] += amp * np.exp(-(di*di + dj*dj) / 2.0)


def test_detect_reference_finds_offset_peak():
    f2, f1 = _grid()
    data = np.random.default_rng(0).normal(0, 1, (200, 200))
    # TSP deslocado: pico diagonal em 0.04 ppm
    _put_diag_peak(data, f2, 0.04, amp=800)
    f2o, f1o = detect_reference(f2, f1, data, ref=0.0, window=0.15)
    assert abs(f2o - 0.04) < 0.02
    assert abs(f1o - 0.04) < 0.02


def test_calibrate_brings_peak_to_zero():
    f2, f1 = _grid()
    data = np.zeros((200, 200))
    _put_diag_peak(data, f2, 0.04, amp=800)
    cal = calibrate_cosy(f2, f1, data, ref=0.0, window=0.15)
    assert cal is not None
    # Offset deve compensar os ~0.04 ppm
    assert abs(cal['df2'] + 0.04) < 0.02
    assert abs(cal['df1'] + 0.04) < 0.02
    # Após calibração, o pico fica perto de 0 no novo eixo
    f2_new = cal['f2_ppm']
    peak_idx = int(np.argmin(np.abs(f2 - 0.04)))
    assert abs(f2_new[peak_idx]) < 0.02


def test_returns_none_when_no_data_in_window():
    f2, f1 = _grid(lo=2.0, hi=10.0)   # eixo não inclui 0 ppm
    data = np.zeros((200, 200))
    assert calibrate_cosy(f2, f1, data, ref=0.0, window=0.1) is None
