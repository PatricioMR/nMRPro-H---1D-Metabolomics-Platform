"""Tests for the Tk-free math of the per-multiplet across-samples view."""
import numpy as np

from gui.widgets.multiplet_sample_view import (
    parse_j_hz, multiplet_window, measure_j, measure_sample,
)

MHZ = 600.0


def _triplet(ppm, centre, j_hz, amp, w=0.0015):
    d = j_hz / MHZ
    def g(c, a):
        return a * np.exp(-((ppm - c) ** 2) / (2 * w ** 2))
    return g(centre - d, amp) + g(centre, 2 * amp) + g(centre + d, amp)


def test_parse_j_hz():
    assert parse_j_hz('7.1 Hz') == 7.1
    assert parse_j_hz(7) == 7.0
    assert parse_j_hz(None) is None
    assert parse_j_hz('n/a') is None


def test_multiplet_window_covers_predicted_lines():
    mp = {'peak_ppm': 0.90,
          'predicted_lines': [{'ppm': 0.888}, {'ppm': 0.900}, {'ppm': 0.912}]}
    dp = {'type': 't', 'j_coupling': '7.0 Hz', 'ppm_center': 0.90}
    lo, hi, centre, db_j, mult = multiplet_window(mp, None, dp, MHZ)
    assert lo < 0.888 and hi > 0.912
    assert abs(centre - 0.90) < 1e-9
    assert db_j == 7.0 and mult == 't'


def test_measure_j_recovers_coupling():
    ppm = np.linspace(0.84, 0.96, 1600)
    y = _triplet(ppm, 0.90, 7.0, 1.0)
    j = measure_j(ppm, y, MHZ)
    assert j is not None
    assert abs(j - 7.0) < 0.6


def test_measure_sample_present_and_absent():
    ppm = np.linspace(0.84, 0.96, 1600)
    noise = 0.01
    lo, hi, centre = 0.86, 0.94, 0.90
    # present sample (tall triplet)
    y = _triplet(ppm, 0.90, 7.0, 1.0) + np.random.default_rng(0).normal(0, noise, ppm.size)
    m = measure_sample(ppm, y, lo, hi, centre, MHZ, 7.0, noise, 3.0)
    assert m['present'] is True
    assert m['j_obs'] is not None and abs(m['dj']) < 1.0
    # absent sample (noise only)
    yn = np.random.default_rng(1).normal(0, noise, ppm.size)
    mn = measure_sample(ppm, yn, lo, hi, centre, MHZ, 7.0, noise, 3.0)
    assert mn['present'] is False


def test_measure_sample_dppm_reflects_shift():
    ppm = np.linspace(0.84, 0.96, 1600)
    shifted = _triplet(ppm, 0.9020, 7.0, 1.0)     # +2 mppm
    m = measure_sample(ppm, shifted, 0.86, 0.94, 0.90, MHZ, 7.0, 0.005, 3.0)
    assert m['present'] is True
    assert 1.0 < m['dppm_mppm'] < 3.0             # ≈ +2 mppm
