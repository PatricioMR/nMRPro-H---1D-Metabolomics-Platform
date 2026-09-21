"""A foot a neighbour bounded stands in the valley, not on the neighbour's
flank.

A weak signal between two strong ones: the midpoint-less-half-a-linewidth
bound put both feet on the neighbours' flanks, the local baseline joined the
two flanks above the signal, and the area came out negative -- the whole
window shaded below the baseline in the quantification stage.
"""
import numpy as np
import pytest

from processing.quant_integration import (
    below_baseline_share, foot_at_valley, integrate_sample, integrate_signal,
    signal_segments)

LW = 0.002
G = LW / 2.0
PPM = np.linspace(3.52, 3.57, 5001)


def _lorentz(c, a):
    return a * G ** 2 / ((PPM - c) ** 2 + G ** 2)


def _peak(rank, c, half=0.003):
    return {'rank': rank, 'ppm': c, 'start_ppm': c - half, 'end_ppm': c + half}


BIG_A, BIG_B, WEAK = 3.550, 3.540, 3.545
Y = _lorentz(BIG_A, 100.0) + _lorentz(BIG_B, 100.0) + _lorentz(WEAK, 4.0)
OWN = [_peak(2, WEAK)]
OTHERS = [_peak(1, BIG_A), _peak(3, BIG_B)]


def test_without_a_trace_the_feet_are_the_bounds():
    (lo, hi), = signal_segments(OWN, linewidth_ppm=LW, other_peaks=OTHERS)
    assert lo == pytest.approx((BIG_B + WEAK) / 2 - LW / 2)
    assert hi == pytest.approx((BIG_A + WEAK) / 2 + LW / 2)


def test_the_bound_is_on_the_neighbours_flank_and_the_trace_under_the_baseline():
    (lo, hi), = signal_segments(OWN, linewidth_ppm=LW, other_peaks=OTHERS)
    assert integrate_sample(PPM, Y, lo, hi, clip=False) < 0
    assert integrate_sample(PPM, Y, lo, hi) >= 0          # nothing above the trace
    assert below_baseline_share(PPM, Y, lo, hi) > 0.8


def test_a_bounded_foot_moves_to_the_valley():
    (lo, hi), = signal_segments(OWN, linewidth_ppm=LW, other_peaks=OTHERS,
                                trace=(PPM, Y))
    m = (PPM >= WEAK - 0.004) & (PPM <= WEAK)
    valley_lo = PPM[m][np.argmin(Y[m])]
    m = (PPM >= WEAK) & (PPM <= WEAK + 0.004)
    valley_hi = PPM[m][np.argmin(Y[m])]
    assert lo == pytest.approx(valley_lo, abs=2e-5)
    assert hi == pytest.approx(valley_hi, abs=2e-5)
    assert lo < WEAK < hi
    # the bump above its floor -- small, squeezed between two tails a
    # hundredfold taller, but the signal's, and positive
    assert integrate_sample(PPM, Y, lo, hi) > 0
    assert below_baseline_share(PPM, Y, lo, hi) < 0.2


def test_a_free_foot_is_not_moved():
    """No neighbour on the low side: the foot stays at its reach."""
    lo, hi = foot_at_valley(WEAK, WEAK - 3 * LW, WEAK + 0.002,
                            WEAK - 3 * LW, WEAK + 3 * LW, LW, (PPM, Y))
    assert lo == pytest.approx(WEAK - 3 * LW)
    assert hi < WEAK + 0.002 or hi == pytest.approx(WEAK + 0.002)


def test_integrate_signal_places_the_feet_on_the_measured_samples():
    X = np.stack([Y, Y * 0.5, np.zeros_like(Y)], axis=1)
    out = integrate_signal(OWN, PPM, X, included=[0, 1], linewidth_ppm=LW,
                           other_peaks=OTHERS)
    lo, hi = out['window']
    assert WEAK - 0.004 < lo < WEAK and WEAK < hi < WEAK + 0.004
    assert all(a > 0 for a in out['areas'].values())
    assert out['below_baseline_share'] < 0.2


def test_the_row_says_when_the_window_ran_under_its_baseline():
    from processing.quant_engine import _baseline_reason
    assert _baseline_reason({'below_baseline_share': 0.05}) == []
    assert _baseline_reason({'below_baseline_share': None}) == []
    (msg,) = _baseline_reason({'below_baseline_share': 0.8})
    assert '80%' in msg and 'counted nothing' in msg
