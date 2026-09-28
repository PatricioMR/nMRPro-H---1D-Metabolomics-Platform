"""Overlapping analytes measured together, with room for what is not known."""
import numpy as np
import pytest

from processing.quant_cascade import build_columns
from scipy.optimize import nnls

from processing.quant_joint_fit import (
    _solve, free_components, joint_measure, refine_part_shifts)

MHZ = 500.0
LW = 2.6 / MHZ
PPM = np.linspace(2.5, 4.5, 8001)


def _met(mid, signals):
    return {'id': mid, 'name': mid,
            'db_peaks': [{'ppm': p, 'type': m, 'multiplicity': m,
                          'protons': nh, 'j_coupling': j}
                         for p, m, nh, j in signals]}


def _cols(*mets, drift=0.0):
    return build_columns(list(mets), field_mhz=MHZ, linewidth_ppm=LW,
                         drift_ppm=drift)


def _trace(cols, amps, *, noise=0.0, seed=0, offsets=None):
    y = np.zeros(PPM.size)
    for c in cols:
        idx, val = c.curve(PPM, (offsets or {}).get(c.mid, 0.0))
        y[idx] += amps[c.mid] * val
    if noise:
        y += np.random.default_rng(seed).normal(0, noise, PPM.size)
    return y


def _lorentz(c, area):
    g = LW / 2
    return area * (g / np.pi) / ((PPM - c) ** 2 + g * g)


def test_two_singlets_on_top_of_each_other_are_divided_by_their_other_signals():
    """A window over the shared line would count both; the fit gives each
    its own, because each has another signal that says how much of it
    there is."""
    a = _met('A', [(3.250, 's', 9, None), (3.900, 's', 2, None)])
    b = _met('B', [(3.253, 's', 2, None), (3.410, 's', 2, None)])
    cols = _cols(a, b)
    X = np.column_stack([_trace(cols, {'A': 1000.0, 'B': 3000.0},
                                noise=5.0, seed=s) for s in range(4)])
    out = joint_measure(cols, [], PPM, X, range(4), sigma=5.0,
                        linewidth_ppm=LW)
    for mid, want in (('A', 1000.0), ('B', 3000.0)):
        got = np.median(list(out[mid]['amplitudes'].values()))
        assert got == pytest.approx(want, rel=0.03)


def test_an_unknown_peak_under_a_known_one_goes_to_a_free_component():
    """Without room for it, an analyte not in the library hands its area to
    the known one it sits beside: +28% on the reference bench at 100
    analytes."""
    a = _met('A', [(3.300, 's', 3, None)])
    cols = _cols(a)
    y = _trace(cols, {'A': 1000.0}) + _lorentz(3.306, 2000.0)
    X = np.column_stack([y + np.random.default_rng(s).normal(0, 2.0, PPM.size)
                         for s in range(3)])
    peaks = [{'ppm': 3.300}, {'ppm': 3.306}]
    out = joint_measure(cols, peaks, PPM, X, range(3), sigma=2.0,
                        linewidth_ppm=LW)
    assert out['A']['free'] == 1
    got = np.median(list(out['A']['amplitudes'].values()))
    assert got == pytest.approx(1000.0, rel=0.05)


def test_an_analyte_that_overlaps_nothing_keeps_its_integration():
    a = _met('A', [(3.000, 's', 3, None)])
    b = _met('B', [(4.000, 's', 3, None)])
    cols = _cols(a, b)
    X = np.column_stack([_trace(cols, {'A': 500.0, 'B': 500.0})] * 3)
    assert joint_measure(cols, [], PPM, X, range(3), sigma=1.0,
                         linewidth_ppm=LW) == {}


def test_a_weak_line_of_a_known_signal_is_not_an_unknown():
    """Every line claims its peak, however weak: the outer lines of
    lactate's quartet stand at under a tenth of its 3H doublet and, judged
    by height, were given free components that took 13% of lactate."""
    lac = _met('L', [(3.000, 'd', 3, '7.0'), (4.100, 'q', 1, '7.0')])
    cols = _cols(lac)
    idx, val = cols[0].curve(PPM, 0.0)
    y = np.zeros(PPM.size)
    y[idx] = 1000.0 * val
    outer = 4.100 + 1.5 * 7.0 / MHZ          # the quartet's outer line
    frees = free_components(cols, [{'ppm': outer}], PPM, y, sigma=0.01,
                            linewidth_ppm=LW)
    assert frees == []


def test_a_signal_placed_off_its_line_is_slid_back():
    """The reference's position is its own pH's: a signal a linewidth off
    is found where it is before anything is fitted."""
    a = _met('A', [(3.250, 's', 3, None), (3.600, 's', 3, None)])
    placed = _cols(a)
    truth_off = 0.7 * LW
    y = np.zeros(PPM.size)
    for comp, _nh in placed[0].parts[:1]:
        comp.shift_ppm = truth_off         # the true spectrum: first signal off
    idx, val = placed[0].curve(PPM, 0.0)
    y[idx] = 1000.0 * val
    cols = _cols(a)                        # the engine's, at the listed place
    idx, _v = cols[0].curve(PPM, 0.0)
    union = np.unique(idx)
    off = refine_part_shifts(cols, None, PPM, y[union], union, LW)
    first = min(off, key=lambda k: k[1])
    assert off[first] == pytest.approx(truth_off, abs=0.3 * LW)


def test_a_known_line_displaced_by_its_ph_keeps_its_peak():
    a = _met('A', [(3.300, 's', 3, None)])
    cols = _cols(a)
    frees = free_components(cols, [{'ppm': 3.300 + 0.8 * LW}], PPM,
                            np.full(PPM.size, 100.0), sigma=1.0,
                            linewidth_ppm=LW)
    assert frees == []


def test_a_second_peak_beside_a_known_line_is_free():
    a = _met('A', [(3.300, 's', 3, None)])
    cols = _cols(a)
    frees = free_components(cols, [{'ppm': 3.300}, {'ppm': 3.306}], PPM,
                            np.full(PPM.size, 100.0), sigma=1.0,
                            linewidth_ppm=LW)
    assert [round(c, 3) for c, _w in frees] == [3.306]


def test_the_reduced_solve_is_the_full_problem():
    rng = np.random.default_rng(3)
    A = np.abs(rng.normal(size=(2000, 12)))
    t = A @ np.r_[np.abs(rng.normal(size=8)), np.zeros(4)]         + rng.normal(0, 0.5, 2000)
    x, r2 = _solve(A, t)
    x0, rn = nnls(A, t)
    assert np.allclose(x, x0, atol=1e-8)
    assert r2 == pytest.approx(rn ** 2, rel=1e-9)


def test_an_unreported_neighbour_in_the_fit_keeps_its_area_off_the_analyte():
    """Why the engine hands the fit every candidate annotation proposed,
    reported or not: without B's column A overlaps nothing, is left to its
    integration, and its window counts B's line under it as A."""
    a = _met('A', [(3.250, 's', 3, None), (3.900, 's', 3, None)])
    b = _met('B', [(3.252, 's', 3, None), (3.410, 's', 2, None)])
    both = _cols(a, b)
    X = np.column_stack([_trace(both, {'A': 1000.0, 'B': 1000.0},
                                noise=2.0, seed=s) for s in range(3)])
    alone = joint_measure([c for c in both if c.mid == 'A'], [], PPM, X,
                          range(3), sigma=2.0, linewidth_ppm=LW)
    with_b = joint_measure(both, [], PPM, X, range(3), sigma=2.0,
                           linewidth_ppm=LW)
    assert 'A' not in alone
    got = np.median(list(with_b['A']['amplitudes'].values()))
    assert got == pytest.approx(1000.0, rel=0.05)
