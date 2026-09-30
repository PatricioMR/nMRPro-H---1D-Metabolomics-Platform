"""The local baseline is read where the cluster of overlapping signals ends,
not at the window's own feet.

A weak signal two linewidths beside a neighbour five times its size: the
window's upper foot stood on the neighbour's flank, the straight line
through the feet passed over the weak signal, and 76% of it was taken away
(all of it beside a neighbour twenty times its size). The neighbour's share
of the window is the model's to take out, not the baseline's.
"""
import numpy as np
import pytest

from processing.quant_deconvolution import fraction_in_window, neighbour_leak
from processing.quant_integration import (
    ANCHOR_MAX_REACH_LINEWIDTHS, BASELINE_STRETCH_LINEWIDTHS,
    WINDOW_REACH_LINEWIDTHS, anchor_levels, baseline_anchors, baseline_under,
    cluster_baseline, integrate_sample, integrate_signal,
    lorentzian_captured_fraction)

FWHM = 2.6 / 600.0
PPM = np.linspace(3.30, 3.80, 20001)


def _lor(c, area):
    g = FWHM / 2.0
    return area * (g / np.pi) / ((PPM - c) ** 2 + g ** 2)


def _peak(rank, c):
    return {'rank': rank, 'ppm': c, 'start_ppm': c - FWHM,
            'end_ppm': c + FWHM}


def _measured(y, own_c, others=(), local_baseline=True):
    """Area of the signal at ``own_c`` as the engine reports it: integrated,
    the known neighbours' leak taken out, divided by what the window holds."""
    own = [_peak(1, own_c)]
    oth = [_peak(2 + i, c) for i, (c, _a) in enumerate(others)]
    info = integrate_signal(own, PPM, y.reshape(-1, 1), linewidth_ppm=FWHM,
                            other_peaks=oth, local_baseline=local_baseline)
    lo, hi = info['window']
    anc = info['baseline_anchors']
    anc2 = anc[:2] if anc else None
    cap = lorentzian_captured_fraction([own_c], [1.0], lo, hi, FWHM / 2.0,
                                       min_trusted=0.0,
                                       segments=info['segments'],
                                       anchors=anc2)
    leak = neighbour_leak((lo, hi), [([c], [1.0], a) for c, a in others],
                          FWHM / 2.0, segments=info['segments'],
                          anchors=anc2)
    return (info['areas'][0] - leak) / cap, info


@pytest.mark.parametrize('sep', [2.0, 3.0, 4.0])
@pytest.mark.parametrize('big', [1.0, 5.0, 20.0])
def test_a_signal_beside_a_larger_one_keeps_its_area(sep, big):
    c, n = 3.545, 3.545 + sep * FWHM
    area, _ = _measured(_lor(c, 1.0) + _lor(n, big), c, [(n, big)])
    assert area == pytest.approx(1.0, rel=0.02)


def test_the_anchors_are_past_the_neighbour_not_on_its_flank():
    c, n = 3.545, 3.545 + 2.0 * FWHM
    _, info = _measured(_lor(c, 1.0) + _lor(n, 5.0), c, [(n, 5.0)])
    lo_a, hi_a, half = info['baseline_anchors']
    assert hi_a - half >= n + WINDOW_REACH_LINEWIDTHS * FWHM - 1e-12
    assert lo_a + half <= c - WINDOW_REACH_LINEWIDTHS * FWHM + 1e-12
    assert info['window'][1] < n        # the window itself did not grow


def test_a_sloped_background_is_removed():
    c = 3.545
    y = _lor(c, 1.0) + 20.0 + 300.0 * (PPM - c)
    area, _ = _measured(y, c)
    assert area == pytest.approx(1.0, rel=0.02)
    off, _ = _measured(y, c, local_baseline=False)
    assert off > 1.05                  # without it the background is area


def test_background_and_neighbour_together():
    c, n = 3.545, 3.545 + 3.0 * FWHM
    y = _lor(c, 1.0) + _lor(n, 5.0) + 20.0 + 300.0 * (PPM - c)
    area, _ = _measured(y, c, [(n, 5.0)])
    assert area == pytest.approx(1.0, rel=0.02)


def test_an_isolated_window_reads_its_own_ends():
    reach = WINDOW_REACH_LINEWIDTHS * FWHM
    lo, hi = 3.545 - reach, 3.545 + reach
    a, b, half = baseline_anchors(lo, hi, [3.545], FWHM)
    assert half == pytest.approx(BASELINE_STRETCH_LINEWIDTHS * FWHM / 2.0)
    assert a == pytest.approx(lo - half)
    assert b == pytest.approx(hi + half)


def test_a_crowded_region_stops_at_the_lowest_stretch_on_the_way():
    # peaks every two linewidths, far past the walk's reach: no free stretch
    comb = np.arange(3.40, 3.70, 2.0 * FWHM)
    dip = 3.545 + 12.0 * FWHM          # a gap in the comb, inside the reach
    keep = [c for c in comb if abs(c - dip) > 1.5 * FWHM]
    y = sum(_lor(c, 1.0) for c in keep)
    a, b, _ = baseline_anchors(3.544, 3.546, keep, FWHM, trace=(PPM, y))
    assert b - 3.546 <= ANCHOR_MAX_REACH_LINEWIDTHS * FWHM + 1e-9
    assert b == pytest.approx(dip, abs=FWHM)


def test_a_crowded_region_without_a_trace_falls_back_to_the_ends():
    comb = list(np.arange(3.40, 3.70, 2.0 * FWHM))
    a, b, half = baseline_anchors(3.544, 3.546, comb, FWHM)
    assert a == pytest.approx(3.544 - half)
    assert b == pytest.approx(3.546 + half)


def test_levels_are_clipped_at_zero_and_refused_off_the_spectrum():
    y = -np.ones_like(PPM)
    (_, v0), (_, v1) = anchor_levels(PPM, y, (3.5, 3.6, FWHM))
    assert v0 == 0.0 and v1 == 0.0
    assert anchor_levels(PPM, y, (3.0, 3.6, FWHM)) is None


def test_the_plot_draws_the_line_the_area_was_measured_against():
    c = 3.545
    background = 20.0 + 300.0 * (PPM - c)
    y = _lor(c, 1.0) + background
    anchors = baseline_anchors(3.535, 3.555, [c], FWHM, trace=(PPM, y))
    x = PPM[(PPM >= 3.535) & (PPM <= 3.555)]
    assert np.allclose(baseline_under(x, PPM, background, anchors),
                       20.0 + 300.0 * (x - c), atol=0.01)
    # the integral is the trace less that line
    base = baseline_under(x, PPM, y, anchors)
    got = integrate_sample(PPM, y, 3.535, 3.555, anchors=anchors, clip=False)
    want = np.trapezoid(np.interp(x, PPM, y) - base, x=x)
    assert got == pytest.approx(want, rel=1e-6)


def test_the_model_draws_one_baseline_under_all_segments():
    # a Lorentzian integrated on two pieces with one line through the span's
    # ends: the model must describe what integrate_sample returns
    c = 3.545
    y = _lor(c, 1.0)
    segs = [(3.538, 3.543), (3.547, 3.552)]
    got = integrate_sample(PPM, y, 3.538, 3.552, segments=segs)
    f = fraction_in_window([c], [1.0], 3.538, 3.552, FWHM / 2.0,
                           local_baseline=True, segments=segs)
    assert got == pytest.approx(f, rel=0.01)


def test_the_spectrum_wide_baseline_is_the_background_under_a_cluster():
    c, n = 3.545, 3.545 + 2.0 * FWHM
    background = 20.0 + 300.0 * (PPM - c)
    y = _lor(c, 1.0) + _lor(n, 5.0) + background
    B = cluster_baseline(PPM, np.column_stack([y, background]), [c, n], FWHM)
    inside = (PPM >= c - 0.01) & (PPM <= n + 0.01)
    # read past the cluster, where the tails are small: the background,
    # plus the few percent the Lorentzians still stand there
    assert np.allclose(B[inside, 1], background[inside], atol=0.01)
    tallest = float((y - background).max())
    assert np.all(B[inside, 0] - background[inside] < 0.02 * tallest)
    # and never the valley between the two lines
    valley = (c + n) / 2.0
    k = int(np.argmin(np.abs(PPM - valley)))
    assert B[k, 0] < 0.5 * (y[k] - background[k]) + background[k]


def test_the_spectrum_wide_baseline_matches_the_window_anchors():
    c = 3.545
    y = _lor(c, 1.0) + 20.0 + 300.0 * (PPM - c)
    reach = WINDOW_REACH_LINEWIDTHS * FWHM
    anchors = baseline_anchors(c - reach, c + reach, [c], FWHM)
    x = PPM[(PPM >= c - reach) & (PPM <= c + reach)]
    B = cluster_baseline(PPM, y, [c], FWHM)[:, 0]
    assert np.allclose(np.interp(x, PPM, B), baseline_under(x, PPM, y, anchors),
                       atol=1e-6)


def test_no_linewidth_or_no_peak_no_spectrum_wide_baseline():
    assert cluster_baseline(PPM, _lor(3.5, 1.0), [], FWHM) is None
    assert cluster_baseline(PPM, _lor(3.5, 1.0), [3.5], 0.0) is None


def test_a_free_foot_one_reach_from_its_apex_does_not_hold_the_walk():
    # the foot sits exactly WINDOW_REACH_LINEWIDTHS from the apex, and
    # rounding put the apex a hair inside it: the walk spun forever
    reach = WINDOW_REACH_LINEWIDTHS * FWHM
    for apex in np.linspace(0.5, 9.5, 2001):
        a, b, _ = baseline_anchors(apex - reach, apex + reach, [apex], FWHM)
        assert a < apex - reach and b > apex + reach


def test_a_fitted_row_carries_the_shape_it_was_measured_with():
    from processing.quant_cascade import Column, fit_curve_record

    class _Comp:
        def __init__(self, c):
            self.c, self.lo, self.hi = c, c - 0.025, c + 0.025
            self.shift_ppm, self.width_scale, self.lines = 0.0, 1.0, [c]

        def model(self, x, shift, width):
            g = FWHM * width / 2.0
            return (g / np.pi) / ((x - self.c - shift) ** 2 + g ** 2)

    col = Column('m1', 'test')
    col.parts += [(_Comp(3.545), 3.0), (_Comp(3.70), 1.0)]
    background = np.column_stack([np.full(PPM.size, 7.0)] * 3)
    amps = {0: 2.0, 1: 0.0, 2: 9.0}
    rec = fit_curve_record(col, PPM, (3.54, 3.55), amps,
                           {'A': [0, 1], 'B': [2]}, label='Joint fit',
                           baseline=background, offsets=[(3.545, 0.0),
                                                         (3.70, FWHM)],
                           offset_limit=FWHM)
    # the table's statistic: mean over the samples where it was found
    assert rec['amp_by_group'] == {'A': 2.0, 'B': 9.0}
    assert rec['n_by_group'] == {'A': [1, 2], 'B': [1, 1]}
    assert np.allclose(rec['baseline_by_group']['A'], 7.0)
    # the signal in focus is the elected one, drawn whole
    assert rec['focus'] == [3.545, 3.545] and rec['focus_matches_window']
    x, u = np.asarray(rec['ppm']), np.asarray(rec['unit_by_group']['A'])
    near = np.abs(x - 3.545) < 0.02
    o = np.argsort(x[near])
    assert np.trapezoid(u[near][o], x=x[near][o]) == pytest.approx(3.0,
                                                                   rel=0.05)
    # the signal held at the edge of the fit's reach is named
    assert rec['at_limit'] == [3.70] and rec['n_signals'] == 2
    far = fit_curve_record(col, PPM, (3.60, 3.61), amps, {'A': [0, 1]},
                           label='Cascade fit')
    assert far['baseline_by_group'] is None
    assert not far['focus_matches_window']


def test_a_fit_that_misplaced_the_shape_does_not_replace_the_integration():
    from processing.quant_engine import joint_fit_refusal
    ok = {'at_limit': [], 'focus_matches_window': True, 'n_signals': 2,
          'focus': [3.40, 3.44]}
    assert joint_fit_refusal(ok) is None
    assert joint_fit_refusal(None) is None
    held = dict(ok, at_limit=[3.426])
    assert 'edge' in joint_fit_refusal(held) and '3.426' in joint_fit_refusal(held)
    other = dict(ok, focus_matches_window=False)
    assert 'not on the lines the election found' in joint_fit_refusal(other)
