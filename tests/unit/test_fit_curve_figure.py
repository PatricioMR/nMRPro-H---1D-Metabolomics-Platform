"""What the report draws of a fitted amount is the number's: the shaded area is
the signal the table's area is read from, the compound's other signals in view
are only outlined, and a fit standing above the trace it was fitted to says so."""
import numpy as np
import pytest

from plotting.fit_curve import FIT_ABOVE_TRACE, draw_fit_curve, fit_over_trace
from processing.quant_cascade import Column, fit_curve_record

FWHM = 2.6 / 600.0
PPM = np.linspace(4.5, 0.5, 40000)


class _Comp:
    def __init__(self, c):
        self.c, self.lo, self.hi = c, c - 0.025, c + 0.025
        self.shift_ppm, self.width_scale, self.lines = 0.0, 1.0, [c]

    def model(self, x, shift, width):
        g = FWHM * width / 2.0
        return (g / np.pi) / ((x - self.c - shift) ** 2 + g ** 2)


def _record(amps=None):
    col = Column('m1', 'test')
    col.parts += [(_Comp(3.545), 3.0), (_Comp(3.60), 1.0)]
    return fit_curve_record(col, PPM, (3.54, 3.55), amps or {0: 2.0, 1: 2.0},
                            {'A': [0, 1]}, label='Joint fit')


def _area(rec, key):
    x = np.asarray(rec['ppm'])
    u = np.asarray(rec[key]['A'])
    o = np.argsort(x)
    return float(np.trapezoid(u[o], x=x[o]))


def test_the_record_holds_the_focus_signal_apart_from_the_whole_fit():
    rec = _record()
    # the whole fit is every signal in view; the focus is the elected one,
    # whose area is its protons -- what the table's area is read from
    assert _area(rec, 'unit_by_group') == pytest.approx(4.0, rel=0.05)
    assert _area(rec, 'focus_unit_by_group') == pytest.approx(3.0, rel=0.05)
    assert rec['focus_nh'] == 3.0


class _Ax:
    def __init__(self):
        self.fills, self.lines = [], []

    def fill_between(self, x, lo, hi, **kw):
        self.fills.append((np.asarray(x), np.asarray(lo), np.asarray(hi)))

    def plot(self, x, y, **kw):
        self.lines.append((np.asarray(x), np.asarray(y), kw.get('label')))


def test_only_the_focus_signal_is_shaded_and_the_whole_fit_is_the_line():
    rec = _record()
    ax = _Ax()
    assert draw_fit_curve(ax, rec, 'A', '#43a047')
    x, lo, hi = ax.fills[0]
    shaded = float(np.trapezoid(hi - lo, x=x))
    assert shaded == pytest.approx(2.0 * 3.0, rel=0.05)
    (_lx, ly, label), = ax.lines
    assert float(np.trapezoid(ly, x=x)) == pytest.approx(2.0 * 4.0, rel=0.05)
    assert 'shaded' in label


def test_a_record_without_a_focus_curve_is_still_drawn_whole():
    rec = _record()
    rec.pop('focus_unit_by_group')
    ax = _Ax()
    assert draw_fit_curve(ax, rec, 'A', '#43a047')
    x, lo, hi = ax.fills[0]
    assert float(np.trapezoid(hi - lo, x=x)) == pytest.approx(2.0 * 4.0, rel=0.05)


def _trace_of(rec, factor):
    x = np.asarray(rec['ppm'])
    u = np.asarray(rec['focus_unit_by_group']['A']) * rec['amp_by_group']['A']
    o = np.argsort(x)
    return x[o][::-1], (factor * u[o])[::-1]


def test_a_fit_holding_more_than_the_trace_is_said_so():
    rec = _record()
    x, y = _trace_of(rec, 0.5)
    ratio = fit_over_trace(rec, 'A', x, y)
    assert ratio == pytest.approx(2.0, rel=0.05) and ratio > FIT_ABOVE_TRACE


def test_a_fit_the_trace_holds_is_not():
    rec = _record()
    x, y = _trace_of(rec, 1.0)
    assert fit_over_trace(rec, 'A', x, y) == pytest.approx(1.0, rel=0.05)


def test_a_trace_with_more_than_the_fit_is_left_to_the_neighbours():
    """What stands over the fit may be another analyte's: not a warning."""
    rec = _record()
    x, y = _trace_of(rec, 3.0)
    assert fit_over_trace(rec, 'A', x, y) < 1.0


def test_no_focus_curve_or_no_trace_gives_no_ratio():
    rec = _record()
    x, y = _trace_of(rec, 0.0)
    assert fit_over_trace(rec, 'A', x, y) is None
    rec.pop('focus_unit_by_group')
    assert fit_over_trace(rec, 'A', x, y) is None
    assert fit_over_trace(_record(), 'Z', x, y) is None


# ── a fit made on a residual ────────────────────────────────────────────

def test_the_residual_a_cascade_analyte_was_fitted_to_leaves_the_others_out():
    from processing.quant_cascade import others_removed
    ppm = np.linspace(2.0, 1.0, 1000)
    cols = {}
    for mid, c in (('A', 1.5), ('B', 1.7)):
        col = Column(mid, mid)
        col.parts.append((_Comp(c), 1.0))
        cols[mid] = col
    Y = np.zeros((ppm.size, 2))
    sel = np.arange(100, 900)
    found = {'A': {'shift_ppm': 0.0, 'per_proton': {0: 5.0, 1: 5.0}},
             'B': {'shift_ppm': 0.0, 'per_proton': {0: 7.0, 1: 0.0}}}
    # Y holds A and B as the cascade found them; for A, B is removed and A stays
    for mid, res in found.items():
        for idx, val in cols[mid].part_curves(ppm, 0.0):
            for s, a in res['per_proton'].items():
                Y[idx, s] += a * val
    R = others_removed(Y, ppm, sel, found, cols, 'A')
    a_only = np.zeros_like(Y)
    for idx, val in cols['A'].part_curves(ppm, 0.0):
        a_only[idx, :] += 5.0 * val[:, None]
    assert np.allclose(R, a_only[sel], atol=1e-9)


def _record_with_residual(resid_factor, fit_unit_height=100.0):
    rec = _record()
    x = np.asarray(rec['ppm'])
    u = np.asarray(rec['focus_unit_by_group']['A']) * rec['amp_by_group']['A']
    rec['residual_by_group'] = {'A': (resid_factor * u).tolist()}
    return rec


def test_a_fit_is_held_against_its_residual_not_against_the_trace():
    rec = _record_with_residual(0.5)
    # a trace a thousand times taller (a neighbour) changes nothing
    x = np.asarray(rec['ppm'])[::-1]
    huge = np.full(x.size, 1e9)
    assert fit_over_trace(rec, 'A', x, huge) == pytest.approx(2.0, rel=0.05)


def test_the_residual_is_not_drawn_and_the_axis_is_cut_on_the_fit():
    from plotting.fit_curve import VIEW_MARGIN, fit_view_top
    rec = _record_with_residual(1.0)
    ax = _Ax()
    assert draw_fit_curve(ax, rec, 'A', '#43a047')
    assert all('other analytes' not in (lab or '') for _x, _y, lab in ax.lines)
    top = fit_view_top(rec, 'A')
    amp_top = rec['amp_by_group']['A'] * max(rec['unit_by_group']['A'])
    assert top == pytest.approx(VIEW_MARGIN * amp_top, rel=0.05)


def test_a_few_points_of_a_neighbours_leftover_do_not_set_the_axis():
    from plotting.fit_curve import fit_view_top
    rec = _record_with_residual(1.0)
    base_top = fit_view_top(rec, 'A')
    resid = np.asarray(rec['residual_by_group']['A'])
    support = np.flatnonzero(np.asarray(rec['focus_unit_by_group']['A'])
                             >= 0.02 * max(rec['focus_unit_by_group']['A']))
    resid[support[:2]] = 1e9
    rec['residual_by_group']['A'] = resid.tolist()
    assert fit_view_top(rec, 'A') == pytest.approx(base_top, rel=0.05)


def test_a_fit_without_a_residual_is_cut_on_itself():
    from plotting.fit_curve import VIEW_MARGIN, fit_view_top
    rec = _record()
    amp_top = rec['amp_by_group']['A'] * max(rec['unit_by_group']['A'])
    assert fit_view_top(rec, 'A') == pytest.approx(VIEW_MARGIN * amp_top, rel=0.05)
    assert fit_view_top(rec, 'Z') is None


def test_the_figure_keeps_the_snr_group_unless_the_fit_holds_next_to_none_there():
    from plotting.fit_curve import fit_group
    fit = {'amp_by_group': {'A': 0.0001, 'B': 0.006, 'C': 0.004}}
    assert fit_group(fit, 'A') == 'B'
    assert fit_group(fit, 'C') == 'C'       # more than half of the highest
    assert fit_group(fit, None) == 'B'
    assert fit_group({'amp_by_group': {}}, 'A') == 'A'
