"""A measured pattern is fitted to the spectrum as a shape: the largest copy
of it that fits under the trace, wherever in the tolerance it fits."""
import numpy as np
import pytest

from processing.pattern_shape import (
    fit_pattern_shape, simulate, trace_apices_near)

FIELD = 600.0
PPM = np.linspace(2.9, 3.5, 6000)            # 0.06 Hz per point
FWHM = 2.6
SIGMA = 1.0
NAD = [(-18.0, 0.3), (-12.0, 0.6), (-6.5, 1.0), (-1.0, 0.9), (4.0, 0.7),
       (9.5, 0.5), (15.0, 0.3)]


def _trace(lines, origin, height, *, seed=0, extra=()):
    y = height * simulate(PPM, lines, origin, FWHM, FIELD)
    for o, h in extra:
        y += h * simulate(PPM, [(0.0, 1.0), (0.01, 0.0)], o, FWHM, FIELD)
    return y + np.random.default_rng(seed).normal(0, SIGMA, PPM.size)


def test_the_pattern_alone_is_found_where_it_is_and_fully_explained():
    y = _trace(NAD, 3.217, 200.0)
    fit = fit_pattern_shape(PPM, y, NAD, db_ppm=3.2, field_mhz=FIELD,
                            fwhm_hz=FWHM, tolerance_ppm=0.05, sigma=SIGMA)
    assert fit['origin_ppm'] == pytest.approx(3.217, abs=0.0005)
    assert fit['shift_ppm'] == pytest.approx(0.017, abs=0.0005)
    assert fit['amplitude'] == pytest.approx(200.0, rel=0.05)
    assert fit['snr'] == pytest.approx(200.0, rel=0.05)
    assert fit['explained'] > 0.9
    assert fit['r2'] > 0.95


def test_a_neighbour_on_top_lowers_explained_but_not_the_amplitude():
    y = _trace(NAD, 3.217, 200.0, extra=[(3.217 - 6.5 / FIELD, 400.0)])
    fit = fit_pattern_shape(PPM, y, NAD, db_ppm=3.2, field_mhz=FIELD,
                            fwhm_hz=FWHM, tolerance_ppm=0.05, sigma=SIGMA)
    assert fit['amplitude'] == pytest.approx(200.0, rel=0.15)
    assert fit['explained'] < 0.85


def test_a_missing_line_caps_the_amplitude():
    """One line of the pattern absent from the trace: the copy that fits
    under it is as tall as that gap allows -- nearly nothing."""
    short = [l for l in NAD if l[0] != 4.0]
    y = _trace(short, 3.217, 200.0)
    fit = fit_pattern_shape(PPM, y, NAD, db_ppm=3.2, field_mhz=FIELD,
                            fwhm_hz=FWHM, tolerance_ppm=0.05, sigma=SIGMA)
    assert fit['amplitude'] < 60.0


def test_noise_alone_holds_no_pattern():
    y = np.random.default_rng(3).normal(0, SIGMA, PPM.size)
    fit = fit_pattern_shape(PPM, y, NAD, db_ppm=3.2, field_mhz=FIELD,
                            fwhm_hz=FWHM, tolerance_ppm=0.05, sigma=SIGMA)
    assert fit['snr'] < 5.0


def test_apices_are_where_the_lines_merge_at_this_width():
    y = _trace(NAD, 3.217, 200.0)
    fit = fit_pattern_shape(PPM, y, NAD, db_ppm=3.2, field_mhz=FIELD,
                            fwhm_hz=FWHM, tolerance_ppm=0.05, sigma=SIGMA)
    # at 2.6 Hz the seven lines of the pattern show fewer than seven apices
    assert 2 <= len(fit['apex_ppms']) <= 7
    found = trace_apices_near(PPM, y, fit['apex_ppms'], tol_hz=2.0,
                              field_mhz=FIELD)
    assert all(f is not None for f in found)


def test_no_width_or_no_trace_is_no_fit():
    y = _trace(NAD, 3.217, 200.0)
    assert fit_pattern_shape(PPM, y, NAD, db_ppm=3.2, field_mhz=FIELD,
                             fwhm_hz=0.0, tolerance_ppm=0.05,
                             sigma=SIGMA) is None
    assert fit_pattern_shape(PPM, y, NAD, db_ppm=9.0, field_mhz=FIELD,
                             fwhm_hz=FWHM, tolerance_ppm=0.05,
                             sigma=SIGMA) is None
    assert fit_pattern_shape(PPM, y, [(0.0, 1.0)], db_ppm=3.2,
                             field_mhz=FIELD, fwhm_hz=FWHM,
                             tolerance_ppm=0.05, sigma=SIGMA) is None


class TestThePatternMeasuredOnTheTrace:
    """measure_pattern_signal fits the shape when the trace is handed over,
    and reads the lines off the trace's own maxima."""

    def _mp(self):
        import json
        from processing.annotation_scoring import PATTERN_MULT
        return {'db_ppm': 3.2, 'type': PATTERN_MULT,
                'pattern_lines': json.dumps([list(l) for l in NAD])}

    def test_present_pattern_is_complete_with_its_fit(self):
        from processing.annotation_scoring import measure_pattern_signal
        y = _trace(NAD, 3.217, 200.0)
        m = measure_pattern_signal(self._mp(), [], field_mhz=FIELD,
                                   dj_gold_hz=0.2, anchor_ppm=3.217,
                                   fwhm_hz=FWHM, trace=(PPM, y), sigma=SIGMA,
                                   tolerance_ppm=0.05, floor=5.0)
        assert m.measured and m.complete
        assert m.pattern_snr == pytest.approx(200.0, rel=0.1)
        assert m.pattern_explained > 0.9
        assert m.origin_ppm == pytest.approx(3.217, abs=0.0005)
        assert m.pattern_residual_hz is not None and m.pattern_residual_hz < 0.5
        assert m.delta_j_hz is not None

    def test_absent_pattern_finds_nothing(self):
        from processing.annotation_scoring import measure_pattern_signal
        y = np.random.default_rng(5).normal(0, SIGMA, PPM.size)
        m = measure_pattern_signal(self._mp(), [], field_mhz=FIELD,
                                   dj_gold_hz=0.2, anchor_ppm=3.217,
                                   fwhm_hz=FWHM, trace=(PPM, y), sigma=SIGMA,
                                   tolerance_ppm=0.05, floor=5.0)
        assert m.measured and not m.complete and m.n_found == 0

    def test_without_a_trace_the_peak_list_is_used(self):
        from processing.annotation_scoring import measure_pattern_signal
        peaks = [{'ppm': 3.217 + o / FIELD, 'height': 100 * r}
                 for o, r in NAD]
        m = measure_pattern_signal(self._mp(), peaks, field_mhz=FIELD,
                                   dj_gold_hz=0.2, anchor_ppm=3.217,
                                   merge_hz=0.0)
        assert m.complete and m.pattern_snr is None
