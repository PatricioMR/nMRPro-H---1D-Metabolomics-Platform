"""A multiplet's measured shape fitted to the spectrum: the largest copy of
it that fits under the trace, wherever in the tolerance it fits."""
import json

import numpy as np
import pytest

from processing.pattern_shape import (
    apex_support, encode_profile, fit_shape, lines_model, parse_profile,
    profile_model)

FIELD = 600.0
PPM = np.linspace(2.9, 3.5, 6000)            # 0.06 Hz per point
FWHM = 2.6                                   # the study's
REF_FWHM = 1.2                               # the reference's
SIGMA = 1.0
NAD = [(-18.0, 0.3), (-12.0, 0.6), (-6.5, 1.0), (-1.0, 0.9), (4.0, 0.7),
       (9.5, 0.5), (15.0, 0.3)]


def _trace(lines, origin, height, *, fwhm=FWHM, seed=0, extra=()):
    y = height * lines_model(PPM, lines, origin, fwhm, FIELD)
    for o, h in extra:
        y += h * lines_model(PPM, [(0.0, 1.0), (0.01, 0.0)], o, fwhm, FIELD)
    return y + np.random.default_rng(seed).normal(0, SIGMA, PPM.size)


def _profile(lines=NAD):
    """The reference's own trace over the multiplet, at its width."""
    offs = np.arange(-30.0, 30.0, 0.1)
    h = lines_model(offs / FIELD, lines, 0.0, REF_FWHM, FIELD)
    return offs, h


def _fit(y, **kw):
    kw.setdefault('db_ppm', 3.2)
    kw.setdefault('field_mhz', FIELD)
    kw.setdefault('fwhm_hz', FWHM)
    kw.setdefault('tolerance_ppm', 0.05)
    kw.setdefault('sigma', SIGMA)
    return fit_shape(PPM, y, **kw)


class TestTheProfile:
    def test_round_trips_through_the_database_encoding(self):
        offs, h = _profile()
        back = parse_profile(encode_profile(offs, h))
        assert back is not None
        assert back[0][0] == pytest.approx(offs[0], abs=1e-3)
        assert len(back[1]) == len(h)
        assert back[1].max() == pytest.approx(1.0)
        assert parse_profile(None) is None
        assert parse_profile('{"start": 0, "h": [1]}') is None

    def test_broadening_widens_and_keeps_the_top_at_one(self):
        prof = _profile()
        narrow = profile_model(PPM, prof, 3.2, 0.0, FIELD)
        wide = profile_model(PPM, prof, 3.2, 3.0, FIELD)
        assert narrow.max() == pytest.approx(1.0)
        assert wide.max() == pytest.approx(1.0)
        assert (wide > 0.5).sum() > (narrow > 0.5).sum()

    def test_the_apex_support_is_the_tops_not_the_flanks(self):
        m = lines_model(PPM, [(-5.0, 1.0), (5.0, 0.5)], 3.2, 2.0, FIELD)
        sup = apex_support(m)
        assert sup.any()
        assert m[sup].min() >= 0.75 * 0.5 - 1e-9
        assert not sup[m < 0.3].any()


class TestTheFit:
    def test_the_pattern_alone_is_found_where_it_is_and_fully_explained(self):
        y = _trace(NAD, 3.217, 200.0)
        fit = _fit(y, profile=_profile(), profile_fwhm_hz=REF_FWHM)
        assert fit['origin_ppm'] == pytest.approx(3.217, abs=0.0005)
        assert fit['amplitude'] == pytest.approx(200.0, rel=0.05)
        assert fit['snr'] == pytest.approx(200.0, rel=0.05)
        assert fit['explained'] > 0.9
        assert fit['r2'] > 0.9

    def test_the_lines_serve_when_there_is_no_profile(self):
        y = _trace(NAD, 3.217, 200.0)
        fit = _fit(y, lines=NAD)
        assert fit['origin_ppm'] == pytest.approx(3.217, abs=0.0005)
        assert fit['explained'] > 0.9

    def test_a_neighbour_on_top_lowers_explained_not_the_amplitude(self):
        y = _trace(NAD, 3.217, 200.0, extra=[(3.217 - 6.5 / FIELD, 400.0)])
        fit = _fit(y, profile=_profile(), profile_fwhm_hz=REF_FWHM)
        assert fit['amplitude'] == pytest.approx(200.0, rel=0.15)
        assert fit['explained'] < 0.85

    def test_a_missing_line_caps_the_amplitude(self):
        short = [l for l in NAD if l[0] != 4.0]
        y = _trace(short, 3.217, 200.0)
        fit = _fit(y, profile=_profile(), profile_fwhm_hz=REF_FWHM)
        assert fit['amplitude'] < 60.0

    def test_a_study_a_little_narrower_than_the_reference_still_fits(self):
        y = _trace(NAD, 3.217, 200.0, fwhm=1.0)
        fit = _fit(y, profile=_profile(), profile_fwhm_hz=REF_FWHM,
                   fwhm_hz=1.0)
        assert fit['amplitude'] > 150.0
        assert fit['explained'] > 0.8

    def test_noise_alone_holds_no_pattern(self):
        y = np.random.default_rng(3).normal(0, SIGMA, PPM.size)
        fit = _fit(y, profile=_profile(), profile_fwhm_hz=REF_FWHM)
        assert fit['snr'] < 5.0

    def test_apices_at_this_width_and_the_model_callable(self):
        y = _trace(NAD, 3.217, 200.0)
        fit = _fit(y, profile=_profile(), profile_fwhm_hz=REF_FWHM)
        assert 2 <= len(fit['apex_ppms']) <= 7
        assert max(fit['apex_rel']) == pytest.approx(1.0, abs=0.02)
        m = fit['model'](PPM)
        assert m.max() == pytest.approx(1.0)

    def test_no_shape_or_no_trace_is_no_fit(self):
        y = _trace(NAD, 3.217, 200.0)
        assert _fit(y) is None
        assert _fit(y, lines=NAD, fwhm_hz=0.0) is None
        assert _fit(y, lines=NAD, db_ppm=9.0) is None
        assert _fit(y, lines=[(0.0, 1.0)]) is None


class TestThePatternMeasuredOnTheTrace:
    """measure_pattern_signal judges a pattern on its shape alone when the
    trace is handed over: here or not (SNR), alone or shared (explained)."""

    def _mp(self, with_profile=True):
        from processing.annotation_scoring import PATTERN_MULT
        mp = {'db_ppm': 3.2, 'type': PATTERN_MULT,
              'pattern_lines': json.dumps([list(l) for l in NAD])}
        if with_profile:
            mp['pattern_profile'] = encode_profile(*_profile())
            mp['profile_fwhm_hz'] = REF_FWHM
        return mp

    def _measure(self, y, mp, **kw):
        from processing.annotation_scoring import measure_pattern_signal
        return measure_pattern_signal(mp, [], field_mhz=FIELD, dj_gold_hz=0.2,
                                      anchor_ppm=3.217, fwhm_hz=FWHM,
                                      trace=(PPM, y), sigma=SIGMA,
                                      tolerance_ppm=0.05, floor=5.0, **kw)

    def test_present_pattern_is_complete_with_its_fit(self):
        m = self._measure(_trace(NAD, 3.217, 200.0), self._mp())
        assert m.measured and m.complete
        assert m.pattern_snr == pytest.approx(200.0, rel=0.1)
        assert m.pattern_explained > 0.9
        assert m.origin_ppm == pytest.approx(3.217, abs=0.0005)
        assert m.pattern_residual_hz < 0.2       # explained -> near zero
        assert m.delta_j_hz is not None and m.delta_j_hz < 0.1

    def test_a_shared_region_reads_as_average_never_bad(self):
        from processing.annotation_scoring import PATTERN_BAD_HZ
        y = _trace(NAD, 3.217, 200.0, extra=[(3.217 - 6.5 / FIELD, 800.0)])
        m = self._measure(y, self._mp())
        assert m.complete
        assert 0.0 < m.pattern_residual_hz < PATTERN_BAD_HZ

    def test_absent_pattern_finds_nothing(self):
        y = np.random.default_rng(5).normal(0, SIGMA, PPM.size)
        m = self._measure(y, self._mp())
        assert m.measured and not m.complete and m.n_found == 0
        assert m.delta_j_hz is None

    def test_the_lines_serve_without_a_profile(self):
        m = self._measure(_trace(NAD, 3.217, 200.0), self._mp(False))
        assert m.complete and m.pattern_explained > 0.9

    def test_without_a_trace_the_peak_list_is_used(self):
        from processing.annotation_scoring import measure_pattern_signal
        peaks = [{'ppm': 3.217 + o / FIELD, 'height': 100 * r}
                 for o, r in NAD]
        m = measure_pattern_signal(self._mp(False), peaks, field_mhz=FIELD,
                                   dj_gold_hz=0.2, anchor_ppm=3.217,
                                   merge_hz=0.0)
        assert m.complete and m.pattern_snr is None


def test_a_short_profile_survives_a_wide_broadening():
    offs = np.arange(-2.0, 2.0, 0.1)
    h = np.exp(-offs ** 2)
    m = profile_model(PPM, (offs, h), 3.2, 12.0, FIELD)
    assert m.shape == PPM.shape and m.max() == pytest.approx(1.0)
