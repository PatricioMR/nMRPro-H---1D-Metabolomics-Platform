"""A complex multiplet is quantified from its fitted shape: no peak claimed,
the copy of the shape that fits under each sample's trace, its area exact."""
import numpy as np
import pytest

from processing.pattern_shape import encode_profile, lines_model
from processing.quant_engine import quantify
from processing.quant_shape import measure_shape, shape_model
from processing.quantification import Candidate, elect

FIELD = 600.0
PPM = np.linspace(3.0, 3.4, 4000)                    # 0.06 Hz per point
LINES = [(-9.0, 0.7), (-1.0, 1.0), (0.5, 0.9), (8.0, 0.8)]
ORIGIN = 3.2
FWHM = 2.6
SIGMA = 1.0


def _profile():
    offs = np.arange(-20.0, 20.0, 0.1)
    return offs, lines_model(offs / FIELD, LINES, 0.0, 1.2, FIELD)


def _shape_ev(*, explained=0.95, protons=1.0):
    offs, h = _profile()
    return {'multiplicity': 'm*', 'delta_j_hz': 0.0, 'ppm_diff': 0.001,
            'n_found': 3, 'n_expected': 3, 'protons': protons,
            'shape_fit': {'origin_ppm': ORIGIN, 'field_mhz': FIELD,
                          'study_linewidth_hz': FWHM,
                          'pattern_profile': encode_profile(offs, h),
                          'profile_fwhm_hz': 1.2, 'pattern_lines': LINES,
                          'snr': 80.0, 'explained': explained,
                          'apex_ppms': [3.185, 3.2, 3.213],
                          'apex_rel': [0.7, 1.0, 0.8]}}


def _study(amps, *, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    cols = []
    for a in amps:
        y = a * lines_model(PPM, LINES, ORIGIN, FWHM, FIELD)
        if extra is not None:
            y = y + extra
        cols.append(y + rng.normal(0, SIGMA, PPM.size))
    return np.column_stack(cols)


class TestMeasureShape:
    def test_the_area_follows_the_amplitude(self):
        X = _study([100.0, 200.0, 400.0])
        out = measure_shape(_shape_ev(), PPM, X, included=[0, 1, 2],
                            noise_sigma=SIGMA)
        a = out['areas']
        assert a[1] / a[0] == pytest.approx(2.0, rel=0.08)
        assert a[2] / a[0] == pytest.approx(4.0, rel=0.08)
        assert out['snr'][2] == pytest.approx(400.0, rel=0.08)
        assert all(out['detected'].values())
        assert out['captured'] is None and out['shape'] is True
        lo, hi = out['window']
        assert lo < ORIGIN - 9.0 / FIELD and hi > ORIGIN + 8.0 / FIELD

    def test_the_area_is_the_amplitude_times_the_shapes_own_area(self):
        X = _study([200.0])
        x, m, _w = shape_model(_shape_ev(), PPM)
        area_m = float(np.trapezoid(m[np.argsort(x)], x=np.sort(x)))
        out = measure_shape(_shape_ev(), PPM, X, included=[0],
                            noise_sigma=SIGMA)
        assert out['areas'][0] == pytest.approx(200.0 * area_m, rel=0.08)

    def test_a_sample_without_the_analyte_is_not_detected(self):
        X = _study([300.0, 0.0])
        out = measure_shape(_shape_ev(), PPM, X, included=[0, 1],
                            noise_sigma=SIGMA)
        assert out['detected'][0] and not out['detected'][1]
        assert out['snr'][1] < 3.0

    def test_the_lines_serve_without_a_profile(self):
        ev = _shape_ev()
        ev['shape_fit']['pattern_profile'] = None
        X = _study([100.0, 300.0])
        out = measure_shape(ev, PPM, X, included=[0, 1], noise_sigma=SIGMA)
        assert out['areas'][1] / out['areas'][0] == pytest.approx(3.0, rel=0.1)

    def test_blanks_are_subtracted(self):
        X = _study([300.0, 300.0, 30.0])
        out = measure_shape(_shape_ev(), PPM, X, included=[0, 1],
                            noise_sigma=SIGMA, blank_indices=[2])
        raw = measure_shape(_shape_ev(), PPM, X, included=[0, 1],
                            noise_sigma=SIGMA)
        assert out['areas'][0] < raw['areas'][0]
        assert np.isfinite(out['blank_area'])

    def test_no_noise_or_no_shape_is_no_measurement(self):
        X = _study([100.0])
        assert measure_shape(_shape_ev(), PPM, X, noise_sigma=None) is None
        assert measure_shape({'shape_fit': {}}, PPM, X,
                             noise_sigma=SIGMA) is None


class TestTheShapeInTheEngine:
    def test_a_shape_is_a_detection_that_claims_no_peak(self):
        shape = Candidate(met_id='S', peak_rank=None, line_ranks=frozenset(),
                          evidence=_shape_ev())
        other = Candidate(met_id='D', peak_rank=1, line_ranks=frozenset({1, 2}),
                          evidence={'multiplicity': 'd', 'delta_j_hz': 0.05,
                                    'ppm_diff': 0.001, 'n_found': 2,
                                    'n_expected': 2, 'protons': 1.0})
        e = elect([shape, other], dj_gold_hz=0.2)
        assert 'S' in e.winners and 'D' in e.winners
        assert not e.claimed_peaks & {None}

    def test_the_row_is_quantified_from_the_shape(self):
        shape = Candidate(met_id='S', peak_rank=None, line_ranks=frozenset(),
                          evidence=_shape_ev(explained=0.95))
        X = _study([100.0, 200.0, 400.0])
        rows = quantify(candidates=[shape], peaks_by_rank={}, ppm=PPM,
                        intensities=X, samples_info=[], included=[0, 1, 2],
                        groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
                        noise_sigma=SIGMA, deconvolve=False)
        row = next(r for r in rows if r['hmdb_id'] == 'S')
        assert not row['_not_quantified']
        assert row['_integ_mode'] == 'Shape'
        assert row['_peak_rank'] is None
        a = row['_areas_per_sample']
        assert a[2] / a[0] == pytest.approx(4.0, rel=0.1)
        assert any('fitted shape' in r for r in row['_status_reasons'])

    def test_a_shared_region_is_reported_as_an_upper_bound(self):
        shape = Candidate(met_id='S', peak_rank=None, line_ranks=frozenset(),
                          evidence=_shape_ev(explained=0.5))
        X = _study([100.0, 200.0, 400.0])
        rows = quantify(candidates=[shape], peaks_by_rank={}, ppm=PPM,
                        intensities=X, samples_info=[], included=[0, 1, 2],
                        groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
                        noise_sigma=SIGMA, deconvolve=False)
        row = next(r for r in rows if r['hmdb_id'] == 'S')
        assert any('Upper bound' in r for r in row['_status_reasons'])


class TestTheShapeSwitch:
    def _cands(self):
        shape = Candidate(met_id='S', peak_rank=None, line_ranks=frozenset(),
                          evidence=_shape_ev())
        other = Candidate(met_id='D', peak_rank=1, line_ranks=frozenset({1, 2}),
                          evidence={'multiplicity': 'd', 'delta_j_hz': 0.05,
                                    'ppm_diff': 0.001, 'n_found': 2,
                                    'n_expected': 2, 'protons': 1.0})
        return shape, other

    def _peaks(self):
        return {1: {'rank': 1, 'ppm': 3.30, 'start_ppm': 3.294, 'end_ppm': 3.306},
                2: {'rank': 2, 'ppm': 3.31, 'start_ppm': 3.304, 'end_ppm': 3.316}}

    def test_off_reports_the_shape_as_not_quantified_and_moves_nothing_else(self):
        from processing.quantification import SHAPES_OFF
        shape, other = self._cands()
        X = _study([100.0, 200.0, 400.0])
        X[:, :] += 50.0 * (lines_model(PPM, [(-3.0, 1.0), (3.0, 1.0)], 3.305,
                                       FWHM, FIELD))[:, None]
        kw = dict(peaks_by_rank=self._peaks(), ppm=PPM, intensities=X,
                  samples_info=[], included=[0, 1, 2], groups={'G': [0, 1, 2]},
                  dj_gold_hz=0.2, noise_sigma=SIGMA, deconvolve=False)
        on = {r['hmdb_id']: r for r in quantify(candidates=[shape, other], **kw)}
        diag = {}
        off = {r['hmdb_id']: r for r in quantify(candidates=[shape, other],
                                                  quantify_shapes=False,
                                                  diagnostics=diag, **kw)}
        assert not on['S']['_not_quantified']
        assert off['S']['_not_quantified']
        assert off['S']['_rejection'] == SHAPES_OFF
        assert 'shape fit' in off['S']['_status_reasons'][0]
        assert off['D']['_areas_per_sample'] == on['D']['_areas_per_sample']
        assert diag['shapes'] == {'off': True, 'left_out': 1}

    def test_off_a_metabolite_with_a_peak_signal_still_competes_on_it(self):
        shape, other = self._cands()
        same = Candidate(met_id='S', peak_rank=1, line_ranks=frozenset({1, 2}),
                         evidence=other.evidence)
        rows = quantify(candidates=[shape, same], peaks_by_rank=self._peaks(),
                        ppm=PPM, intensities=_study([100.0]), samples_info=[],
                        included=[0], groups={'G': [0]}, dj_gold_hz=0.2,
                        noise_sigma=SIGMA, deconvolve=False,
                        quantify_shapes=False)
        row = next(r for r in rows if r['hmdb_id'] == 'S')
        assert row['_integ_mode'] == 'Multiplet'
