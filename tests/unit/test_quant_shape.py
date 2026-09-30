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

    def test_a_shape_with_no_apex_of_its_own_is_not_quantified_and_says_so(self):
        """A shape competes for no peak and is not fitted alone: elected
        unopposed on a fit that any shape passes, it was worth 41 false
        positives on the mixture bench. Without a free apex it has a row
        that says what it would take."""
        from processing.quantification import SHAPE_UNIDENTIFIED
        shape = Candidate(met_id='S', peak_rank=None, line_ranks=frozenset(),
                          evidence=_shape_ev(explained=0.95))
        X = _study([100.0, 200.0, 400.0])
        rows = quantify(candidates=[shape], peaks_by_rank={}, ppm=PPM,
                        intensities=X, samples_info=[], included=[0, 1, 2],
                        groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
                        noise_sigma=SIGMA, deconvolve=False)
        row = next(r for r in rows if r['hmdb_id'] == 'S')
        assert row['_not_quantified']
        assert row['_rejection'] == SHAPE_UNIDENTIFIED
        assert 'apex' in row['_status_reasons'][0]

    def test_an_eligible_analyte_only_in_the_remainder_pool_keeps_a_row(self):
        """The application hands every shape to the remainder pool, never to
        the election. One that no remainder takes had no step that wrote its
        row, and 16 eligible analytes left the bench's table that way."""
        from processing.quantification import SHAPE_UNIDENTIFIED
        shape = Candidate(met_id='S', peak_rank=None, line_ranks=frozenset(),
                          evidence={**_shape_ev(explained=0.95),
                                    'eligible': True})
        stranger = Candidate(met_id='N', peak_rank=None, line_ranks=frozenset(),
                             evidence={**_shape_ev(), 'eligible': False})
        rows = quantify(candidates=[], peaks_by_rank={}, ppm=PPM,
                        intensities=_study([100.0, 200.0, 400.0]),
                        samples_info=[], included=[0, 1, 2],
                        groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
                        noise_sigma=SIGMA, deconvolve=False,
                        residual_pool=[shape, stranger])
        by = {r['hmdb_id']: r for r in rows}
        assert by['S']['_not_quantified']
        assert by['S']['_rejection'] == SHAPE_UNIDENTIFIED
        assert 'N' not in by

    def test_a_shape_on_a_free_apex_is_quantified_at_the_apex_s_amount(self):
        """With the resolution on, a detected apex nobody holds, at which the
        shape scaled to it fits under the trace over its whole width,
        identifies the analyte; its amount follows the apex, sample by
        sample."""
        # a second signal of the analyte, a singlet at 3.35, is in the
        # study at the same amount: alone on one apex the analyte must
        # have another signal seen at the amount this one implies
        second = lines_model(PPM, [(0.0, 1.0)], 3.35, FWHM, FIELD)
        X = _study([100.0, 200.0, 400.0])
        X = X + np.column_stack([a * second for a in (100.0, 200.0, 400.0)])
        shape = Candidate(met_id='S', peak_rank=7, line_ranks=frozenset({7}),
                          evidence={**_shape_ev(explained=0.95),
                                    'db_ppm': ORIGIN, 'expected_ppm': ORIGIN,
                                    'predicted_ppms': [3.185, 3.2, 3.213]},
                          payload={'met': {'db_peaks': [
                              {'ppm_center': 3.35, 'type': 's', 'protons': 1}]}})
        peaks = {7: {'rank': 7, 'ppm': 3.2, 'height': float(X[:, 2].max())}}
        rows = quantify(candidates=[shape], peaks_by_rank=peaks, ppm=PPM,
                        intensities=X, samples_info=[], included=[0, 1, 2],
                        groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
                        noise_sigma=SIGMA, deconvolve=False,
                        resolve_shared=True, linewidth_ppm=FWHM / FIELD)
        row = next(r for r in rows if r['hmdb_id'] == 'S')
        assert not row['_not_quantified']
        assert row['_by_residual'] == 'spectrum'
        assert row['_confidence'] == 1
        a = row['_areas_per_sample']
        assert a[2] / a[0] == pytest.approx(4.0, rel=0.15)
        assert any('free apex' in r for r in row['_status_reasons'])


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
        # on: not quantified either (no free apex, resolution off), but the
        # row names what it would take; off: the switch is named
        from processing.quantification import SHAPE_UNIDENTIFIED
        assert on['S']['_not_quantified']
        assert on['S']['_rejection'] == SHAPE_UNIDENTIFIED
        assert off['S']['_not_quantified']
        assert off['S']['_rejection'] == SHAPES_OFF
        assert 'Quantify complex multiplets' in off['S']['_status_reasons'][0]
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


class TestTheSwitchCoversEveryComplexMultiplet:
    """Off means no m* competes -- not only the ones fitted to the trace. A
    pattern matched on the peak list carries no shape_fit but is a complex
    multiplet all the same, and with the switch off it took peaks from real
    triplets on the mixture bench."""

    def test_a_peak_list_pattern_is_left_out_with_the_switch_off(self):
        from processing.quantification import SHAPES_OFF
        peaks = {1: {'rank': 1, 'ppm': 3.30, 'start_ppm': 3.294, 'end_ppm': 3.306},
                 2: {'rank': 2, 'ppm': 3.31, 'start_ppm': 3.304, 'end_ppm': 3.316}}
        pattern = Candidate(met_id='P', peak_rank=1, line_ranks=frozenset({1, 2}),
                            evidence={'multiplicity': 'm*', 'delta_j_hz': 0.02,
                                      'ppm_diff': 0.001, 'n_found': 3,
                                      'n_expected': 3, 'protons': 1.0})
        triplet = Candidate(met_id='T', peak_rank=1, line_ranks=frozenset({1, 2}),
                            evidence={'multiplicity': 't', 'delta_j_hz': 0.10,
                                      'ppm_diff': 0.002, 'n_found': 3,
                                      'n_expected': 3, 'protons': 1.0})
        X = _study([100.0, 200.0, 400.0])
        kw = dict(peaks_by_rank=peaks, ppm=PPM, intensities=X,
                  samples_info=[], included=[0, 1, 2], groups={'G': [0, 1, 2]},
                  dj_gold_hz=0.2, noise_sigma=SIGMA, deconvolve=False)
        on = {r['hmdb_id']: r for r in quantify(candidates=[pattern, triplet], **kw)}
        off = {r['hmdb_id']: r for r in quantify(candidates=[pattern, triplet],
                                                  quantify_shapes=False, **kw)}
        # on or off, a pattern never competes for a peak: the triplet has
        # its peak both ways, and the pattern's row says why it has none
        from processing.quantification import SHAPE_UNIDENTIFIED
        assert on['P']['_rejection'] == SHAPE_UNIDENTIFIED
        assert not on['T']['_not_quantified']
        assert off['P']['_rejection'] == SHAPES_OFF
        assert not off['T']['_not_quantified']
