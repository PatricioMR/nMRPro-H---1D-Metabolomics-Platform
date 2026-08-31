"""From spectra to the verdict: fit each sample, agree on δ and J, then score.

The four stages that used to be one number off the mean spectrum:

    fit_across_samples      → which apices are this signal's lines, per sample
    signal_across_samples   → what each sample measured
    consensus               → the signal's own δ and J, median over the samples
    apply_per_sample_consensus → the score uses those instead of the master ones

Real traces, no mocks: the samples are built from Lorentzian lines so the
apices the fit finds are the ones a spectrum actually gives.
"""
import numpy as np
import pytest

from processing.annotation_scoring import apply_per_sample_consensus
from processing.multiplet_line_fit import fit_across_samples
from processing.per_sample_signal import signal_across_samples

MHZ = 600.0
J_DB = 7.0
D = J_DB / MHZ
CENTRE = 1.470
PARAMS = {'dj_gold_hz': 0.20, 'dj_bad_hz': 0.50, 'dj_neutral': 0.0,
          'dppm_gold': 0.05}


def _axis():
    return np.arange(1.60, 1.35, -0.0002)


def _lorentzian(ppm, peaks, width=0.0012):
    y = np.zeros_like(ppm)
    for p, h in peaks:
        y = y + h * (width ** 2) / ((ppm - p) ** 2 + width ** 2)
    return y


def _doublet_sample(ppm, centre, j_hz, height=1.0):
    d = j_hz / MHZ
    return _lorentzian(ppm, [(centre - d / 2, height), (centre + d / 2, height)])


def _line_peaks(n_samples):
    """The two master peaks, carrying only the areas this path still reads."""
    return [{'ppm': CENTRE - D / 2, 'ppm_per_sample': [CENTRE - D / 2] * n_samples,
             'area_per_sample': [1.0] * n_samples},
            {'ppm': CENTRE + D / 2, 'ppm_per_sample': [CENTRE + D / 2] * n_samples,
             'area_per_sample': [1.0] * n_samples}]


def _run(columns, *, included=None, noise_floor=0.0):
    ppm = _axis()
    arr = np.column_stack(columns)
    exp = [CENTRE - D / 2, CENTRE + D / 2]
    fits = fit_across_samples(exp, ppm, arr, included=included, field_mhz=MHZ,
                              dj_bad_hz=PARAMS['dj_bad_hz'],
                              dj_gold_hz=PARAMS['dj_gold_hz'],
                              max_drift_ppm=PARAMS['dppm_gold'],
                              noise_floor=noise_floor)
    return signal_across_samples(_line_peaks(arr.shape[1]), fits=fits,
                                 included=included, db_ppm=CENTRE,
                                 db_j_hz=J_DB, field_mhz=MHZ, n_lines=2)


class TestCleanStudy:
    def test_every_sample_measures_the_coupling(self):
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE, J_DB) for _ in range(4)]
        out = _run(cols)
        assert out['n_present'] == 4
        for row in out['per_sample']:
            assert row['j_obs_hz'] == pytest.approx(J_DB, abs=0.3)

    def test_the_consensus_lands_on_the_database_value(self):
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE, J_DB) for _ in range(4)]
        out = _run(cols)
        assert out['observed_j_hz'] == pytest.approx(J_DB, abs=0.3)
        assert out['delta_j_hz'] < PARAMS['dj_gold_hz']
        assert out['j_n'] == 4


class TestFoldingCannotHappen:
    """The defect that motivated all of this: at 600 MHz a 7 Hz doublet's legs
    sit 0.0117 ppm apart, well inside the ±0.02 ppm the old per-peak apex
    search scanned, so the taller leg won BOTH searches and the doublet was
    recorded twice at one position — J = 0, centre off by half a coupling.
    """

    def test_an_uneven_doublet_still_yields_two_positions(self):
        ppm = _axis()
        d = J_DB / MHZ
        # One leg ten times the other, the shape that used to fold.
        col = _lorentzian(ppm, [(CENTRE - d / 2, 10.0), (CENTRE + d / 2, 1.0)])
        out = _run([col])
        row = out['per_sample'][0]
        assert row['n_found'] == 2
        assert row['j_obs_hz'] == pytest.approx(J_DB, abs=0.4)
        assert row['j_obs_hz'] > 0

    def test_the_centre_is_not_displaced_onto_the_taller_leg(self):
        ppm = _axis()
        d = J_DB / MHZ
        col = _lorentzian(ppm, [(CENTRE - d / 2, 10.0), (CENTRE + d / 2, 1.0)])
        out = _run([col])
        row = out['per_sample'][0]
        assert row['ppm_centre'] == pytest.approx(CENTRE, abs=1e-3)
        # Folding would have put it a half-coupling away.
        assert abs(row['ppm_centre'] - (CENTRE - d / 2)) > d / 4


class TestSamplesWithoutTheAnalyte:
    def test_an_empty_trace_measures_nothing(self):
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE, J_DB), np.zeros_like(ppm)]
        out = _run(cols, noise_floor=0.01)
        assert out['n_present'] == 1
        assert out['per_sample'][1]['ppm_centre'] is None
        assert out['per_sample'][1]['j_obs_hz'] is None

    def test_the_consensus_rests_only_on_what_was_measured(self):
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE, J_DB) for _ in range(3)]
        cols.append(np.zeros_like(ppm))
        out = _run(cols, noise_floor=0.01)
        assert out['j_n'] == 3
        assert out['observed_j_hz'] == pytest.approx(J_DB, abs=0.3)


class TestDriftAcrossSamples:
    def test_a_drifting_multiplet_keeps_its_coupling(self):
        """δ moves with pH; J does not. The pattern travels whole."""
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE + s * 0.004, J_DB) for s in range(4)]
        out = _run(cols)
        assert out['j_sd'] < 0.3
        assert out['ppm_spread'] == pytest.approx(0.012, abs=1e-3)


class TestIntoTheScore:
    def _mp(self, out, **kw):
        mp = {'type': 'd', 'ppm_diff': 0.04, 'delta_j_hz': 0.9,
              'per_sample_consensus': {
                  k: out[k] for k in ('ppm_centre', 'ppm_diff', 'ppm_n',
                                      'ppm_sd', 'observed_j_hz', 'delta_j_hz',
                                      'j_n', 'j_sd')}}
        mp.update(kw)
        return mp

    def test_the_samples_overrule_the_mean_spectrum(self):
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE, J_DB) for _ in range(5)]
        out = _run(cols)
        mp = self._mp(out)
        s_dj, conf = apply_per_sample_consensus(mp, params=PARAMS)
        assert conf == 1.0 and s_dj == pytest.approx(1.0)
        assert mp['delta_j_hz'] < PARAMS['dj_gold_hz']
        assert mp['ppm_diff'] < 0.01

    def test_samples_that_measured_nothing_change_nothing(self):
        ppm = _axis()
        out = _run([np.zeros_like(ppm)] * 3, noise_floor=0.01)
        mp = self._mp(out)
        assert apply_per_sample_consensus(mp, params=PARAMS) is None
        assert mp['delta_j_hz'] == pytest.approx(0.9)
        assert mp['ppm_diff'] == pytest.approx(0.04)

    def test_a_measurable_disagreement_is_reported_as_it_is(self):
        """The fit chooses among apices, it does not move them: a doublet that
        is really 7.4 Hz reports 7.4 Hz. Its ΔJ is a measurement, not the
        smallest number the window could be talked into."""
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE, 7.4) for _ in range(4)]
        out = _run(cols)
        assert out['observed_j_hz'] == pytest.approx(7.4, abs=0.25)
        assert out['delta_j_hz'] > PARAMS['dj_gold_hz']
        mp = self._mp(out)
        s_dj, _ = apply_per_sample_consensus(mp, params=PARAMS)
        assert 0.0 < s_dj < 1.0          # decayed, not perfect, not zero

    def test_a_pattern_that_is_not_this_multiplet_is_refused(self):
        """A doublet really 11 Hz wide is not a 7 Hz doublet drawn badly. The
        spacing misses by 4 Hz, far past dj_bad_hz, so no assignment is called
        a fit and the samples report nothing rather than a forced reading. The
        stored ΔJ from the master lines then stands, and it still disagrees —
        the disagreement is not lost, only measured somewhere else.
        """
        ppm = _axis()
        cols = [_doublet_sample(ppm, CENTRE, 11.0) for _ in range(4)]
        out = _run(cols)
        assert out['n_present'] == 0 and out['j_n'] == 0
        assert out['observed_j_hz'] is None
        mp = self._mp(out)
        assert apply_per_sample_consensus(mp, params=PARAMS) is None
        assert mp['delta_j_hz'] == pytest.approx(0.9)
