"""A winner with no detected peak is measured on the window its reference
predicts, and the row says so.

The window is the signal's own width, slid over the study's drift; a window in
an excluded region is the solvent's and is not measured; the background under
the window is removed before the integral is read.
"""
import numpy as np
import pytest

from processing.quant_predicted_window import (
    drift_offsets, integration_noise, predicted_window, project_drift,
    project_noise, window_is_excluded)
from processing.quantification import (
    EXCLUDED_REGION, NO_MEASURABLE_WINDOW, NOT_DETECTED, WINDOW_INTEGRAL_TEXT)


class TestIntegrationNoise:
    def test_it_grows_as_the_root_of_the_window(self):
        """Random error partially cancels across the window. Summing σ over N
        points instead would make a broad multiplet look noisier for no
        reason but its width."""
        a = integration_noise(1.0, 0.001, 100)
        b = integration_noise(1.0, 0.001, 400)
        assert b / a == pytest.approx(2.0)

    def test_it_scales_with_the_noise(self):
        assert (integration_noise(2.0, 0.001, 100)
                == pytest.approx(2 * integration_noise(1.0, 0.001, 100)))

    @pytest.mark.parametrize('sigma,n', [(0, 100), (None, 100), (1.0, 0)])
    def test_no_noise_estimate_is_not_a_zero_noise(self, sigma, n):
        """A zero noise would give every window an infinite SNR."""
        assert np.isnan(integration_noise(sigma, 0.001, n))


class TestTheWindow:
    def test_it_spans_the_predicted_lines(self):
        lo, hi = predicted_window([1.474, 1.486], linewidth_ppm=0.0)
        assert (lo, hi) == pytest.approx((1.474, 1.486))

    def test_it_is_padded_by_the_linewidth_only(self):
        """One linewidth each side, so the outer lines' tails fall inside."""
        lo, hi = predicted_window([1.480], linewidth_ppm=0.004)
        assert (lo, hi) == pytest.approx((1.476, 1.484))

    def test_the_drift_does_not_widen_it(self):
        """Where an analyte sits and how wide it is are different questions.

        Padding by the drift made the window 25.7 Hz for a 3.2 Hz line on the
        reference study — 180 points instead of 45 of noise and neighbours."""
        plain = predicted_window([1.480], linewidth_ppm=0.004)
        drifted = predicted_window([1.480], linewidth_ppm=0.004,
                                   drift_ppm=0.016)
        assert drifted == pytest.approx(plain)

    def test_a_singlet_does_not_get_a_zero_width_window(self):
        lo, hi = predicted_window([1.480], linewidth_ppm=0.004)
        assert hi > lo

    def test_no_predicted_positions_means_no_window(self):
        assert predicted_window([], linewidth_ppm=0.004) is None


class TestSlidingRatherThanWidening:
    """A resonance is near its database position, not on it. The window slides
    over the plausible range and the best position is kept, instead of growing
    to cover the range and paying for all of it in noise."""

    def test_it_always_tries_the_database_position(self):
        assert 0.0 in drift_offsets(0.016, 0.005)

    def test_it_covers_the_drift_in_both_directions(self):
        off = drift_offsets(0.016, 0.005)
        assert min(off) <= -0.014 and max(off) >= 0.014

    def test_the_step_cannot_let_a_line_fall_between_positions(self):
        """Half a linewidth, so any line is inside some window."""
        off = sorted(drift_offsets(0.016, 0.005))
        assert max(b - a for a, b in zip(off, off[1:])) <= 0.005

    def test_no_drift_means_one_position(self):
        assert drift_offsets(0.0, 0.005) == [0.0]
        assert drift_offsets(None, 0.005) == [0.0]

    def test_a_large_drift_does_not_become_hundreds_of_measurements(self):
        assert len(drift_offsets(5.0, 0.001)) <= 17


class TestAnExcludedRegionIsNotAssessable:
    """Nothing measured in an excluded region belongs to the analyte. In the
    water window it is the suppressed solvent, orders of magnitude above any
    metabolite. On the reference study four analytes had their windows
    inside 4.50-5.00 ppm."""

    def test_a_window_inside_the_region_is_caught(self):
        assert window_is_excluded(4.62, 4.67, [(4.50, 5.00)])

    def test_a_window_that_merely_touches_it_is_caught(self):
        assert window_is_excluded(4.45, 4.52, [(4.50, 5.00)])

    def test_a_window_outside_is_not(self):
        assert not window_is_excluded(1.47, 1.49, [(4.50, 5.00)])

    def test_no_regions_excludes_nothing(self):
        assert not window_is_excluded(4.62, 4.67, [])
        assert not window_is_excluded(4.62, 4.67, None)


class TestProjectMeasurements:
    def test_the_linewidth_comes_from_the_trace_not_from_the_peak_widths(self):
        """width_ppm is valley-to-valley and moves with min_distance, so a
        window padded by it would be wide or narrow according to a detection
        setting rather than to the spectrum. One linewidth per project."""
        import inspect

        from gui.stages import _quant_adapter as a
        src = inspect.getsource(a.quantify_project)
        assert 'measure_linewidth_ppm(ppm, arr, peaks)' in src
        assert 'project_linewidth' not in src

    def test_noise_is_the_median_detection_estimate(self):
        assert project_noise([{'noise': 2.0}, {'noise': 4.0}]) == 3.0

    def test_no_noise_estimate_is_not_a_number(self):
        assert np.isnan(project_noise([]))

    def test_drift_is_measured_not_assumed(self):
        mets = [{'matched_peaks': [{'ppm_diff': 0.002}, {'ppm_diff': -0.006}]}]
        assert project_drift(mets) == pytest.approx(0.004)

    def test_nothing_matched_means_no_drift_padding_invented(self):
        assert project_drift([]) == 0.0


# ── Through the engine ───────────────────────────────────────────────────────

def _cand(mid, predicted, protons=1.0):
    from processing.quantification import Candidate
    return Candidate(met_id=mid, peak_rank=None, line_ranks=frozenset(),
                     evidence={'multiplicity': 's', 'delta_j_hz': None,
                               'ppm_diff': 0.001, 'n_found': 1,
                               'n_expected': 1, 'quality': 'good',
                               'score': 0.8, 'protons': protons,
                               'predicted_ppms': predicted})


def _quantify(y, **kw):
    from processing.quant_engine import quantify
    ppm = np.linspace(0.0, 10.0, 10001)
    args = dict(candidates=[_cand('ghost', [5.500])], peaks_by_rank={},
                ppm=ppm, intensities=np.asarray(y).reshape(-1, 1),
                samples_info=[{}], included=[0], groups={'A': [0]},
                dj_gold_hz=0.5, noise_sigma=1.0, linewidth_ppm=0.006,
                drift_ppm=0.002, local_baseline=False)
    args.update(kw)
    return quantify(**args)


def _ghost(y, **kw):
    return next(r for r in _quantify(y, **kw) if r['hmdb_id'] == 'ghost')


PPM = np.linspace(0.0, 10.0, 10001)


def _bump(height, centre=5.5):
    return height * np.exp(-0.5 * ((PPM - centre) / 0.004) ** 2)


def test_a_signal_with_no_peak_is_quantified_at_one_dot_and_says_so():
    row = _ghost(_bump(500.0))
    assert not row['_not_quantified'] and row.get('_window_integral')
    assert row['conc_mm'] > 0 and row['_concs_by_group']['A'][0] > 0
    assert row['_confidence'] == 1
    assert row['_status_reasons'] == [WINDOW_INTEGRAL_TEXT]
    assert row['_included'] == [0]


def test_a_small_signal_is_quantified_the_same_way():
    """No limit decides between small and large: both are the window's
    integral, at one dot."""
    small, large = _ghost(_bump(3.0)), _ghost(_bump(500.0))
    assert not small['_not_quantified'] and small['_confidence'] == 1
    assert small['conc_mm'] / large['conc_mm'] == pytest.approx(3.0 / 500.0,
                                                               rel=0.05)


def test_no_row_carries_a_limit():
    for y in (np.zeros_like(PPM), _bump(3.0), _bump(500.0)):
        row = _ghost(y)
        for k in ('_lod_mm', '_loq_mm', '_censored', '_below_limit',
                  '_detected_per_sample', '_strength'):
            assert k not in row, k


def test_a_window_with_nothing_above_the_baseline_is_not_detected():
    row = _ghost(np.zeros_like(PPM))
    assert row['_not_quantified'] and row['_rejection'] == NOT_DETECTED
    assert np.isnan(row['conc_mm']) and row['_concs_by_group'] == {}
    assert 'nothing above the local baseline' in row['_status_reasons'][0]


def test_the_snr_is_read_against_the_noise_of_the_integral():
    row = _ghost(_bump(500.0))
    assert np.isfinite(row['snr']) and row['snr'] > 10
    assert np.isnan(_ghost(_bump(500.0), noise_sigma=None)['snr'])


def test_without_a_noise_estimate_it_is_still_measured():
    row = _ghost(_bump(500.0), noise_sigma=None)
    assert not row['_not_quantified'] and row['conc_mm'] > 0


def test_a_candidate_with_no_predicted_position_is_not_given_a_window():
    row = _ghost(np.zeros_like(PPM), candidates=[_cand('ghost', [])])
    assert row['_rejection'] == NO_MEASURABLE_WINDOW


def test_the_row_says_which_region_it_looked_at():
    lo, hi = _ghost(_bump(500.0))['_multiplet_window']
    assert lo < 5.500 < hi


def test_an_excluded_region_is_not_measured():
    y = 5e5 * np.exp(-0.5 * ((PPM - 4.70) / 0.02) ** 2)
    row = _ghost(y, candidates=[_cand('ghost', [4.70])],
                 excluded_regions=[(4.50, 5.00)])
    assert row['_not_quantified'] and row['_rejection'] == EXCLUDED_REGION
    assert 'excluded from' in row['_status_reasons'][0].lower()


def test_the_amount_travels_the_same_chain_as_a_concentration():
    """Standard, dilution and correction reach it as they reach any row."""
    common = dict(standard_areas={0: 100.0}, standard_nh=9.0,
                  standard_conc=1.0, external_standard_samples=[0])
    plain = _ghost(_bump(500.0), **common)['conc_mm']
    diluted = _ghost(_bump(500.0), dilution={0: 4.0}, **common)['conc_mm']
    assert diluted == pytest.approx(plain * 4.0, rel=1e-6)


class TestTheBackgroundIsRemovedBeforeTheIntegral:
    """With no peak to bound the window, the background under it is the part
    of its area that is surely not the analyte. Measured on a 197-point
    window: pure noise on a +2 sigma offset integrated to three times what
    the noise alone gives."""

    def _row(self, offset=0.0, peak_height=0.0, seed=0):
        from processing.quant_engine import quantify
        ppm = np.linspace(0.0, 10.0, 40001)
        y = np.random.default_rng(seed).normal(0.0, 1.0, ppm.size) + offset
        if peak_height:
            y = y + peak_height * np.exp(-0.5 * ((ppm - 5.500) / 0.002) ** 2)
        rows = quantify(candidates=[_cand('ghost', [5.500])], peaks_by_rank={},
                        ppm=ppm, intensities=y.reshape(-1, 1),
                        samples_info=[{}], included=[0], groups={'A': [0]},
                        dj_gold_hz=0.5, noise_sigma=1.0, linewidth_ppm=0.010,
                        drift_ppm=0.005, local_baseline=False)
        return next(r for r in rows if r['hmdb_id'] == 'ghost')

    def _amount(self, **kw):
        r = self._row(**kw)
        return 0.0 if r['_not_quantified'] else r['conc_mm']

    def test_a_raised_baseline_adds_nothing_a_peak_would(self):
        peak = self._amount(peak_height=40.0)
        assert peak > 0
        for offset in (2.0, 10.0):
            assert self._amount(offset=offset) < 0.05 * peak

    def test_it_does_not_follow_the_runs_integration_setting(self):
        """local_baseline=False governs how a quantified area is measured;
        here the background is removed whatever it says."""
        assert self._amount(offset=5.0) < 0.05 * self._amount(peak_height=40.0)


class TestTheNoiseIsTheRegionsOwn:
    """How crowded a window's surroundings are, measured as INTERFERENCE.

    Taurine, second most concentrated analyte of the reference study, had
    this come back 193x the blank -- the band flanking its window holds a
    skirt 16.9 million counts tall and no detected peak. A region that
    crowded cannot be measured down to the noise floor, and the row says so.
    """

    PPM = np.linspace(6.0, 8.0, 8001)
    LO, HI = 7.0926, 7.1025
    LW = 0.0054

    def _noise(self, seed=0):
        return np.random.default_rng(seed).normal(0, 5.0, self.PPM.size)

    def _peak(self, centre, height, width=0.004):
        return height * np.exp(-0.5 * ((self.PPM - centre) / width) ** 2)

    def _sigma(self, y, **kw):
        from processing.quant_predicted_window import local_noise_sigma
        args = dict(linewidth_ppm=self.LW, global_sigma=5.0)
        args.update(kw)
        return local_noise_sigma(self.PPM, y, self.LO, self.HI, **args)

    def test_a_quiet_region_sits_at_the_instrument_floor(self):
        assert self._sigma(self._noise()) == pytest.approx(5.0, rel=0.3)

    def test_a_crowded_region_reports_a_higher_uncertainty(self):
        crowded = self._noise() + self._peak(7.13, 3000.0)
        assert self._sigma(crowded) > 10 * 5.0

    def test_a_signal_does_not_raise_its_own_reading(self):
        """The band flanks the window and excludes it: the residual water,
        orders of magnitude above any metabolite, would read as its own
        surroundings."""
        inside = self._noise() + self._peak(7.0975, 3000.0)
        assert self._sigma(inside) < 3 * 5.0

    def test_it_never_promises_better_than_the_instrument(self):
        """A local estimate below the noise floor would claim a region quieter
        than the spectrometer."""
        assert self._sigma(self._noise(), global_sigma=500.0) == 500.0

    def test_the_scale_is_robust_to_the_peaks_in_the_band(self):
        """A plain standard deviation would report the neighbours themselves as
        noise. The MAD reflects the baseline while under half the band is peak,
        compared over the same points."""
        y = self._noise() + self._peak(7.13, 3000.0)
        band = (((self.PPM >= self.LO - 8 * self.LW) & (self.PPM < self.LO))
                | ((self.PPM > self.HI) & (self.PPM <= self.HI + 8 * self.LW)))
        plain_sd = float(np.std(y[band]))
        assert self._sigma(y) < plain_sd

    def test_no_linewidth_means_no_local_estimate(self):
        assert self._sigma(self._noise(), linewidth_ppm=0.0) == 5.0

    def test_a_band_with_too_few_points_falls_back_to_the_floor(self):
        from processing.quant_predicted_window import local_noise_sigma
        ppm = np.linspace(7.09, 7.11, 6)
        assert local_noise_sigma(ppm, np.zeros(6), 7.0926, 7.1025,
                                 linewidth_ppm=self.LW,
                                 global_sigma=5.0) == 5.0

    def test_the_engine_measures_it_and_reports_it(self):
        """Measured and not reaching the analyst is this code base's
        recurring failure. It reaches them as a reason on the row."""
        import inspect

        from processing import quant_engine as qe
        for fn in (qe._window_integral_row, qe._window_snr):
            src = inspect.getsource(fn)
            assert 'local_noise_sigma(' in src, fn.__name__

    def test_a_crowded_window_says_so_on_the_row(self):
        from processing.quant_predicted_window import INTERFERENCE_NOTE
        from processing.quant_engine import _interference_reason
        assert _interference_reason({'interference': 1.0}) == []
        assert _interference_reason({}) == []
        said = _interference_reason({'interference': 193.0})
        assert said and 'not noise' in said[0]
        assert _interference_reason({'interference': INTERFERENCE_NOTE})


class TestTheResultIsWorthTheWorseOfItsEvidences:
    """Independent things spoil a number and none makes up for another.
    Averaging would let either hide the other. The size of the area is not
    one of them: a small number is a small quantity, not a doubtful one."""

    def _sig(self, mult='d', n_found=2, n_expected=2, dj=0.1):
        from processing.quantification import Candidate
        return Candidate(met_id='M', peak_rank=1,
                         evidence={'multiplicity': mult, 'n_found': n_found,
                                   'n_expected': n_expected,
                                   'delta_j_hz': dj})

    def test_a_perfect_multiplet_is_strong_whatever_its_size(self):
        from processing.quant_engine import _confidence
        assert _confidence(self._sig(dj=0.05))[0] == 3

    def test_a_big_singlet_is_not_strong(self):
        """Its identity is what is doubtful, and size cannot answer that."""
        from processing.quant_engine import _confidence
        assert _confidence(self._sig(mult='s', n_found=1, n_expected=1,
                                     dj=None), identity=1)[0] == 1

    def test_a_shared_window_holds_it_at_two(self):
        from processing.quant_engine import _confidence
        assert _confidence(self._sig(dj=0.05), shared=True)[0] == 2


def test_a_small_quantified_signal_keeps_its_dots():
    """A doublet barely above the noise used to be capped at two dots for
    standing under three times the limit of quantification."""
    from processing.quant_engine import quantify
    from processing.quantification import Candidate
    ppm = np.linspace(0.0, 10.0, 10001)
    y = (3.0 * np.exp(-0.5 * ((ppm - 5.0) / 0.004) ** 2)
         + np.random.default_rng(0).normal(0, 1.0, ppm.size)).reshape(-1, 1)
    peak = {'rank': 1, 'ppm': 5.0, 'start_ppm': 4.99, 'end_ppm': 5.01,
            'noise': 1.0, 'snr': 3.0}
    cand = Candidate(met_id='M', peak_rank=1, line_ranks=frozenset([1]),
                     evidence={'multiplicity': 'd', 'n_found': 2,
                               'n_expected': 2, 'delta_j_hz': 0.05,
                               'protons': 1.0})
    rows = quantify(candidates=[cand], peaks_by_rank={1: peak}, ppm=ppm,
                    intensities=y, samples_info=[{}], included=[0],
                    groups={'A': [0]}, dj_gold_hz=0.5, noise_sigma=1.0,
                    linewidth_ppm=0.006, local_baseline=False)
    row = next(r for r in rows if r['hmdb_id'] == 'M')
    assert not row['_not_quantified']
    assert not any('limit of quantification' in t
                   for t in row['_status_reasons'])
    assert '_lod_mm' not in row and '_strength' not in row
