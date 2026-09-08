"""An analyte the experiment could not see is reported as the bound it is.

The engine this replaces integrated a database-positioned window and reported
whatever came out as a concentration. On the reference study 15 numbers came
from that path, and 8 more were refused only because their window collided with
a region another analyte was already integrating — meaning they had been about
to report a neighbour's area.
"""
import numpy as np
import pytest

from processing.quant_detection_limit import (
    BELOW_LOD, BELOW_LOQ, EXCLUDED_REGION, K_LOD, K_LOQ, UNRESOLVED, censor,
    censor_text, detection_limits, drift_offsets, integration_noise,
    predicted_window, project_drift, project_noise, window_is_excluded)


class TestIntegrationNoise:
    def test_it_grows_as_the_root_of_the_window(self):
        """Random error partially cancels across the window. Summing σ over N
        points instead would make a broad multiplet look undetectable for no
        reason but its width."""
        a = integration_noise(1.0, 0.001, 100)
        b = integration_noise(1.0, 0.001, 400)
        assert b / a == pytest.approx(2.0)

    def test_it_scales_with_the_noise(self):
        assert (integration_noise(2.0, 0.001, 100)
                == pytest.approx(2 * integration_noise(1.0, 0.001, 100)))

    @pytest.mark.parametrize('sigma,n', [(0, 100), (None, 100), (1.0, 0)])
    def test_no_noise_estimate_is_not_a_zero_limit(self, sigma, n):
        """A zero limit would call every analyte quantifiable."""
        assert np.isnan(integration_noise(sigma, 0.001, n))


class TestLimits:
    def test_the_qnmr_convention(self):
        """SNR 3 to detect, 10 to quantify — Bharti & Roy 2012."""
        lod, loq = detection_limits(1.0, 0.001, 100)
        assert loq / lod == pytest.approx(K_LOQ / K_LOD)

    def test_they_are_not_numbers_when_the_noise_is_unknown(self):
        lod, loq = detection_limits(None, 0.001, 100)
        assert np.isnan(lod) and np.isnan(loq)


class TestTheWindow:
    def test_it_spans_the_predicted_lines(self):
        lo, hi = predicted_window([1.474, 1.486], linewidth_ppm=0.0)
        assert (lo, hi) == pytest.approx((1.474, 1.486))

    def test_it_is_padded_by_the_linewidth_only(self):
        """One linewidth each side, so the outer lines' tails fall inside."""
        lo, hi = predicted_window([1.480], linewidth_ppm=0.004)
        assert (lo, hi) == pytest.approx((1.476, 1.484))

    def test_the_drift_does_not_widen_it(self):
        """Where an analyte sits and how wide it is are different questions,
        and only the second belongs in a detection limit.

        Padding by the drift made the window 25.7 Hz for a 3.2 Hz line on the
        reference study — 180 points instead of 45 — and since the noise on an
        integral grows as the root of the point count, every limit came back
        twice what the spectrum supports."""
        plain = predicted_window([1.480], linewidth_ppm=0.004)
        drifted = predicted_window([1.480], linewidth_ppm=0.004,
                                   drift_ppm=0.016)
        assert drifted == pytest.approx(plain)

    def test_a_singlet_does_not_get_a_zero_width_window(self):
        lo, hi = predicted_window([1.480], linewidth_ppm=0.004)
        assert hi > lo

    def test_no_predicted_positions_means_no_window(self):
        assert predicted_window([], linewidth_ppm=0.004) is None

    def test_a_wider_window_costs_a_higher_limit(self):
        """Which is why the width is the signal's and not the search range's:
        noise comes in with the extra points, signal does not."""
        narrow = detection_limits(1.0, 0.001, 45)[1]
        wide = detection_limits(1.0, 0.001, 180)[1]
        assert wide == pytest.approx(narrow * 2.0, rel=0.01)


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
    metabolite. On the reference study four analytes were given limits or
    marked as overlaps on windows sitting inside 4.50-5.00 ppm."""

    def test_a_window_inside_the_region_is_caught(self):
        assert window_is_excluded(4.62, 4.67, [(4.50, 5.00)])

    def test_a_window_that_merely_touches_it_is_caught(self):
        assert window_is_excluded(4.45, 4.52, [(4.50, 5.00)])

    def test_a_window_outside_is_not(self):
        assert not window_is_excluded(1.47, 1.49, [(4.50, 5.00)])

    def test_no_regions_excludes_nothing(self):
        assert not window_is_excluded(4.62, 4.67, [])
        assert not window_is_excluded(4.62, 4.67, None)

    def test_the_engine_gives_it_no_limit_and_no_signal_verdict(self):
        from processing.quant_engine import quantify
        ppm = np.linspace(0.0, 10.0, 10001)
        y = (5e5 * np.exp(-0.5 * ((ppm - 4.70) / 0.02) ** 2)).reshape(-1, 1)
        rows = quantify(candidates=[_cand('ghost', [4.70])], peaks_by_rank={},
                        ppm=ppm, intensities=y, samples_info=[{}], included=[0],
                        groups={}, dj_gold_hz=0.5, noise_sigma=1.0,
                        linewidth_ppm=0.006, local_baseline=False,
                        excluded_regions=[(4.50, 5.00)])
        row = next(r for r in rows if r['hmdb_id'] == 'ghost')
        assert row['_censored'] == EXCLUDED_REGION
        assert row['status'] == 'excluded'
        assert not np.isfinite(row.get('_loq_mm', float('nan')))

    def test_without_the_exclusion_the_solvent_gets_a_misleading_verdict(self):
        """Which is why the exclusion exists rather than a cleverer verdict.

        Left unexcluded, the residual water is judged like an analyte: its own
        flanks make the local noise enormous, so the limit is enormous, and it
        reports as below the limit of detection — a mountain, called absent.
        Before the local noise estimate it reported as an unresolved analyte
        instead. Both readings are wrong, and neither can be fixed by tuning:
        nothing measured in a suppressed region belongs to any analyte.
        """
        from processing.quant_engine import quantify
        ppm = np.linspace(0.0, 10.0, 10001)
        y = (5e5 * np.exp(-0.5 * ((ppm - 4.70) / 0.02) ** 2)).reshape(-1, 1)
        rows = quantify(candidates=[_cand('ghost', [4.70])], peaks_by_rank={},
                        ppm=ppm, intensities=y, samples_info=[{}], included=[0],
                        groups={}, dj_gold_hz=0.5, noise_sigma=1.0,
                        linewidth_ppm=0.006, local_baseline=False)
        row = next(r for r in rows if r['hmdb_id'] == 'ghost')
        assert row['_censored'] != EXCLUDED_REGION


class TestTheThreeOutcomes:
    def test_nothing_above_the_noise_is_below_the_detection_limit(self):
        assert censor(0.5, 1.0, 3.3) == BELOW_LOD

    def test_signal_that_is_not_enough_to_measure(self):
        assert censor(2.0, 1.0, 3.3) == BELOW_LOQ

    def test_area_well_above_the_limit_with_no_peak_is_not_a_concentration(self):
        """Something is there that detection could not resolve — most often an
        overlapped shoulder. Calling it a concentration is the mistake this
        module exists to stop."""
        assert censor(100.0, 1.0, 3.3) == UNRESOLVED

    def test_an_unknown_area_is_not_silently_quantified(self):
        assert censor(float('nan'), 1.0, 3.3) == BELOW_LOD

    def test_the_boundaries_belong_to_the_lower_verdict(self):
        assert censor(1.0, 1.0, 3.3) == BELOW_LOD
        assert censor(3.3, 1.0, 3.3) == UNRESOLVED


class TestTheReason:
    def test_it_carries_the_numbers(self):
        """"Below the limit of detection" without saying what the limit was
        leaves the analyst unable to tell an informative absence from an
        insensitive measurement."""
        txt = censor_text(BELOW_LOQ, 0.01, 0.033)
        assert '0.01' in txt and '0.033' in txt

    def test_an_unknown_limit_says_so_rather_than_printing_nan(self):
        assert 'nan' not in censor_text(BELOW_LOD, float('nan'),
                                        float('nan')).lower()

    def test_the_unresolved_reason_tells_the_analyst_to_look(self):
        assert 'sobreposi' in censor_text(UNRESOLVED, 0.01, 0.03).lower()


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


def test_an_empty_region_is_reported_as_below_the_detection_limit():
    ppm = np.linspace(0.0, 10.0, 10001)
    rows = _quantify(np.zeros_like(ppm))
    row = next(r for r in rows if r['hmdb_id'] == 'ghost')
    assert row['_censored'] == BELOW_LOD
    assert row['status'] == '< LOD'


def test_the_bound_is_reported_in_the_tables_own_units():
    """A limit in area units cannot be compared with the concentrations beside
    it, and a limit nobody can compare is not a limit."""
    ppm = np.linspace(0.0, 10.0, 10001)
    rows = _quantify(np.zeros_like(ppm), standard_areas={0: 100.0},
                     standard_nh=9.0, standard_conc=1.0)
    row = next(r for r in rows if r['hmdb_id'] == 'ghost')
    assert np.isfinite(row['_loq_mm'])
    assert row['_loq_mm'] < 1.0


def test_a_buried_peak_is_flagged_rather_than_quantified():
    """The case the old gap-fill reported as a clean concentration. Here it
    identifies itself: area well over the limit, no peak detected anywhere in
    the window."""
    ppm = np.linspace(0.0, 10.0, 10001)
    y = 500.0 * np.exp(-0.5 * ((ppm - 5.500) / 0.004) ** 2)
    row = next(r for r in _quantify(y) if r['hmdb_id'] == 'ghost')
    assert row['_censored'] == UNRESOLVED
    assert row['_not_quantified']
    assert np.isnan(row['conc_mm'])


def test_no_censored_row_ever_carries_a_concentration():
    """It is a bound, not a measurement. A number here would flow into every
    group mean and every statistic as if it had been measured."""
    ppm = np.linspace(0.0, 10.0, 10001)
    for y in (np.zeros_like(ppm),
              3.0 * np.exp(-0.5 * ((ppm - 5.5) / 0.004) ** 2),
              500.0 * np.exp(-0.5 * ((ppm - 5.5) / 0.004) ** 2)):
        row = next(r for r in _quantify(y) if r['hmdb_id'] == 'ghost')
        assert row['_not_quantified']
        assert np.isnan(row['conc_mm'])
        assert row['_concs_by_group'] == {}


def test_without_a_noise_estimate_nothing_is_censored():
    """A bound computed from an unknown noise level is not a bound."""
    ppm = np.linspace(0.0, 10.0, 10001)
    row = next(r for r in _quantify(np.zeros_like(ppm), noise_sigma=None)
               if r['hmdb_id'] == 'ghost')
    assert row.get('_censored') is None
    assert row['_rejection'] == 'no_measurable_window'


def test_a_candidate_with_no_predicted_position_is_not_given_a_window():
    ppm = np.linspace(0.0, 10.0, 10001)
    row = next(r for r in _quantify(np.zeros_like(ppm),
                                    candidates=[_cand('ghost', [])])
               if r['hmdb_id'] == 'ghost')
    assert row.get('_censored') is None


def test_the_censored_row_says_which_region_it_looked_at():
    """A bound whose window nobody can see is not auditable."""
    ppm = np.linspace(0.0, 10.0, 10001)
    row = next(r for r in _quantify(np.zeros_like(ppm))
               if r['hmdb_id'] == 'ghost')
    lo, hi = row['_multiplet_window']
    assert lo < 5.500 < hi


# ── The limit describes the NET signal ───────────────────────────────────────

class TestTheBackgroundIsRemovedBeforeTheComparison:
    """Currie's limits describe what stands above the background, so an
    integral that still contains the background cannot be compared with a limit
    built from noise alone.

    Measured on a 197-point window: pure noise on a +2 sigma offset, with no
    peak in it at all, integrates to three times the quantification limit. All
    28 of the reference study's censored analytes came back 'overlap?' — a
    hundred percent in one category, which is a classifier that is not
    classifying.
    """

    def _window(self, offset=0.0, peak_height=0.0, seed=0):
        ppm = np.linspace(0.0, 10.0, 40001)
        rng = np.random.default_rng(seed)
        y = rng.normal(0.0, 1.0, ppm.size) + offset
        if peak_height:
            y = y + peak_height * np.exp(-0.5 * ((ppm - 5.500) / 0.002) ** 2)
        return ppm, y.reshape(-1, 1)

    def _verdict(self, **kw):
        from processing.quant_engine import quantify
        ppm, y = self._window(**kw)
        rows = quantify(candidates=[_cand('ghost', [5.500])], peaks_by_rank={},
                        ppm=ppm, intensities=y, samples_info=[{}], included=[0],
                        groups={'A': [0]}, dj_gold_hz=0.5, noise_sigma=1.0,
                        linewidth_ppm=0.010, drift_ppm=0.005,
                        local_baseline=False)
        return next(r for r in rows if r['hmdb_id'] == 'ghost')['_censored']

    def test_a_raised_baseline_with_no_peak_is_still_below_the_limit(self):
        assert self._verdict(offset=2.0) == BELOW_LOD

    def test_even_a_steeply_raised_one(self):
        assert self._verdict(offset=10.0) == BELOW_LOD

    def test_a_real_peak_is_still_seen(self):
        """The correction must not make the criterion blind."""
        assert self._verdict(peak_height=40.0) == UNRESOLVED

    def test_it_does_not_follow_the_runs_integration_setting(self):
        """That flag governs how a quantified area is measured, where it costs
        a known ~5% on the ratio between two analytes. Here there is no choice
        to make, so passing local_baseline=False must change nothing."""
        assert self._verdict(offset=5.0) == BELOW_LOD


def test_the_limit_travels_the_same_chain_as_a_concentration():
    """A limit on a different scale from the numbers beside it cannot be
    compared, and a limit nobody can compare is not a limit. It left out the
    dilution and the correction factor."""
    from processing.quant_engine import quantify
    ppm = np.linspace(0.0, 10.0, 10001)
    y = np.zeros((ppm.size, 1))
    common = dict(candidates=[_cand('ghost', [5.500])], peaks_by_rank={},
                  ppm=ppm, intensities=y, samples_info=[{}], included=[0],
                  groups={}, dj_gold_hz=0.5, noise_sigma=1.0,
                  standard_areas={0: 100.0}, standard_nh=9.0,
                  standard_conc=1.0, external_standard_samples=[0])
    plain = next(r for r in quantify(**common)
                 if r['hmdb_id'] == 'ghost')['_loq_mm']
    diluted = next(r for r in quantify(**common, dilution={0: 4.0})
                   if r['hmdb_id'] == 'ghost')['_loq_mm']
    assert diluted == pytest.approx(plain * 4.0, rel=1e-6)


class TestTheNoiseIsTheRegionsOwn:
    """A limit built on the spectrometer's quiet region answers "how small a
    signal would stand out over an empty baseline". In a crowded region that is
    the wrong question, and it comes back optimistic exactly where it matters.

    The symptom: six analytes claiming the histidine imidazole region at
    7.0926-7.1025 ppm were handed identical limits, differing only by proton
    count, because the window width and the global sigma were the same for all
    of them.
    """

    PPM = np.linspace(6.0, 8.0, 8001)
    LO, HI = 7.0926, 7.1025
    LW = 0.0054

    def _noise(self, seed=0):
        return np.random.default_rng(seed).normal(0, 5.0, self.PPM.size)

    def _peak(self, centre, height, width=0.004):
        return height * np.exp(-0.5 * ((self.PPM - centre) / width) ** 2)

    def _sigma(self, y, **kw):
        from processing.quant_detection_limit import local_noise_sigma
        args = dict(linewidth_ppm=self.LW, global_sigma=5.0)
        args.update(kw)
        return local_noise_sigma(self.PPM, y, self.LO, self.HI, **args)

    def test_a_quiet_region_sits_at_the_instrument_floor(self):
        assert self._sigma(self._noise()) == pytest.approx(5.0, rel=0.3)

    def test_a_crowded_region_reports_a_higher_uncertainty(self):
        crowded = self._noise() + self._peak(7.13, 3000.0)
        assert self._sigma(crowded) > 10 * 5.0

    def test_a_signal_does_not_raise_its_own_limit(self):
        """The band flanks the window and excludes it. Including it would let a
        real signal lift the bar until it could never clear it — the residual
        water came back reported as below the limit of detection."""
        inside = self._noise() + self._peak(7.0975, 3000.0)
        assert self._sigma(inside) < 3 * 5.0

    def test_it_never_promises_better_than_the_instrument(self):
        """A local estimate below the noise floor would claim a sensitivity the
        spectrometer does not have."""
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
        from processing.quant_detection_limit import local_noise_sigma
        ppm = np.linspace(7.09, 7.11, 6)
        assert local_noise_sigma(ppm, np.zeros(6), 7.0926, 7.1025,
                                 linewidth_ppm=self.LW,
                                 global_sigma=5.0) == 5.0

    def test_the_engine_uses_it(self):
        """Measured and not reaching the decision is this code base's recurring
        failure."""
        import inspect

        from processing import quant_engine as qe
        src = inspect.getsource(qe._censored_row)
        assert 'local_noise_sigma(' in src
        assert 'detection_limits(sigma_here' in src

    def test_two_analytes_in_regions_of_different_crowding_get_different_limits(self):
        """The point of the whole change."""
        from processing.quant_detection_limit import detection_limits
        quiet = self._sigma(self._noise())
        crowded = self._sigma(self._noise() + self._peak(7.13, 3000.0))
        assert detection_limits(crowded, 0.00025, 41)[1] >             detection_limits(quiet, 0.00025, 41)[1] * 5


class TestTheResultIsWorthTheWorseOfItsTwoEvidences:
    """Two independent things spoil a number and neither makes up for the other.

    A textbook multiplet measured at the quantification limit is a confident
    identity and an imprecise quantity; a towering singlet with a rival on its
    peak is the reverse. Averaging would let either hide the other.

    Until now the dots read only the shape: a metabolite at 1.1x its limit got
    the same three dots as one at a hundred times, and no quantified analyte
    was given a limit at all — only the ones that could not be measured were.
    """

    def _sig(self, mult='d', n_found=2, n_expected=2, dj=0.1):
        from processing.quantification import Candidate
        return Candidate(met_id='M', peak_rank=1,
                         evidence={'multiplicity': mult, 'n_found': n_found,
                                   'n_expected': n_expected,
                                   'delta_j_hz': dj})

    def test_a_perfect_multiplet_at_the_limit_is_not_strong(self):
        from processing.quant_engine import _confidence
        assert _confidence(self._sig(dj=0.05), strength=2)[0] == 2

    def test_a_big_singlet_is_not_strong_either(self):
        """Its identity is what is doubtful, and size cannot answer that."""
        from processing.quant_engine import _confidence
        assert _confidence(self._sig(mult='s', n_found=1, n_expected=1,
                                     dj=None), strength=3)[0] == 1

    def test_a_perfect_multiplet_well_above_the_limit_is_strong(self):
        from processing.quant_engine import _confidence
        assert _confidence(self._sig(dj=0.05), strength=3)[0] == 3

    def test_without_a_strength_the_shape_decides_alone(self):
        """A missing noise estimate must not silently mark everything weak."""
        from processing.quant_engine import _confidence
        assert _confidence(self._sig(dj=0.05))[0] == 3


class TestStrengthNeverCallsAnIdentifiedAnalyteBad:
    """It floors at two dots. An analyte whose peak was identified and whose
    contest it won is not made doubtful by being scarce: under the limit says
    the quantity is small and the number imprecise, not that the assignment was
    wrong."""

    def test_far_below_the_limit_still_floors_at_two(self):
        from processing.quant_detection_limit import strength_band
        assert strength_band(0.001, 10.0) == 2

    def test_at_the_limit_is_two(self):
        from processing.quant_detection_limit import strength_band
        assert strength_band(10.0, 10.0) == 2

    def test_the_agreed_multiple_is_where_strong_starts(self):
        from processing.quant_detection_limit import (
            STRONG_ABOVE_LOQ, strength_band)
        assert STRONG_ABOVE_LOQ == 3.0
        assert strength_band(30.0, 10.0) == 3
        assert strength_band(29.9, 10.0) == 2

    def test_an_unknown_limit_does_not_penalise(self):
        from processing.quant_detection_limit import strength_band
        assert strength_band(50.0, float('nan')) == 3
        assert strength_band(50.0, None) == 3


def test_every_quantified_row_carries_its_own_limits():
    """Without them nobody can tell a number ten times the limit from one at
    1.1x, and the confidence had no way to say which it was."""
    from processing.quant_engine import quantify
    ppm = np.linspace(0.0, 10.0, 10001)
    y = (1e4 * np.exp(-0.5 * ((ppm - 5.0) / 0.004) ** 2)
         + np.random.default_rng(0).normal(0, 1.0, ppm.size)).reshape(-1, 1)
    peak = {'rank': 1, 'ppm': 5.0, 'start_ppm': 4.99, 'end_ppm': 5.01,
            'noise': 1.0, 'snr': 100.0}
    from processing.quantification import Candidate
    cand = Candidate(met_id='M', peak_rank=1, line_ranks=frozenset([1]),
                     evidence={'multiplicity': 's', 'n_found': 1,
                               'n_expected': 1, 'delta_j_hz': None,
                               'protons': 1.0})
    rows = quantify(candidates=[cand], peaks_by_rank={1: peak}, ppm=ppm,
                    intensities=y, samples_info=[{}], included=[0],
                    groups={'A': [0]}, dj_gold_hz=0.5, noise_sigma=1.0,
                    linewidth_ppm=0.006, local_baseline=False)
    row = next(r for r in rows if r['hmdb_id'] == 'M')
    assert not row['_not_quantified']
    assert np.isfinite(row['_lod_mm']) and np.isfinite(row['_loq_mm'])
    assert row['_strength'] in (2, 3)


def test_a_weak_row_says_why_it_is_capped():
    """Two dots with no reason is a verdict the analyst cannot check."""
    from processing.quant_engine import quantify
    ppm = np.linspace(0.0, 10.0, 10001)
    y = (3.0 * np.exp(-0.5 * ((ppm - 5.0) / 0.004) ** 2)
         + np.random.default_rng(0).normal(0, 1.0, ppm.size)).reshape(-1, 1)
    peak = {'rank': 1, 'ppm': 5.0, 'start_ppm': 4.99, 'end_ppm': 5.01,
            'noise': 1.0, 'snr': 3.0}
    from processing.quantification import Candidate
    cand = Candidate(met_id='M', peak_rank=1, line_ranks=frozenset([1]),
                     evidence={'multiplicity': 'd', 'n_found': 2,
                               'n_expected': 2, 'delta_j_hz': 0.05,
                               'protons': 1.0})
    rows = quantify(candidates=[cand], peaks_by_rank={1: peak}, ppm=ppm,
                    intensities=y, samples_info=[{}], included=[0],
                    groups={'A': [0]}, dj_gold_hz=0.5, noise_sigma=1.0,
                    linewidth_ppm=0.006, local_baseline=False)
    row = next(r for r in rows if r['hmdb_id'] == 'M')
    if row['_strength'] == 2:
        assert row['_confidence'] == 2
        assert 'limite de quantifica' in row['_status_reasons'][0]
