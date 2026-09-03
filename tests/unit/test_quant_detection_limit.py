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
    BELOW_LOD, BELOW_LOQ, K_LOD, K_LOQ, UNRESOLVED, censor, censor_text,
    detection_limits, integration_noise, predicted_window, project_drift,
    project_linewidth, project_noise)


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

    def test_it_is_padded_by_the_measured_linewidth_and_drift(self):
        """Both come from the study. A resonance is not at its database
        position in any real sample — it is near it."""
        lo, hi = predicted_window([1.480], linewidth_ppm=0.004, drift_ppm=0.002)
        assert (lo, hi) == pytest.approx((1.474, 1.486))

    def test_a_singlet_does_not_get_a_zero_width_window(self):
        lo, hi = predicted_window([1.480], linewidth_ppm=0.004)
        assert hi > lo

    def test_no_predicted_positions_means_no_window(self):
        assert predicted_window([], linewidth_ppm=0.004) is None

    def test_padding_only_makes_the_bound_more_conservative(self):
        """A wider window integrates more noise, so the limit rises. A bound
        that is too tight would call an analyte absent that merely drifted."""
        narrow = detection_limits(1.0, 0.001, 20)[1]
        wide = detection_limits(1.0, 0.001, 80)[1]
        assert wide > narrow


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
    def test_linewidth_is_the_median_of_what_detection_measured(self):
        peaks = [{'width_ppm': 0.004}, {'width_ppm': 0.006},
                 {'width_ppm': 0.008}]
        assert project_linewidth(peaks) == pytest.approx(0.006)

    def test_no_peaks_falls_back_rather_than_failing(self):
        assert project_linewidth([]) > 0

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
