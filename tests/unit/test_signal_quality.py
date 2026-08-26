"""Signal quality grading: Good / Average / Poor, hard gates, and components."""
import numpy as np
import pytest

from processing.signal_quality import (
    classify_signal_quality, corrected_profile, measure_fwhm_points,
    _WEIGHTS, GATE_MIN_SNR, GATE_MIN_PTS_FWHM,
    SIGNAL_GOOD, SIGNAL_AVERAGE, SIGNAL_POOR,
)

NOISE = 0.001


def _lorentz(x, amp, cen, hwhm):
    return amp * hwhm ** 2 / ((x - cen) ** 2 + hwhm ** 2)


def _grade(y, *, left=None, right=None, cluster=None, shape_r2=0.99,
           noise=NOISE, median_fwhm_pts=None, height=None):
    """Grade a whole array as one signal, with sensible index defaults."""
    n = len(y)
    left = 0 if left is None else left
    right = n - 1 if right is None else right
    c_lo, c_hi = cluster if cluster else (left, right)
    if height is None:
        y_c = corrected_profile(y, 0, n - 1, left, right)
        height = float(y_c.max())
    if median_fwhm_pts is None:
        median_fwhm_pts = measure_fwhm_points(
            corrected_profile(y, 0, n - 1, left, right))
    return classify_signal_quality(
        y, start_idx=0, end_idx=n - 1, left_idx=left, right_idx=right,
        cluster_lo_idx=c_lo, cluster_hi_idx=c_hi,
        height=height, shape_r2=shape_r2, noise_std=noise,
        median_fwhm_pts=median_fwhm_pts)


# ── Weights ───────────────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9


# ── Hard gates ────────────────────────────────────────────────────────────────

def test_straight_ramp_fails_snr_gate():
    """A pure ramp has no height above its own baseline → not a signal."""
    y = np.linspace(0.2, 0.8, 121)
    tier, score, detail = _grade(y, shape_r2=0.05)
    assert tier == SIGNAL_POOR
    assert score == 0.0
    assert detail['gate_failed'] == 'snr'


def test_low_snr_fails_gate():
    x = np.linspace(-1, 1, 121)
    y = _lorentz(x, 2.5 * NOISE, 0.0, 0.15)      # SNR = 2.5 < 3
    tier, score, detail = _grade(y)
    assert tier == SIGNAL_POOR
    assert detail['gate_failed'] == 'snr'
    assert detail['snr'] < GATE_MIN_SNR


def test_undersampled_signal_fails_digital_resolution_gate():
    """Too few points across the FWHM → the integral is digitisation-limited."""
    x = np.linspace(-1, 1, 41)
    y = _lorentz(x, 1.0, 0.0, 0.02)              # very narrow on a coarse grid
    tier, score, detail = _grade(y)
    assert detail['fwhm_pts'] < GATE_MIN_PTS_FWHM
    assert tier == SIGNAL_POOR
    assert detail['gate_failed'] == 'digital_resolution'


# ── Clean signals ─────────────────────────────────────────────────────────────

def test_clean_isolated_lorentzian_is_good():
    x = np.linspace(-1, 1, 201)
    y = _lorentz(x, 1.0, 0.0, 0.12)
    tier, score, detail = _grade(y)
    assert tier == SIGNAL_GOOD, (score, detail)
    assert detail['gate_failed'] is None
    assert detail['components']['sym'] > 0.95
    assert detail['components']['shoulder'] > 0.9


def test_resolved_multiplet_line_can_be_good():
    """A cluster member is graded on the cluster's real baseline, not the valley.

    Two well-separated doublet lines: the left line's right foot is the shared
    valley, which sits above baseline.  Measuring tilt there would condemn every
    multiplet line, so the cluster's outer feet are used instead.
    """
    x = np.linspace(-1.0, 1.0, 401)
    y = _lorentz(x, 1.0, -0.30, 0.06) + _lorentz(x, 1.0, 0.30, 0.06)
    valley = int(np.argmin(y[np.argmax(y[:200]):np.argmax(y[200:]) + 200])
                 + np.argmax(y[:200]))
    tier, score, detail = _grade(
        y[:valley + 1], left=0, right=valley,
        cluster=(0, valley),                 # left foot and the valley
        median_fwhm_pts=measure_fwhm_points(
            corrected_profile(y[:valley + 1], 0, valley, 0, valley)))
    # The valley of a resolved doublet is near baseline, so the line grades well.
    assert tier in (SIGNAL_GOOD, SIGNAL_AVERAGE)
    assert score >= 0.5, detail


# ── Individual components ─────────────────────────────────────────────────────

def test_hidden_shoulder_is_penalised():
    """An unresolved component inside the bounds lowers the shoulder score."""
    x = np.linspace(-1, 1, 201)
    clean = _lorentz(x, 1.0, 0.0, 0.10)
    with_shoulder = clean + _lorentz(x, 0.45, 0.22, 0.07)

    _, _, d_clean = _grade(clean)
    _, _, d_shoulder = _grade(with_shoulder, shape_r2=0.80)

    assert d_shoulder['n_shoulders'] >= 1
    assert (d_shoulder['components']['shoulder']
            < d_clean['components']['shoulder'])


def test_asymmetry_from_shoulder_is_classified_as_shoulder():
    x = np.linspace(-1, 1, 201)
    y = _lorentz(x, 1.0, 0.0, 0.10) + _lorentz(x, 0.5, 0.25, 0.08)
    _, _, detail = _grade(y, shape_r2=0.80)
    assert detail['asym_kind'] == 'shoulder'
    assert detail['components']['sym'] < 0.95


def test_dispersive_residual_is_classified_as_phase():
    """A phase error drives the profile below its own baseline."""
    x = np.linspace(-1, 1, 201)
    hwhm = 0.10
    absorptive = _lorentz(x, 1.0, 0.0, hwhm)
    dispersive = 1.0 * hwhm * x / (x ** 2 + hwhm ** 2)
    y = 0.85 * absorptive + 0.35 * dispersive
    _, _, detail = _grade(y, shape_r2=0.70)
    assert detail['asym_kind'] == 'phase'


def test_tilted_baseline_lowers_tilt_score():
    x = np.linspace(-1, 1, 201)
    flat = _lorentz(x, 1.0, 0.0, 0.12)
    tilted = flat + np.linspace(0.0, 0.5, 201)
    _, _, d_flat = _grade(flat)
    _, _, d_tilt = _grade(tilted, shape_r2=0.85)
    assert d_tilt['components']['base_tilt'] < d_flat['components']['base_tilt']


def test_signal_on_broad_hump_loses_grounding():
    """Feet well above the noise ceiling → the integral rests on something.

    Grounding must read this signal's OWN feet, not the cluster's outer ones:
    the cluster eventually returns to baseline far away, yet the integral here
    still sits on top of the hump and is inflated.
    """
    x = np.linspace(-1, 1, 401)
    y = _lorentz(x, 0.4, 0.0, 2.0) + _lorentz(x, 1.0, 0.0, 0.04)
    apex = int(np.argmax(y))
    own_l, own_r = apex - 25, apex + 25          # feet still up on the hump
    _, _, detail = _grade(y, left=own_l, right=own_r,
                          cluster=(0, len(y) - 1), shape_r2=0.90)
    assert detail['components']['base_ground'] < 0.9, detail


def test_noise_ripples_are_not_counted_as_shoulders():
    """Residual noise must not manufacture unresolved components."""
    rng = np.random.default_rng(3)
    x = np.linspace(-1, 1, 201)
    y = _lorentz(x, 1.0, 0.0, 0.12) + rng.normal(0, NOISE, 201)
    _, _, detail = _grade(y, shape_r2=0.98)
    assert detail['n_shoulders'] == 0, detail
    assert detail['components']['shoulder'] == 1.0


def test_one_failed_component_caps_the_grade_at_average():
    """A line three times broader than the spectrum median cannot be Good."""
    x = np.linspace(-1, 1, 401)
    y = _lorentz(x, 1.0, 0.0, 0.20)
    fwhm = measure_fwhm_points(corrected_profile(y, 0, 400, 0, 400))
    tier, score, detail = _grade(y, shape_r2=1.0, median_fwhm_pts=fwhm / 5.0)
    assert detail['components']['width'] == 0.0
    assert score >= 0.80                       # the average alone would pass
    assert tier == SIGNAL_AVERAGE, (score, detail)
    assert detail['weakest_component'] == 'width'


def test_anomalously_broad_line_is_penalised():
    x = np.linspace(-1, 1, 201)
    broad = _lorentz(x, 1.0, 0.0, 0.30)
    fwhm_broad = measure_fwhm_points(corrected_profile(broad, 0, 200, 0, 200))
    _, _, d_ref = _grade(broad, median_fwhm_pts=fwhm_broad)
    _, _, d_wide = _grade(broad, median_fwhm_pts=fwhm_broad / 3.0)
    assert d_ref['components']['width'] == 1.0
    assert d_wide['components']['width'] < 1.0


def test_missing_median_fwhm_drops_width_component():
    x = np.linspace(-1, 1, 201)
    y = _lorentz(x, 1.0, 0.0, 0.12)
    _, score, detail = _grade(y, median_fwhm_pts=0)
    assert 'width' not in detail['components']
    assert 0.0 <= score <= 1.0


# ── Tier boundaries ───────────────────────────────────────────────────────────

def test_degraded_signal_lands_in_average():
    """Mediocre shape + tilted baseline: neither clean nor rejectable."""
    x = np.linspace(-1, 1, 201)
    y = (_lorentz(x, 1.0, 0.0, 0.12)
         + _lorentz(x, 0.55, 0.26, 0.09)
         + np.linspace(0.0, 0.35, 201))
    tier, score, detail = _grade(y, shape_r2=0.55)
    assert tier == SIGNAL_AVERAGE, (score, detail)


# ── Annotation eligibility ────────────────────────────────────────────────────

from processing.signal_quality import (  # noqa: E402
    ANNOTATABLE_TIERS,
    effective_tier,
    filter_annotatable,
    is_annotatable,
)


def test_annotatable_tiers_exclude_poor():
    assert ANNOTATABLE_TIERS == {SIGNAL_GOOD, SIGNAL_AVERAGE}
    assert SIGNAL_POOR not in ANNOTATABLE_TIERS


@pytest.mark.parametrize('tier,expected', [
    (SIGNAL_GOOD, True), (SIGNAL_AVERAGE, True), (SIGNAL_POOR, False)])
def test_is_annotatable_by_automatic_grade(tier, expected):
    assert is_annotatable({'signal_quality': tier}) is expected


def test_manual_status_overrides_automatic_grade():
    """The analyst's call in the peak list wins over the computed tier."""
    assert is_annotatable({'status': 'Good', 'signal_quality': SIGNAL_POOR})
    assert not is_annotatable({'status': 'Poor', 'signal_quality': SIGNAL_GOOD})


def test_unknown_status_falls_back_to_average():
    """A vocabulary mismatch must never silently discard a signal."""
    assert effective_tier({'status': 'Undefined'}) == SIGNAL_AVERAGE
    assert effective_tier({}) == SIGNAL_AVERAGE
    assert is_annotatable({'status': 'Undefined'})


def test_effective_tier_is_case_insensitive():
    assert effective_tier({'signal_quality': 'poor'}) == SIGNAL_POOR
    assert effective_tier({'status': 'GOOD'}) == SIGNAL_GOOD


def test_filter_annotatable_splits_and_preserves_order():
    peaks = [{'ppm': 1.0, 'signal_quality': SIGNAL_GOOD},
             {'ppm': 2.0, 'signal_quality': SIGNAL_POOR},
             {'ppm': 3.0, 'signal_quality': SIGNAL_AVERAGE}]
    usable, rejected = filter_annotatable(peaks)
    assert [p['ppm'] for p in usable] == [1.0, 3.0]
    assert [p['ppm'] for p in rejected] == [2.0]


def test_filter_annotatable_empty_input():
    assert filter_annotatable(None) == ([], [])


# ── configurable annotation grades ────────────────────────────────────────────

from processing.signal_quality import normalize_tiers  # noqa: E402


def test_default_selection_is_good_and_average():
    assert normalize_tiers(None) == ANNOTATABLE_TIERS


@pytest.mark.parametrize('choice,expected', [
    (['Good'],                    {SIGNAL_GOOD}),
    (['Average'],                 {SIGNAL_AVERAGE}),
    (['Good', 'Poor'],            {SIGNAL_GOOD, SIGNAL_POOR}),
    (['good', 'average', 'poor'], {SIGNAL_GOOD, SIGNAL_AVERAGE, SIGNAL_POOR}),
])
def test_selection_is_honoured_case_insensitively(choice, expected):
    assert set(normalize_tiers(choice)) == expected


@pytest.mark.parametrize('bad', [[], (), None, ['typo'], ['']])
def test_unusable_selection_falls_back_instead_of_annotating_nothing(bad):
    """A typo in a saved project must not silently produce zero annotations."""
    assert normalize_tiers(bad) == ANNOTATABLE_TIERS


def test_only_good_excludes_average_signals():
    peaks = [{'ppm': 1.0, 'signal_quality': SIGNAL_GOOD},
             {'ppm': 2.0, 'signal_quality': SIGNAL_AVERAGE},
             {'ppm': 3.0, 'signal_quality': SIGNAL_POOR}]
    usable, rejected = filter_annotatable(peaks, ['Good'])
    assert [p['ppm'] for p in usable] == [1.0]
    assert [p['ppm'] for p in rejected] == [2.0, 3.0]


def test_only_average_excludes_good_signals():
    """The selection is a set, not a floor — 'Average only' means only Average."""
    peaks = [{'ppm': 1.0, 'signal_quality': SIGNAL_GOOD},
             {'ppm': 2.0, 'signal_quality': SIGNAL_AVERAGE}]
    usable, _ = filter_annotatable(peaks, [SIGNAL_AVERAGE])
    assert [p['ppm'] for p in usable] == [2.0]


def test_manual_status_still_wins_under_a_restricted_selection():
    assert is_annotatable({'status': 'Good', 'signal_quality': SIGNAL_POOR},
                          ['Good'])
    assert not is_annotatable({'status': 'Poor', 'signal_quality': SIGNAL_GOOD},
                              ['Good'])
