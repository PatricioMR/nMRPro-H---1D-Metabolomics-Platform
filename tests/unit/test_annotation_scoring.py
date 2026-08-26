"""Unit tests for the pure annotation scoring core (processing/annotation_scoring.py).

These lock in the behaviour extracted from AnnotationResultsWindow._compute_scores
so the upcoming model changes have a parity baseline.
"""
import pytest

from processing.annotation_scoring import (
    LINE_COUNTS,
    dedup_coincident,
    dj_score,
    drop_worst_j_outlier,
    expected_line_count,
    is_singlet_like,
    measure_multiplet,
    parse_first_j,
    ppm_score,
    snap_lines_to_peaks,
    spacings_hz,
    sync_predicted_lines,
    synthetic_lines,
)

FIELD = 600.0
TOL = 0.05


def _peaks(*ppms):
    return [{'ppm': p, 'rank': i} for i, p in enumerate(ppms)]


# ── multiplicity vocabulary ───────────────────────────────────────────────────

@pytest.mark.parametrize('mult', ['s', 'Singlet', '', 'm', 'multiplet', 'br s'])
def test_singlet_like_broad(mult):
    assert is_singlet_like(mult)


def test_singlet_like_strict_excludes_multiplet():
    assert is_singlet_like('m') is True
    assert is_singlet_like('m', strict=True) is False
    assert is_singlet_like('s', strict=True) is True


def test_doublet_is_not_singlet_like():
    assert not is_singlet_like('d')


@pytest.mark.parametrize('mult,n', list(LINE_COUNTS.items()))
def test_expected_line_count_known(mult, n):
    assert expected_line_count(mult) == n


def test_expected_line_count_unknown_uses_fallback():
    assert expected_line_count('sept', fallback=7) == 7


# ── J parsing ─────────────────────────────────────────────────────────────────

def test_parse_first_j_takes_first_positive():
    assert parse_first_j('J = 7.2, 1.5 Hz') == 7.2


def test_parse_first_j_none_when_absent():
    assert parse_first_j('') is None
    assert parse_first_j(None) is None
    assert parse_first_j('J = 0 Hz') is None


# ── component scores ──────────────────────────────────────────────────────────

def test_ppm_score_perfect_match_is_one():
    assert ppm_score(0.0, TOL / 2) == pytest.approx(1.0)


def test_ppm_score_decays_monotonically():
    s = [ppm_score(d, TOL / 2) for d in (0.0, 0.01, 0.02, 0.05)]
    assert s == sorted(s, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in s)


def test_dj_score_bands():
    assert dj_score(0.1, 0.20, 0.50) == 1.0          # gold
    assert dj_score(0.35, 0.20, 0.50) == pytest.approx(0.5)   # midpoint decay
    assert dj_score(0.50, 0.20, 0.50) == 0.0         # at the bad gate
    assert dj_score(2.0, 0.20, 0.50) == 0.0


def test_dj_score_unmeasured_is_neutral():
    assert dj_score(None, 0.20, 0.50) == 0.0
    assert dj_score(None, 0.20, 0.50, neutral=0.7) == 0.7


# ── line snapping ─────────────────────────────────────────────────────────────

def test_snap_consumes_each_peak_once():
    """One detected peak cannot satisfy two predicted lines."""
    pred = [{'ppm': 1.30}, {'ppm': 1.31}]
    found = snap_lines_to_peaks(pred, _peaks(1.305), TOL)
    assert len(found) == 1


def test_snap_ignores_peaks_beyond_tolerance():
    assert snap_lines_to_peaks([{'ppm': 1.30}], _peaks(2.00), TOL) == []


def test_dedup_collapses_parent_and_subpeak():
    assert dedup_coincident([1.300, 1.3005, 1.320]) == [1.300, 1.320]


def test_dedup_keeps_genuine_multiplet_lines():
    """J = 6 Hz at 600 MHz → 0.01 ppm apart; must survive."""
    assert len(dedup_coincident([1.300, 1.310])) == 2


def test_spacings_hz_sorted_and_converted():
    assert spacings_hz([1.31, 1.30], FIELD) == pytest.approx([6.0])


# ── J-consistency outlier removal ─────────────────────────────────────────────

def test_drop_outlier_removes_spurious_line():
    # Clean doublet at J = 6 Hz plus an unrelated peak far to the right.
    positions = [1.300, 1.310, 1.400]
    kept = drop_worst_j_outlier(positions, 6.0, FIELD)
    assert kept == [1.300, 1.310]


def test_drop_outlier_keeps_consistent_triplet():
    positions = [1.300, 1.310, 1.320]
    assert drop_worst_j_outlier(positions, 6.0, FIELD) == positions


# ── predicted-line syncing ────────────────────────────────────────────────────

def test_sync_reanchors_lines_and_normalises_intensity():
    original = [{'ppm': 1.300, 'rel_intensity': 0.5},
                {'ppm': 1.310, 'rel_intensity': 1.0}]
    out = sync_predicted_lines(original, [1.3005, 1.3105], TOL)
    assert [l['ppm'] for l in out] == [1.3005, 1.3105]
    assert max(l['rel_intensity'] for l in out) == pytest.approx(1.0)


def test_sync_without_found_lines_returns_original():
    original = [{'ppm': 1.30, 'rel_intensity': 1.0}]
    assert sync_predicted_lines(original, [], TOL) == original


# ── synthetic targets ─────────────────────────────────────────────────────────

def test_synthetic_lines_spacing_matches_j():
    lines = synthetic_lines(1.30, 6.0, 2, FIELD)
    assert spacings_hz([l['ppm'] for l in lines], FIELD) == pytest.approx([6.0])


def test_synthetic_lines_centred_on_hint():
    lines = synthetic_lines(1.30, 6.0, 3, FIELD)
    assert lines[1]['ppm'] == pytest.approx(1.30)


# ── end-to-end measurement ────────────────────────────────────────────────────

def test_measure_complete_doublet():
    m = measure_multiplet(
        [{'ppm': 1.300, 'rel_intensity': 1.0}, {'ppm': 1.310, 'rel_intensity': 1.0}],
        _peaks(1.300, 1.310),
        mult='d', j_coupling='J = 6.0 Hz', db_ppm=1.305, centre_hint=1.305,
        n_expected=2, tolerance=TOL, field_mhz=FIELD)
    assert m.complete
    assert m.n_found == 2
    assert m.observed_j_hz == pytest.approx(6.0)
    assert m.delta_j_hz == pytest.approx(0.0, abs=1e-9)
    assert m.centre_ppm == pytest.approx(1.305)


def test_measure_incomplete_reports_no_delta_j():
    """Missing lines: a stale ΔJ must never survive as evidence."""
    m = measure_multiplet(
        [{'ppm': 1.300}, {'ppm': 1.310}],
        _peaks(1.300),
        mult='d', j_coupling='J = 6.0 Hz', db_ppm=1.305, centre_hint=1.305,
        n_expected=2, tolerance=TOL, field_mhz=FIELD)
    assert m.measured
    assert not m.complete
    assert m.n_found == 1
    assert m.delta_j_hz is None
    assert m.observed_j_hz is None


def test_measure_without_predicted_lines_is_unmeasured():
    m = measure_multiplet([], _peaks(1.30), mult='d', j_coupling='',
                          db_ppm=1.30, centre_hint=1.30, n_expected=2,
                          tolerance=TOL, field_mhz=FIELD)
    assert m.measured is False
    assert m.n_found is None


def test_measure_builds_synthetic_targets_when_lines_truncated():
    """A sub-peak annotation may store a single line; the DB J still defines the rest."""
    m = measure_multiplet(
        [{'ppm': 1.300}],
        _peaks(1.300, 1.310),
        mult='d', j_coupling='J = 6.0 Hz', db_ppm=1.305, centre_hint=1.305,
        n_expected=2, tolerance=TOL, field_mhz=FIELD)
    assert m.n_found == 2
    assert m.delta_j_hz == pytest.approx(0.0, abs=1e-9)


def test_measure_no_db_j_leaves_delta_j_none():
    m = measure_multiplet(
        [{'ppm': 1.300}, {'ppm': 1.310}],
        _peaks(1.300, 1.310),
        mult='d', j_coupling='', db_ppm=1.305, centre_hint=1.305,
        n_expected=2, tolerance=TOL, field_mhz=FIELD)
    assert m.complete
    assert m.observed_j_hz == pytest.approx(6.0)
    assert m.delta_j_hz is None


# ── regression: signal quality must not corrupt the J measurement ─────────────

def test_poor_line_still_counts_as_a_coupling_partner():
    """A Poor-graded line of a doublet is still a line of that doublet.

    Alanine's CH3 doublet (J ≈ 7.2 Hz): if the second line is dropped because it
    graded Poor, the surviving line pairs with an unrelated neighbour and the
    reported J is wrong. Measurement reads every detected peak; the Poor
    exclusion applies to the anchor that carries the annotation, not here.
    """
    j_db = 7.2
    j_ppm = j_db / FIELD
    centre = 1.4750
    line_lo, line_hi = centre - j_ppm / 2, centre + j_ppm / 2
    peaks = [
        {'ppm': line_lo, 'rank': 0, 'signal_quality': 'Good'},
        {'ppm': line_hi, 'rank': 1, 'signal_quality': 'Poor'},   # still a line
        {'ppm': centre + 0.0155, 'rank': 2, 'signal_quality': 'Good'},  # neighbour
    ]
    pred = [{'ppm': line_lo, 'rel_intensity': 1.0},
            {'ppm': line_hi, 'rel_intensity': 1.0}]

    m = measure_multiplet(pred, peaks, mult='d', j_coupling=f'J = {j_db} Hz',
                          db_ppm=centre, centre_hint=centre, n_expected=2,
                          tolerance=TOL, field_mhz=FIELD)
    assert m.complete
    assert m.observed_j_hz == pytest.approx(j_db, abs=0.01)
    assert m.delta_j_hz == pytest.approx(0.0, abs=0.01)


def test_dropping_the_poor_line_fabricates_a_wrong_coupling():
    """Documents the failure mode the regression above guards against."""
    j_db = 7.2
    j_ppm = j_db / FIELD
    centre = 1.4750
    line_lo, line_hi = centre - j_ppm / 2, centre + j_ppm / 2
    without_poor = [
        {'ppm': line_lo, 'rank': 0},
        {'ppm': centre + 0.0155, 'rank': 2},   # unrelated neighbour
    ]
    pred = [{'ppm': line_lo, 'rel_intensity': 1.0},
            {'ppm': line_hi, 'rel_intensity': 1.0}]

    m = measure_multiplet(pred, without_poor, mult='d',
                          j_coupling=f'J = {j_db} Hz', db_ppm=centre,
                          centre_hint=centre, n_expected=2, tolerance=TOL,
                          field_mhz=FIELD)
    # The neighbour is inside the 0.05 ppm window, so the doublet still looks
    # complete — with a coupling constant that was never measured.
    assert m.complete
    assert m.delta_j_hz > 1.0
