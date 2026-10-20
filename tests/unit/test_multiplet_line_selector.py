# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Unit tests for processing.multiplet_line_selector."""
import pytest
import numpy as np
from processing.multiplet_line_selector import (
    select_clean_lines,
    _nnls_mode,
    _ratio_mode,
    _collect_matched_ranks,
    matched_line_ranks,
    line_match_tolerance,
    _build_result,
    _is_edge_peak,
    NNLS_CLEAN_THRESHOLD,
    NNLS_MARGINAL_THRESHOLD,
    RATIO_MAX_DR_UP,
    RATIO_MAX_DR_DOWN,
    MIN_CLEAN_FRACTION,
    EDGE_PPM_MARGIN,
    LINE_MATCH_FLOOR,
    LINE_MATCH_CAP,
)

PPM_LIMITS = (0.0, 10.0)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _peak(rank, ppm, height=100.0, gq='good', grounded=True):
    return {
        'rank': rank, 'ppm': ppm, 'height': height,
        'gaussian_quality': gq,
        'baseline_grounded': grounded,
        'start_idx': 0, 'end_idx': 10,
    }


def _line(ppm, rel):
    return {'ppm': ppm, 'rel_intensity': rel}


def _hit(mult, predicted, matched_peaks=None, peak_rank=None):
    h = {'multiplicity': mult, 'predicted_lines': predicted, '_mid': 'MET001'}
    if matched_peaks:
        h['matched_peaks'] = matched_peaks
    if peak_rank is not None:
        h['peak_rank'] = peak_rank
    return h


# ── Group 1: _compute_deviation_ratios (via _ratio_mode) ─────────────────────

class TestRatioMode:
    def test_clean_doublet_perfect_1_1(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00, 100.0), 11: _peak(11, 1.02, 100.0)}
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert result['mode'] == 'ratio'
        assert set(result['clean_ranks']) == {10, 11}
        assert result['contaminated_ranks'] == []
        assert abs(result['scale_factor'] - 1.0) < 1e-6
        assert abs(result['clean_intensity_fraction'] - 1.0) < 1e-6

    def test_left_line_contaminated_60pct_inflation(self):
        # Left line is 1.6× expected → DR[0] = 0.6 > 0.35 → contaminated
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00, 160.0), 11: _peak(11, 1.02, 100.0)}
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert 11 in result['clean_ranks']
        assert 10 in result['contaminated_ranks']
        assert abs(result['scale_factor'] - 2.0) < 1e-6  # 1 of 2 lines clean

    def test_right_line_suppressed_40pct(self):
        # Right line is 0.4× the left line.  With minimum-based H_ref, H_ref
        # anchors on the smaller value (h2=40).  h1=100 then appears as
        # DR=+1.5 → contaminated; h2 is the reference and appears clean.
        # Limitation: cannot distinguish "h1 contaminated" from "h2 suppressed"
        # on ratio analysis alone — algorithm conservatively flags the taller line.
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00, 100.0), 11: _peak(11, 1.02, 40.0)}
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert 10 in result['contaminated_ranks']
        assert 11 in result['clean_ranks']

    def test_uniform_contamination_undetectable_by_ratio(self):
        # When all lines are uniformly inflated (e.g. both 3×), the intra-multiplet
        # ratios are still correct (1:1), so ratio analysis reports all lines clean.
        # Uniform contamination requires absolute-level comparison (e.g. vs IS),
        # which is outside the scope of this module.
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00, 300.0), 11: _peak(11, 1.02, 300.0)}
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert result['mode'] == 'ratio'
        assert set(result['clean_ranks']) == {10, 11}

    def test_triplet_perfect_1_2_1(self):
        predicted = [_line(0.99, 0.5), _line(1.00, 1.0), _line(1.01, 0.5)]
        pkr = {
            10: _peak(10, 0.99, 50.0),
            11: _peak(11, 1.00, 100.0),
            12: _peak(12, 1.01, 50.0),
        }
        result = _ratio_mode(predicted, [10, 11, 12], pkr, PPM_LIMITS)
        assert result['mode'] == 'ratio'
        assert len(result['clean_ranks']) == 3
        assert abs(result['scale_factor'] - 1.0) < 1e-6

    def test_triplet_centre_contaminated(self):
        # Centre line 3× expected → contaminated; outer lines clean
        predicted = [_line(0.99, 0.5), _line(1.00, 1.0), _line(1.01, 0.5)]
        pkr = {
            10: _peak(10, 0.99, 50.0),
            11: _peak(11, 1.00, 300.0),   # contaminated
            12: _peak(12, 1.01, 50.0),
        }
        result = _ratio_mode(predicted, [10, 11, 12], pkr, PPM_LIMITS)
        assert 10 in result['clean_ranks']
        assert 12 in result['clean_ranks']
        assert 11 in result['contaminated_ranks']
        # clean_rel = 0.5+0.5 = 1.0; total_rel = 2.0 → scale = 2.0; cif = 0.5
        assert abs(result['scale_factor'] - 2.0) < 1e-6
        assert abs(result['clean_intensity_fraction'] - 0.5) < 1e-6

    def test_overlap_quality_always_contaminated_despite_dr(self):
        # DR is fine but gaussian_quality='overlap' → contaminated
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {
            10: _peak(10, 1.00, 100.0, gq='overlap'),
            11: _peak(11, 1.02, 100.0),
        }
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert 10 in result['contaminated_ranks']
        assert 11 in result['clean_ranks']

    def test_poor_quality_tighter_threshold(self):
        # DR = 0.25 > POOR_QUALITY_DR_UP (0.20) but < RATIO_MAX_DR_UP (0.35)
        # → contaminated because gaussian_quality='poor'
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {
            10: _peak(10, 1.00, 125.0, gq='poor'),   # DR = 0.25
            11: _peak(11, 1.02, 100.0),
        }
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert 10 in result['contaminated_ranks']

    def test_unmatched_line_is_contaminated(self):
        predicted = [_line(0.99, 0.5), _line(1.00, 1.0), _line(1.01, 0.5)]
        pkr = {
            10: _peak(10, 0.99, 50.0),
            11: _peak(11, 1.00, 100.0),
            # no peak for line at 1.01
        }
        result = _ratio_mode(predicted, [10, 11, None], pkr, PPM_LIMITS)
        assert None not in result['clean_ranks']
        assert -1 in result['contaminated_ranks']


# ── Group 2: _collect_matched_ranks ──────────────────────────────────────────

class TestMatchLines:
    def test_exact_match_triplet(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0), _line(1.04, 1.0)]
        pkr = {
            10: _peak(10, 1.00), 11: _peak(11, 1.02), 12: _peak(12, 1.04),
        }
        hit = _hit('t', predicted,
                   matched_peaks=[{'rank': 10}, {'rank': 11}, {'rank': 12}])
        ranks = _collect_matched_ranks(hit, pkr)
        assert ranks == [10, 11, 12]

    def test_no_peak_for_one_line(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0), _line(1.04, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02)}  # no rank 12
        hit = _hit('t', predicted,
                   matched_peaks=[{'rank': 10}, {'rank': 11}])
        ranks = _collect_matched_ranks(hit, pkr)
        assert ranks[0] == 10
        assert ranks[1] == 11
        assert ranks[2] is None

    def test_greedy_no_double_assignment(self):
        # Two predicted lines at almost the same ppm; only one actual peak exists.
        # The higher-rel line should win the match.
        predicted = [_line(1.000, 0.5), _line(1.001, 1.0)]
        pkr = {10: _peak(10, 1.001)}
        hit = _hit('dd', predicted, matched_peaks=[{'rank': 10}])
        ranks = _collect_matched_ranks(hit, pkr)
        # Higher rel (1.0 at 1.001) wins rank 10; lower rel gets None
        assert ranks[1] == 10
        assert ranks[0] is None


# ── Group 2a: line_match_tolerance (adaptive, spacing-derived) ────────────────

class TestLineMatchTolerance:
    def test_doublet_half_spacing(self):
        # spacing 0.02 → 0.5 × 0.02 = 0.01
        assert line_match_tolerance([1.00, 1.02]) == pytest.approx(0.01)

    def test_triplet_uses_min_gap(self):
        # min gap 0.006 → 0.003
        assert line_match_tolerance([0.994, 1.000, 1.010]) == pytest.approx(0.003)

    def test_singlet_returns_floor(self):
        assert line_match_tolerance([1.00]) == LINE_MATCH_FLOOR
        assert line_match_tolerance([]) == LINE_MATCH_FLOOR

    def test_degenerate_spacing_clamped_to_floor(self):
        # spacing 0.001 → 0.0005, below floor → floor
        assert line_match_tolerance([1.000, 1.001]) == LINE_MATCH_FLOOR

    def test_wide_spacing_capped(self):
        # spacing 0.10 → 0.05, above cap → cap
        assert line_match_tolerance([1.00, 1.10]) == LINE_MATCH_CAP

    def test_independent_of_order(self):
        assert line_match_tolerance([1.02, 1.00]) == pytest.approx(0.01)


# ── Group 2b: matched_line_ranks (greedy claim of own legs only) ─────────────

class TestMatchedLineRanks:
    def test_returns_only_own_legs(self):
        # Doublet at 1.00/1.02 owns ranks 10,11. A neighbour peak at 1.05 (rank 99)
        # falls outside the J tolerance and must NOT be claimed.
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02), 99: _peak(99, 1.05)}
        hit = _hit('d', predicted,
                   matched_peaks=[{'rank': 10}, {'rank': 11}])
        assert set(matched_line_ranks(hit, pkr)) == {10, 11}

    def test_drops_unmatched_lines(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0), _line(1.04, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02)}  # no rank for 1.04
        hit = _hit('t', predicted, matched_peaks=[{'rank': 10}, {'rank': 11}])
        result = matched_line_ranks(hit, pkr)
        assert None not in result
        assert set(result) == {10, 11}

    def test_does_not_steal_neighbour_in_window(self):
        # Two predicted lines but a foreign peak (rank 50) sits BETWEEN them.
        # Greedy ppm matching binds 10/11 to the predicted lines; 50 stays free.
        predicted = [_line(1.00, 1.0), _line(1.06, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.06), 50: _peak(50, 1.03)}
        hit = _hit('d', predicted,
                   matched_peaks=[{'rank': 10}, {'rank': 11}, {'rank': 50}])
        assert 50 not in matched_line_ranks(hit, pkr)

    def test_tight_tolerance_rejects_near_neighbour(self):
        # Doublet 1.00/1.02 (spacing 0.02 → tol 0.01). A foreign peak at 1.032
        # is 0.012 ppm from the 1.02 line: the OLD fixed 0.015 tolerance would
        # have claimed it (no real peak at 1.02); the spacing-derived tolerance
        # rejects it, so only the genuine leg (rank 10) is claimed.
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00), 99: _peak(99, 1.032)}
        hit = _hit('d', predicted,
                   matched_peaks=[{'rank': 10}, {'rank': 99}])
        assert set(matched_line_ranks(hit, pkr)) == {10}


# ── Group 3: dispatch for singlet/m ──────────────────────────────────────────

class TestSingletDispatch:
    def _call(self, mult, predicted=None):
        hit = _hit(mult, predicted or [])
        return select_clean_lines(hit, {}, {}, 'MET001', False, PPM_LIMITS)

    def test_singlet_returns_singlet_mode(self):
        r = self._call('s')
        assert r['mode'] == 'singlet'
        assert r['scale_factor'] == 1.0
        assert r['clean_intensity_fraction'] == 1.0

    def test_m_multiplet_returns_singlet_mode(self):
        assert self._call('m')['mode'] == 'singlet'

    def test_empty_predicted_returns_singlet_mode(self):
        assert self._call('d', [])['mode'] == 'singlet'

    def test_single_predicted_line_returns_singlet_mode(self):
        assert self._call('d', [_line(1.0, 1.0)])['mode'] == 'singlet'


# ── Group 4: NNLS mode ────────────────────────────────────────────────────────

class TestNNLSMode:
    def test_both_clean(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02)}
        jr = {10: {'MET001': 0.95}, 11: {'MET001': 0.92}}
        r = _nnls_mode(predicted, [10, 11], pkr, jr, 'MET001')
        assert r['mode'] == 'nnls'
        assert set(r['clean_ranks']) == {10, 11}
        assert abs(r['scale_factor'] - 1.0) < 1e-6

    def test_one_contaminated_line(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02)}
        jr = {10: {'MET001': 0.95}, 11: {'MET001': 0.20}}
        r = _nnls_mode(predicted, [10, 11], pkr, jr, 'MET001')
        assert r['clean_ranks'] == [10]
        assert 11 in r['contaminated_ranks']
        assert abs(r['scale_factor'] - 2.0) < 1e-6
        assert abs(r['clean_intensity_fraction'] - 0.5) < 1e-6

    def test_both_below_skip_threshold(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02)}
        jr = {10: {'MET001': 0.10}, 11: {'MET001': 0.15}}
        r = _nnls_mode(predicted, [10, 11], pkr, jr, 'MET001')
        assert r['mode'] == 'skip'

    def test_no_nnls_entry_falls_back_to_gaussian_quality(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00, gq='good'), 11: _peak(11, 1.02, gq='overlap')}
        jr = {}  # no NNLS data at all
        r = _nnls_mode(predicted, [10, 11], pkr, jr, 'MET001')
        assert 10 in r['clean_ranks']
        assert 11 in r['contaminated_ranks']

    def test_marginal_excluded(self):
        # own_frac = 0.55 (< CLEAN_THRESHOLD 0.70) → excluded as contaminated (F3),
        # so its >40% contamination is not injected. The genuinely clean line
        # (0.95) drives the area; scale_factor rescales to the full multiplet.
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02)}
        jr = {10: {'MET001': 0.95}, 11: {'MET001': 0.55}}
        r = _nnls_mode(predicted, [10, 11], pkr, jr, 'MET001')
        assert 10 in r['clean_ranks']
        assert 11 in r['contaminated_ranks']
        assert abs(r['scale_factor'] - 2.0) < 1e-6

    def test_unmatched_line_is_contaminated(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00)}
        jr = {10: {'MET001': 0.95}}
        r = _nnls_mode(predicted, [10, None], pkr, jr, 'MET001')
        assert 10 in r['clean_ranks']
        assert -1 in r['contaminated_ranks']


# ── Group 5: _build_result ────────────────────────────────────────────────────

class TestBuildResult:
    def test_scale_factor_quartet_2_of_4_lines_clean(self):
        # Quartet: pred_rel = [0.33, 1.0, 1.0, 0.33] (approx 1:3:3:1 / max)
        # Inner 2 lines clean (rel=1.0 each), outer contaminated
        total_rel = 0.33 + 1.0 + 1.0 + 0.33   # ≈ 2.66
        clean_rel = 1.0 + 1.0                   # = 2.0
        r = _build_result('ratio', [11, 12], [10, 13], total_rel, clean_rel, '')
        expected_sf = total_rel / clean_rel
        assert abs(r['scale_factor'] - expected_sf) < 1e-6

    def test_singlet_scale_factor_one(self):
        r = _build_result('ratio', [10], [], 1.0, 1.0, '')
        assert r['scale_factor'] == 1.0

    def test_below_min_clean_fraction_returns_skip(self):
        # clean_rel = 0.10, total = 1.0 → cif = 0.10 < 0.25
        r = _build_result('ratio', [10], [11, 12, 13], 1.0, 0.10, '')
        assert r['mode'] == 'skip'

    def test_zero_clean_returns_skip(self):
        r = _build_result('ratio', [], [10, 11], 2.0, 0.0, '')
        assert r['mode'] == 'skip'


# ── Group 6: edge peak handling ───────────────────────────────────────────────

class TestEdgePeak:
    def test_edge_peak_flagged_in_report(self):
        # Peak at ppm = 0.01 is within EDGE_PPM_MARGIN of 0.0
        predicted = [_line(0.01, 1.0), _line(0.03, 1.0)]
        pkr = {10: _peak(10, 0.01, 100.0), 11: _peak(11, 0.03, 100.0)}
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert '[edge]' in result['contamination_report']

    def test_edge_peak_not_auto_skip(self):
        # Edge peak with clean DR should not become 'skip'
        predicted = [_line(0.01, 1.0), _line(0.03, 1.0)]
        pkr = {10: _peak(10, 0.01, 100.0), 11: _peak(11, 0.03, 100.0)}
        result = _ratio_mode(predicted, [10, 11], pkr, PPM_LIMITS)
        assert result['mode'] != 'skip'

    def test_is_edge_peak(self):
        assert _is_edge_peak(0.01, (0.0, 10.0)) is True
        assert _is_edge_peak(9.99, (0.0, 10.0)) is True
        assert _is_edge_peak(5.00, (0.0, 10.0)) is False


# ── Group 7: select_clean_lines integration ───────────────────────────────────

class TestSelectCleanLines:
    def test_nnls_path_selected_when_use_nnls_true(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00), 11: _peak(11, 1.02)}
        jr = {10: {'MET001': 0.95}, 11: {'MET001': 0.85}}
        hit = _hit('d', predicted,
                   matched_peaks=[{'rank': 10, 'ppm': 1.00},
                                  {'rank': 11, 'ppm': 1.02}])
        r = select_clean_lines(hit, pkr, jr, 'MET001', True, PPM_LIMITS)
        assert r['mode'] == 'nnls'

    def test_ratio_path_selected_when_use_nnls_false(self):
        predicted = [_line(1.00, 1.0), _line(1.02, 1.0)]
        pkr = {10: _peak(10, 1.00, 100.0), 11: _peak(11, 1.02, 100.0)}
        hit = _hit('d', predicted,
                   matched_peaks=[{'rank': 10, 'ppm': 1.00},
                                  {'rank': 11, 'ppm': 1.02}])
        r = select_clean_lines(hit, pkr, {}, 'MET001', False, PPM_LIMITS)
        assert r['mode'] == 'ratio'

    def test_singlet_bypasses_all_logic(self):
        hit = _hit('s', [_line(1.00, 1.0), _line(1.02, 1.0)])
        r = select_clean_lines(hit, {}, {}, 'MET001', True, PPM_LIMITS)
        assert r['mode'] == 'singlet'

    def test_no_matched_peaks_returns_singlet(self):
        hit = _hit('d', [_line(1.00, 1.0), _line(1.02, 1.0)])
        # No matched_peaks key and no peak_rank → empty candidate set
        r = select_clean_lines(hit, {}, {}, 'MET001', False, PPM_LIMITS)
        assert r['mode'] == 'singlet'
