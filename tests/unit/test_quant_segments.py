# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A multiplet is integrated on its own lines, and a foreign peak between them
is left out.

A triplet with somebody's doublet in its middle used to be integrated foot
to foot, doublet included, because the window was one span from the first
line to the last. Each line is now a window of its own, bounded by the
foreign peaks the way the outer lines always were, and the gap a foreign peak
sits in is not summed. What the segments leave outside is put back by the
same capture correction, summed over the segments.
"""
import numpy as np
import pytest

from processing.quant_deconvolution import fraction_in_window, neighbour_leak
from processing.quant_engine import quantify
from processing.quant_integration import (
    foreign_between_lines, integrate_sample, integrate_signal,
    lorentzian_captured_fraction, signal_segments, signal_window)
from processing.quantification import Candidate

LW = 0.003                                    # 1.8 Hz at 600 MHz
G = LW / 2.0
PPM = np.linspace(1.40, 1.60, 8000)


def _lorentz(c, a):
    return a * G ** 2 / ((PPM - c) ** 2 + G ** 2)


def _peak(rank, centre, half=0.004):
    return {'rank': rank, 'ppm': centre,
            'start_ppm': centre - half, 'end_ppm': centre + half}


TRIPLET = [1.470, 1.482, 1.494]
OWN = [_peak(1, 1.470), _peak(2, 1.482), _peak(3, 1.494)]
INTRUDER = _peak(9, 1.476)                    # between the first two lines


class TestTheSegments:
    def test_without_a_foreign_peak_the_lines_make_one_segment(self):
        segs = signal_segments(OWN, linewidth_ppm=LW, other_peaks=OWN)
        assert len(segs) == 1
        assert segs[0] == signal_window(OWN, linewidth_ppm=LW, other_peaks=OWN)

    def test_a_foreign_peak_between_two_lines_keeps_them_apart(self):
        segs = signal_segments(OWN, linewidth_ppm=LW,
                               other_peaks=OWN + [INTRUDER])
        assert len(segs) == 2
        (lo1, hi1), (lo2, hi2) = segs
        # each line stops at the midpoint to the intruder, less half a
        # linewidth -- the rule the outer lines already obeyed
        assert hi1 == pytest.approx((1.470 + 1.476) / 2 + LW / 2)
        assert lo2 == pytest.approx((1.476 + 1.482) / 2 - LW / 2)
        assert hi1 < 1.476 < lo2
        # the outer bounds are what the whole window had
        assert (lo1, hi2) == signal_window(OWN, linewidth_ppm=LW,
                                           other_peaks=OWN + [INTRUDER])

    def test_the_span_is_unchanged_by_a_peak_inside(self):
        """The window the report shows is the same span it always was."""
        assert signal_window(OWN, linewidth_ppm=LW, other_peaks=OWN) == \
            signal_window(OWN, linewidth_ppm=LW, other_peaks=OWN + [INTRUDER])

    def test_lines_closer_than_two_linewidths_cannot_be_cut(self):
        """A foreign peak between two unresolved lines has nowhere to cut:
        each line keeps half a linewidth past its apex, and the two halves
        meet."""
        own = [_peak(1, 1.470), _peak(2, 1.475)]
        segs = signal_segments(own, linewidth_ppm=LW,
                               other_peaks=own + [_peak(9, 1.4725)])
        assert len(segs) == 1

    def test_without_a_linewidth_the_feet_stand(self):
        """No linewidth, no reach and no neighbour rule: each line's own feet,
        and the span is the feet's span."""
        segs = signal_segments(OWN, linewidth_ppm=None, other_peaks=[INTRUDER])
        assert segs == [(1.466, 1.474), (1.478, 1.486), (1.490, 1.498)]
        assert signal_window(OWN, linewidth_ppm=None) == (1.466, 1.498)

    def test_the_foreign_peaks_between_the_lines_are_named(self):
        assert foreign_between_lines(OWN, OWN + [INTRUDER]) == [1.476]
        # a neighbour outside the outer apices is not "between"
        assert foreign_between_lines(OWN, OWN + [_peak(9, 1.52)]) == []
        # the signal's own lines are never foreign, by rank or identity
        assert foreign_between_lines(OWN, OWN + [dict(OWN[1])]) == []


class TestTheCaptureOverSegments:
    def test_the_sum_over_segments_is_the_sum_of_each(self):
        segs = [(1.461, 1.4745), (1.4775, 1.503)]
        whole = lorentzian_captured_fraction(TRIPLET, [1, 2, 1], 0, 0, G,
                                             min_trusted=0, segments=segs)
        parts = sum(lorentzian_captured_fraction(TRIPLET, [1, 2, 1], a, b, G,
                                                 min_trusted=0)
                    for a, b in segs)
        assert whole == pytest.approx(parts, rel=1e-9)

    def test_the_outer_lines_of_a_triplet_alone_hold_half(self):
        """Pascal's row says 1:2:1 -- the outer lines are half the multiplet,
        and dividing by that half is what puts the middle line back."""
        wide = [(1.470 - 0.05, 1.470 + 0.005), (1.494 - 0.005, 1.494 + 0.05)]
        f = lorentzian_captured_fraction(TRIPLET, [1, 2, 1], 0, 0, G,
                                         min_trusted=0, segments=wide)
        assert f == pytest.approx(0.5, abs=0.06)

    def test_the_leak_model_sums_over_segments_too(self):
        segs = [(1.461, 1.4745), (1.4775, 1.503)]
        whole = fraction_in_window([1.476], [1.0], 0, 0, G, segments=segs)
        parts = sum(fraction_in_window([1.476], [1.0], a, b, G)
                    for a, b in segs)
        assert whole == pytest.approx(parts, rel=1e-9)
        assert neighbour_leak((1.461, 1.503), [([1.476], [1.0], 100.0)], G,
                              segments=segs) == pytest.approx(100.0 * whole)


class TestTheBaselineIsTheWindows:
    def test_the_baseline_is_drawn_through_the_window_not_the_cut(self):
        """A baseline through a segment's own ends would be drawn through the
        foreign peak's flank, and measured that took the area negative."""
        trace = sum(_lorentz(c, a) for c, a in zip(TRIPLET, (1, 2, 1)))
        trace = trace + _lorentz(1.476, 2.0)
        segs = [(1.461, 1.4745), (1.4775, 1.503)]
        one_baseline = integrate_sample(PPM, trace, 1.461, 1.503, segments=segs)
        per_piece = sum(integrate_sample(PPM, trace, a, b) for a, b in segs)
        assert one_baseline > 0
        assert per_piece < one_baseline

    def test_segments_sum_only_what_is_inside_them(self):
        trace = np.ones_like(PPM)
        whole = integrate_sample(PPM, trace, 1.46, 1.50, local_baseline=False)
        half = integrate_sample(PPM, trace, 1.46, 1.50, local_baseline=False,
                                segments=[(1.46, 1.48)])
        assert half == pytest.approx(whole / 2.0, rel=0.01)


class TestTheTripletAroundSomebodysPeak:
    """Lorentzian lines, so the capture correction is exact and the number
    can be checked against the area that went in."""

    def _cand(self):
        return Candidate(met_id='T', peak_rank=2, line_ranks=frozenset([1, 2, 3]),
                         evidence={'multiplicity': 't', 'delta_j_hz': 0.05,
                                   'ppm_diff': 0.001, 'n_found': 3,
                                   'n_expected': 3, 'protons': 3.0})

    def _run(self, trace, peaks):
        X = np.column_stack([trace, trace * 2.0, trace * 3.0])
        return quantify(candidates=[self._cand()], peaks_by_rank=peaks,
                        ppm=PPM, intensities=X, samples_info=[],
                        included=[0, 1, 2], groups={'G': [0, 1, 2]},
                        dj_gold_hz=0.20, linewidth_ppm=LW,
                        deconvolve=False)[0]

    def test_the_intruder_is_left_out_and_said(self):
        triplet = sum(_lorentz(c, a) for c, a in zip(TRIPLET, (1, 2, 1)))
        own = {p['rank']: p for p in OWN}
        clean = self._run(triplet, own)
        with_intruder = self._run(triplet + _lorentz(1.476, 1.0),
                                  {**own, 9: INTRUDER})
        whole = integrate_signal(OWN, PPM, triplet.reshape(-1, 1),
                                 linewidth_ppm=LW, other_peaks=OWN)
        # integrated whole, the intruder's peak would be 26% of the triplet
        assert with_intruder['_foreign_inside'] == [1.476]
        assert len(with_intruder['_integ_segments']) == 2
        assert len(clean['_integ_segments']) == 1
        ratio = with_intruder['raw_area'] / clean['raw_area']
        assert ratio < 1.15
        reasons = ' '.join(with_intruder['_status_reasons'])
        assert 'foreign peak at 1.4760 ppm' in reasons
        assert '2 pieces' in reasons
        assert whole is not None

    def test_a_clean_triplet_is_measured_as_before(self):
        triplet = sum(_lorentz(c, a) for c, a in zip(TRIPLET, (1, 2, 1)))
        r = self._run(triplet, {p['rank']: p for p in OWN})
        assert r['_foreign_inside'] == []
        assert r['_status_reasons'] == [] or all(
            'foreign' not in t for t in r['_status_reasons'])
        true = float(np.trapezoid(triplet, x=PPM))
        assert r['raw_area'] / (2.0 * true) == pytest.approx(1.0, abs=0.12)


class TestThePatternWithLinesMissing:
    """A multiplet detection placed only part of: the area of the placed
    lines, scaled by the pattern to the whole -- or refused when the placed
    part is too small to scale from."""

    def test_the_missing_line_stands_where_it_was_predicted(self):
        from processing.quant_integration import pattern_lines
        ppms, w, mask, share = pattern_lines(TRIPLET, [1.4701, 1.4819])
        assert ppms == [1.4701, 1.4819, 1.494]
        assert w == [1, 2, 1]
        assert mask == [True, True, False]
        assert share == pytest.approx(0.75)

    def test_the_middle_line_alone_is_half_the_pattern(self):
        from processing.quant_integration import pattern_lines
        _, _, mask, share = pattern_lines(TRIPLET, [1.482])
        assert mask == [False, True, False] and share == pytest.approx(0.5)

    def test_all_lines_placed_is_the_pattern_itself(self):
        from processing.quant_integration import pattern_lines
        ppms, w, mask, share = pattern_lines(TRIPLET, [1.470, 1.482, 1.494])
        assert share == 1.0 and all(mask) and w == [1, 2, 1]

    def test_no_prediction_to_scale_by_leaves_the_lines_alone(self):
        from processing.quant_integration import pattern_lines
        ppms, w, mask, share = pattern_lines([], [1.470, 1.482])
        assert ppms == [1.470, 1.482] and share == 1.0

    def _cand(self, ranks):
        return Candidate(met_id='T', peak_rank=2, line_ranks=frozenset(ranks),
                         evidence={'multiplicity': 't', 'delta_j_hz': 0.05,
                                   'ppm_diff': 0.001, 'n_found': 3,
                                   'n_expected': 3, 'protons': 3.0,
                                   'predicted_ppms': list(TRIPLET)})

    def _run(self, cand, trace, peaks):
        X = np.column_stack([trace, trace * 2.0, trace * 3.0])
        # No local baseline: the capture model has no term for the
        # trapezium it removes, and a window ending on a missing line's
        # flank loses a large one (see ADR-016, pending). The scaling
        # itself is what is under test here.
        return quantify(candidates=[cand], peaks_by_rank=peaks, ppm=PPM,
                        intensities=X, samples_info=[], included=[0, 1, 2],
                        groups={'G': [0, 1, 2]}, dj_gold_hz=0.20,
                        linewidth_ppm=LW, deconvolve=False,
                        local_baseline=False)[0]

    def test_two_of_three_lines_are_scaled_to_the_triplet(self):
        triplet = sum(_lorentz(c, a) for c, a in zip(TRIPLET, (1, 2, 1)))
        own = {p['rank']: p for p in OWN}
        whole = self._run(self._cand([1, 2, 3]), triplet, own)
        # the outer line at 1.494 was never placed by detection
        part = self._run(self._cand([1, 2]), triplet, {1: own[1], 2: own[2]})
        assert part['_missing_lines'] == [1.494]
        assert part['_pattern_share'] == pytest.approx(0.75)
        assert part['raw_area'] == pytest.approx(whole['raw_area'], rel=0.05)
        assert any('Partial multiplet' in t for t in part['_status_reasons'])

    def test_one_line_of_three_is_not_quantified_and_says_why(self):
        from processing.quantification import PARTIAL_MULTIPLET
        triplet = sum(_lorentz(c, a) for c, a in zip(TRIPLET, (1, 2, 1)))
        own = {p['rank']: p for p in OWN}
        r = self._run(self._cand([2]), triplet, {2: own[2]})
        assert r['_not_quantified'] is True
        assert r['_rejection'] == PARTIAL_MULTIPLET
        txt = ' '.join(r['_status_reasons'])
        assert '1 of 3 lines' in txt and '50%' in txt and '1.4700' in txt

