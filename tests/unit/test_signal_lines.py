# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_signal_lines.py
"""The lines the analyst chose for a signal: read by every reader of the
signal's lines, kept by position, reversible, and marked Good with the
verdict they replaced kept."""
import numpy as np
import pytest

from controllers.signal_lines import SignalLineEditor
from processing.multiplet_line_selector import multiplet_line_peaks
from processing.signal_comparison import pattern_share
from processing.signal_lines import (
    MANUAL_LINES_KEY, clear_manual_lines, lines_before, manual_lines, peaks_at,
    set_manual_lines)


def _peaks():
    return {r: {'rank': r, 'ppm': ppm, 'snr': snr} for r, ppm, snr in (
        (1, 3.812, 40.0), (2, 3.794, 60.0), (3, 3.786, 30.0), (4, 3.774, 55.0),
        (5, 3.762, 50.0), (6, 3.758, 20.0), (7, 1.480, 300.0), (8, 1.468, 290.0))}


def _quartet():
    """Alanine's quartet as an annotation matched it: four predicted lines
    that landed on peaks 1, 2, 4 and 6 -- two of them another compound's."""
    return {'peak_rank': 4, 'peak_ppm': 3.774, 'type': 'q', 'protons': 1,
            'annotation_quality': 'average',
            'predicted_lines': [{'ppm': x} for x in (3.812, 3.794, 3.774, 3.758)]}


class TestManualLines:

    def test_a_signal_without_a_choice_keeps_the_matched_lines(self):
        mp = _quartet()
        assert manual_lines(mp) is None
        assert [p['rank'] for p in multiplet_line_peaks(mp, _peaks())] == [1, 2, 4, 6]

    def test_the_chosen_lines_replace_the_matched_ones(self):
        mp, pk = _quartet(), _peaks()
        set_manual_lines(mp, [pk[2], pk[3], pk[4], pk[5]])
        assert manual_lines(mp) == [3.794, 3.786, 3.774, 3.762]
        assert sorted(p['rank'] for p in multiplet_line_peaks(mp, pk)) == [2, 3, 4, 5]

    def test_the_anchor_moves_to_the_chosen_line_nearest_their_centre(self):
        mp, pk = _quartet(), _peaks()
        rank = set_manual_lines(mp, [pk[1], pk[2], pk[3]])       # the anchor (4) left out
        assert rank == 3 and mp['peak_rank'] == 3                 # 3.786, nearest the centre 3.774

    def test_the_anchor_stays_when_it_is_one_of_the_chosen_lines(self):
        mp, pk = _quartet(), _peaks()
        assert set_manual_lines(mp, [pk[2], pk[3], pk[4], pk[5]]) == 4

    def test_the_same_lines_leave_the_position_where_the_annotation_put_it(self):
        mp, pk = _quartet(), _peaks()
        set_manual_lines(mp, [pk[1], pk[2], pk[4], pk[6]])
        assert mp['peak_ppm'] == pytest.approx(3.774)

    def test_one_line_of_the_pattern_does_not_become_its_centre(self):
        mp, pk = _quartet(), _peaks()
        set_manual_lines(mp, [pk[1]])                            # on its predicted line
        assert mp['peak_ppm'] == pytest.approx(3.774)

    def test_lines_moved_from_the_prediction_move_the_centre_by_as_much(self):
        mp, pk = _quartet(), _peaks()
        set_manual_lines(mp, [pk[3], pk[5]])                     # 3.786, 3.762: 0.008 and 0.004 off
        assert mp['peak_ppm'] == pytest.approx(3.774 + np.median([-0.008, 0.004]))

    def test_a_signal_without_predicted_lines_stands_at_the_mean_of_its_lines(self):
        mp, pk = dict(_quartet(), predicted_lines=[]), _peaks()
        set_manual_lines(mp, [pk[7], pk[8]])
        assert mp['peak_ppm'] == pytest.approx((1.480 + 1.468) / 2)

    def test_a_line_is_found_by_position_after_a_new_detection_renumbers_the_peaks(self):
        mp, pk = _quartet(), _peaks()
        set_manual_lines(mp, [pk[7], pk[8]])
        renumbered = {r + 100: dict(p, rank=r + 100, ppm=p['ppm'] + 0.0005) for r, p in pk.items()}
        assert sorted(p['rank'] for p in multiplet_line_peaks(mp, renumbered)) == [107, 108]

    def test_a_line_gone_is_not_replaced_by_a_neighbour(self):
        mp, pk = _quartet(), _peaks()
        set_manual_lines(mp, [pk[7], pk[8]])
        del pk[8]
        assert [p['rank'] for p in multiplet_line_peaks(mp, pk)] == [7]

    def test_peaks_at_uses_each_peak_once(self):
        pk = _peaks()
        assert [p['rank'] for p in peaks_at([3.7745, 3.7735], pk)] == [4]

    def test_restoring_gives_back_the_anchor_and_the_matched_lines(self):
        mp, pk = _quartet(), _peaks()
        set_manual_lines(mp, [pk[2], pk[3]])
        set_manual_lines(mp, [pk[3], pk[4]])                    # a second edit keeps the first state
        assert lines_before(mp)['peak_rank'] == 4
        assert clear_manual_lines(mp) == 4
        assert mp['peak_rank'] == 4 and mp['peak_ppm'] == 3.774 and MANUAL_LINES_KEY not in mp
        assert [p['rank'] for p in multiplet_line_peaks(mp, pk)] == [1, 2, 4, 6]
        assert clear_manual_lines(mp) is None

    def test_a_signal_needs_a_line(self):
        with pytest.raises(ValueError):
            set_manual_lines(_quartet(), [])

    @pytest.mark.parametrize('bad', [[], 'x', [None], ['a']])
    def test_a_malformed_choice_is_no_choice(self, bad):
        assert manual_lines({MANUAL_LINES_KEY: bad}) is None


class TestPatternShare:
    """The share of a pattern some lines hold, by the quantification's own
    weights (quant_integration.pattern_lines)."""

    EV = {'predicted_ppms': [3.812, 3.794, 3.774, 3.758], 'line_weights': None}

    def test_an_outer_and_an_inner_line_of_a_quartet(self):
        assert pattern_share(self.EV, [3.812], 0.002) == pytest.approx(1 / 8)
        assert pattern_share(self.EV, [3.794], 0.002) == pytest.approx(3 / 8)

    def test_the_lines_together_and_the_whole_pattern(self):
        assert pattern_share(self.EV, [3.794, 3.774], 0.002) == pytest.approx(6 / 8)
        assert pattern_share(self.EV, [3.812, 3.794, 3.774, 3.758], 0.002) == 1.0

    def test_a_peak_on_no_predicted_line_has_no_share(self):
        assert pattern_share(self.EV, [3.786], 0.002) is None

    def test_a_singlet_is_its_line_within_the_tolerance(self):
        ev = {'predicted_ppms': [3.03]}
        assert pattern_share(ev, [3.031], 0.002) == 1.0
        assert pattern_share(ev, [3.05], 0.002) is None

    def test_nothing_predicted_gives_no_share(self):
        assert pattern_share({}, [3.03], 0.002) is None


class TestEditor:

    def _setup(self, mark=None):
        mp = _quartet()
        mets = [{'id': 'HMDB0000161', 'name': 'L-alanine', 'matched_peaks': [mp]}]
        marks = {} if mark is None else {('HMDB0000161', 4): mark}
        return SignalLineEditor(mets, marks), mp, marks

    def test_the_chosen_signal_is_marked_good_on_its_new_anchor(self):
        ed, mp, marks = self._setup()
        pk = _peaks()
        rank = ed.apply('HMDB0000161', 0, [pk[2], pk[3], pk[4], pk[5]])
        assert marks == {('HMDB0000161', rank): 'good'} and mp['annotation_quality'] == 'good'
        assert ed.has_manual('HMDB0000161', 0)

    def test_restoring_puts_back_the_verdict_and_the_mark_it_replaced(self):
        ed, mp, marks = self._setup(mark='bad')
        pk = _peaks()
        ed.apply('HMDB0000161', 0, [pk[3], pk[4]])
        ed.apply('HMDB0000161', 0, [pk[2], pk[3]])
        ed.restore('HMDB0000161', 0)
        assert marks == {('HMDB0000161', 4): 'bad'} and mp['annotation_quality'] == 'average'
        assert not ed.has_manual('HMDB0000161', 0) and mp['peak_rank'] == 4

    def test_restoring_a_signal_never_edited_changes_nothing(self):
        ed, mp, marks = self._setup(mark='bad')
        ed.restore('HMDB0000161', 0)
        assert marks == {('HMDB0000161', 4): 'bad'} and mp['annotation_quality'] == 'average'

    def test_an_unknown_analyte_is_an_error_not_a_silent_skip(self):
        ed, _mp, _marks = self._setup()
        with pytest.raises(KeyError):
            ed.apply('nobody', 0, [_peaks()[1]])
