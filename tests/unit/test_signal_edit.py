"""Splitting a window that holds two signals, adding one detection missed,
and undoing either -- the Signal List's hand edits."""
import pickle

import numpy as np
import pytest

from processing.peak_detection import as_app_peak, peak_measures
from processing.signal_edit import (
    ORIGIN_ADDED, ORIGIN_SPLIT, add_signal, measure_window, next_rank,
    remove_added_signal, split_signal, undo_split, valley_near)

PPM = np.linspace(4.0, 0.0, 16000)          # 0.00025 ppm per point
NOISE = 0.05


def _lorentz(c, h, w=0.002):
    return h * w ** 2 / ((PPM - c) ** 2 + w ** 2)


def _idx(ppm):
    return int(np.argmin(np.abs(PPM - ppm)))


@pytest.fixture
def merged():
    """Two lines 0.012 ppm apart that detection left in one window, in three
    samples, plus a broad pedestal the baseline model removed."""
    corr = np.column_stack([_lorentz(2.000, 100.0 * k) + _lorentz(1.988, 40.0 * k)
                            for k in (1, 2, 3)])
    raw = corr + 5.0 * np.exp(-((PPM - 2.0) / 0.3) ** 2)[:, None]
    lo, hi = _idx(2.03), _idx(1.96)
    m = measure_window(PPM, raw, corr, lo, hi, NOISE, index=_idx(2.000))
    big = as_app_peak(dict(m, rank=1, signal_quality='Good',
                           signal_quality_score=0.9))
    other = as_app_peak(dict(measure_window(PPM, raw, corr, _idx(3.02),
                                            _idx(2.98), NOISE), rank=2))
    peaks = [big, other]
    return peaks, big, raw, corr


def test_a_split_keeps_the_rank_on_the_piece_that_holds_the_apex(merged):
    peaks, big, raw, corr = merged
    lo, hi = big['start_idx'], big['end_idx']
    cut = valley_near(corr.mean(axis=1), _idx(1.994), lo, hi)
    piece = split_signal(peaks, big, cut, PPM, raw, corr, NOISE)

    assert peaks[0] is big and big['rank'] == 1
    assert piece['rank'] == 3 and piece in peaks
    assert piece['manual_origin'] == ORIGIN_SPLIT
    assert piece['split_from_rank'] == 1
    assert big['ppm'] == pytest.approx(2.000, abs=0.0005)
    assert piece['ppm'] == pytest.approx(1.988, abs=0.0005)
    # the two windows meet at the valley, as detection's neighbours do
    assert big['end_idx'] == cut == piece['start_idx']
    assert big['start_idx'] == lo and piece['end_idx'] == hi
    # each piece is measured by detection's arithmetic on its own window
    for p in (big, piece):
        m = peak_measures(PPM, corr, p['start_idx'], p['end_idx'], NOISE,
                          index=p['index'])
        assert p['area'] == pytest.approx(m['area'])
        assert p['snr'] == pytest.approx(m['snr'])
    assert big['boundaries_edited'] is True


def test_the_cut_snaps_to_the_valley_between_the_two_lines(merged):
    peaks, big, raw, corr = merged
    mean = corr.mean(axis=1)
    lo, hi = big['start_idx'], big['end_idx']
    a, b = _idx(2.000), _idx(1.988)
    true_valley = a + int(np.argmin(mean[a:b + 1]))
    near = valley_near(mean, true_valley + 2, lo, hi)
    assert near == true_valley


def test_a_split_piece_has_no_grade_it_did_not_earn(merged):
    peaks, big, raw, corr = merged
    piece = split_signal(peaks, big, _idx(1.994), PPM, raw, corr, NOISE)
    assert piece['status'] is None
    assert piece['signal_quality_score'] != piece['signal_quality_score']
    assert big['signal_quality'] == 'Good'          # kept, flagged as old bounds


def test_per_sample_apexes_stay_inside_each_piece(merged):
    peaks, big, raw, corr = merged
    piece = split_signal(peaks, big, _idx(1.994), PPM, raw, corr, NOISE)
    for p in (big, piece):
        lo_ppm, hi_ppm = sorted((p['start_ppm'], p['end_ppm']))
        assert all(lo_ppm <= c <= hi_ppm for c in p['ppm_per_sample'])
        assert all(p['aligned_per_sample'])
    assert piece['ppm_per_sample'][0] == pytest.approx(1.988, abs=0.0005)


def test_the_baseline_share_is_recorded_for_a_hand_made_window(merged):
    peaks, big, raw, corr = merged
    piece = split_signal(peaks, big, _idx(1.994), PPM, raw, corr, NOISE)
    assert 0.0 < piece['broad_removed_frac'] < 1.0
    assert len(piece['broad_removed_frac_per_sample']) == 3


def test_undo_restores_the_parent_exactly(merged):
    peaks, big, raw, corr = merged
    before = pickle.dumps(big)
    piece = split_signal(peaks, big, _idx(1.994), PPM, raw, corr, NOISE)
    parent = undo_split(peaks, piece)
    assert parent is big and piece not in peaks and len(peaks) == 2
    assert pickle.dumps(big) == before


def test_undo_survives_a_saved_project(merged):
    peaks, big, raw, corr = merged
    split_signal(peaks, big, _idx(1.994), PPM, raw, corr, NOISE)
    loaded = pickle.loads(pickle.dumps(peaks))
    piece = next(p for p in loaded if p.get('manual_origin') == ORIGIN_SPLIT)
    parent = undo_split(loaded, piece)
    assert len(loaded) == 2
    assert (parent['start_idx'], parent['end_idx']) == (_idx(2.03), _idx(1.96))


def test_undo_refuses_while_a_later_split_of_the_window_stands(merged):
    peaks, big, raw, corr = merged
    first = split_signal(peaks, big, _idx(1.994), PPM, raw, corr, NOISE)
    split_signal(peaks, big, _idx(2.015), PPM, raw, corr, NOISE)
    with pytest.raises(ValueError, match='later'):
        undo_split(peaks, first)


def test_a_cut_too_close_to_an_edge_is_refused(merged):
    peaks, big, raw, corr = merged
    with pytest.raises(ValueError):
        split_signal(peaks, big, big['start_idx'] + 1, PPM, raw, corr, NOISE)
    assert len(peaks) == 2


def test_a_new_signal_cannot_cover_another_but_may_meet_it(merged):
    peaks, big, raw, corr = merged
    with pytest.raises(ValueError, match='#1'):
        add_signal(peaks, _idx(2.00), _idx(1.90), PPM, raw, corr, NOISE)
    p = add_signal(peaks, big['end_idx'], _idx(1.90), PPM, raw, corr, NOISE)
    assert p['manual_origin'] == ORIGIN_ADDED and p['rank'] == 3
    assert p['start_idx'] == big['end_idx']


def test_an_added_signal_is_removed_and_a_detected_one_is_not(merged):
    peaks, big, raw, corr = merged
    p = add_signal(peaks, _idx(1.5), _idx(1.4), PPM, raw, corr, NOISE)
    remove_added_signal(peaks, p)
    assert p not in peaks
    with pytest.raises(ValueError):
        remove_added_signal(peaks, big)


def test_next_rank_never_reuses_one():
    assert next_rank([{'rank': 4}, {'rank': 9}, {'rank': '_sub_1'}]) == 10
    assert next_rank([]) == 1
