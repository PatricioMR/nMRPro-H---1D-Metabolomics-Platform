"""Valley-bounded segment alignment (processing.peak_alignment)."""
import numpy as np

from processing.peak_alignment import (
    representative_sample, valley_segment_align, valley_segments)

PPM = np.linspace(4.0, 0.0, 16000)            # descending, as NMR is
STEP = abs(PPM[1] - PPM[0])


def _lor(c, h, w=0.0015):
    return h / (1.0 + ((PPM - c) / w) ** 2)


def _study(shifts_by_signal, heights=None, noise=0.0, seed=0):
    """Columns of signals that each move by their own amount per sample:
    ``shifts_by_signal[k][s]`` is signal k's offset (ppm) in sample s."""
    rng = np.random.default_rng(seed)
    centres = [3.56, 1.92, 3.04]
    heights = heights or [1.0, 0.6, 2.0]
    n_s = len(shifts_by_signal[0])
    cols = []
    for s in range(n_s):
        y = sum(_lor(c + sh[s], h) for c, sh, h
                in zip(centres, shifts_by_signal, heights))
        cols.append(y + rng.normal(0, noise, PPM.size))
    return np.column_stack(cols), centres


def _apex(y, c, r=0.02):
    k = np.abs(PPM - c) <= r
    return float(PPM[k][np.argmax(y[k])])


def test_each_signal_is_brought_onto_the_reference_by_its_own_amount():
    # glycine moves one way, acetate the other, creatine stays: one rigid
    # shift per sample cannot do this
    sh = [[0.0, 0.006, -0.004, 0.005],
          [0.0, -0.009, 0.008, -0.007],
          [0.0, 0.0, 0.001, 0.0]]
    X, centres = _study(sh, noise=1e-3)
    out, rec = valley_segment_align(PPM, X, noise_sigma=1e-3, ref_idx=0,
                                    max_shift_ppm=0.015)
    for c in centres:
        ref = _apex(out[:, 0], c)
        for s in range(1, X.shape[1]):
            assert abs(_apex(out[:, s], c) - ref) <= 1.5 * STEP


def test_a_moved_peak_keeps_its_area_exactly():
    sh = [[0.0, 0.006], [0.0, -0.009], [0.0, 0.0]]
    X, centres = _study(sh)
    out, _ = valley_segment_align(PPM, X, noise_sigma=1e-4, ref_idx=0)
    for c, moved in zip(centres, (0.006, -0.009, 0.0)):
        after = out[np.abs(PPM - c) <= 0.03, 1].sum()
        before = X[np.abs(PPM - (c + moved)) <= 0.03, 1].sum()
        assert np.isclose(after, before, rtol=2e-3)


def test_no_peak_is_copied_when_a_neighbour_moves():
    # a tall peak beside a small one, both moving in the sample: the moved
    # segment must not carry a second copy of the tall one
    X = np.column_stack([_lor(3.040, 10.0) + _lor(3.034, 0.5),
                         _lor(3.045, 10.0) + _lor(3.039, 0.5)])
    out, _ = valley_segment_align(PPM, X, noise_sigma=1e-3, ref_idx=0)
    k = (PPM > 3.0) & (PPM < 3.08)
    tall = out[k, 1] > 5.0
    # one contiguous run of the tall peak, not two
    assert np.count_nonzero(np.diff(tall.astype(int)) == 1) == 1


def test_a_signal_absent_from_a_sample_is_not_dragged_onto_noise():
    rng = np.random.default_rng(1)
    X = np.column_stack([_lor(2.0, 1.0), rng.normal(0, 1e-3, PPM.size)])
    out, rec = valley_segment_align(PPM, X, noise_sigma=1e-3, ref_idx=0)
    assert np.array_equal(out[:, 1], X[:, 1])
    assert not rec['shifts_pts'][:, 1].any()


def test_segments_tile_the_axis_and_never_cut_a_peak():
    y = _lor(3.56, 1.0) + _lor(1.92, 1.0)
    segs = valley_segments(y, 1e-3, step_ppm=STEP, max_seg_ppm=0.06)
    assert segs[0][0] == 0 and segs[-1][1] == PPM.size
    assert all(a2 == b1 for (_, b1), (a2, _) in zip(segs, segs[1:]))
    cuts = [a for a, _ in segs[1:]]
    for c in (3.56, 1.92):
        apex = int(np.argmin(np.abs(PPM - c)))
        assert all(abs(cut - apex) > 10 for cut in cuts)


def test_a_wide_cluster_is_split_only_at_a_deep_valley():
    # two singlets 0.05 ppm apart with a deep valley between: split; a
    # doublet 0.012 apart (one blob): not split
    pair = _lor(3.00, 1.0) + _lor(2.95, 1.0)
    segs = valley_segments(pair, 1e-4, step_ppm=STEP, max_seg_ppm=0.03)
    inside = [a for a, _ in segs if 2.95 < PPM[a] < 3.00]
    assert inside
    doublet = _lor(3.006, 1.0) + _lor(2.994, 1.0)
    segs = valley_segments(doublet, 1e-4, step_ppm=STEP, max_seg_ppm=0.06)
    assert not [a for a, _ in segs if 2.994 < PPM[a] < 3.006]


def test_the_reference_is_a_real_sample_that_agrees_most_with_the_rest():
    base = _lor(2.0, 1.0)
    odd = _lor(1.5, 1.0)
    X = np.column_stack([odd, base, base * 1.1, base * 0.9])
    assert representative_sample(X) in (1, 2, 3)
    assert representative_sample(X, columns=[0, 3]) in (0, 3)


def test_the_record_says_what_was_moved():
    sh = [[0.0, 0.006], [0.0, -0.009], [0.0, 0.0]]
    X, _ = _study(sh)
    _out, rec = valley_segment_align(PPM, X, noise_sigma=1e-4, ref_idx=0)
    assert rec['ref_idx'] == 0
    assert rec['shifts_pts'].shape == (len(rec['segments_ppm']), 2)
    moved = rec['shifts_pts'][:, 1]
    got = sorted({int(round(abs(m) * STEP * 1000)) for m in moved if m})
    assert got == [6, 9]
