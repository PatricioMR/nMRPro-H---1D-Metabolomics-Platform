"""A metabolite's signals stand in the proportion of their protons; a
candidate peak far under what the metabolite's tallest signal predicts is
not that signal, whatever its position."""
from processing.annotation_consistency import consistent_candidates

PEAKS = {1: {'rank': 1, 'ppm': 3.928, 'snr': 189.0},     # creatine CH2, 2H
         2: {'rank': 2, 'ppm': 3.026, 'snr': 8.0},       # noise near 3.024
         3: {'rank': 3, 'ppm': 3.044, 'snr': 268.0},     # creatine CH3, 3H
         4: {'rank': 4, 'ppm': 1.470, 'snr': 90.0}}


def _c(mid, sig, rank, score, protons):
    return {'bmrb_id': mid, 'ppm_center': sig, 'peak_rank': rank,
            'composite_score': score, 'protons': protons}


def test_the_peak_the_other_singlet_predicts_is_kept_and_the_noise_dropped():
    ch2 = _c('CRE', 3.920, 1, 0.9, 2)
    near = _c('CRE', 3.024, 2, 0.95, 3)      # nearer, scores higher
    tall = _c('CRE', 3.024, 3, 0.7, 3)
    out = consistent_candidates([ch2, near, tall], PEAKS)
    assert tall in out and ch2 in out
    assert near not in out


def test_a_signal_with_one_candidate_is_untouched():
    ch2 = _c('CRE', 3.920, 1, 0.9, 2)
    only = _c('CRE', 3.024, 2, 0.95, 3)
    assert consistent_candidates([ch2, only], PEAKS) == [ch2, only]


def test_the_reference_is_never_judged_against_itself():
    a = _c('X', 3.024, 3, 0.7, 3)
    b = _c('X', 3.024, 2, 0.9, 3)
    out = consistent_candidates([a, b], PEAKS)
    assert a in out and b not in out


def test_overlap_makes_a_peak_taller_never_shorter_so_taller_passes():
    ref = _c('Y', 1.470, 4, 0.9, 3)
    ref.update({'_n_found_lines': 2, '_n_exp_lines': 2})   # a verified doublet
    small = _c('Y', 3.024, 2, 0.95, 1)     # expected 30, has 8: over a fifth
    big = _c('Y', 3.024, 3, 0.6, 1)        # 268 where 30 expected: overlap
    out = consistent_candidates([ref, small, big], PEAKS)
    assert small in out and big in out


def test_without_protons_or_peaks_nothing_is_dropped():
    a = _c('Z', 3.024, 2, 0.9, None)
    b = _c('Z', 3.024, 3, 0.8, None)
    assert consistent_candidates([a, b], PEAKS) == [a, b]
    c = _c('W', 3.024, None, 0.9, 3)
    d = _c('W', 3.024, 3, 0.8, 3)
    assert consistent_candidates([c, d], PEAKS) == [c, d]
    assert consistent_candidates([], PEAKS) == []


def test_a_verified_multiplet_is_the_reference_over_a_taller_stray_peak():
    d = _c('V', 1.470, 4, 0.9, 3)
    d.update({'_n_found_lines': 2, '_n_exp_lines': 2})
    stray = _c('V', 3.024, 3, 0.6, 1)      # 268 where the doublet predicts 30
    small = _c('V', 3.024, 2, 0.95, 1)     # 8: over a fifth of 30, consistent
    out = consistent_candidates([d, stray, small], PEAKS)
    assert small in out and stray in out and d in out
