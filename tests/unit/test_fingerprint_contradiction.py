"""If it were here, would I see the big one?

Every annotation gate judges a signal on its own. On the reference study eight
para-substituted phenols passed on the same doublet at 6.9 ppm — 2 of 2 lines,
ΔJ under 0.5 Hz, Δppm under 0.05, every one Good — and every one is right
about the signal: the ortho protons of any of them give that doublet. What
tells them apart is the signals they do not share, and 4-hydroxybenzaldehyde's
aldehyde proton at 9.7 ppm is the plainest case: if the doublet is its 2H, the
singlet is half that area, and the region holds noise.

Absence is evidence only when absence was measurable, so the gate fires under
four conditions: the missing signal would have been unmistakable, its region
is not excluded, no detected peak sits in its window, and what is there falls
short by a factor and not by a little.
"""
import numpy as np
import pytest

from processing.annotation_detectability import (
    CONTRADICTION_MIN_EXPECTED_SNR, CONTRADICTION_RATIO,
    fingerprint_contradictions)

PPM = np.linspace(0.0, 10.0, 20001)
NOISE = 1.0


def _lorentz(centre, height, hwhm=0.002):
    return height * hwhm ** 2 / ((PPM - centre) ** 2 + hwhm ** 2)


def _spectrum(*lines, seed=0):
    rng = np.random.default_rng(seed)
    # A noise tail above 9.5 ppm is what estimate_noise reads, so the noise
    # must reach there and the lines must not.
    y = rng.normal(0.0, NOISE, PPM.size)
    for centre, height in lines:
        y += _lorentz(centre, height)
    return y


def _peak(rank, ppm):
    return {'rank': rank, 'ppm': float(ppm)}


def _met(name, matched, db, *, quality='good'):
    return {
        'id': name, 'name': name,
        'matched_peaks': [
            {'peak_rank': r, 'peak_ppm': p, 'db_ppm': p, 'type': mult,
             'protons': nh, 'annotation_quality': quality}
            for r, p, mult, nh in matched],
        'db_peaks': [{'ppm_center': p, 'type': mult, 'protons': nh}
                     for p, mult, nh in db],
    }


def _aldehyde(y_extra=(), *, aldehyde_height=0.0):
    """4-hydroxybenzaldehyde as the database has it: a 2H doublet at 6.92 and
    a 1H aldehyde singlet at 9.72 (a 2H doublet at 7.8 too, left out here).
    Only the 6.92 doublet was matched."""
    lines = [(6.915, 200.0), (6.929, 200.0)] + list(y_extra)
    if aldehyde_height:
        lines.append((9.72, aldehyde_height))
    y = _spectrum(*lines)
    peaks = [_peak(1, 6.915), _peak(2, 6.929)]
    if aldehyde_height:
        peaks.append(_peak(3, 9.72))
    met = _met('4-Hydroxybenzaldehyde',
               matched=[(1, 6.92, 'd', 2.0)],
               db=[(6.92, 'd', 2.0), (9.72, 's', 1.0)])
    # As annotation leaves it: peak_ppm is the anchored line's apex, db_ppm
    # the multiplet's centre. The centre of a doublet is a valley.
    met['matched_peaks'][0]['peak_ppm'] = 6.915
    return y, peaks, met


def _run(y, peaks, mets, **kw):
    args = dict(tolerance_ppm=0.05, excluded_regions=[(4.5, 5.0)])
    args.update(kw)
    return fingerprint_contradictions(mets, PPM, y, peaks, **args)


def test_a_missing_aldehyde_singlet_contradicts_the_doublet():
    """The doublet reads SNR 200 per line, 2H over two lines; the 1H singlet
    would then read 200 too. The region holds noise."""
    y, peaks, met = _aldehyde()
    assert _run(y, peaks, [met]) == 1
    assert all(mp['annotation_quality'] == 'bad'
               for mp in met['matched_peaks'])
    c = met['_fingerprint_contradictions']
    assert len(c) == 1 and c[0]['ppm'] == pytest.approx(9.72)
    assert c[0]['expected_snr'] > CONTRADICTION_MIN_EXPECTED_SNR
    assert 'Fingerprint contradicted' in met['matched_peaks'][0][
        '_status_reasons'][0]


def test_an_aldehyde_singlet_that_is_there_confirms_it():
    y, peaks, met = _aldehyde(aldehyde_height=180.0)
    assert _run(y, peaks, [met]) == 0
    assert met['matched_peaks'][0]['annotation_quality'] == 'good'


def test_a_peak_of_anybody_in_the_window_is_not_absence():
    """Something there. Overlap, or this very signal drifted further than
    allowed for — either way, nothing was measured absent."""
    y, peaks, met = _aldehyde(y_extra=[(9.74, 30.0)])
    peaks.append(_peak(9, 9.74))
    assert _run(y, peaks, [met]) == 0


def test_a_peak_of_noise_in_the_window_excuses_nothing():
    """Creatinine's second singlet, expected at SNR 268, had a peak of SNR 7
    in its window and was excused by it. A detected peak stands in for the
    signal only when it is tall enough to be it."""
    y, peaks, met = _aldehyde(y_extra=[(9.74, 7.0)])
    noise_peak = _peak(9, 9.74)
    noise_peak['snr'] = 7.0
    peaks.append(noise_peak)
    stats = {}
    assert _run(y, peaks, [met], stats=stats) == 1
    assert stats['contradicted'] == 1 and stats['peak_in_window'] == 0


def test_a_tall_enough_peak_in_the_window_still_excuses():
    y, peaks, met = _aldehyde(y_extra=[(9.74, 300.0)])
    tall = _peak(9, 9.74)
    tall['snr'] = 300.0
    peaks.append(tall)
    assert _run(y, peaks, [met]) == 0


def test_a_signal_matched_to_a_peak_of_noise_is_looked_at_as_missing():
    """Creatinine's second singlet was matched to a peak of SNR 7 where its
    anchor said 268, and being matched it was never questioned."""
    y, peaks, met = _aldehyde(y_extra=[(9.74, 7.0)])
    noise_peak = _peak(9, 9.74)
    noise_peak['snr'] = 7.0
    peaks.append(noise_peak)
    # the aldehyde singlet is "matched" -- to the noise peak
    met['matched_peaks'].append({'db_ppm': 9.72, 'peak_ppm': 9.74,
                                 'peak_rank': 9, 'type': 's', 'protons': 1,
                                 'annotation_quality': None})
    stats = {}
    assert _run(y, peaks, [met], stats=stats) == 1
    assert stats.get('matched_to_noise') == 1


def test_a_tall_peak_beyond_the_metabolites_own_drift_excuses_nothing():
    """Creatinine's two singlets sat +0.010 and +0.013 from the database; a
    third at +0.060 is not the same molecule. Lactate's quartet line 0.05
    ppm away must not stand in for the missing one."""
    y, peaks, met = _aldehyde(y_extra=[(9.77, 300.0)])     # +0.05 from 9.72
    tall = _peak(9, 9.77)
    tall['snr'] = 300.0
    peaks.append(tall)
    stats = {}
    # the doublet sits +0.0075 from the database (own drift); with the
    # linewidth the window is a few millippm, and 9.77 is outside it
    assert _run(y, peaks, [met], stats=stats, linewidth_ppm=0.003) == 1
    # without a linewidth the study window stands, as before
    assert _run(y, peaks, [met]) == 0


def test_a_signal_under_the_water_is_not_absence():
    """Lactate's CH quartet at 4.11 is the canonical case: 3H methyl doublet
    found, 1H quartet expected at 1/3 of it, and it sits where the solvent
    was suppressed. Nothing in that region counts for or against."""
    y = _spectrum((1.324, 600.0), (1.336, 600.0))
    peaks = [_peak(1, 1.324), _peak(2, 1.336)]
    met = _met('L-Lactic acid', matched=[(1, 1.33, 'd', 3.0)],
               db=[(1.33, 'd', 3.0), (4.11, 'q', 1.0)])
    assert _run(y, peaks, [met], excluded_regions=[(4.05, 5.0)]) == 0
    assert met['matched_peaks'][0]['annotation_quality'] == 'good'


def test_a_signal_that_would_be_faint_is_not_absence():
    """A 1H line expected at SNR 12 off a 3H anchor at 36: below the floor,
    'absent' and 'buried' are the same observation."""
    y = _spectrum((1.324, 36.0), (1.336, 36.0))
    peaks = [_peak(1, 1.324), _peak(2, 1.336)]
    met = _met('X', matched=[(1, 1.33, 'd', 3.0)],
               db=[(1.33, 'd', 3.0), (7.5, 's', 1.0)])
    assert _run(y, peaks, [met]) == 0


def test_a_weak_line_where_a_strong_one_was_expected_is_not_contradiction():
    """The ratio, not the floor. Expected 200, found 60: short, but a line is
    there and no detected peak marks it — annotation may have missed it.
    Contradiction needs noise, not a disappointment."""
    y, peaks, met = _aldehyde(y_extra=[(9.72, 60.0)])
    assert _run(y, peaks, [met]) == 0


def test_the_multiplicity_sets_the_expected_height():
    """A 3H triplet anchor at apex 100 is 1.5 proton-lines; a 1H singlet from
    it is expected at 100 / 1.5 ≈ 67, not at 33. Getting the share wrong
    would let a contradiction hide behind a mis-expected height."""
    y = _spectrum((2.09, 50.0), (2.10, 100.0), (2.11, 50.0))
    peaks = [_peak(1, 2.09), _peak(2, 2.10), _peak(3, 2.11)]
    met = _met('T', matched=[(2, 2.10, 't', 3.0)],
               db=[(2.10, 't', 3.0), (8.0, 's', 1.0)])
    assert _run(y, peaks, [met]) == 1
    assert met['_fingerprint_contradictions'][0]['expected_snr'] == (
        pytest.approx(100.0 / 1.5, rel=0.1))


def test_a_manual_mark_is_never_overwritten():
    y, peaks, met = _aldehyde()
    marks = {('4-Hydroxybenzaldehyde', 1): 'good'}
    assert _run(y, peaks, [met], manual_marks=marks) == 0
    assert met['matched_peaks'][0]['annotation_quality'] == 'good'
    # The contradiction is still recorded for the card to show.
    assert met['_fingerprint_contradictions']


def test_an_unresolved_multiplet_in_the_database_says_nothing():
    """'m' carries no line pattern and no J; its absence is not measurable."""
    y, peaks, met = _aldehyde()
    met['db_peaks'][1]['type'] = 'm'
    assert _run(y, peaks, [met]) == 0


def test_an_analyte_already_all_bad_is_left_alone():
    y, peaks, met = _aldehyde()
    met['matched_peaks'][0]['annotation_quality'] = 'bad'
    assert _run(y, peaks, [met]) == 0


def test_a_cleared_contradiction_is_removed():
    """Rerun after the analyst changed something: a stale contradiction must
    not survive on the metabolite."""
    y, peaks, met = _aldehyde()
    _run(y, peaks, [met])
    assert met['_fingerprint_contradictions']
    for mp in met['matched_peaks']:
        mp['annotation_quality'] = 'good'
    y2, peaks2, _ = _aldehyde(aldehyde_height=180.0)
    _run(y2, peaks2, [met])
    assert '_fingerprint_contradictions' not in met


def test_the_thresholds_are_conservative():
    """The floor sits three times detection's own SNR of 6, and pure noise
    peak to peak reads about 4 sigma — so the ratio only condemns a region
    holding the noise itself."""
    assert CONTRADICTION_MIN_EXPECTED_SNR >= 18.0
    assert CONTRADICTION_RATIO >= 4.0


class TestWhereAMissingSignalIsLookedFor:
    """The annotation tolerance answers "could this be the same signal" and is
    generous by design. Used as the width of the window, it held a detected
    peak of somebody nearly every time — the reference study's peaks sit
    0.026 ppm apart on average — and the gate fired on nothing that mattered.
    """

    def test_the_window_is_the_drift_with_margin(self):
        from processing.annotation_detectability import (
            DRIFT_MULTIPLE, search_half_width)
        assert search_half_width(tolerance_ppm=0.05, drift_ppm=0.004,
                                 linewidth_ppm=0.003) == pytest.approx(
            DRIFT_MULTIPLE * 0.004)

    def test_never_narrower_than_the_line(self):
        from processing.annotation_detectability import search_half_width
        assert search_half_width(tolerance_ppm=0.05, drift_ppm=0.0,
                                 linewidth_ppm=0.004) == pytest.approx(0.008)

    def test_never_wider_than_the_tolerance(self):
        """The search would not have matched anything further out either."""
        from processing.annotation_detectability import search_half_width
        assert search_half_width(tolerance_ppm=0.05, drift_ppm=0.03,
                                 linewidth_ppm=0.004) == pytest.approx(0.05)

    def test_a_peak_beyond_the_drift_is_not_this_signal(self):
        """A neighbour 0.04 ppm away no longer excuses an absence when the
        study's own drift is 0.003 ppm."""
        y, peaks, met = _aldehyde(y_extra=[(9.76, 30.0)])
        peaks.append(_peak(9, 9.76))
        assert _run(y, peaks, [met], search_ppm=0.01) == 1

    def test_the_stats_say_where_each_signal_stopped(self):
        y, peaks, met = _aldehyde()
        stats: dict = {}
        _run(y, peaks, [met], stats=stats)
        assert stats['predicted'] == 2
        assert stats['matched'] == 1
        assert stats['contradicted'] == 1
        assert stats['analytes_marked'] == 1


def test_a_missing_signal_is_looked_for_where_its_siblings_say_it_is():
    """A molecule's resonances drift together. On the reference study
    4-hydroxyphenylacetic acid's matched doublet sat +0.043 ppm from the
    database; its unmatched sibling, looked for at the database position,
    found noise, and the analyte was contradicted by its own drift."""
    y = _spectrum((6.955, 200.0), (6.969, 200.0),      # matched, drifted +0.04
                  (6.885, 150.0), (6.899, 150.0))      # sibling, drifted too
    peaks = [_peak(1, 6.955), _peak(2, 6.969), _peak(3, 6.885), _peak(4, 6.899)]
    met = _met('4-Hydroxyphenylacetic acid',
               matched=[(1, 6.92, 'd', 2.0)],
               db=[(6.92, 'd', 2.0), (6.852, 'd', 2.0)])
    met['matched_peaks'][0]['peak_ppm'] = 6.96
    # Narrow search: at the database position 6.852 +/- 0.01 there is nothing.
    assert _run(y, peaks, [met], search_ppm=0.01) == 0
    assert met['_fingerprint_verdicts']['own_shift_ppm'] == pytest.approx(0.04)


def test_detections_noise_is_used_when_given():
    """The floor is a multiple of detection's SNR threshold, so it must be
    measured against detection's noise or it is not the floor it says it is.
    With a noise five times the tail's, an expected SNR of 200 becomes 40 and
    the region's 4 becomes 0.8 -- still contradicted, but on the right scale."""
    y, peaks, met = _aldehyde()
    stats: dict = {}
    _run(y, peaks, [met], noise=5.0 * NOISE, stats=stats)
    assert stats['noise'] == pytest.approx(5.0 * NOISE)
    c = met['_fingerprint_contradictions'][0]
    assert c['expected_snr'] == pytest.approx(40.0, rel=0.15)


def test_a_neighbour_on_the_tallest_match_does_not_set_the_scale():
    """Creatinine's methyl at 3.034 sits on creatine's, 25 times its own size:
    read from that peak, creatinine's 4.04 singlet "should" stand at hundreds
    and stands at its true height, well clear of the noise -- and creatinine
    was marked Bad on a signal matched 0.1 mppm from the drift. Each matched
    peak is an upper bound on the amount; the lowest of them that is a real
    peak sets it."""
    y = _spectrum((3.034, 2500.0), (4.042, 60.0))
    peaks = [{'rank': 1, 'ppm': 3.034, 'snr': 2500.0},
             {'rank': 2, 'ppm': 4.042, 'snr': 60.0}]
    met = _met('creatinine', [(1, 3.034, 's', 3), (2, 4.042, 's', 2)],
               [(3.034, 's', 3), (4.042, 's', 2)])
    assert _run(y, peaks, [met]) == 0
