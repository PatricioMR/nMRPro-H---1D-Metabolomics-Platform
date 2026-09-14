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
