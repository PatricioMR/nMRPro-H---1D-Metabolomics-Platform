"""Two gates on a winner's measured amount (ADR-025).

A winner whose amount its own fingerprint contradicts -- another listed
signal, at that amount, would stand threefold above what its region holds
-- is withdrawn and the election run again without it. A singlet on its own
is held to a minimum signal-to-noise and to one other signal seen at the
amount it implies.
"""
import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.quant_engine import quantify
from processing.quantification import (
    AMOUNT_CONTRADICTED, Candidate, SINGLET_UNSUPPORTED)

FIELD = 600.0
PPM = np.linspace(1.0, 4.0, 30000)
FWHM = 2.6
LW = FWHM / FIELD
SIGMA = 1.0


def _line(centre, height, lines=((0.0, 1.0),)):
    return height * lines_model(PPM, list(lines), centre, FWHM, FIELD)


def _study(parts, n=3, seed=0):
    rng = np.random.default_rng(seed)
    cols = []
    for k in range(n):
        y = np.zeros(PPM.size)
        for centre, height, lines in parts:
            y += _line(centre, height * (k + 1), lines)
        cols.append(y + rng.normal(0, SIGMA, PPM.size))
    return np.column_stack(cols)


def _peak(rank, centre, half=0.006):
    return {'rank': rank, 'ppm': centre, 'start_ppm': centre - half,
            'end_ppm': centre + half, 'height': 1.0}


def _cand(mid, rank, centre, mult, protons, db_peaks, **extra):
    ev = {'multiplicity': mult, 'ppm_diff': 0.001, 'protons': protons,
          'db_ppm': centre, 'study_drift_ppm': 0.0, **extra}
    if mult == 's':
        ev.update({'n_found': 1, 'n_expected': 1, 'peak_ppm': centre,
                   'predicted_ppms': [centre]})
    return Candidate(met_id=mid, peak_rank=rank, line_ranks=frozenset({rank}),
                     evidence=ev, payload={'met': {'db_peaks': db_peaks}})


def _run(cands, peaks, X, **kw):
    return {r['hmdb_id']: r for r in quantify(
        candidates=cands, peaks_by_rank=peaks, ppm=PPM, intensities=X,
        samples_info=[], included=[0, 1, 2], groups={'G': [0, 1, 2]},
        dj_gold_hz=0.2, noise_sigma=SIGMA, linewidth_ppm=LW,
        deconvolve=False, **kw)}


class TestTheSingletGate:
    def test_a_singlet_seen_with_a_second_signal_is_quantified(self):
        X = _study([(2.0, 100.0, ((0.0, 1.0),)), (3.0, 100.0, ((0.0, 1.0),))])
        c = _cand('A', 1, 2.0, 's', 3, [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
                                        {'ppm_center': 3.0, 'type': 's', 'protons': 3}])
        rows = _run([c], {1: _peak(1, 2.0)}, X)
        assert not rows['A']['_not_quantified']

    def test_a_singlet_whose_second_signal_is_missing_is_not(self):
        """The 3.0 ppm singlet the database lists is not in the spectrum."""
        X = _study([(2.0, 100.0, ((0.0, 1.0),))])
        c = _cand('A', 1, 2.0, 's', 3, [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
                                        {'ppm_center': 3.0, 'type': 's', 'protons': 3}])
        rows = _run([c], {1: _peak(1, 2.0)}, X)
        assert rows['A']['_not_quantified']
        assert rows['A']['_rejection'] == AMOUNT_CONTRADICTED

    def test_a_singlet_of_noise_height_is_not(self):
        X = _study([(2.0, 3.0, ((0.0, 1.0),))])
        c = _cand('A', 1, 2.0, 's', 3, [{'ppm_center': 2.0, 'type': 's', 'protons': 3}])
        rows = _run([c], {1: _peak(1, 2.0)}, X)
        assert rows['A']['_rejection'] == SINGLET_UNSUPPORTED

    def test_a_singlet_whose_other_signals_are_too_faint_to_check_passes_on_snr(self):
        """At 0.05 of the amount, the listed second singlet would stand under
        the noise: nothing could be checked, and 'not checked' is not
        'not seen'."""
        X = _study([(2.0, 30.0, ((0.0, 1.0),))])
        c = _cand('A', 1, 2.0, 's', 30, [{'ppm_center': 2.0, 'type': 's', 'protons': 30},
                                         {'ppm_center': 3.0, 'type': 's', 'protons': 1}])
        rows = _run([c], {1: _peak(1, 2.0)}, X)
        assert not rows['A']['_not_quantified']

    def test_a_lone_listed_singlet_needs_only_its_signal_to_noise(self):
        X = _study([(2.0, 100.0, ((0.0, 1.0),))])
        c = _cand('A', 1, 2.0, 's', 3, [{'ppm_center': 2.0, 'type': 's', 'protons': 3}])
        rows = _run([c], {1: _peak(1, 2.0)}, X)
        assert not rows['A']['_not_quantified']

    def test_a_manual_good_mark_overrides_the_gate(self):
        X = _study([(2.0, 3.0, ((0.0, 1.0),))])
        c = _cand('A', 1, 2.0, 's', 3, [{'ppm_center': 2.0, 'type': 's', 'protons': 3}],
                  manual_quality='good')
        rows = _run([c], {1: _peak(1, 2.0)}, X)
        assert not rows['A']['_not_quantified']


class TestTheAmountAgainstTheFingerprint:
    def test_a_winner_its_fingerprint_contradicts_gives_the_peak_to_the_next(self):
        """Two analytes claim one singlet at 2.0. The first is listed with a
        big 9H singlet at 3.5 that the spectrum does not hold; the second is
        listed with a singlet at 3.0 that it does. The first wins the
        election on its evidence, is withdrawn on its amount, and the second
        takes the peak."""
        X = _study([(2.0, 100.0, ((0.0, 1.0),)), (3.0, 100.0, ((0.0, 1.0),))])
        first = _cand('F', 1, 2.0, 's', 3,
                      [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
                       {'ppm_center': 3.5, 'type': 's', 'protons': 9}],
                      ppm_diff=0.0005, score=0.9)
        second = _cand('S', 1, 2.0, 's', 3,
                       [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
                        {'ppm_center': 3.0, 'type': 's', 'protons': 3}],
                       ppm_diff=0.002, score=0.5)
        rows = _run([first, second], {1: _peak(1, 2.0)}, X)
        assert rows['F']['_rejection'] == AMOUNT_CONTRADICTED
        assert '3.50 ppm' in rows['F']['_status_reasons'][0]
        assert not rows['S']['_not_quantified']
