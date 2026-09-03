"""The same physical lines cannot be quantified twice.

The old allocation blocked a candidate whose ANCHOR peak was already won. A
multiplet also claims its line peaks — but the next candidate was only ever
tested on its anchor, so a different anchor over the same lines walked straight
in. On the reference study 16 of 111 quantified metabolites were integrating
another's lines: L-alanine and hexanoic acid both on 1.4801/1.4919, L-valine,
L-isoleucine and 2-hydroxy-3-methylbutyric acid all on 0.9912/1.0028, and
nicotinamide with nicotinuric acid on 8.1938/8.2071.

These were source-text assertions on that allocation. The election is now a
pure function, so the same claims are made about what it does.
"""
import numpy as np
import pytest

from processing.quantification import (
    LOST_COMPETITION, Candidate, elect)


def _cand(mid, rank, lines=(), *, dj=0.0, ppm=0.0, mult='d', n=2):
    return Candidate(
        met_id=mid, peak_rank=rank, line_ranks=frozenset(lines),
        evidence={'multiplicity': mult, 'delta_j_hz': dj, 'ppm_diff': ppm,
                  'n_found': n, 'n_expected': n, 'quality': 'good',
                  'score': 1.0})


def test_a_candidate_is_tested_on_its_lines_not_only_its_anchor():
    """Different anchors over the same lines is the same signal counted twice,
    however different the anchors look."""
    win = _cand('winner', 10, [10, 11], dj=0.0)
    other = _cand('other', 11, [10, 11], dj=5.0)
    e = elect([win, other], dj_gold_hz=0.5)
    assert 'winner' in e.winners
    assert 'other' not in e.winners
    assert e.rejected['other'] == LOST_COMPETITION


def test_two_winners_never_share_a_line():
    """The property the whole allocation exists for."""
    cands = [_cand(f'm{i}', 10 + i, [10 + i, 11 + i], dj=i * 0.1)
             for i in range(6)]
    e = elect(cands, dj_gold_hz=0.5)
    seen = set()
    for c in e.winners.values():
        assert not (c.line_ranks & seen)
        seen |= set(c.line_ranks)


def test_an_anchor_already_won_is_not_handed_out_again():
    e = elect([_cand('a', 5, [5], dj=0.0), _cand('b', 5, [5], dj=9.0)],
              dj_gold_hz=0.5)
    assert set(e.winners) == {'a'}


def test_losing_one_signal_costs_the_signal_not_the_analyte():
    """A metabolite keeps competing on its other resonances. Dropping the
    analyte because one of its peaks was contested loses a measurable analyte
    for a reason that has nothing to do with the peak it would have used."""
    e = elect([_cand('rival', 10, [10], dj=0.0),
               _cand('mine', 10, [10], dj=9.0),
               _cand('mine', 20, [20], dj=0.1)],
              dj_gold_hz=0.5)
    assert 'mine' in e.winners
    assert e.winners['mine'].peak_rank == 20


class TestTheOtherWaysBackIn:
    """There are none. The old engine had three — a rescue pass, an APQ
    fallback and a fill-in loop — and each re-entered a metabolite that had
    already lost, on evidence the election had already weighed."""

    def test_a_beaten_metabolite_stays_beaten(self):
        e = elect([_cand('a', 1, [1], dj=0.0), _cand('b', 1, [1], dj=9.0)],
                  dj_gold_hz=0.5)
        assert 'b' not in e.winners
        assert e.rejected['b'] == LOST_COMPETITION

    def test_the_engine_exposes_no_second_allocation(self):
        """One function decides, so there is no second place for the rule to
        drift to. Four call sites answering this question is how the anchor
        check and the line check came to disagree."""
        import processing.quantification as q
        allocators = [n for n in dir(q)
                      if n.startswith(('rescue', 'fallback', 'fill'))]
        assert allocators == []


def test_claimed_peaks_are_visible_to_every_path():
    """The claim set is returned, not kept private to the loop that built it —
    that is what let a later pass integrate a region already spoken for."""
    e = elect([_cand('a', 1, [1, 2])], dj_gold_hz=0.5)
    assert e.claimed_peaks == frozenset({1, 2})


class TestGapFillCannotTakeAnOccupiedRegion:
    """It cannot take any region: there is no gap-fill.

    A candidate with no detected apex was integrated over a window the database
    predicted. On the reference study 23 of 73 placed analytes had no apex and
    15 produced a number that way — a quarter of the results measured where a
    peak was expected rather than where one was found.
    """

    def test_a_candidate_with_no_detected_peak_wins_nothing_to_integrate(self):
        from processing.quant_engine import quantify
        ppm = np.linspace(0.0, 2.0, 401)
        y = np.exp(-0.5 * ((ppm - 1.0) / 0.01) ** 2).reshape(-1, 1) * 100
        rows = quantify(candidates=[_cand('ghost', None, [])],
                        peaks_by_rank={}, ppm=ppm, intensities=y,
                        samples_info=[{}], included=[0], groups={'A': [0]},
                        dj_gold_hz=0.5)
        row = next(r for r in rows if r['hmdb_id'] == 'ghost')
        assert row['_not_quantified']
        assert np.isnan(row['conc_mm'])

    def test_every_quantified_row_rests_on_a_detected_peak(self):
        from processing.quant_engine import quantify
        ppm = np.linspace(0.0, 2.0, 401)
        y = (np.exp(-0.5 * ((ppm - 1.0) / 0.01) ** 2) * 100).reshape(-1, 1)
        peak = {'rank': 1, 'ppm': 1.0, 'start_ppm': 0.95, 'end_ppm': 1.05}
        rows = quantify(candidates=[_cand('real', 1, [1])],
                        peaks_by_rank={1: peak}, ppm=ppm, intensities=y,
                        samples_info=[{}], included=[0], groups={'A': [0]},
                        dj_gold_hz=0.5)
        for r in rows:
            if not r.get('_not_quantified') and not r.get('_is_is'):
                assert r['_multiplet_window'] is not None
                assert r['_peak_rank'] is not None
