"""One peak, one metabolite — and a truthful reason for everyone else.

The engine this replaces asked the same question in four places that
disagreed: a greedy pass, a rescue pass, an APQ fallback and a fill-in loop.
Peaks were handed out by anchor while the lines underneath were shared, an
ambiguous rank still elected a winner by tie-break, and an analyte withdrawn
before competing was told it had been outscored.
"""
import pytest

from processing.quantification import (
    AMBIGUOUS_RANK, Candidate, LOST_COMPETITION, NO_ELIGIBLE_SIGNAL,
    REJECTION_TEXT, UNVERIFIED_MULTIPLET, elect)

GOLD = 0.20


def _c(mid, rank, *, lines=(), mult='d', dj=0.05, ppm=0.001,
       n_found=2, n_exp=2, **ev):
    e = dict(multiplicity=mult, delta_j_hz=dj, ppm_diff=ppm,
             n_found=n_found, n_expected=n_exp)
    e.update(ev)
    return Candidate(met_id=mid, peak_rank=rank,
                     line_ranks=frozenset(lines or ([rank] if rank else [])),
                     evidence=e)


class TestOnePeakOneMetabolite:
    def test_the_better_evidence_wins_the_peak(self):
        out = elect([_c('A', 10, dj=2.0), _c('B', 10, dj=0.05)],
                    dj_gold_hz=GOLD)
        assert set(out.winners) == {'B'}
        assert out.rejected['A'] == LOST_COMPETITION

    def test_a_loser_keeps_competing_on_its_other_signals(self):
        """Losing a peak costs a signal, never the analyte."""
        out = elect([_c('A', 10, dj=2.0), _c('B', 10, dj=0.05),
                     _c('A', 20, dj=0.05)], dj_gold_hz=GOLD)
        assert set(out.winners) == {'A', 'B'}
        assert out.winners['A'].peak_rank == 20
        assert 'A' not in out.rejected

    def test_sharing_the_lines_is_sharing_the_signal(self):
        """Different anchors over the same lines is the same peak twice. Three
        metabolites were found on the lines at 0.9912 and 1.0028 from anchors
        90, 58 and 83."""
        out = elect([_c('A', 90, lines=(90, 91), dj=0.05),
                     _c('B', 58, lines=(90, 91), dj=0.10),
                     _c('C', 83, lines=(90, 91), dj=0.30)], dj_gold_hz=GOLD)
        assert set(out.winners) == {'A'}
        assert out.rejected['B'] == out.rejected['C'] == LOST_COMPETITION

    def test_a_peak_free_of_claims_is_still_given_out(self):
        out = elect([_c('A', 10, lines=(10, 11)), _c('B', 30, lines=(30, 31))],
                    dj_gold_hz=GOLD)
        assert set(out.winners) == {'A', 'B'}
        assert out.claimed_peaks == {10, 11, 30, 31}


class TestWhoMayCompete:
    def test_an_unverified_multiplet_stands_aside(self):
        """Not all lines found, so the multiplicity was never verified — it
        cannot take a peak from a signal that was."""
        out = elect([_c('A', 10, n_found=2, n_exp=3, mult='t', dj=None),
                     _c('B', 10)], dj_gold_hz=GOLD)
        assert set(out.winners) == {'B'}
        assert out.rejected['A'] == UNVERIFIED_MULTIPLET

    def test_a_manual_good_mark_lets_an_unverified_multiplet_compete(self):
        """The analyst verified what the lines could not."""
        out = elect([_c('A', 10, n_found=2, n_exp=3, mult='t', dj=None,
                        manual_quality='good'),
                     _c('B', 10)], dj_gold_hz=GOLD)
        assert set(out.winners) == {'A'}

    def test_an_ambiguous_rank_has_no_winner(self):
        out = elect([_c('A', 10), _c('B', 10)],
                    ambiguous_ranks={10}, dj_gold_hz=GOLD)
        assert out.winners == {}
        assert out.rejected['A'] == out.rejected['B'] == AMBIGUOUS_RANK

    def test_ambiguity_only_costs_that_peak(self):
        out = elect([_c('A', 10), _c('B', 10), _c('A', 20)],
                    ambiguous_ranks={10}, dj_gold_hz=GOLD)
        assert set(out.winners) == {'A'}
        assert out.winners['A'].peak_rank == 20
        assert out.rejected['B'] == AMBIGUOUS_RANK

    def test_a_metabolite_that_never_stood_is_not_called_a_loser(self):
        out = elect([_c('A', 10, n_found=1, n_exp=3, mult='t', dj=None)],
                    dj_gold_hz=GOLD)
        assert out.rejected['A'] == UNVERIFIED_MULTIPLET

    def test_standing_and_losing_outranks_having_a_withdrawn_signal(self):
        """One signal withdrawn and another beaten means it was beaten."""
        out = elect([_c('A', 10, n_found=1, n_exp=3, mult='t', dj=None),
                     _c('A', 20, dj=2.0), _c('B', 20, dj=0.05)],
                    dj_gold_hz=GOLD)
        assert out.rejected['A'] == LOST_COMPETITION

    def test_nothing_at_all(self):
        out = elect([], dj_gold_hz=GOLD)
        assert out.winners == {} and out.rejected == {}


class TestDeterminism:
    def test_a_tie_resolves_the_same_way_every_run(self):
        """A coin toss here would move concentrations between runs with
        nothing in the data changing."""
        cands = [_c('B', 10), _c('A', 10)]
        first = elect(cands, dj_gold_hz=GOLD).winners
        for _ in range(5):
            assert set(elect(list(reversed(cands)),
                             dj_gold_hz=GOLD).winners) == set(first)


def test_every_reason_can_be_shown_to_the_analyst():
    """English, like the rest of the interface, and each one names both what
    happened and what would change it."""
    for reason in (LOST_COMPETITION, UNVERIFIED_MULTIPLET, AMBIGUOUS_RANK,
                   NO_ELIGIBLE_SIGNAL):
        assert REJECTION_TEXT[reason].startswith('Not quantified')


def test_the_reasons_say_different_things():
    assert len(set(REJECTION_TEXT.values())) == len(REJECTION_TEXT)


class TestEligibilityAsksWhetherSomethingWasSeen:
    """Not whether the search was confident about what it was.

    'Good' is what the annotation search says when it has settled an identity.
    A signal it left undecided is not thereby absent — it is a peak that was
    found and not adjudicated. Requiring 'good' meant a metabolite whose real
    signals were all undecided could only get in through a signal that was NOT
    undecided, and until 2026-09-08 that was routinely a singlet with no
    detected peak: no peak means no rivals and an empty window, so "alone and
    uncontested" promoted it to Good and made it an anchor.

    Measured on the reference study, end to end:

        rule                          quantified   strong  fair  weak
        good-only, with the ghosts        48         15      9    24
        peak-bearing and not bad          97         15      9    73

    Every analyte the wider rule adds is a singlet marked weak. The strong and
    fair counts do not move, which is the property that makes the widening
    honest: it reports more, at a confidence that says so.
    """

    def _met(self, mid, mps):
        return {'id': mid, 'matched_peaks': mps}

    def test_a_signal_with_a_peak_and_no_verdict_is_enough(self):
        from processing.quantification import eligible_metabolite_ids
        met = self._met('M1', [{'peak_rank': 7, 'annotation_quality': None}])
        assert eligible_metabolite_ids([met]) == {'M1'}

    def test_a_signal_with_no_peak_is_not(self):
        """The hole this closes: nothing was seen, so nothing was identified."""
        from processing.quantification import eligible_metabolite_ids
        met = self._met('M1', [{'peak_rank': None,
                                'annotation_quality': None}])
        assert eligible_metabolite_ids([met]) == set()

    def test_a_bad_signal_is_not(self):
        from processing.quantification import eligible_metabolite_ids
        met = self._met('M1', [{'peak_rank': 7,
                                'annotation_quality': 'bad'}])
        assert eligible_metabolite_ids([met]) == set()

    def test_one_usable_signal_carries_the_metabolite(self):
        from processing.quantification import eligible_metabolite_ids
        met = self._met('M1', [{'peak_rank': 7, 'annotation_quality': 'bad'},
                               {'peak_rank': 9, 'annotation_quality': None}])
        assert eligible_metabolite_ids([met]) == {'M1'}

    def test_good_still_gets_in_without_a_rank(self):
        """A manual or search Good is a judgement about identity, and it is not
        this function's place to overrule it."""
        from processing.quantification import eligible_metabolite_ids
        met = self._met('M1', [{'peak_rank': None,
                                'annotation_quality': 'good'}])
        assert eligible_metabolite_ids([met]) == {'M1'}

    def test_eligibility_is_not_a_verdict_on_identity(self):
        """Which signal is measured is the election's; how much the number is
        worth is the confidence's. Letting a metabolite in says only that
        something was found to measure."""
        from processing.quant_engine import _confidence
        from processing.quantification import Candidate
        singlet = Candidate(met_id='M1', peak_rank=7,
                            evidence={'multiplicity': 's', 'n_found': 1,
                                      'n_expected': 1, 'delta_j_hz': None})
        # its shape says nothing either way -- there is no coupling to
        # confirm -- so what it is worth is left to its identity, which
        # this call does not give it
        assert _confidence(singlet)[0] == 3
        assert _confidence(singlet, identity=1)[0] == 1


class TestWhatWasSeenComesFirst:
    """A signal matched to no peak cannot be judged.

    Never having been matched, its Delta-ppm is zero by default — the best
    value the ranking can see. Absence read as perfection, so it outranked the
    same metabolite's real signals. On the reference study 11 analytes won on a
    signal with no peak while each held one with a peak, among them
    L-histidine, L-carnosine, ATP and hypoxanthine, and the peak they held went
    unmeasured.
    """

    def _c(self, mid, rank, lines=(), ppm_diff=0.01):
        return Candidate(met_id=mid, peak_rank=rank,
                         line_ranks=frozenset(lines),
                         evidence={'multiplicity': 's', 'n_found': 1,
                                   'n_expected': 1, 'delta_j_hz': None,
                                   'ppm_diff': ppm_diff, 'quality': None})

    def test_a_detected_signal_beats_the_metabolites_own_ghost(self):
        """Even when the ghost's zero Delta-ppm looks better."""
        ghost = self._c('M', None, ppm_diff=0.0)
        real = self._c('M', 7, [7], ppm_diff=0.02)
        e = elect([ghost, real], dj_gold_hz=0.5)
        assert e.winners['M'].peak_rank == 7

    def test_losing_a_contested_peak_is_losing(self):
        """Not an occasion to fall back on a signal nobody saw. Reporting the
        ghost instead would turn a defeat into a measurement."""
        rival = self._c('R', 7, [7], ppm_diff=0.001)
        mine_real = self._c('M', 7, [7], ppm_diff=0.03)
        mine_ghost = self._c('M', None, ppm_diff=0.0)
        e = elect([rival, mine_real, mine_ghost], dj_gold_hz=0.5)
        assert e.winners['R'].peak_rank == 7
        assert 'M' not in e.winners
        assert e.rejected['M'] == LOST_COMPETITION

    def test_a_metabolite_with_only_ghosts_still_gets_its_bound(self):
        """It is an ordering, not a bar. An analyte the spectrum shows nothing
        of is reported as an upper limit, which is a real measurement; what it
        must never do is take precedence over a peak that is there."""
        e = elect([self._c('M', None, ppm_diff=0.0)], dj_gold_hz=0.5)
        assert 'M' in e.winners
        assert e.winners['M'].peak_rank is None

    def test_a_ghost_never_blocks_another_metabolite(self):
        """It claims nothing, so it cannot."""
        e = elect([self._c('G', None), self._c('R', 7, [7])], dj_gold_hz=0.5)
        assert e.winners['R'].peak_rank == 7
        assert 7 in e.claimed_peaks


class TestAWinnerKeepsItsOtherSignals:
    """Lactate was elected on its quartet at 4.11 ppm and left the election;
    its methyl doublet at 1.33 -- the strongest signal in the region -- then
    went uncontested to N-acetyl-L-alanine, reported at 4.5 mM in plasma off
    the same molecule that stood at 2.3 mM four rows down under its own name.
    """

    def _cands(self):
        from processing.quantification import Candidate
        ev = {'multiplicity': 'd', 'n_found': 2, 'n_expected': 2,
              'delta_j_hz': 0.0, 'ppm_diff': 0.0}
        return [
            # lactate: quartet first in the ordering (better score), doublet
            # second -- both verified, both with peaks
            Candidate('LAC', 1, frozenset({1, 2, 3, 4}),
                      {**ev, 'multiplicity': 'q', 'n_found': 4,
                       'n_expected': 4, 'score': 1.0}),
            Candidate('LAC', 5, frozenset({5, 6}), {**ev, 'score': 0.9}),
            # a rival whose only signal is lactate's doublet
            Candidate('NAA', 5, frozenset({5, 6}), {**ev, 'score': 0.8}),
        ]

    def test_the_rival_does_not_get_the_winners_other_peak(self):
        from processing.quantification import LOST_COMPETITION, elect
        e = elect(self._cands(), dj_gold_hz=0.2)
        assert 'LAC' in e.winners
        assert e.rejected.get('NAA') == LOST_COMPETITION
        assert {5, 6} <= set(e.claimed_peaks)

    def test_losing_still_costs_only_a_signal(self):
        """Rule 4 stands: a metabolite that loses one peak competes on the
        next. Claiming a winner's other signals must not turn into claiming a
        loser's."""
        from processing.quantification import Candidate, elect
        ev = {'multiplicity': 's', 'n_found': 1, 'n_expected': 1,
              'delta_j_hz': 0.0}
        cands = [Candidate('A', 1, frozenset({1}), {**ev, 'score': 1.0}),
                 Candidate('B', 1, frozenset({1}), {**ev, 'score': 0.5}),
                 Candidate('B', 2, frozenset({2}), {**ev, 'score': 0.4})]
        e = elect(cands, dj_gold_hz=0.2)
        assert e.winners['A'].peak_rank == 1
        assert e.winners['B'].peak_rank == 2


class TestALossSaysWhoAndWhy:
    """'Lost the competition' says nothing an analyst can act on. 'Lost peak
    #5 to N-acetyl-L-alanine on a manual Good mark' does: it names the rival
    and the one criterion that decided it, so the analyst knows whether to
    look at the spectrum, at the mark, or at the database."""

    def _pair(self, **naa):
        from processing.quantification import Candidate
        ev = {'multiplicity': 'd', 'n_found': 2, 'n_expected': 2,
              'delta_j_hz': 0.0, 'ppm_diff': 0.0, 'quality': 'good'}
        return [Candidate('LAC', 5, frozenset({5, 6}), dict(ev)),
                Candidate('NAA', 5, frozenset({5, 6}), {**ev, **naa})]

    def test_a_manual_mark_is_named_as_the_reason(self):
        from processing.quantification import elect
        e = elect(self._pair(delta_j_hz=0.35, ppm_diff=0.05,
                             manual_quality='good'), dj_gold_hz=0.2)
        assert e.beaten['LAC'] == [(5, 'NAA', 'a manual Good mark')]

    def test_delta_j_is_named_when_it_decided(self):
        from processing.quantification import elect
        e = elect(self._pair(delta_j_hz=-0.0), dj_gold_hz=0.2)
        # identical keys: decided by id -- and said so
        assert e.beaten.get('NAA', e.beaten.get('LAC'))[0][2].startswith(
            'a tie')
        e = elect(self._pair(delta_j_hz=0.6), dj_gold_hz=0.2)
        assert e.beaten['NAA'][0][2] == 'delta-J'

    def test_the_rejected_row_carries_it(self):
        import numpy as np
        from processing.quant_engine import quantify
        ppm = np.linspace(0, 10, 2001)
        y = np.zeros((2001, 1)); y[1000, 0] = 100.0
        peaks = {5: {'ppm': 5.0, 'rank': 5, 'start_ppm': 4.98,
                     'end_ppm': 5.02},
                 6: {'ppm': 5.01, 'rank': 6, 'start_ppm': 4.99,
                     'end_ppm': 5.03}}
        cands = self._pair(delta_j_hz=0.35, ppm_diff=0.05,
                           manual_quality='good')
        rows = {r['metabolite']: r for r in quantify(
            candidates=cands, peaks_by_rank=peaks, ppm=ppm, intensities=y,
            samples_info=[{'label': 'A', 'type': 'Sample'}], included=[0],
            groups={'A': [0]}, dj_gold_hz=0.2, use_pqn=False,
            deconvolve=False)}
        assert any('Lost peak #5 to NAA on a manual Good mark' in t
                   for t in rows['LAC']['_status_reasons'])
