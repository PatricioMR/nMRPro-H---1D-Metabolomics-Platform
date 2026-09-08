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
    for reason in (LOST_COMPETITION, UNVERIFIED_MULTIPLET, AMBIGUOUS_RANK,
                   NO_ELIGIBLE_SIGNAL):
        assert REJECTION_TEXT[reason].startswith('Não quantificado')


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
        assert _confidence(singlet)[0] == 1
