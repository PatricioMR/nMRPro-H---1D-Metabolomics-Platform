"""Two analytes on one peak: what each of them owns of it.

The election gives a contested peak to a single owner, and it is right to —
two metabolites integrating the same area is what this engine was built to
stop. But the area belongs to both, and until the split existed the winner was
credited with all of it while the other was credited with nothing. On the
reference study 61 analytes ended with ``lost_competition`` and no number.

The accounting is linear and introduces no new measurement. A window's area is
the sum over its claimants of concentration times protons; a claimant measured
on a clean window of its own says what it must be contributing here; the rest
belongs to whoever is left. One unaccounted claimant is one equation with one
unknown. Two is one equation with two, and nothing is reported.

    Targeted profiling, as in Weljie et al. (2006), Analytical Chemistry
    78:4430-4442 — the concentration of a metabolite fixed on one resonance
    constrains every other resonance it has.
"""
import numpy as np
import pytest

from processing.quant_deconvolution import (
    MIN_CLAIM_FRACTION, MIN_RESIDUE_SHARE, anchor_per_proton,
    claimants_in_window, resolve_contested)


def _w(key, claimants, **areas):
    return {'key': key, 'claimants': claimants,
            'areas': {int(k[1:]): v for k, v in areas.items()}}


class TestAnchors:
    def test_a_window_with_one_claimant_measures_that_claimant(self):
        a = anchor_per_proton([_w('w1', [('lac', 3.0)], s0=300.0)])
        assert a == {'lac': {0: 100.0}}

    def test_a_contested_window_anchors_nobody(self):
        assert anchor_per_proton(
            [_w('w1', [('lac', 3.0), ('ala', 3.0)], s0=300.0)]) == {}

    def test_several_clean_windows_are_averaged_by_the_median(self):
        """One bad window must not set the scale for the rest."""
        a = anchor_per_proton([_w('w1', [('lac', 3.0)], s0=300.0),
                               _w('w2', [('lac', 1.0)], s0=100.0),
                               _w('w3', [('lac', 1.0)], s0=400.0)])
        assert a['lac'][0] == pytest.approx(100.0)

    def test_a_claimant_with_no_protons_is_not_an_anchor(self):
        assert anchor_per_proton([_w('w1', [('x', 0.0)], s0=300.0)]) == {}


class TestSplit:
    def test_the_known_claimant_gives_the_rest_away(self):
        w = [_w('w1', [('lac', 3.0)], s0=300.0),
             _w('w2', [('lac', 3.0), ('ala', 3.0)], s0=450.0)]
        shares, reasons = resolve_contested(w)
        assert shares['w2']['ala'][0] == pytest.approx(150.0)
        assert not reasons

    def test_the_winner_is_corrected_and_not_only_the_loser(self):
        """The whole window comes back, because the winner was credited with
        all of it and 300 of these 450 are not its own."""
        shares, _ = resolve_contested(
            [_w('w1', [('lac', 3.0)], s0=300.0),
             _w('w2', [('lac', 3.0), ('ala', 3.0)], s0=450.0)])
        assert shares['w2']['lac'][0] == pytest.approx(300.0)

    def test_it_cascades(self):
        """A metabolite solved on one window anchors the next, so a chain of
        overlaps unwinds from whatever clean signal it touches."""
        shares, _ = resolve_contested(
            [_w('w1', [('lac', 3.0)], s0=300.0),
             _w('w2', [('lac', 3.0), ('ala', 3.0)], s0=450.0),
             _w('w3', [('ala', 3.0), ('gli', 2.0)], s0=250.0)])
        assert shares['w3']['gli'][0] == pytest.approx(100.0)

    def test_two_unknowns_report_nothing(self):
        shares, reasons = resolve_contested(
            [_w('w1', [('x', 1.0), ('y', 1.0)], s0=80.0)])
        assert shares == {}
        assert 'two unknowns' in reasons['w1'] or '2 unknowns' in reasons['w1']

    def test_a_residue_smaller_than_its_own_error_is_refused(self):
        """It is the difference of two larger numbers and carries the sum of
        their errors; reporting it would build a concentration out of another
        analyte's uncertainty."""
        total = 300.0 / (1.0 - MIN_RESIDUE_SHARE / 2.0)
        shares, reasons = resolve_contested(
            [_w('w1', [('lac', 3.0)], s0=300.0),
             _w('w2', [('lac', 3.0), ('ala', 3.0)], s0=total)])
        assert 'ala' not in shares.get('w2', {})
        assert reasons['w2']

    def test_a_negative_residue_is_refused(self):
        shares, reasons = resolve_contested(
            [_w('w1', [('lac', 3.0)], s0=300.0),
             _w('w2', [('lac', 3.0), ('ala', 3.0)], s0=250.0)])
        assert shares == {}
        assert reasons['w2']

    def test_every_claimant_already_measured_adds_nothing(self):
        shares, reasons = resolve_contested(
            [_w('w1', [('lac', 3.0)], s0=300.0),
             _w('w2', [('ala', 3.0)], s0=150.0),
             _w('w3', [('lac', 3.0), ('ala', 3.0)], s0=450.0)])
        assert 'w3' not in shares
        assert reasons['w3']

    def test_each_sample_is_split_on_its_own(self):
        shares, _ = resolve_contested(
            [_w('w1', [('lac', 3.0)], s0=300.0, s1=600.0),
             _w('w2', [('lac', 3.0), ('ala', 3.0)], s0=450.0, s1=900.0)])
        assert shares['w2']['ala'] == {0: pytest.approx(150.0),
                                       1: pytest.approx(300.0)}

    def test_nothing_is_invented_where_there_is_no_anchor(self):
        shares, _ = resolve_contested(
            [_w('w1', [('lac', 3.0), ('ala', 3.0)], s0=450.0)])
        assert shares == {}


class TestClaimants:
    """Who counts as being in a window, and for how many protons.

    A window measures ``concentration * protons * fraction`` of each claimant,
    the fraction being how much of that claimant's signal falls inside. That
    product is what the balance needs, so it is what this returns: one linear
    solver then serves windows whose claimants are each cut off at a different
    place.
    """

    def test_a_claimant_counts_for_the_part_of_it_the_window_holds(self):
        assert claimants_in_window([('a', 3.0, 0.5)]) == [('a', 1.5)]

    def test_two_signals_of_one_metabolite_in_a_window_add_up(self):
        """Two of its lines inside one window is twice the signal."""
        assert claimants_in_window(
            [('a', 2.0, 1.0), ('a', 3.0, 1.0)]) == [('a', 5.0)]

    def test_a_contribution_below_the_floor_is_not_a_claimant(self):
        """Counting it adds an unknown -- and an unknown is what stops a
        window being solved -- for a term smaller than the error on the terms
        that matter."""
        assert claimants_in_window(
            [('a', 3.0, 1.0),
             ('b', 3.0, MIN_CLAIM_FRACTION / 2.0)]) == [('a', 3.0)]

    def test_a_contribution_at_the_floor_counts(self):
        got = dict(claimants_in_window([('b', 3.0, MIN_CLAIM_FRACTION)]))
        assert 'b' in got

    def test_the_order_does_not_depend_on_the_input_order(self):
        a = claimants_in_window([('z', 1.0, 1.0), ('a', 2.0, 1.0)])
        b = claimants_in_window([('a', 2.0, 1.0), ('z', 1.0, 1.0)])
        assert a == b == [('a', 2.0), ('z', 1.0)]

    def test_nothing_at_all_claims_nothing(self):
        assert claimants_in_window([]) == []


class TestInTheEngine:
    """The split as ``quantify`` applies it, on a spectrum built to be shared.

    Two metabolites: A with a clean singlet of its own and a second signal
    under B's peak, and B whose only signal is that shared peak. A is the only
    one that can be anchored, so whichever of them loses the peak is
    measurable only by subtraction.
    """

    @staticmethod
    def _study():
        ppm = np.linspace(0.0, 10.0, 20001)
        y = np.zeros_like(ppm)

        def line(centre, area, hwhm=0.002):
            return area / np.pi * hwhm / ((ppm - centre) ** 2 + hwhm ** 2)

        # A: 3 protons alone at 1.00, 3 protons at 3.00 sharing with B.
        # B: 2 protons at 3.00 and nowhere else.
        y += line(1.000, 300.0)
        y += line(3.000, 300.0) + line(3.000, 200.0)
        return ppm, y.reshape(-1, 1)

    def _run(self, foot=0.050, a_prefers_the_shared_peak=False, **kw):
        from processing.quantification import Candidate
        ppm, y = self._study()
        # Feet at 25 half-widths by default: far enough out that the straight
        # local baseline has almost no Lorentzian pedestal to eat, so what is
        # left in the numbers is the split and not the integration.
        peaks = {0: {'ppm': 1.000, 'rank': 0,
                     'start_ppm': 1.000 - foot, 'end_ppm': 1.000 + foot},
                 1: {'ppm': 3.000, 'rank': 1,
                     'start_ppm': 3.000 - foot, 'end_ppm': 3.000 + foot}}
        ev = {'multiplicity': 's', 'n_found': 1, 'n_expected': 1,
              'delta_j_hz': 0.0}
        # Score is the election's last tie-break, so it is what decides which
        # of the two signals A is elected on -- and therefore which analyte
        # ends up owning the shared peak and which has to be recovered from it.
        a_clean, a_shared = (0.5, 1.0) if a_prefers_the_shared_peak else (1.0, 0.5)
        cands = [
            Candidate('A', 0, frozenset({0}),
                      {**ev, 'protons': 3.0, 'score': a_clean}),
            Candidate('A', 1, frozenset({1}),
                      {**ev, 'protons': 3.0, 'score': a_shared}),
            Candidate('B', 1, frozenset({1}),
                      {**ev, 'protons': 2.0, 'score': 0.1}),
        ]
        from processing.quant_engine import quantify
        rows = quantify(candidates=cands, peaks_by_rank=peaks, ppm=ppm,
                        intensities=y,
                        samples_info=[{'label': 'A', 'type': 'Sample'}],
                        included=[0], groups={'A': [0]}, dj_gold_hz=0.2,
                        use_pqn=False, linewidth_ppm=0.004,
                        noise_sigma=1e-6, **kw)
        return {r['metabolite']: r for r in rows}

    @staticmethod
    def _area(row):
        return float(np.mean(list(row['_areas_per_sample'].values())))

    # ── the winner of a shared peak was credited with all of it ──────────────

    def test_the_winner_of_a_shared_peak_keeps_only_its_own_share(self):
        """B wins the peak at 3.00 and used to be credited with the whole 500,
        300 of which is A's."""
        rows = self._run()
        assert self._area(rows['B']) == pytest.approx(200.0, rel=0.05)

    def test_the_claimant_that_did_not_win_it_gets_a_number_too(self):
        """A is elected on its clean singlet, so the shared peak is not what
        measures it -- but its part of that peak is now accounted for and not
        silently handed to B."""
        rows = self._run()
        assert self._area(rows['A']) == pytest.approx(300.0, rel=0.05)

    def test_the_two_shares_are_in_the_proportion_they_were_built_in(self):
        """Whatever the integration loses at the feet it loses from both, so
        the proportion is the part the split is answerable for."""
        rows = self._run()
        assert (self._area(rows['B']) / self._area(rows['A'])
                == pytest.approx(200.0 / 300.0, rel=0.01))

    def test_a_shared_area_never_reads_three_dots(self):
        """It rests on another analyte's concentration and on the claimant list
        being complete, neither of which the multiplet's shape can vouch for.
        """
        rows = self._run()
        assert rows['B']['_confidence'] <= 2
        assert any('Shared window' in t for t in rows['B']['_status_reasons'])

    def test_the_reason_names_who_it_is_shared_with(self):
        rows = self._run()
        assert rows['B']['_shared_with'] == ['A']

    # ── the analyte that lost the peak outright ──────────────────────────────

    def test_the_analyte_that_lost_the_competition_now_has_a_number(self):
        """The 61 on the reference study. A wins the shared peak, so B has no
        peak of its own anywhere and the election gives it nothing. Its
        concentration exists all the same: A is anchored on its clean singlet,
        and what A cannot account for in the shared window is B."""
        rows = self._run(a_prefers_the_shared_peak=True)
        assert not rows['B']['_not_quantified'], rows['B']['_status_reasons']
        assert self._area(rows['B']) == pytest.approx(200.0, rel=0.05)

    def test_recovering_it_does_not_leave_a_second_row_saying_it_lost(self):
        rows = self._run(a_prefers_the_shared_peak=True)
        assert rows['B'].get('_rejection') is None

    # ── narrow windows, which is what a crowded region produces ──────────────

    def test_a_narrow_window_still_splits(self):
        """Foot to foot in a crowded region is narrow by construction: the feet
        are the valleys and the valleys are close when the neighbours are.
        Requiring a window to hold 75% of its lines -- the threshold the
        capture correction uses to decide whether it is worth applying --
        threw away 69 of the reference study's 113 windows, the uncontested
        ones among them, and left nothing to solve anything against.

        At two half-widths a Lorentzian window holds about 70%.
        """
        rows = self._run(foot=0.004)
        assert self._area(rows['B']) == pytest.approx(200.0, rel=0.10)

    def test_the_narrow_window_accounts_for_the_baseline_it_subtracts(self):
        """The local baseline is a straight line through the signal's own
        tails, so it takes away a trapezium of the area being measured -- at
        two half-widths, most of what the window holds. Modelling only the
        Lorentzian tail and not the trapezium put this at 127 instead of 200.
        """
        rows = self._run(foot=0.004, local_baseline=True)
        assert self._area(rows['B']) == pytest.approx(200.0, rel=0.10)

    def test_a_window_holding_less_than_half_of_a_signal_is_not_split(self):
        """Dividing by the fraction then multiplies the error faster than it
        corrects the area. The winner keeps the whole window, and the row says
        the number is an upper bound rather than leaving it to be read as a
        concentration."""
        rows = self._run(foot=0.0012)      # 0.6 half-widths, about 37%
        assert not rows['B']['_shared_with']
        assert any('not split' in t for t in rows['B']['_status_reasons'])
        assert rows['B']['_confidence'] <= 2
