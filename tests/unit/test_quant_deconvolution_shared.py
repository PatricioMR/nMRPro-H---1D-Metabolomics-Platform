"""Two analytes on one peak: what each of them owns of it.

The election gives a contested peak to a single owner, and it is right to —
two metabolites integrating the same area is what this engine was built to
stop. But the area belongs to both, and until the split existed the winner was
credited with all of it while the other was credited with nothing. On the
reference study 61 analytes ended with ``lost_competition`` and no number.

Every window is one equation: its area is the sum over its claimants of
concentration times protons times the fraction of that claimant the window
holds. Nothing in the equation is fitted to the trace — its shape comes from
the database and from detection, and only the concentrations are unknown.

One window at a time this reaches almost nothing: a window with two claimants
and no independent measurement of either has one equation and two unknowns,
and 215 of the reference study's 242 contested windows were of that kind. But
a metabolite appears in as many windows as it has signals, each holding a
different fraction of it, so together they are 291 equations in about 130
unknowns and overdetermined.

The difficulty is not solving them; it is that a solver returns a number for
every column whether or not the data determines it. Two metabolites appearing
in the same windows in the same proportions cannot be told apart by any amount
of arithmetic. That is what the old JG-Deconv stage had no answer for, and it
is the whole of what these tests are about.

    Lawson & Hanson (1974), Solving Least Squares Problems, ch. 23 — NNLS.
    Belsley, Kuh & Welsch (1980), Regression Diagnostics — collinearity.
    Weljie et al. (2006), Analytical Chemistry 78:4430 — targeted profiling.
"""
import numpy as np
import pytest

from processing.quant_deconvolution import (
    MAX_VIF, MIN_CLAIM_FRACTION, claimants_in_window, solve_windows)


def _w(key, claimants, **areas):
    return {'key': key, 'claimants': claimants,
            'areas': {int(k[1:]): v for k, v in areas.items()}}


class TestTheJointSolution:
    def test_a_window_with_one_claimant_measures_that_claimant(self):
        pp, _, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0)])
        assert pp['lac'][0] == pytest.approx(100.0)

    def test_the_known_claimant_gives_the_rest_away(self):
        pp, why, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0),
                                 _w('w2', [('lac', 3.0), ('ala', 3.0)],
                                    s0=450.0)])
        assert pp['ala'][0] == pytest.approx(50.0)
        assert not why

    def test_the_winner_is_corrected_and_not_only_the_loser(self):
        """300 of that window's 450 are lactate's, and the analyte elected on
        it was credited with all 450."""
        pp, _, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0),
                               _w('w2', [('lac', 3.0), ('ala', 3.0)],
                                  s0=450.0)])
        assert pp['lac'][0] == pytest.approx(100.0)

    def test_it_reaches_through_a_chain(self):
        """Alanine is measured only through the window it shares with lactate,
        and glycine only through the one it shares with alanine."""
        pp, _, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0),
                               _w('w2', [('lac', 3.0), ('ala', 3.0)],
                                  s0=450.0),
                               _w('w3', [('ala', 3.0), ('gli', 2.0)],
                                  s0=250.0)])
        assert pp['gli'][0] == pytest.approx(50.0)

    def test_two_claimants_and_nothing_else_report_nothing(self):
        """One equation, two unknowns, and no other window to constrain them.
        This is the case the one-window-at-a-time cascade was limited to."""
        pp, why, _st = solve_windows([_w('w1', [('x', 1.0), ('y', 1.0)], s0=80.0)])
        assert pp == {}
        assert 'do not separate it from' in why['x']

    def test_but_two_windows_in_different_proportions_do_report(self):
        """The whole reason for solving them together. Neither analyte has a
        window of its own; between them the two windows still say which is
        which, because they are not in the same proportions."""
        pp, why, _st = solve_windows([_w('w1', [('x', 1.0), ('y', 2.0)], s0=50.0),
                                 _w('w2', [('x', 3.0), ('y', 1.0)], s0=50.0)])
        assert pp['x'][0] == pytest.approx(10.0)
        assert pp['y'][0] == pytest.approx(20.0)

    def test_a_refusal_names_what_it_could_not_be_separated_from(self):
        _pp, why, _st = solve_windows([_w('w1', [('x', 1.0), ('y', 1.0)], s0=80.0)])
        assert 'y' in why['x'] and 'x' in why['y']

    def test_a_window_nobody_kept_does_not_spoil_the_rest(self):
        """Its whole area would otherwise become a residual and inflate the
        standard error of every column that WAS determined."""
        pp, _, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0),
                               _w('w2', [('lac', 3.0), ('ala', 3.0)],
                                  s0=450.0),
                               _w('w3', [('x', 1.0), ('y', 1.0)], s0=80.0)])
        assert pp['ala'][0] == pytest.approx(50.0)

    def test_each_sample_is_solved_on_its_own(self):
        pp, _, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0, s1=600.0),
                               _w('w2', [('lac', 3.0), ('ala', 3.0)],
                                  s0=450.0, s1=900.0)])
        assert pp['ala'] == {0: pytest.approx(50.0), 1: pytest.approx(100.0)}

    def test_no_concentration_comes_back_negative(self):
        """A negative one would let an analyte pay for another's excess."""
        pp, _, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0),
                               _w('w2', [('lac', 3.0), ('ala', 3.0)],
                                  s0=250.0)])
        assert all(v >= 0 for d in pp.values() for v in d.values())

    def test_an_analyte_the_balance_leaves_nothing_gets_no_number(self):
        pp, why, _st = solve_windows([_w('w1', [('lac', 3.0)], s0=300.0),
                                 _w('w2', [('lac', 3.0), ('ala', 3.0)],
                                    s0=300.0)])
        assert 'ala' not in pp
        assert why['ala']

    def test_a_noisier_window_counts_for_less(self):
        """Two windows disagreeing about one analyte; the answer must sit
        nearer the one whose area is better measured."""
        clean = {**_w('w1', [('lac', 3.0)], s0=300.0), 'weight': 10.0}
        noisy = {**_w('w2', [('lac', 3.0)], s0=600.0), 'weight': 0.1}
        pp, _, _st = solve_windows([clean, noisy])
        assert pp['lac'][0] < 150.0

    def test_nothing_at_all_solves_nothing(self):
        assert solve_windows([]) == ({}, {}, {})

    def test_the_design_threshold_is_not_the_precision_threshold(self):
        """Two questions get confused here. "Can these be told apart at all"
        is about the design and needs columns that are near-multiples of each
        other; "how well is this pinned down" is about the data and is the
        standard error's job. Refusing at the classical VIF of 10 -- 90%
        reproducible, so imprecise rather than indeterminate -- threw out 157
        of the reference study's 159 analytes, clean windows included."""
        assert MAX_VIF >= 1000.0


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
        """A, present on its clean singlet, keeps the shared peak as its own
        and is elected on it here; it used to be credited with the whole
        500, 200 of which is B's."""
        rows = self._run(a_prefers_the_shared_peak=True)
        assert self._area(rows['A']) == pytest.approx(300.0, rel=0.05)
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

    def test_a_window_too_narrow_for_its_line_is_not_split(self):
        """At 0.6 half-widths the window holds 30% of the line and the local
        baseline takes 25 of those, so nothing in it is worth attributing.
        The winner keeps the whole window and the row says the number is an
        upper bound rather than leaving it to be read as a concentration.

        A is elected on the shared peak here; B, whose only signal it is,
        lost it and has no row to mark."""
        rows = self._run(foot=0.0012, a_prefers_the_shared_peak=True)
        assert not rows['A']['_shared_with']
        assert any('not split' in t for t in rows['A']['_status_reasons']), (
            rows['A']['_status_reasons'])
        assert rows['A']['_confidence'] <= 2


class TestTheSwitch:
    """Overlap correction is the analyst's choice, and off must mean off.

    A setting that half applies is worse than either state: the number moves
    and the reason it moved is not in the run's record.
    """

    def test_off_leaves_the_winner_the_whole_window(self):
        """A, elected on the shared peak, is credited with all 500 of it --
        300 its own and 200 that are B's."""
        rows = TestInTheEngine()._run(deconvolve=False,
                                      a_prefers_the_shared_peak=True)
        area = TestInTheEngine._area(rows['A'])
        assert area == pytest.approx(500.0, rel=0.05)

    def test_off_gives_the_analyte_that_lost_nothing(self):
        rows = TestInTheEngine()._run(a_prefers_the_shared_peak=True,
                                      deconvolve=False)
        assert rows['B']['_not_quantified']

    def test_off_marks_no_row_as_shared(self):
        rows = TestInTheEngine()._run(deconvolve=False)
        assert not any(r.get('_shared_with') for r in rows.values())

    def test_on_is_the_default(self):
        """It corrects a number that is otherwise too large, and refuses what
        it cannot determine, so it is not a setting an analyst should have to
        find."""
        import inspect

        from processing.quant_engine import quantify
        assert (inspect.signature(quantify).parameters['deconvolve'].default
                is True)

    def test_the_run_records_that_it_was_off(self):
        """Otherwise a log with no split in it reads the same as a split that
        found nothing."""
        from processing.quant_engine import quantify
        diag: dict = {}
        TestInTheEngine()._run(deconvolve=False, diagnostics=diag)
        assert diag['shared'] == {'off': True}

    def test_a_split_row_says_it_used_the_overlap_correction(self):
        rows = TestInTheEngine()._run()
        assert rows['B']['_used_j_resolve'] is True
        assert rows['B']['_j_resolve_frac'] < 1.0


class TestWhatARefusalCosts:
    """A refusal must cost confidence only where it is about the measurement.

    Refusing at the classical VIF of 10 marked 157 of the reference study's
    159 analytes, and every one of them then read two dots — including
    analytes measured on windows nobody else is in, whose own measurement the
    joint system never touched. Losing precision in a system is not the same
    as a window holding somebody else's signal, and only the second is a
    statement about the number in the table.
    """

    def test_an_analyte_on_a_window_of_its_own_is_not_marked(self):
        rows = TestInTheEngine()._run()
        assert not rows['A']['_shared_with']
        assert not any('Shared window' in t
                       for t in rows['A']['_status_reasons'])

    def test_an_analyte_sharing_a_peak_is_marked_when_it_is_not_split(self):
        """Its number is then the whole of a window it holds with another, so
        it is an upper bound and the row has to say so."""
        rows = TestInTheEngine()._run(foot=0.0012,
                                      a_prefers_the_shared_peak=True)
        assert any('not split' in t for t in rows['A']['_status_reasons'])
        assert rows['A']['_confidence'] <= 2


class TestWhatIsNotThere:
    """An undetected signal is a measurement, and the one that was missing.

    On the reference study half the columns were exact multiples of another:
    an analyte with one signal in one window, shared with an analyte in the
    same position. Nothing in those windows tells them apart. But if one of
    them should also show a line at 7 ppm and there is nothing there, its
    concentration is bounded by the noise in that region and the other's is
    not — and that is a row in which their columns differ.
    """

    @staticmethod
    def _spectrum(area_at_seven):
        ppm = np.linspace(0.0, 10.0, 20001)
        y = np.zeros_like(ppm)

        def line(centre, area, hwhm=0.002):
            return area / np.pi * hwhm / ((ppm - centre) ** 2 + hwhm ** 2)

        # One shared peak at 3.00 holding 500. A has a second signal predicted
        # at 7.00; what is there is the caller's choice.
        y += line(3.000, 500.0)
        if area_at_seven:
            y += line(7.000, area_at_seven)
        return ppm, y.reshape(-1, 1)

    def _run(self, area_at_seven, *, predict=True):
        from processing.quantification import Candidate
        from processing.quant_engine import quantify
        ppm, y = self._spectrum(area_at_seven)
        peaks = {1: {'ppm': 3.000, 'rank': 1,
                     'start_ppm': 2.950, 'end_ppm': 3.050}}
        ev = {'multiplicity': 's', 'n_found': 1, 'n_expected': 1,
              'delta_j_hz': 0.0}
        cands = [
            Candidate('A', 1, frozenset({1}), {**ev, 'protons': 2.0}),
            Candidate('B', 1, frozenset({1}), {**ev, 'protons': 2.0}),
        ]
        if predict:
            cands.append(Candidate('A', None, frozenset(),
                                   {**ev, 'n_found': 0, 'protons': 2.0,
                                    'predicted_ppms': [7.000]}))
        diag: dict = {}
        rows = quantify(candidates=cands, peaks_by_rank=peaks, ppm=ppm,
                        intensities=y,
                        samples_info=[{'label': 'A', 'type': 'Sample'}],
                        included=[0], groups={'A': [0]}, dj_gold_hz=0.2,
                        use_pqn=False, linewidth_ppm=0.004,
                        noise_sigma=1e-3, diagnostics=diag)
        return {r['metabolite']: r for r in rows}, diag['shared']

    def test_without_it_the_two_cannot_be_told_apart(self):
        rows, diag = self._run(0.0, predict=False)
        assert diag['resolved'] == 0
        assert diag['refused_indeterminate'] == 2

    def test_nothing_where_a_signal_should_be_gives_the_peak_to_the_other(self):
        """A should show 2 protons at 7 ppm and there is noise. So A is at most
        noise-sized, and the 500 at 3 ppm are B's."""
        rows, diag = self._run(0.0)
        assert diag['bounds'] == 1
        b = float(np.mean(list(rows['B']['_areas_per_sample'].values())))
        assert b == pytest.approx(500.0, rel=0.05), rows['B']['_status_reasons']

    def test_a_real_signal_where_it_was_predicted_is_not_a_bound(self):
        """If A's second signal is there, A is there, and the window at 7 ppm
        measures it rather than bounding it away. Same protons at both
        positions, so 300 at 7 ppm means 300 of the shared 500 are A's."""
        rows, diag = self._run(300.0)
        a = float(np.mean(list(rows['A']['_areas_per_sample'].values())))
        b = float(np.mean(list(rows['B']['_areas_per_sample'].values())))
        assert a == pytest.approx(300.0, rel=0.05)
        assert b == pytest.approx(200.0, rel=0.05)

    def test_an_analyte_detected_nowhere_gets_no_bound_window(self):
        """A database shift is not where a real sample puts the line. Without
        one detected signal to say the analyte is present at all, a window at
        the predicted position would be measuring the database, not the
        sample."""
        from processing.quantification import Candidate
        from processing.quant_engine import quantify
        ppm, y = self._spectrum(0.0)
        peaks = {1: {'ppm': 3.000, 'rank': 1,
                     'start_ppm': 2.950, 'end_ppm': 3.050}}
        ev = {'multiplicity': 's', 'n_found': 1, 'n_expected': 1,
              'delta_j_hz': 0.0}
        cands = [Candidate('B', 1, frozenset({1}), {**ev, 'protons': 2.0}),
                 Candidate('C', None, frozenset(),
                           {**ev, 'n_found': 0, 'protons': 2.0,
                            'predicted_ppms': [7.000]})]
        diag: dict = {}
        quantify(candidates=cands, peaks_by_rank=peaks, ppm=ppm,
                 intensities=y, samples_info=[{'label': 'A', 'type': 'Sample'}],
                 included=[0], groups={'A': [0]}, dj_gold_hz=0.2,
                 use_pqn=False, linewidth_ppm=0.004, noise_sigma=1e-3,
                 diagnostics=diag)
        assert diag['shared']['bounds'] == 0
        assert diag['shared']['bounds_no_anchor'] == 1

    def test_the_bound_follows_the_drift(self):
        """The window slides over the drift the study measured and keeps the
        position with the MOST area, so a line that moved is found and the
        bound can only ever be too loose."""
        from processing.quantification import Candidate
        from processing.quant_engine import quantify
        ppm = np.linspace(0.0, 10.0, 20001)
        y = np.zeros_like(ppm)
        g = 0.002
        y += 500.0 / np.pi * g / ((ppm - 3.000) ** 2 + g ** 2)
        y += 300.0 / np.pi * g / ((ppm - 7.006) ** 2 + g ** 2)   # drifted
        y = y.reshape(-1, 1)
        peaks = {1: {'ppm': 3.000, 'rank': 1,
                     'start_ppm': 2.950, 'end_ppm': 3.050}}
        ev = {'multiplicity': 's', 'n_found': 1, 'n_expected': 1,
              'delta_j_hz': 0.0}
        cands = [Candidate('A', 1, frozenset({1}), {**ev, 'protons': 2.0}),
                 Candidate('B', 1, frozenset({1}), {**ev, 'protons': 2.0}),
                 Candidate('A', None, frozenset(),
                           {**ev, 'n_found': 0, 'protons': 2.0,
                            'predicted_ppms': [7.000]})]
        rows = {r['metabolite']: r for r in quantify(
            candidates=cands, peaks_by_rank=peaks, ppm=ppm, intensities=y,
            samples_info=[{'label': 'A', 'type': 'Sample'}], included=[0],
            groups={'A': [0]}, dj_gold_hz=0.2, use_pqn=False,
            linewidth_ppm=0.004, noise_sigma=1e-3, drift_ppm=0.008)}
        a = float(np.mean(list(rows['A']['_areas_per_sample'].values())))
        assert a == pytest.approx(300.0, rel=0.10)
