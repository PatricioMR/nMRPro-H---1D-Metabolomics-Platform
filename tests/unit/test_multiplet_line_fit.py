# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Fitting a multiplet against the apices ONE sample really has.

The defect this replaces: each line was located by its own apex search, so the
taller leg of a doublet won both searches and the pattern was recorded twice at
one position. The assignment here is injective, which makes that unsayable.
"""
import numpy as np
import pytest

from processing.multiplet_line_fit import (
    candidate_apices, fit_across_samples, fit_lines, fit_window, observed_j_hz)

MHZ = 600.0
J = 7.0
D = J / MHZ
GOLD, BAD = 0.20, 0.50


def _axis(lo=1.40, hi=1.55, step=0.0002):
    """A descending ppm axis, the convention every spectrum here uses."""
    return np.arange(hi, lo, -step)


def _spectrum(ppm, peaks, width=0.0015):
    """Lorentzian lines at ``peaks`` = [(ppm, height), ...]."""
    y = np.zeros_like(ppm)
    for p, h in peaks:
        y = y + h * (width ** 2) / ((ppm - p) ** 2 + width ** 2)
    return y


class TestCandidateApices:
    def test_finds_both_legs_of_a_doublet(self):
        ppm = _axis()
        y = _spectrum(ppm, [(1.47 - D / 2, 1.0), (1.47 + D / 2, 1.0)])
        got = [p for p, _ in candidate_apices(ppm, y, 1.44, 1.50)]
        assert len(got) == 2
        assert got[0] == pytest.approx(1.47 - D / 2, abs=3e-4)
        assert got[1] == pytest.approx(1.47 + D / 2, abs=3e-4)

    def test_returned_in_ascending_ppm(self):
        ppm = _axis()
        y = _spectrum(ppm, [(1.46, 1.0), (1.48, 1.0), (1.50, 1.0)])
        got = [p for p, _ in candidate_apices(ppm, y, 1.44, 1.52)]
        assert got == sorted(got)

    def test_noise_is_not_an_apex(self):
        """A sample without this analyte must give an empty list, not noise."""
        ppm = _axis()
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.01, ppm.size)
        assert candidate_apices(ppm, y, 1.44, 1.50, noise_floor=0.05) == []

    def test_a_window_outside_the_axis_finds_nothing(self):
        ppm = _axis()
        y = _spectrum(ppm, [(1.47, 1.0)])
        assert candidate_apices(ppm, y, 9.0, 9.5) == []


class TestInjectiveAssignment:
    """The whole point: one apex cannot be two lines."""

    def test_two_lines_never_share_an_apex(self):
        cands = [(1.47, 10.0), (1.47 + D, 1.0)]
        fit = fit_lines([1.47, 1.47 + D], cands, field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD)
        assert len(set(fit['ppms'])) == len(fit['ppms']) == 2

    def test_the_taller_leg_cannot_win_both(self):
        """The old per-peak search returned the same position twice here."""
        ppm = _axis()
        y = _spectrum(ppm, [(1.47 - D / 2, 10.0), (1.47 + D / 2, 1.0)])
        cands = candidate_apices(ppm, y, 1.44, 1.50)
        fit = fit_lines([1.47 - D / 2, 1.47 + D / 2], cands, field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD)
        assert fit['fitted'] is True
        assert fit['ppms'][0] != fit['ppms'][1]
        j = observed_j_hz(fit['ranks'], fit['ppms'], MHZ)
        assert j == pytest.approx(J, abs=0.3)

    def test_one_apex_cannot_serve_a_doublet(self):
        """A single line in the window is one line, never a folded pair."""
        cands = [(1.47, 10.0)]
        fit = fit_lines([1.47 - D / 2, 1.47 + D / 2], cands, field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD)
        assert fit['ppms'] == [] and fit['fitted'] is False


class TestChoosingTheBestSpacing:
    def test_the_pair_matching_the_coupling_wins(self):
        """A foreign resonance sits between the legs; the fit must not take it
        just because it is closer to the expected centre."""
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(1.47 - D / 2, 1.0), (1.4702, 5.0), (1.47 + D / 2, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD)
        assert fit['ppms'] == pytest.approx([exp[0], exp[1]])

    def test_a_drifted_multiplet_is_not_penalised_for_drifting(self):
        """Translation is free: only the gaps between lines are judged."""
        exp = [1.47 - D / 2, 1.47 + D / 2]
        shift = 0.02
        cands = [(exp[0] + shift, 1.0), (exp[1] + shift, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is True
        assert fit['residual_hz'] == pytest.approx(0.0, abs=1e-6)
        assert fit['drift_ppm'] == pytest.approx(shift, abs=1e-9)

    def test_drift_beyond_the_tolerance_is_refused(self):
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(exp[0] + 0.4, 1.0), (exp[1] + 0.4, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is False and fit['ppms'] == []

    def test_a_window_with_no_such_pattern_does_not_fit(self):
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(1.46, 1.0), (1.50, 1.0)]      # 24 Hz apart, not 7
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is False

    def test_the_worst_pair_decides_not_the_average(self):
        """One leg on a foreign resonance must not be averaged away by the
        others agreeing."""
        exp = [1.47 - D, 1.47, 1.47 + D]
        cands = [(1.47 - D, 1.0), (1.47, 1.0), (1.47 + D + 0.004, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert len(fit['ppms']) == 2       # drops the displaced line
        assert fit['fitted'] is True


class TestPartialPatterns:
    def test_more_lines_win_when_the_spacing_holds(self):
        exp = [1.47 - D, 1.47, 1.47 + D]
        cands = [(p, 1.0) for p in exp]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['ranks'] == [0, 1, 2]

    def test_outer_lines_alone_still_fit_and_give_j(self):
        exp = [1.47 - D, 1.47, 1.47 + D]
        cands = [(1.47 - D, 1.0), (1.47 + D, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['ranks'] == [0, 2]
        assert observed_j_hz(fit['ranks'], fit['ppms'], MHZ) == pytest.approx(
            J, abs=1e-6)


class TestSinglets:
    def test_the_nearest_apex_is_the_reading(self):
        fit = fit_lines([3.20], [(3.19, 1.0), (3.205, 1.0)], field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['ppms'] == pytest.approx([3.205])
        assert fit['fitted'] is True

    def test_a_nearer_apex_of_noise_is_not_the_singlet(self):
        # the signal of height 190 sits 0.008 out; an apex of 12 sits 0.006 out
        fit = fit_lines([3.936], [(3.928, 190.0), (3.942, 12.0)], field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['ppms'] == [3.928]

    def test_a_singlet_has_no_coupling(self):
        fit = fit_lines([3.20], [(3.20, 1.0)], field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert observed_j_hz(fit['ranks'], fit['ppms'], MHZ) is None

    def test_rival_apices_are_counted(self):
        fit = fit_lines([3.20], [(3.199, 1.0), (3.201, 1.0), (3.30, 1.0)],
                        field_mhz=MHZ, dj_bad_hz=BAD, dj_gold_hz=GOLD,
                        max_drift_ppm=0.01)
        assert fit['n_alternatives'] == 1

    def test_nothing_within_the_tolerance_is_no_reading(self):
        fit = fit_lines([3.20], [(3.40, 1.0)], field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['fitted'] is False


class TestUncertainty:
    def test_a_lone_good_fit_has_no_alternatives(self):
        exp = [1.47 - D / 2, 1.47 + D / 2]
        fit = fit_lines(exp, [(exp[0], 1.0), (exp[1], 1.0)], field_mhz=MHZ,
                        dj_bad_hz=BAD, dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['n_alternatives'] == 0

    def test_a_second_pair_at_the_same_spacing_is_reported(self):
        """Two pairs 7 Hz apart in one window: the reading stands, but it was
        not the only one the window allowed."""
        exp = [1.47 - D / 2, 1.47 + D / 2]
        cands = [(exp[0], 1.0), (exp[1], 1.0),
                 (exp[0] + 0.01, 1.0), (exp[1] + 0.01, 1.0)]
        fit = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD,
                        dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert fit['n_alternatives'] >= 1


class TestWindow:
    def test_spans_the_pattern_plus_the_drift_allowed(self):
        lo, hi = fit_window([1.46, 1.48], 0.01)
        assert (lo, hi) == pytest.approx((1.45, 1.49))

    def test_no_lines_no_window(self):
        assert fit_window([], 0.01) is None


class TestAcrossSamples:
    def test_each_sample_is_fitted_on_its_own_trace(self):
        ppm = _axis()
        a = _spectrum(ppm, [(1.47 - D / 2, 1.0), (1.47 + D / 2, 1.0)])
        b = _spectrum(ppm, [(1.48 - D / 2, 1.0), (1.48 + D / 2, 1.0)])
        fits = fit_across_samples([1.47 - D / 2, 1.47 + D / 2], ppm,
                                  np.column_stack([a, b]), field_mhz=MHZ,
                                  dj_bad_hz=BAD, dj_gold_hz=GOLD,
                                  max_drift_ppm=0.05)
        assert float(np.mean(fits[0]['ppms'])) == pytest.approx(1.47, abs=5e-4)
        assert float(np.mean(fits[1]['ppms'])) == pytest.approx(1.48, abs=5e-4)

    def test_a_sample_without_the_analyte_reports_no_fit(self):
        ppm = _axis()
        a = _spectrum(ppm, [(1.47 - D / 2, 1.0), (1.47 + D / 2, 1.0)])
        b = np.zeros_like(ppm)
        fits = fit_across_samples([1.47 - D / 2, 1.47 + D / 2], ppm,
                                  np.column_stack([a, b]), field_mhz=MHZ,
                                  dj_bad_hz=BAD, dj_gold_hz=GOLD,
                                  max_drift_ppm=0.05, noise_floor=0.01)
        assert fits[0]['fitted'] is True
        assert fits[1]['fitted'] is False

    def test_excluded_samples_are_not_fitted(self):
        ppm = _axis()
        cols = np.column_stack([_spectrum(ppm, [(1.47, 1.0)])] * 3)
        fits = fit_across_samples([1.47], ppm, cols, included=[0, 2],
                                  field_mhz=MHZ, dj_bad_hz=BAD,
                                  dj_gold_hz=GOLD, max_drift_ppm=0.05)
        assert sorted(fits) == [0, 2]


def _exhaustive(expected_ppms, candidates, *, field_mhz, dj_bad_hz,
                dj_gold_hz, max_drift_ppm=None):
    """The assignment as first written: every order-preserving chain within
    dj_bad_hz, of every length, then the hierarchical choice. Exponential on
    dense patterns, kept here only as the reference the pruned search must
    reproduce exactly."""
    from processing.multiplet_line_fit import _cap
    exp = sorted(float(p) for p in expected_ppms)
    cands = _cap([(float(p), float(h)) for p, h in candidates])
    cp = [p for p, _ in cands]
    scored = []

    def _ext(ranks, picks, res):
        for r in range(ranks[-1] + 1, len(exp)):
            for c in range(picks[-1] + 1, len(cp)):
                pr = abs((cp[c] - cp[picks[-1]]) - (exp[r] - exp[ranks[-1]])) \
                    * field_mhz
                if pr > dj_bad_hz:
                    continue
                w = max(pr, res)
                ch = [cp[i] for i in picks + [c]]
                rk = ranks + [r]
                drift = abs(sum(ch) / len(ch) - sum(exp[i] for i in rk) / len(rk))
                if max_drift_ppm is None or drift <= max_drift_ppm:
                    scored.append((len(rk), w, drift, rk, ch))
                _ext(rk, picks + [c], w)

    for r in range(len(exp)):
        for c in range(len(cp)):
            _ext([r], [c], 0.0)
    if not scored:
        return None
    best = min(scored, key=lambda s: (-s[0], s[1], s[2]))
    alts = sum(1 for s in scored if s[0] == best[0] and s[4] != best[4]
               and abs(s[1] - best[1]) <= dj_gold_hz)
    return best[3], best[4], best[1], best[2], alts


def test_the_pruned_search_chooses_exactly_what_the_exhaustive_one_did():
    rng = np.random.default_rng(7)
    for _ in range(400):
        ne, nc = int(rng.integers(2, 9)), int(rng.integers(2, 10))
        j = rng.uniform(1.0, 10.0) / MHZ
        exp = list(1.5 + (np.arange(ne) * j if rng.random() < 0.5
                          else np.sort(rng.uniform(0, ne * j, ne))))
        cands = [(float(p), 1.0) for p in
                 (1.5 + np.arange(nc) * rng.uniform(1, 8) / MHZ
                  if rng.random() < 0.3
                  else rng.uniform(1.49, 1.5 + ne * j + 0.01, nc))]
        kw = dict(field_mhz=MHZ, dj_bad_hz=float(rng.choice([0.5, 1.0, 2.0])),
                  dj_gold_hz=GOLD,
                  max_drift_ppm=float(rng.choice([0.005, 0.02])))
        ref = _exhaustive(exp, cands, **kw)
        f = fit_lines(exp, cands, **kw)
        if ref is None:
            assert not f['fitted']
            continue
        assert (f['ranks'], f['ppms'], f['residual_hz'], f['drift_ppm'],
                f['n_alternatives']) == ref


def test_a_dense_pattern_with_evenly_spaced_apices_does_not_stall():
    """58 lines 0.2-1 Hz apart (a BMRB pattern like L-citrulline) match
    almost any spacing to 0.5 Hz; twelve apices 2.5 Hz apart in it took the
    exhaustive search 9 s for one sample, and the reading of a real study
    stalled on such signals."""
    import time
    rng = np.random.default_rng(3)
    exp = list(3.70 + np.cumsum(rng.uniform(0.2, 1.0, 58)) / MHZ)
    mid = (exp[0] + exp[-1]) / 2
    cands = [(mid + (i - 5.5) * 2.5 / MHZ, 1.0) for i in range(12)]
    t0 = time.perf_counter()
    f = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD, dj_gold_hz=GOLD,
                  max_drift_ppm=0.02)
    assert time.perf_counter() - t0 < 0.5
    assert f['fitted'] and len(f['ranks']) >= 2


def test_a_list_far_denser_than_the_apices_is_reported_not_read():
    """The doublet of 2-isopropylmalic acid arrived with 223 lines, some
    0.03 Hz apart, against 12 apices 3-7 Hz apart; searching one sample to
    the end took 110 s. Past the budget the sample is not read -- no
    assignment is picked from an unfinished search."""
    import time
    rng = np.random.default_rng(5)
    exp = list(0.80 + np.cumsum(rng.uniform(0.02, 0.7, 223)) / MHZ)
    gaps = [0, 3.65, 2.84, 4.12, 4.31, 6.27, 6.16, 5.3, 3.73, 6.98, 3.26, 7.16]
    c0 = (exp[0] + exp[-1]) / 2
    cands = [(c0 + (x - 27.0) / MHZ, 1.0) for x in np.cumsum(gaps)]
    t0 = time.perf_counter()
    f = fit_lines(exp, cands, field_mhz=MHZ, dj_bad_hz=BAD, dj_gold_hz=GOLD,
                  max_drift_ppm=0.05)
    assert time.perf_counter() - t0 < 10
    assert f['too_ambiguous'] is True
    assert f['fitted'] is False and f['ppms'] == [] and f['ranks'] == []


def test_an_ordinary_fit_is_not_flagged():
    exp = [1.5 - D, 1.5, 1.5 + D]
    f = fit_lines(exp, [(p, 1.0) for p in exp], field_mhz=MHZ, dj_bad_hz=BAD,
                  dj_gold_hz=GOLD, max_drift_ppm=0.02)
    assert f['fitted'] and f['too_ambiguous'] is False
