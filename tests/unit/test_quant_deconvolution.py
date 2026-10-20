# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A window is not a container for one signal.

A Lorentzian's tails fall away as the square of the distance and never reach
zero, so every line puts a little of itself into every window and a strong
neighbour a short way off puts in a great deal.

Measured on a study whose answer was known: L-alanine sits 0.148 ppm from
L-lactic acid, whose line is eight times larger, and after every other
correction alanine still read 6.4% high. Removing the neighbours took the whole
study's mean absolute error from 1.7% to 0.8%, and alanine's from 6.4% to 2.9%.
"""
import numpy as np
import pytest

from processing.quant_deconvolution import (
    fraction_in_window, neighbour_leak, subtract_neighbours)

G = 0.0027                      # hwhm of a 3.2 Hz line at 600 MHz


class TestHowMuchOfASignalAWindowHolds:
    def test_a_window_on_the_line_holds_most_of_it(self):
        assert fraction_in_window([1.0], [1.0], 0.98, 1.02, G) > 0.9

    def test_a_window_far_away_holds_almost_none(self):
        assert fraction_in_window([1.0], [1.0], 1.5, 1.52, G) < 0.001

    def test_the_tail_falls_away_but_never_to_zero(self):
        """Which is the whole reason this module exists."""
        near = fraction_in_window([1.0], [1.0], 1.05, 1.07, G)
        far = fraction_in_window([1.0], [1.0], 1.20, 1.22, G)
        assert near > far > 0.0

    def test_the_leak_matters_because_the_neighbour_is_larger(self):
        """A tenth of a percent of lactate is percents of alanine.

        At the 0.148 ppm that separates them, a window on alanine holds
        about 0.1% of lactate's area — negligible against lactate, and not
        against alanine, which is eight times smaller. That is why the
        correction is applied as a share of the RECEIVING signal.
        """
        f = fraction_in_window([1.318], [1.0], 1.454, 1.479, G)
        assert 0.0005 < f < 0.005, f
        assert f * 8 > 0.005, 'eight times over, it is percent-level'

    def test_it_is_the_capture_correction_read_the_other_way(self):
        """One model, so a leak subtracted from one analyte is the quantity the
        other was credited with."""
        from processing.quant_integration import lorentzian_captured_fraction
        own = lorentzian_captured_fraction([1.0], [1.0], 0.98, 1.02, G)
        same = fraction_in_window([1.0], [1.0], 0.98, 1.02, G)
        assert own == pytest.approx(same, rel=1e-9)

    def test_no_linewidth_means_no_claim(self):
        assert fraction_in_window([1.0], [1.0], 0.98, 1.02, 0) == 0.0


class TestTheLeak:
    def test_it_scales_with_how_much_neighbour_there_is(self):
        w = (1.454, 1.479)
        one = neighbour_leak(w, [([1.318], [1.0], 100.0)], G)
        ten = neighbour_leak(w, [([1.318], [1.0], 1000.0)], G)
        assert ten == pytest.approx(10 * one)

    def test_neighbours_add_up(self):
        w = (1.454, 1.479)
        a = neighbour_leak(w, [([1.318], [1.0], 100.0)], G)
        b = neighbour_leak(w, [([1.600], [1.0], 100.0)], G)
        both = neighbour_leak(w, [([1.318], [1.0], 100.0),
                                  ([1.600], [1.0], 100.0)], G)
        assert both == pytest.approx(a + b)

    def test_a_neighbour_with_no_measured_area_contributes_nothing(self):
        assert neighbour_leak((1.454, 1.479),
                              [([1.318], [1.0], float('nan'))], G) == 0.0


class TestSubtracting:
    def _two(self, big=1000.0, small=100.0):
        return {
            'strong': {'window': (1.310, 1.326), 'lines': [1.318],
                       'weights': [1.0], 'areas': {0: big, 1: big}},
            'weak': {'window': (1.458, 1.474), 'lines': [1.466],
                     'weights': [1.0], 'areas': {0: small, 1: small}},
        }

    def test_the_weak_one_loses_more_than_the_strong_one(self):
        """Its neighbour is ten times larger, so the same absolute leak is ten
        times the share."""
        _, removed = subtract_neighbours(self._two(), G)
        assert removed['weak'] > removed['strong']

    def test_the_area_comes_down_by_the_share_that_was_not_its_own(self):
        out, removed = subtract_neighbours(self._two(), G)
        assert out['weak'][0] == pytest.approx(100.0 * (1 - removed['weak']))

    def test_every_sample_is_corrected_by_the_same_share(self):
        """The leak is a property of the geometry, not of one injection."""
        out, _ = subtract_neighbours(self._two(), G)
        assert out['weak'][0] == pytest.approx(out['weak'][1])

    def test_a_lone_signal_has_nothing_to_lose(self):
        one = {'a': {'window': (1.0, 1.02), 'lines': [1.01], 'weights': [1.0],
                     'areas': {0: 50.0}}}
        out, removed = subtract_neighbours(one, G)
        assert out['a'] == {0: 50.0} and removed['a'] == 0.0

    def test_a_window_that_is_mostly_somebody_else_is_left_alone(self):
        """At that point subtracting would report the remainder as an analyte.
        That is the shared peak, which the election settles and this does not.
        """
        sigs = self._two(big=100000.0, small=1.0)
        out, removed = subtract_neighbours(sigs, G)
        assert removed['weak'] == 0.0
        assert out['weak'] == {0: 1.0, 1: 1.0}

    def test_it_does_not_iterate_against_itself(self):
        """One pass. The leaks are a few percent, so a second pass would move
        things by a few percent of a few percent, and iterating a model against
        its own output invites it to converge on its assumptions."""
        import inspect
        src = inspect.getsource(subtract_neighbours)
        assert 'while' not in src and 'for _ in range' not in src
