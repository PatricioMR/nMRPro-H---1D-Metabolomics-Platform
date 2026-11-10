# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_signal_comparison.py
"""A signal's quantification against the reference's: the gap, the spread and
the distance, on signals whose relation is known."""
import numpy as np
import pytest

from processing.signal_comparison import (
    closest_first, compare_to_reference, per_proton, quantity_gap)

N = 16


def _study(seed=0):
    rng = np.random.default_rng(seed)
    conc = 1000 * np.exp(rng.normal(0, 0.4, N))
    noise = lambda: np.exp(rng.normal(0, 0.02, N))
    return rng, conc, 1 * conc * noise(), noise               # the reference holds 1 H


class TestQuantityGap:

    def test_a_signal_with_the_area_its_protons_call_for_is_at_zero(self):
        assert quantity_gap(3.0, 1, 3) == pytest.approx(0.0)       # 3 H, 3x the area

    def test_a_signal_with_too_little_area_for_its_protons_is_below(self):
        assert quantity_gap(2.0, 1, 3) == pytest.approx(2 / 3 - 1)  # 3 H but 2x the area

    def test_two_lines_of_one_signal_hold_the_protons_between_them(self):
        assert quantity_gap(0.5, 1, 1) == pytest.approx(-0.5)

    @pytest.mark.parametrize('ratio', [np.nan, 0.0, -1.0])
    def test_an_unreadable_ratio_is_nan_not_a_number(self, ratio):
        assert np.isnan(quantity_gap(ratio, 1, 2))

    @pytest.mark.parametrize('n_ref,n_signal', [(0, 2), (None, 2), (1, 0), (1, None)])
    def test_a_missing_count_of_protons_gives_nan(self, n_ref, n_signal):
        assert np.isnan(quantity_gap(2.0, n_ref, n_signal))


class TestCompare:

    def test_the_other_signal_of_the_molecule_is_near_and_another_molecule_is_not(self):
        rng, conc, ref, noise = _study()
        three = 3 * conc * noise()
        other = 700 * np.exp(rng.normal(0, 0.4, N))
        r = compare_to_reference(ref, np.vstack([three, other]), 5.0,
                                 np.array([5.0, 5.0]), 1, [3, 3])
        assert abs(r[0]['gap']) < 0.05 and r[0]['distance'] < 0.12
        assert r[1]['distance'] > 0.3 and r[1]['sd'] > 0.25

    def test_the_same_area_read_as_more_protons_is_below_the_reference(self):
        _rng, conc, ref, noise = _study(5)
        r = compare_to_reference(ref, (conc * noise())[None, :], 5.0, np.array([5.0]), 1, [2])
        assert r[0]['gap'] == pytest.approx(-0.5, abs=0.05)

    def test_a_dilution_cancels_in_the_ratio(self):
        rng, conc, ref, noise = _study(1)
        d = np.exp(rng.normal(0, 0.5, N))
        r = compare_to_reference(ref * d, (2 * conc * noise() * d)[None, :], 5.0,
                                 np.array([5.0]), 1, [2])
        assert abs(r[0]['gap']) < 0.05 and r[0]['sd'] < 0.1

    def test_a_row_without_protons_has_a_ratio_but_no_gap_or_distance(self):
        _rng, conc, ref, noise = _study(2)
        r = compare_to_reference(ref, (2 * conc * noise())[None, :], 5.0, np.array([5.0]), 1, [None])
        assert r[0]['ratio'] == pytest.approx(2.0, rel=0.05)
        assert np.isnan(r[0]['gap']) and np.isnan(r[0]['distance'])

    def test_too_few_samples_over_the_noise_gives_nan(self):
        _rng, _conc, ref, _n = _study(3)
        r = compare_to_reference(ref, np.full(N, 1.0)[None, :], 5.0, np.array([5.0]), 1, [1])
        assert r[0]['n'] == 0 and np.isnan(r[0]['distance']) and np.isnan(r[0]['gap'])

    def test_the_reference_against_itself_is_a_ratio_of_one(self):
        _rng, _conc, ref, _n = _study(4)
        r = compare_to_reference(ref, ref[None, :], 5.0, np.array([5.0]), 2, [2])
        assert r[0]['ratio'] == pytest.approx(1.0) and r[0]['distance'] == pytest.approx(0.0, abs=1e-9)


class TestOrder:

    def test_nearest_first_and_the_unreadable_last(self):
        assert closest_first([0.3, np.nan, 0.05, 0.1]) == [2, 3, 0, 1]

    def test_a_tie_keeps_the_input_order(self):
        assert closest_first([0.1, 0.1, np.nan, np.nan]) == [0, 1, 2, 3]

    def test_nothing_to_order(self):
        assert closest_first([]) == []


def test_per_proton_divides_the_areas():
    assert list(per_proton([6.0, 9.0], 3)) == [2.0, 3.0]
