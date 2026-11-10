# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_signal_structure.py
"""Blocks of signals, axes of the samples, and what each does between groups."""
import numpy as np
import pytest

from processing.signal_structure import (
    first_component, group_means, group_tests, mean_pairwise_r,
    sample_components, sample_distance, signal_modules, split_factors)

G4 = (['Flight_Younger'] * 6 + ['Flight_Older'] * 6 + ['HC_Younger'] * 6
      + ['HC_Older'] * 6)


class TestFactors:

    def test_two_token_labels_split_in_two_factors(self):
        f = split_factors(G4)
        assert set(f) == {'A', 'B'}
        assert set(f['A']) == {'Flight', 'HC'} and set(f['B']) == {'Younger', 'Older'}

    @pytest.mark.parametrize('labels', [
        ['a', 'b', 'a', 'b'],                       # one token
        ['x_a', 'x_b', 'x_a', 'x_b'],              # first token never varies
        ['x_a', 'y_b_c', 'x_a', 'y_b_c'],          # unequal token counts
    ])
    def test_otherwise_the_labels_are_one_factor(self, labels):
        assert split_factors(labels) == {}


class TestGroupTests:

    def test_one_way_finds_a_difference_and_a_non_difference(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 3, 0, 3], float).repeat(6) + rng.normal(0, 0.5, 24)
        assert group_tests(y, G4)['group'] < 1e-6
        assert group_tests(rng.normal(size=24), G4)['group'] > 0.01

    def test_a_flight_effect_is_factor_A_and_not_B_or_the_interaction(self):
        rng = np.random.default_rng(1)
        flight = np.array([1, 1, 0, 0], float).repeat(6)
        y = 2.0 * flight + rng.normal(0, 0.4, 24)
        p = group_tests(y, G4)
        assert p['A'] < 1e-6
        assert p['B'] > 0.01 and p['AxB'] > 0.01

    def test_an_interaction_is_seen_as_one(self):
        rng = np.random.default_rng(2)
        y = np.array([3, 0, 0, 0], float).repeat(6) + rng.normal(0, 0.4, 24)
        assert group_tests(y, G4)['AxB'] < 1e-4

    def test_single_factor_labels_have_only_the_group_test(self):
        p = group_tests(np.arange(12.0), ['a'] * 4 + ['b'] * 4 + ['c'] * 4)
        assert set(p) == {'group'}

    def test_a_group_of_one_or_no_variation_is_nan_not_an_error(self):
        assert np.isnan(group_tests(np.ones(6), ['a'] * 3 + ['b'] * 3)['group'])
        assert np.isnan(group_tests(np.arange(5.0), ['a'] * 4 + ['b'])['group'])

    def test_group_means_are_in_sorted_label_order(self):
        m = group_means([1.0, 2.0, 10.0, 20.0], ['b', 'b', 'a', 'a'])
        assert list(m) == ['a', 'b'] and m == {'a': 15.0, 'b': 1.5}


class TestModules:

    def _R(self):
        # blocks {0,1,2} and {3,4,5,6}; 7 alone; 8 flat (NaN)
        R = np.eye(9) * 1.0
        for blk, r in (([0, 1, 2], 0.95), ([3, 4, 5, 6], 0.9)):
            for i in blk:
                for j in blk:
                    if i != j:
                        R[i, j] = r
        R[8, :] = R[:, 8] = np.nan
        R[8, 8] = 1.0
        return R

    def test_blocks_are_recovered_biggest_first(self):
        blocks, lab = signal_modules(self._R(), r_cut=0.8)
        assert [sorted(b.tolist()) for b in blocks] == [[3, 4, 5, 6], [0, 1, 2]]
        assert len(set(lab)) == 4              # two blocks + 7 + the flat one

    def test_a_flat_signal_relates_to_nothing(self):
        blocks, lab = signal_modules(self._R(), r_cut=0.5)
        assert all(8 not in b for b in blocks)

    def test_a_stricter_cut_splits_a_loose_block(self):
        R = np.full((4, 4), 0.85)
        np.fill_diagonal(R, 1.0)
        assert len(signal_modules(R, r_cut=0.8)[0]) == 1
        assert signal_modules(R, r_cut=0.9)[0] == []

    def test_min_size_keeps_pairs_out(self):
        R = np.eye(4)
        R[0, 1] = R[1, 0] = 0.99
        assert signal_modules(R, r_cut=0.8)[0] == []
        assert len(signal_modules(R, r_cut=0.8, min_size=2)[0]) == 1

    def test_too_few_signals(self):
        assert signal_modules(np.eye(1))[0] == []

    def test_mean_pairwise_r(self):
        assert mean_pairwise_r(self._R(), [3, 4, 5, 6]) == pytest.approx(0.9)
        assert np.isnan(mean_pairwise_r(self._R(), [0]))


class TestComponents:

    def test_first_component_of_a_block_carries_it_and_rises_with_it(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=20)
        X = np.vstack([a * z + rng.normal(0, 0.05, 20) for a in (1, 2, -3)])
        sc, share, load = first_component(X)
        assert share > 0.95 and abs(np.corrcoef(sc, z)[0, 1]) > 0.99
        assert np.sum(load > 0) >= np.sum(load < 0)       # sign: most rise

    def test_flat_rows_are_ignored_not_fatal(self):
        sc, share, load = first_component(np.vstack([np.arange(8.0), np.ones(8)]))
        assert share == pytest.approx(1.0) and load[1] == 0.0

    def test_all_flat_gives_nan(self):
        sc, share, _ = first_component(np.ones((2, 5)))
        assert np.isnan(share) and np.isnan(sc).all()

    def test_pca_separates_two_clusters_of_samples(self):
        rng = np.random.default_rng(4)
        grp = np.array([0.0] * 10 + [4.0] * 10)
        X = np.vstack([grp * a + rng.normal(0, 0.5, 20) for a in (1, 1.5, -1, 2)]
                      + [rng.normal(size=20) for _ in range(6)])
        pc = sample_components(X, k=3)
        assert pc['scores'].shape == (20, 3) and pc['loadings'].shape == (10, 3)
        assert 0 < pc['variance'].sum() <= 1.0 + 1e-9
        s1 = pc['scores'][:, 0]
        assert abs(s1[:10].mean() - s1[10:].mean()) > 3 * s1.std() / 2

    def test_the_outlier_stands_furthest(self):
        sc = np.vstack([np.random.default_rng(5).normal(size=(15, 2)), [[9.0, 9.0]]])
        assert int(np.argmax(sample_distance(sc))) == 15
