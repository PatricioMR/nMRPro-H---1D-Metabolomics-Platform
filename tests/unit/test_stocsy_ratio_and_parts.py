# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy_ratio_and_parts.py
"""The fixed-ratio test and the split of a correlation into where it comes
from (all samples / within groups / between group means)."""
import numpy as np
import pytest

from processing.stocsy import (
    AREA_NOISE_KAPPA, area_noise, covariation_parts, fixed_ratio)

N = 24


def _areas(seed=0, cv=0.4):
    rng = np.random.default_rng(seed)
    a = 1000 * np.exp(rng.normal(0, cv, N))
    b = 1000 * np.exp(rng.normal(0, cv, N))
    return a, b, rng


class TestAreaNoise:

    def test_it_is_kappa_times_white_noise_and_grows_with_the_window(self):
        a = area_noise(10.0, 0.001, 100)
        assert a == pytest.approx(AREA_NOISE_KAPPA * 10.0 * 0.001 * 10.0)
        assert area_noise(10.0, 0.001, 400) == pytest.approx(2 * a)

    def test_dx_sign_does_not_matter(self):
        assert area_noise(5.0, -0.002, 50) == area_noise(5.0, 0.002, 50)


class TestFixedRatio:

    def test_a_fixed_ratio_reads_as_noise_only(self):
        a, _b, rng = _areas()
        sig = 5.0
        y = 0.4 * a + rng.normal(0, sig, N)
        x = a + rng.normal(0, sig, N)
        fr = fixed_ratio(x, y[None, :], sig, np.array([sig]))
        assert fr['ratio'][0] == pytest.approx(0.4, rel=0.03)
        assert fr['sd'][0] < 0.04 and fr['excess_sd'][0] < 0.03
        assert 0.3 < fr['q'][0] < 3 and fr['p'][0] > 0.01
        assert fr['n'][0] == N

    def test_independent_molecules_have_a_ratio_that_varies_beyond_noise(self):
        a, b, rng = _areas(1)
        fr = fixed_ratio(a, b[None, :], 5.0, np.array([5.0]))
        assert fr['sd'][0] > 0.3 and fr['excess_sd'][0] > 0.25
        assert fr['p'][0] < 1e-6

    def test_a_dilution_cancels_in_the_ratio_but_not_in_r(self):
        rng = np.random.default_rng(2)
        d = np.exp(rng.normal(0, 0.5, N))
        a = 1000 * np.exp(rng.normal(0, 0.05, N)) * d
        b = 800 * np.exp(rng.normal(0, 0.05, N)) * d
        fr = fixed_ratio(a, b[None, :], 1.0, np.array([1.0]))
        assert fr['sd'][0] < 0.1

    def test_a_noisy_signal_is_not_called_inconsistent_for_its_noise(self):
        """With a big noise the same ratio is read with a big spread: the test
        is against what noise leaves, so it must not reject."""
        a, _b, rng = _areas(3)
        sig = 150.0
        x = a + rng.normal(0, sig, N)
        y = 0.5 * a + rng.normal(0, sig, N)
        fr = fixed_ratio(x, y[None, :], sig, np.array([sig]))
        assert fr['p'][0] > 0.01
        assert fr['noise_sd'][0] > 0.15

    def test_samples_under_the_noise_floor_are_not_read(self):
        a, _b, rng = _areas(4)
        y = 0.5 * a
        a2 = a.copy()
        a2[:10] = 2.0                         # under 3 sigma
        fr = fixed_ratio(a2, y[None, :], 5.0, np.array([5.0]), min_valid=8)
        assert fr['n'][0] == N - 10

    def test_too_few_samples_read_gives_nan_not_a_number(self):
        a = np.full(N, 2.0)
        a[:5] = 1000.0
        fr = fixed_ratio(a, (0.5 * a)[None, :], 5.0, np.array([5.0]), min_valid=8)
        assert fr['n'][0] == 5
        assert all(np.isnan(fr[k][0]) for k in ('sd', 'excess_sd', 'p', 'ratio'))

    def test_one_row_per_signal(self):
        a, b, _ = _areas(5)
        M = np.vstack([b, 2 * a, a])
        fr = fixed_ratio(a, M, 1.0, np.ones(3))
        assert fr['sd'].shape == (3,)
        assert fr['sd'][2] == pytest.approx(0.0, abs=1e-9)
        assert fr['sd'][1] < fr['sd'][0]


class TestCovariationParts:

    G = ['A'] * 8 + ['B'] * 8 + ['C'] * 8

    def test_a_correlation_made_only_of_a_group_effect_is_all_between(self):
        rng = np.random.default_rng(6)
        step = np.repeat([0.0, 3.0, 6.0], 8)
        x = step + rng.normal(0, 0.5, N)
        y = step + rng.normal(0, 0.5, N)
        p = covariation_parts(x, y[None, :], self.G)
        assert p['all'][0] > 0.9 and p['between'][0] > 0.95
        assert abs(p['within'][0]) < 0.5

    def test_a_shared_sample_factor_survives_within_groups(self):
        rng = np.random.default_rng(7)
        z = rng.normal(0, 1, N)
        x = z + rng.normal(0, 0.1, N)
        y = 2 * z + rng.normal(0, 0.1, N)
        p = covariation_parts(x, y[None, :], self.G)
        assert p['within'][0] > 0.9

    def test_with_two_groups_the_between_part_is_undefined(self):
        x = np.arange(N, dtype=float)
        p = covariation_parts(x, x[None, :], ['A'] * 12 + ['B'] * 12)
        assert np.isnan(p['between'][0]) and p['all'][0] == pytest.approx(1.0)

    def test_without_usable_groups_within_is_nan(self):
        x = np.arange(N, dtype=float)
        p = covariation_parts(x, x[None, :], ['A'] * N)
        assert np.isnan(p['within'][0])

    def test_pqn_factors_are_applied_to_every_part(self):
        rng = np.random.default_rng(8)
        d = np.exp(rng.normal(0, 0.5, N))
        x = rng.uniform(1, 2, N) * d
        y = rng.uniform(1, 2, N) * d
        raw = covariation_parts(x, y[None, :], self.G)['all'][0]
        pqn = covariation_parts(x, y[None, :], self.G, factors=d)['all'][0]
        assert raw > 0.5 and abs(pqn) < 0.5
