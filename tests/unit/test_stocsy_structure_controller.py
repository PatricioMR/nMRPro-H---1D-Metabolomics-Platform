# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy_structure_controller.py
"""Modules, axes and samples of a study the controller builds, on a study whose
structure is known: a molecule that differs between groups, a pool of molecules
that follow one per-sample factor, a reference and a water tail."""
import numpy as np
import pytest

from controllers.stocsy_controller import StocsyAnalysis

N_PER = 8
GROUPS = (['Flight_Younger'] * N_PER + ['Flight_Older'] * N_PER
          + ['HC_Younger'] * N_PER + ['HC_Older'] * N_PER)
N = len(GROUPS)


def _study(seed=0, blanks=4, dilution_sd=0.0):
    rng = np.random.default_rng(seed)
    noise = lambda: np.exp(rng.normal(0, 0.05, N))
    dil = np.exp(rng.normal(0, dilution_sd, N)) if dilution_sd else np.ones(N)
    flight = np.array([g.startswith('Flight') for g in GROUPS], float)
    pool_f = rng.normal(0, 0.5, N)                  # a per-sample factor
    ref_f = np.exp(rng.normal(0, 0.15, N))
    tail_f = np.exp(rng.normal(0, 0.3, N))
    mol_a = 1000 * np.exp(1.0 * flight) * np.exp(rng.normal(0, 0.1, N))
    areas = {}
    # molecule A: two signals, a fixed ratio, higher in Flight
    areas[1] = (mol_a * noise(), 3.2)
    areas[2] = (0.5 * mol_a * noise(), 3.9)
    areas[15] = (0.25 * mol_a * noise(), 5.5)
    # a pool: three different molecules following the same per-sample factor
    for k, ppm in zip((3, 4, 5), (1.04, 1.48, 2.15)):
        areas[k] = (600 * np.exp(pool_f) * noise(), ppm)
    # the reference: 0 ppm and its satellites
    for k, ppm in zip((6, 7, 8), (0.0, 0.1, -0.1)):
        areas[k] = (3000 * ref_f * noise(), ppm)
    # the water tail beside the excluded region
    for k, ppm in zip((9, 10, 11), (4.45, 4.52, 4.58)):
        areas[k] = (900 * tail_f * noise(), ppm)
    # unrelated signals
    for k, ppm in zip((12, 13, 14), (6.1, 7.2, 8.3)):
        areas[k] = (500 * np.exp(rng.normal(0, 0.4, N)), ppm)
    peaks = []
    for k, (a, ppm) in areas.items():
        a = a * dil
        peaks.append({'rank': k, 'ppm': ppm, 'snr': 60.0 + k,
                      'start_idx': 100 * k, 'end_idx': 100 * k + 40,
                      'area_per_sample': np.append(a, np.zeros(blanks)),
                      'assigned_metabolite': {1: 'MolA', 2: 'MolA'}.get(k)})
    ppm_axis = np.linspace(10.0, -0.5, 2000)
    X = np.random.default_rng(seed + 1).normal(0, 1.0, (ppm_axis.size, N + blanks))
    info = [{'index': i, 'label': GROUPS[i], 'type': 'Sample', 'name': f'S{i}'}
            for i in range(N)]
    info += [{'index': N + i, 'label': 'BLK', 'type': 'BLK', 'name': f'B{i}'}
             for i in range(blanks)]
    return ppm_axis, X, peaks, info


def _an(**kw):
    opts = {k: kw.pop(k) for k in ('dilution_sd',) if k in kw}
    ppm, X, peaks, info = _study(**opts)
    return StocsyAnalysis(ppm, X, peaks, info, excluded_regions=[(4.7, 5.0)],
                          noise_std=1.0, **kw)


def _module_with(an, rank, **kw):
    return next((m for m in an.modules(**kw) if rank in m['ranks']), None)


class TestModules:

    def test_the_molecule_is_one_module_and_it_is_a_group_signal(self):
        m = _module_with(_an(), 1, r_cut=0.8, min_snr=10)
        assert m is not None and set(m['ranks']) >= {1, 2, 15}
        assert m['p']['A'] < 1e-6                       # Flight vs HC
        assert m['p']['B'] > 0.01
        assert m['names'] == {'MolA': 2}                 # annotation counts
        assert max(m['means'], key=m['means'].get).startswith('Flight')

    def test_the_pool_is_a_module_that_does_not_differ_between_groups(self):
        m = _module_with(_an(), 3, min_snr=10)
        assert m is not None and {3, 4, 5} <= set(m['ranks'])
        assert m['mean_r'] > 0.8
        assert m['p']['group'] > 0.01 and m['artefact'] is None

    def test_the_reference_and_the_water_tail_are_named_as_such(self):
        an = _an()
        assert _module_with(an, 6, min_snr=10)['artefact'] == 'reference'
        assert _module_with(an, 9, min_snr=10)['artefact'] == 'water tail'

    def test_modules_come_biggest_first_with_ids_in_that_order(self):
        mods = _an().modules(min_snr=10)
        assert [m['n'] for m in mods] == sorted((m['n'] for m in mods), reverse=True)
        assert [m['id'] for m in mods] == list(range(1, len(mods) + 1))

    def test_members_are_listed_high_delta_first(self):
        m = _module_with(_an(), 9, min_snr=10)
        ppms = [_an().ppm_of(r) for r in m['ranks']]
        assert ppms == sorted(ppms, reverse=True)

    def test_a_signal_under_the_snr_bar_is_not_read(self):
        an = _an()
        assert _module_with(an, 1, min_snr=500) is None
        assert an.modules(min_snr=500) == []

    def test_the_options_of_the_correlation_reach_the_modules(self):
        """Raw areas under a 60% dilution: everything that follows the sample
        joins one block; PQN gives the blocks back."""
        an = _an(dilution_sd=0.6)
        raw = _module_with(an, 12, r_cut=0.5, min_snr=10)
        an.set_options(normalize='pqn')
        pqn = _module_with(an, 12, r_cut=0.5, min_snr=10)
        assert raw is not None and raw['n'] >= 8
        assert pqn is None or pqn['n'] < raw['n']

    def test_modules_are_cached_until_the_options_change(self):
        an = _an()
        assert an.modules(min_snr=10) is an.modules(min_snr=10)
        first = an.modules(min_snr=10)
        an.set_options(method='spearman')
        assert an.modules(min_snr=10) is not first


class TestAxesAndSamples:

    def test_an_axis_follows_the_group_difference(self):
        comps = _an().components(4, min_snr=10)
        assert len(comps) == 4
        assert sum(c['variance'] for c in comps) <= 1.0 + 1e-9
        assert min(c['p']['A'] for c in comps) < 1e-4

    def test_loadings_name_the_signals_with_their_delta(self):
        c = _an().components(2, min_snr=10)[0]
        assert len(c['loadings']) == 8
        assert all(len(t) == 3 and isinstance(t[0], int) for t in c['loadings'])

    def test_too_few_signals_or_samples_gives_no_axes(self):
        assert _an().components(3, min_snr=1e6) == []

    def test_one_row_per_analytical_sample_with_its_pqn_factor(self):
        rows = _an().samples(3, min_snr=10)
        assert len(rows) == N                           # blanks are not samples
        assert {'name', 'group', 'pqn_factor', 'scores', 'distance'} <= set(rows[0])
        assert len(rows[0]['scores']) == 3
        assert all(r['pqn_factor'] > 0 for r in rows)

    def test_the_sample_that_is_odd_stands_out(self):
        ppm, X, peaks, info = _study(seed=3)
        for p in peaks:
            if p['rank'] in (12, 13, 14, 1, 2):
                p['area_per_sample'][5] *= 12.0         # one odd sample
        an = StocsyAnalysis(ppm, X, peaks, info, excluded_regions=[(4.7, 5.0)],
                            noise_std=1.0)
        rows = an.samples(3, min_snr=10)
        assert max(rows, key=lambda r: r['distance'])['name'] == 'S5'

    def test_a_study_whose_groups_differ_in_dilution_says_so(self):
        ppm, X, peaks, info = _study(seed=4)
        for p in peaks:                                 # Flight_Younger half as concentrated
            p['area_per_sample'][:N_PER] *= 0.5
        an = StocsyAnalysis(ppm, X, peaks, info, noise_std=1.0)
        div = an.dilution_by_group()
        assert div['diverges'] and div['ratio'] > 1.3
        an.set_options(normalize='pqn')
        assert any('PQN factors differ between groups' in n for n in an.notes())

    def test_the_note_is_about_pqn_only(self):
        ppm, X, peaks, info = _study(seed=4)
        for p in peaks:
            p['area_per_sample'][:N_PER] *= 0.5
        an = StocsyAnalysis(ppm, X, peaks, info, noise_std=1.0)
        assert not any('PQN factors differ' in n for n in an.notes())


class TestPartnerColumns:

    def test_a_group_effect_is_between_and_the_molecule_is_within(self):
        an = _an()
        by = {d['rank']: d for d in an.partners(1)}
        assert by[2]['r'] > 0.9 and by[2]['r_within'] > 0.8       # same molecule
        assert by[2]['ratio'] == pytest.approx(0.5, rel=0.1)
        assert by[2]['ratio_sd'] < 0.12
        assert np.isfinite(by[2]['ratio_noise_sd'])
        pool = [by[k] for k in (3, 4, 5)]
        assert all(abs(d['r_between']) < 0.99 for d in pool)

    def test_a_pool_member_shares_a_per_sample_factor_not_a_ratio(self):
        by = {d['rank']: d for d in _an().partners(3)}
        assert by[4]['r_within'] > 0.8
        assert by[4]['ratio_sd'] > 0.0

    def test_export_carries_the_new_columns(self):
        rows = _an().export_rows(1)
        assert {'r_within_groups', 'r_between_groups', 'ratio_to_seed',
                'ratio_sd_log', 'ratio_noise_sd'} <= set(rows[0])


class TestAreaNoise:

    def test_kappa_is_read_from_the_blanks_when_there_are_enough(self):
        ppm, X, peaks, info = _study(blanks=5)
        rng = np.random.default_rng(0)
        for p in peaks:                                 # blanks that hold noise only
            a = np.asarray(p['area_per_sample'], float)
            a[N:] = rng.normal(0, 30.0, 5)
            p['area_per_sample'] = a
        an = StocsyAnalysis(ppm, X, peaks, info, noise_std=1.0)
        assert 1.0 <= an.kappa <= 6.0 and an.kappa != 2.2

    def test_without_blanks_the_default_kappa_is_used(self):
        ppm, X, peaks, info = _study(blanks=0)
        assert StocsyAnalysis(ppm, X, peaks, info, noise_std=1.0).kappa == 2.2

    def test_without_a_noise_level_it_is_estimated_from_the_spectrum(self):
        ppm, X, peaks, info = _study()
        an = StocsyAnalysis(ppm, X, peaks, info)
        assert np.all(an.area_sigma > 0) and len(an.area_sigma) == len(an.ranks)
