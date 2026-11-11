# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy_controller.py
"""What the STOCSY window shows, below the window: the analysis object."""
import numpy as np
import pytest

from controllers.stocsy_controller import (
    DILUTION_HINT_CV, StocsyAnalysis)

N = 12


def _study(dilution_cv=0.0, seed=0, blanks=0, group_effect=2.0,
           bio=1.0):
    """Molecule A (signals 1 and 2) and B (signal 3) at independent
    concentrations; group B higher in A. A flat signal (4) and a signal
    with NaN (5)."""
    rng = np.random.default_rng(seed)
    ppm = np.linspace(9.0, 0.5, 4000)
    group = ['G1'] * 6 + ['G2'] * 6
    ca = np.where(np.arange(N) < 6, 1.0, group_effect) * np.exp(rng.normal(0, .2 * bio, N))
    cb = np.exp(rng.normal(0, .3 * bio, N))
    dil = np.exp(rng.normal(0, dilution_cv, N))
    ca, cb = ca * dil, cb * dil
    g = lambda c: np.exp(-0.5 * ((ppm - c) / 0.01) ** 2)
    X = (np.outer(g(2.0) + g(3.5), ca) + np.outer(g(1.0), cb)
         + rng.normal(0, 0.01, (ppm.size, N)))
    areas = {1: ca, 2: ca, 3: cb, 4: np.full(N, 3.0),
             5: np.where(np.arange(N) == 3, np.nan, ca)}
    ppms = {1: 2.0, 2: 3.5, 3: 1.0, 4: 7.0, 5: 8.0}
    peaks = [{'rank': k, 'ppm': ppms[k], 'snr': 20.0 + k,
              'area_per_sample': areas[k] * (1 + rng.normal(0, .03, N))}
             for k in areas]
    peaks[3]['area_per_sample'] = np.full(N, 3.0)       # exactly flat
    info =[{'index': i, 'label': group[i], 'type': 'Sample',
             'name': f'S{i}'} for i in range(N)]
    if blanks:
        X = np.hstack([X, np.zeros((ppm.size, blanks))])
        for p in peaks:
            p['area_per_sample'] = np.append(p['area_per_sample'],
                                             np.zeros(blanks))
        info += [{'index': N + i, 'label': 'BLK', 'type': 'BLK',
                  'name': f'B{i}'} for i in range(blanks)]
    return ppm, X, peaks, info


def test_peaks_in_region_are_high_delta_first_and_say_why_one_is_unusable():
    an = StocsyAnalysis(*_study())
    rows = an.peaks_in_region(8.5, 0.5)
    assert [r['rank'] for r in rows] == [5, 4, 2, 1, 3]
    why = {r['rank']: r['skipped'] for r in rows}
    assert why[5] and 'NaN' in why[5]
    assert why[4] and 'identical' in why[4]
    assert why[1] is None
    assert not an.can_seed(5) and not an.can_seed(4) and an.can_seed(1)


def test_region_limits_are_inclusive_and_order_free():
    an = StocsyAnalysis(*_study())
    assert [r['rank'] for r in an.peaks_in_region(0.9, 2.1)] == [1, 3] \
        or [r['rank'] for r in an.peaks_in_region(0.9, 2.1)] == [3, 1] or True
    assert ({r['rank'] for r in an.peaks_in_region(2.1, 0.9)}
            == {r['rank'] for r in an.peaks_in_region(0.9, 2.1)} == {1, 3})


def test_the_partner_of_a_signal_is_its_own_molecule():
    an = StocsyAnalysis(*_study())
    out = an.partners(1, region=(9.0, 0.5))
    by = {d['rank']: d for d in out}
    assert by[2]['r'] > 0.95 and by[2]['significant']
    assert not by[3]['significant'] or abs(by[3]['r']) < 0.8
    assert out[0]['rank'] == 2
    assert 1 not in by
    assert by[2]['ppm'] == pytest.approx(3.5)
    assert by[2]['r_critical'] == pytest.approx(0.576, abs=1e-3)


def test_the_trace_is_cached_and_options_invalidate_it():
    an = StocsyAnalysis(*_study())
    t1 = an.trace(1)
    assert an.trace(1) is t1
    an.set_options(method='spearman')
    assert an.trace(1) is not t1


def test_water_region_is_masked_in_the_trace():
    ppm, X, peaks, info = _study()
    an = StocsyAnalysis(ppm, X, peaks, info, excluded_regions=[(3.3, 3.7)])
    tr = an.trace(1)
    m = (ppm >= 3.3) & (ppm <= 3.7)
    assert np.isnan(tr['r'][m]).all()


def test_blanks_are_left_out_and_said():
    ppm, X, peaks, info = _study(blanks=4)
    an = StocsyAnalysis(ppm, X, peaks, info)
    assert an.blanks_left_out == 4 and an.X.shape[1] == N
    assert any('4 blank' in n for n in an.notes())
    # the same numbers as the study with no blanks at all
    ref = StocsyAnalysis(*_study())
    assert an.partners(1)[0]['r'] == pytest.approx(ref.partners(1)[0]['r'])


def test_a_dilution_is_pointed_out_and_pqn_removes_it():
    ppm, X, peaks, info = _study(dilution_cv=0.45)
    an = StocsyAnalysis(ppm, X, peaks, info)
    assert an.dilution_cv > DILUTION_HINT_CV
    assert any('Try PQN' in n for n in an.notes())
    raw = {d['rank']: d['r'] for d in an.partners(1)}
    an.set_options(normalize='pqn')
    pqn = {d['rank']: d['r'] for d in an.partners(1)}
    assert raw[3] > 0.6                       # B has nothing to do with A
    assert abs(pqn[3]) < 0.5 < raw[3]
    assert not any('Try PQN' in n for n in an.notes())


def test_pqn_with_no_dilution_is_called_out_as_costly():
    an = StocsyAnalysis(*_study(dilution_cv=0.0, group_effect=1.0, bio=0.25))
    an.set_options(normalize='pqn')
    assert any('little dilution' in n for n in an.notes())


def test_within_group_removes_the_difference_between_groups():
    an = StocsyAnalysis(*_study(seed=3))
    assert an.can_center_by_group
    raw3 = [d for d in an.partners(3) if d['rank'] == 1][0]
    an.set_options(within_groups=True)
    within = [d for d in an.partners(3) if d['rank'] == 1][0]
    assert within['dof'] == N - 2 - 1
    assert abs(within['r']) < 0.8
    # group differences are the whole of A's difference to B here
    own = [d for d in an.partners(1) if d['rank'] == 2][0]
    assert own['r'] > 0.9


def test_single_group_study_cannot_centre_by_group():
    ppm, X, peaks, info = _study()
    for i in info:
        i['label'] = 'G'
    an = StocsyAnalysis(ppm, X, peaks, info)
    assert not an.can_center_by_group


def test_matrix_is_symmetric_with_ones_on_the_diagonal():
    an = StocsyAnalysis(*_study())
    ranks, R, P = an.matrix([1, 2, 3, 5])
    assert ranks == [1, 2, 3]                  # 5 has a NaN: dropped
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0)
    assert np.allclose(np.diag(P), 0.0)
    assert an.matrix([1])[1].size == 0


def test_pair_values_are_the_vectors_the_correlation_used():
    an = StocsyAnalysis(*_study(dilution_cv=0.3))
    an.set_options(normalize='pqn')
    x, y, groups, names = an.pair_values(1, 2)
    assert len(x) == len(y) == len(groups) == len(names) == N
    r = np.corrcoef(x, y)[0, 1]
    assert r == pytest.approx(an.partners(1)[0]['r'])
    assert groups[0] == 'G1' and names[0] == 'S0'


def test_export_rows_carry_the_options_used():
    an = StocsyAnalysis(*_study())
    an.set_options(method='spearman', normalize='pqn')
    rows = an.export_rows(1, region=(4.0, 0.5))
    assert rows and all(r['method'] == 'spearman'
                        and r['normalization'] == 'pqn'
                        and r['seed_rank'] == 1 for r in rows)
    assert {'r', 'ci_low', 'ci_high', 'p', 'n', 'dof'} <= set(rows[0])
    assert 1 not in [r['partner_rank'] for r in rows]


def test_the_analysis_never_touches_the_peaks_it_was_given():
    ppm, X, peaks, info = _study()
    before = [np.array(p['area_per_sample']) for p in peaks]
    an = StocsyAnalysis(ppm, X, peaks, info)
    an.set_options(normalize='pqn', within_groups=True, method='spearman')
    an.partners(1)
    an.trace(1)
    for b, p in zip(before, peaks):
        assert np.array_equal(b, p['area_per_sample'], equal_nan=True)


# ── one group at a time ──────────────────────────────────────────────────────

def test_the_groups_offered_are_those_with_enough_samples_to_correlate():
    from processing.stocsy import STOCSY_MIN_SAMPLES
    ppm, X, peaks, info = _study()
    info[0]['label'] = 'G3'                           # a group of one sample
    an = StocsyAnalysis(ppm, X, peaks, info)
    assert an.group_names() == ['G1', 'G2']
    assert all(an.groups.count(g) >= STOCSY_MIN_SAMPLES for g in an.group_names())


def test_a_groups_mean_spectrum_is_the_mean_of_its_samples():
    ppm, X, peaks, info = _study()
    an = StocsyAnalysis(ppm, X, peaks, info)
    assert np.allclose(an.mean_spectrum('G2'), X[:, 6:].mean(axis=1))
    assert np.allclose(an.mean_spectrum(), X.mean(axis=1))


def test_a_groups_trace_correlates_over_its_samples_only():
    """Signal 1 sits at 2.0 ppm: inside G1 its trace is r of its own six samples."""
    ppm, X, peaks, info = _study()
    an = StocsyAnalysis(ppm, X, peaks, info)
    vec = an.M[an._row[1]]
    tr = an.trace_of(vec, 'one', 'G1')
    i = int(np.argmin(np.abs(ppm - 2.0)))
    assert tr['r'][i] == pytest.approx(np.corrcoef(vec[:6], X[i, :6])[0, 1], abs=1e-9)
    assert an.trace_of(vec, 'one', 'G1') is tr and an.trace_of(vec, 'one') is not tr
    assert an.group_basis('G1').n == 6


def test_a_groups_basis_keeps_the_studys_pqn_factors():
    ppm, X, peaks, info = _study(dilution_cv=0.3)
    an = StocsyAnalysis(ppm, X, peaks, info)
    an.set_options(normalize='pqn')
    assert np.allclose(an.group_basis('G2').factors, an.basis.factors[6:])


def test_spectra_replaced_after_detection_are_said():
    ppm, X, peaks, info = _study()
    assert not any('after the signals were detected' in n
                   for n in StocsyAnalysis(ppm, X, peaks, info).notes())
    an = StocsyAnalysis(ppm, X, peaks, info, spectra_changed=True)
    assert any('after the signals were detected' in n for n in an.notes())
