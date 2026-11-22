# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The resolution plot (RP-Method): the spectrum less what was resolved of
it, the residual's missing and over-claimed signals, and the candidates for
a missing one."""
import numpy as np
import pytest

from processing import resolution_plot as rp

PPM = np.linspace(3.0, 1.0, 4001)          # stored decreasing, as Bruker does
SIGMA = 1.0


def _lor(x, c, w=0.003):
    return w ** 2 / ((x - c) ** 2 + w ** 2)


def _fit_row(mid, centre, amp, base=0.0):
    x = np.linspace(centre - 0.1, centre + 0.1, 801)
    rec = {'ppm': x, 'focus': (centre, centre), 'amp_by_group': {'A': amp},
           'unit_by_group': {'A': _lor(x, centre)}}
    if base:
        rec['baseline_by_group'] = {'A': np.full_like(x, base)}
    return {'hmdb_id': mid, 'metabolite': mid, '_joint_fit': True, '_fit_curve': rec}


@pytest.fixture
def study():
    # a fitted peak at 2.0 explained, a peak at 1.5 nobody fitted, and a fit
    # at 2.5 claiming twice the peak that is there
    trace = 100 * _lor(PPM, 2.0) + 60 * _lor(PPM, 1.5) + 40 * _lor(PPM, 2.5)
    rows = [_fit_row('GOOD', 2.0, 100.0), _fit_row('OVER', 2.5, 80.0),
            {'hmdb_id': 'NQ', 'metabolite': 'NQ', '_not_quantified': True,
             '_rejection': 'fit_misplaced'}]
    return trace, rows


def test_the_residual_is_the_spectrum_less_what_was_resolved(study):
    trace, rows = study
    out = rp.resolve(rows, PPM, trace, 'A', lambda a: 1.0)
    assert {c['id'] for c in out['curves']} == {'GOOD', 'OVER'}
    i2 = np.argmin(abs(PPM - 2.0))
    assert abs(out['residual'][i2]) < 1.0
    i15 = np.argmin(abs(PPM - 1.5))
    assert out['residual'][i15] == pytest.approx(60.0, rel=0.02)


def test_a_fit_baseline_is_taken_off_the_residual():
    trace = 100 * _lor(PPM, 2.0) + 5.0 * ((PPM > 1.9) & (PPM < 2.1))
    out = rp.resolve([_fit_row('X', 2.0, 100.0, base=5.0)], PPM, trace, 'A', lambda a: 1.0)
    inside = (PPM > 1.92) & (PPM < 2.08)
    assert np.all(abs(out['residual'][inside]) < 0.5)


def test_regions_say_what_is_missing_and_what_is_over_claimed(study):
    trace, rows = study
    out = rp.resolve(rows, PPM, trace, 'A', lambda a: 1.0)
    regs = rp.residual_regions(PPM, trace, out['residual'], SIGMA, min_snr=5.0)
    kinds = {(g['kind'], round(g['apex_ppm'], 2)) for g in regs}
    assert kinds == {('missing', 1.5), ('over', 2.5)}
    miss = next(g for g in regs if g['kind'] == 'missing')
    assert miss['snr'] == pytest.approx(60.0, rel=0.02)
    assert miss['share'] == pytest.approx(1.0, abs=0.01)
    assert regs[0]['snr'] >= regs[-1]['snr']


def test_an_excluded_region_and_the_noise_floor_hide_a_residual(study):
    trace, rows = study
    res = rp.resolve(rows, PPM, trace, 'A', lambda a: 1.0)['residual']
    regs = rp.residual_regions(PPM, trace, res, SIGMA, min_snr=5.0, excluded=[(1.4, 1.6)])
    assert all(g['kind'] != 'missing' for g in regs)
    assert rp.residual_regions(PPM, trace, res, 100.0, min_snr=5.0) == []


def test_a_row_without_a_number_resolves_nothing(study):
    trace, rows = study
    curves, _b = rp.analyte_curves(rows, PPM, trace, 'A', lambda a: 1.0)
    assert 'NQ' not in {c['id'] for c in curves}


def test_the_resolved_share_counts_only_what_the_spectrum_holds(study):
    trace, rows = study
    out = rp.resolve(rows, PPM, trace, 'A', lambda a: 1.0)
    share = rp.resolved_share(PPM, trace, out['resolved'], out['baseline'])
    # 100 + 40 of 200 resolved (the over-claim cannot count twice)
    assert share == pytest.approx(140 / 200, abs=0.02)


def test_candidates_are_the_annotations_signals_near_the_residual(study):
    _trace, rows = study
    region = {'lo': 1.49, 'hi': 1.51, 'apex_ppm': 1.5}
    mets = [
        {'id': 'MISS', 'name': 'Missing one', 'composite_score': 0.7,
         'db_peaks': [{'ppm_center': 1.503, 'ppm_start': 1.503, 'ppm_end': 1.503,
                       'type': 'd', 'protons': 3}],
         'matched_peaks': [{'db_ppm': 1.503, 'peak_ppm': 1.5, 'peak_rank': 7,
                            'annotation_quality': None}]},
        {'id': 'NQ', 'name': 'NQ', 'db_peaks': [{'ppm_center': 1.52}], 'matched_peaks': []},
        {'id': 'GOOD', 'name': 'GOOD', 'db_peaks': [{'ppm_center': 1.508}, {'ppm_center': 2.0}],
         'matched_peaks': []},
        {'id': 'FAR', 'name': 'far', 'db_peaks': [{'ppm_center': 1.9}], 'matched_peaks': []},
    ]
    cands = rp.candidates(region, mets, rows, tolerance=0.05, ppm=PPM)
    assert [c['met_id'] for c in cands] == ['MISS', 'NQ', 'GOOD']
    miss = cands[0]
    assert miss['mp_index'] == 0 and miss['matched_ppm'] == 1.5
    assert miss['status'] == 'not in the result'
    assert miss['delta_ppm'] == pytest.approx(0.003)
    assert cands[1]['status'] == 'no number' and cands[1]['detail'] == 'fit_misplaced'
    assert cands[1]['mp_index'] is None
    assert cands[2]['status'] == 'quantified'


def test_a_run_holding_two_signals_is_two_regions():
    trace = 50 * _lor(PPM, 1.50, 0.004) + 50 * _lor(PPM, 1.52, 0.004)
    regs = rp.residual_regions(PPM, trace, trace, SIGMA, min_snr=5.0)
    assert sorted(round(g['apex_ppm'], 2) for g in regs) == [1.5, 1.52]
    a, b = sorted(regs, key=lambda g: g['apex_ppm'])
    assert a['hi'] < b['lo']


def test_a_residual_on_an_integrated_analytes_other_signal_is_named(study):
    _trace, rows = study
    integ = {'hmdb_id': 'INT', 'metabolite': 'Integrated', '_integ_mode': 'Multiplet',
             '_multiplet_window': (1.99, 2.01)}
    rows = rows + [integ]
    mets = [{'id': 'INT', 'name': 'Integrated',
             'matched_peaks': [{'peak_ppm': 2.0}, {'peak_ppm': 1.501}]},
            {'id': 'GOOD', 'name': 'Fitted', 'matched_peaks': [{'peak_ppm': 1.502}]},
            {'id': 'NQ', 'name': 'NQ', 'matched_peaks': [{'peak_ppm': 1.5}]}]
    region = {'lo': 1.49, 'hi': 1.51, 'apex_ppm': 1.5}
    # the fitted one's record covers 1.9-2.1 only: 1.502 is another signal of
    # it the plot does not model; the one without a number has none
    assert rp.quantified_signals_in(region, mets, rows, PPM) == ['Integrated', 'Fitted']
    near = {'lo': 2.05, 'hi': 2.06, 'apex_ppm': 2.055}
    mets_near = [{'id': 'GOOD', 'name': 'Fitted', 'matched_peaks': [{'peak_ppm': 2.055}]}]
    assert rp.quantified_signals_in(near, mets_near, rows, PPM) == []
    # on the window its own number came from, a residual is that number short
    own = {'lo': 1.995, 'hi': 2.005, 'apex_ppm': 2.0}
    assert rp.quantified_signals_in(own, mets, rows, PPM) == []


def test_a_candidate_carries_the_annotations_j_and_its_fingerprint(study):
    _trace, rows = study
    prof = '{"start": -10.0, "step": 0.5, "h": [0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0]}'
    mets = [{'id': 'T', 'name': 'Triplet',
             'db_peaks': [{'ppm_center': 1.50, 'type': 't', 'protons': 2, 'j_coupling': '7.2',
                           'pattern_profile': prof,
                           'pattern_lines': '[[-7.2, 0.5], [0.0, 1.0], [7.2, 0.5]]'}],
             'matched_peaks': [{'db_ppm': 1.50, 'peak_ppm': 1.501, 'ppm_center': 1.502,
                                'observed_j_hz': 7.4, 'expected_j_hz': 7.2, 'delta_j_hz': 0.2,
                                '_n_found_lines': 3, '_n_exp_lines': 3,
                                'predicted_lines': [{'ppm': 1.49, 'rel_intensity': 0.5}]}]}]
    c = rp.candidates({'lo': 1.49, 'hi': 1.51, 'apex_ppm': 1.5}, mets, rows, 0.05, PPM)[0]
    assert (c['j_ref'], c['j_obs'], c['j_exp'], c['delta_j'], c['lines']) == ('7.2', 7.4, 7.2, 0.2, (3, 3))
    assert c['centre'] == 1.502 and c['predicted'][0]['ppm'] == 1.49
    x, h, kind = rp.fingerprint(c, 600.0)
    assert kind == 'profile' and h.max() == 1.0
    assert x[0] == pytest.approx(1.502 - 10.0 / 600.0) and np.all(np.diff(x) > 0)
    c['profile'] = None
    x, h, kind = rp.fingerprint(c, 600.0)
    assert kind == 'lines' and x[1] == pytest.approx(1.502)
    assert x[2] - x[1] == pytest.approx(7.2 / 600.0)
    assert rp.fingerprint(c, None) is None


def test_a_candidate_is_fitted_into_the_residual_at_the_studys_width():
    # a residual triplet (J 7 Hz, 2 Hz wide at 600 MHz) and a candidate whose
    # reference lists that triplet: the copy fits it almost whole
    f, lw, c0 = 600.0, 2.0, 1.5
    lines = [(-7.0, 0.5), (0.0, 1.0), (7.0, 0.5)]
    from processing.pattern_shape import lines_model
    res = 80.0 * lines_model(PPM, lines, c0, lw, f)
    # placed 4 mppm off by the annotation: slid back onto the residual
    cand = {'centre': c0 + 0.004, 'pattern_lines': '[[-7.0, 0.5], [0.0, 1.0], [7.0, 0.5]]',
            'profile': None, 'field_mhz': f, 'study_lw_hz': lw, 'tolerance_ppm': 0.01}
    region = {'lo': 1.47, 'hi': 1.53}
    fit = rp.fit_in_residual(cand, PPM, res, SIGMA, region)
    assert fit['origin'] == pytest.approx(c0, abs=0.0005)
    assert fit['shift_ppm'] == pytest.approx(-0.004, abs=0.0005)
    assert fit['y'].max() == pytest.approx(80.0, rel=0.05)
    assert fit['share'] > 0.9
    # a doublet laid on the triplet explains clearly less of it
    cand2 = dict(cand, pattern_lines='[[-3.5, 1.0], [3.5, 1.0]]')
    assert rp.fit_in_residual(cand2, PPM, res, SIGMA, region)['share'] < fit['share'] - 0.2
    assert rp.fit_in_residual(dict(cand, study_lw_hz=None), PPM, res, SIGMA, region) is None
    # without the annotation's tolerance the caller's is used
    assert rp.fit_in_residual(dict(cand, tolerance_ppm=None), PPM, res, SIGMA, region) is None
    assert rp.fit_in_residual(dict(cand, tolerance_ppm=None), PPM, res, SIGMA, region,
                              tolerance_ppm=0.01) is not None


def test_an_over_claim_names_who_claims_it(study):
    trace, rows = study
    out = rp.resolve(rows, PPM, trace, 'A', lambda a: 1.0)
    over = next(g for g in rp.residual_regions(PPM, trace, out['residual'], SIGMA, min_snr=5.0)
                if g['kind'] == 'over')
    assert rp.claimed_by(over, out['curves'], SIGMA) == ['OVER']


def test_a_quantified_candidates_number_predicts_its_other_signal():
    from processing.pattern_shape import lines_model
    f, lw, c0 = 600.0, 2.0, 1.5
    lines = [(-7.0, 0.5), (0.0, 1.0), (7.0, 0.5)]
    shape = lines_model(PPM, lines, c0, lw, f)
    o = np.argsort(PPM)
    unit = np.trapezoid(shape[o], PPM[o])
    res = 300.0 * 2 * shape / unit             # 2 H at 300 per proton
    row = {'hmdb_id': 'Q', '_by_cascade': True,
           '_fit_curve': {'ppm': PPM, 'focus': (0.9, 0.9), 'amp_by_group': {'A': 300.0},
                          'unit_by_group': {'A': np.zeros_like(PPM)}}}
    cand = {'centre': c0, 'pattern_lines': '[[-7.0, 0.5], [0.0, 1.0], [7.0, 0.5]]',
            'profile': None, 'field_mhz': f, 'study_lw_hz': lw, 'protons': 2}
    exp = rp.expected_from_number(cand, row, PPM, res, 'A', {'lo': 1.45, 'hi': 1.55})
    assert exp['ratio'] == pytest.approx(1.0, rel=0.05)
    # its number holding half: half the residual
    row['_fit_curve']['amp_by_group']['A'] = 150.0
    assert rp.expected_from_number(cand, row, PPM, res, 'A', {'lo': 1.45, 'hi': 1.55})['ratio'] \
        == pytest.approx(0.5, rel=0.05)
    # a number that is not a fit's says nothing here
    assert rp.expected_from_number(cand, {'hmdb_id': 'Q'}, PPM, res, 'A', {'lo': 1.45, 'hi': 1.55}) is None


def _confirmed_with_a_second_signal(amp, conf=2):
    """A confirmed analyte fitted at 2.0 (its record reaches 1.9-2.1) whose
    reference lists a second signal at 1.5, 2 H, a triplet."""
    row = _fit_row('CONF', 2.0, 100.0)
    row['_confidence'] = conf
    row['_fit_curve']['amp_by_group'] = {'A': amp}
    met = {'id': 'CONF', 'name': 'CONF',
           'db_peaks': [{'ppm_center': 2.0, 'protons': 1, 'type': 's'},
                        {'ppm_center': 1.5, 'protons': 2, 'type': 't',
                         'pattern_lines': '[[-7.0, 0.5], [0.0, 1.0], [7.0, 0.5]]'}],
            'matched_peaks': [{'db_ppm': 1.5, 'peak_ppm': 1.5, 'study_linewidth_hz': 2.0,
                               'pattern_tolerance_ppm': 0.01, 'pattern_field_mhz': 600.0,
                               'type': 't', '_confirmed': True, '_n_found_lines': 3,
                               '_n_exp_lines': 3, 'delta_j_hz': 0.1, 'ppm_diff': 0.002}]}
    return row, met


def test_a_confirmed_analytes_other_signal_is_taken_off_the_residual():
    from processing.pattern_shape import lines_model
    shape = lines_model(PPM, [(-7.0, 0.5), (0.0, 1.0), (7.0, 0.5)], 1.5, 2.0, 600.0)
    o = np.argsort(PPM)
    unit = np.trapezoid(shape[o], PPM[o])
    res = 50.0 * 2 * shape / unit                     # 2 H at 50 per proton
    row, met = _confirmed_with_a_second_signal(50.0)
    sec = rp.secondary_signals([row], [met], PPM, res, 'A', SIGMA)
    assert [s['signal_ppm'] for s in sec] == [1.5]
    assert sec[0]['taken'] == pytest.approx(sec[0]['predicted'], rel=0.1)
    left = res - rp.on_axis(sec[0]['x'], sec[0]['y'], PPM)
    assert np.max(np.abs(left)) < 0.1 * res.max()


def test_a_signal_is_taken_only_up_to_what_the_residual_holds():
    from processing.pattern_shape import lines_model
    shape = lines_model(PPM, [(-7.0, 0.5), (0.0, 1.0), (7.0, 0.5)], 1.5, 2.0, 600.0)
    res = 30.0 * shape
    row, met = _confirmed_with_a_second_signal(500.0)     # predicts far more than is there
    sec = rp.secondary_signals([row], [met], PPM, res, 'A', SIGMA)
    left = res - rp.on_axis(sec[0]['x'], sec[0]['y'], PPM)
    assert sec[0]['taken'] < 0.5 * sec[0]['predicted']
    assert left.min() >= -2.0 * SIGMA - 1e-9


def test_one_dot_analytes_are_not_references():
    row, met = _confirmed_with_a_second_signal(50.0, conf=1)
    assert rp.secondary_signals([row], [met], PPM, np.ones_like(PPM), 'A', SIGMA) == []


def test_a_singlet_is_fitted_and_subtracted_like_a_multiplet():
    from processing.pattern_shape import lines_model
    res = 40.0 * lines_model(PPM, [(0.0, 1.0)], 1.5, 2.0, 600.0)
    cand = {'centre': 1.502, 'pattern_lines': '[[0.0, 1.0]]', 'profile': None,
            'field_mhz': 600.0, 'study_lw_hz': 2.0, 'tolerance_ppm': 0.01}
    fit = rp.fit_in_residual(cand, PPM, res, SIGMA, {'lo': 1.49, 'hi': 1.51})
    assert fit['origin'] == pytest.approx(1.5, abs=0.0005)
    assert fit['share'] > 0.9


def test_an_unconfirmed_multiplet_is_not_taken():
    from processing.pattern_shape import lines_model
    res = 30.0 * lines_model(PPM, [(-7.0, 0.5), (0.0, 1.0), (7.0, 0.5)], 1.5, 2.0, 600.0)
    row, met = _confirmed_with_a_second_signal(50.0)
    met['matched_peaks'][0]['_confirmed'] = False
    assert rp.secondary_signals([row], [met], PPM, res, 'A', SIGMA) == []
    met['matched_peaks'][0]['_confirmed'] = True
    sec = rp.secondary_signals([row], [met], PPM, res, 'A', SIGMA)
    assert sec and sec[0]['why'].startswith('t confirmed: 3/3 lines')


def test_a_singlet_is_taken_only_when_stocsy_linked_it():
    from processing.pattern_shape import lines_model
    res = 30.0 * lines_model(PPM, [(0.0, 1.0)], 1.5, 2.0, 600.0)
    row, met = _confirmed_with_a_second_signal(50.0)
    met['db_peaks'][1] = {'ppm_center': 1.5, 'protons': 3, 'type': 's',
                          'pattern_lines': '[[0.0, 1.0]]'}
    met['matched_peaks'][0].update(type='s', _confirmed=False)
    assert rp.secondary_signals([row], [met], PPM, res, 'A', SIGMA) == []
    row['_signal_links'] = [{'db_ppm': 1.5, 'type': 's', 'rank': 3, 'r': 0.94, 'gap': 0.05}]
    sec = rp.secondary_signals([row], [met], PPM, res, 'A', SIGMA)
    assert [x['why'] for x in sec] == ['singlet linked by STOCSY (r 0.94)']



def test_resolution_is_resolve_with_the_share_and_the_regions(study):
    """What the window and the report both show: one call."""
    trace, rows = study
    base = rp.resolve(rows, PPM, trace, 'A', lambda a: 1.0)
    out = rp.resolution(rows, PPM, trace, 'A', lambda a: 1.0, sigma=SIGMA, min_snr=5.0)
    assert np.allclose(out['residual'], base['residual'])
    assert out['secondary'] == []
    assert out['share'] == pytest.approx(
        rp.resolved_share(PPM, trace, base['resolved'], base['baseline']))
    assert {(g['kind'], round(g['apex_ppm'], 2)) for g in out['regions']} == {
        (g['kind'], round(g['apex_ppm'], 2))
        for g in rp.residual_regions(PPM, trace, base['residual'], SIGMA, min_snr=5.0)}
    # without a noise level there are no regions to find, and nothing to subtract
    bare = rp.resolution(rows, PPM, trace, 'A', lambda a: 1.0)
    assert bare['regions'] == [] and bare['secondary'] == []


def test_the_studys_linewidth_is_the_annotations():
    mets = [{'matched_peaks': [{}]}, {'matched_peaks': [{'study_linewidth_hz': 1.8}]}]
    assert rp.study_linewidth(mets) == 1.8
    assert rp.study_linewidth([]) is None
