# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_signal_links.py
"""Whether the rest of an analyte's listing moves with what was measured for it,
on spectra whose answer is known: a doublet and a singlet of one molecule beside
another compound's singlet standing nearer the listing."""
import contextlib
import copy
import io

import numpy as np
import pytest

from processing import signal_links as sl
from processing.annotation_quality import promote_singlets
from processing.annotation_scoring import SCORING_DEFAULTS
from processing.peak_detection import as_app_peak, detect_peaks

N = 16
PPM = np.arange(0.5, 4.5, 0.0005)
FWHM = 0.0044
NOISE = 2.0
INFO = [{'index': i, 'label': 'A' if i < N // 2 else 'B', 'type': 'Sample'} for i in range(N)]
TOL = SCORING_DEFAULTS['dppm_gold']


def _line(x0, area):
    hw = FWHM / 2
    return area * (hw / np.pi) / ((PPM - x0) ** 2 + hw ** 2)


def _background(rng):
    """Noise and the rest of an extract: compounds that vary on their own, the
    majority PQN reads a sample's dilution over."""
    X = rng.normal(0, NOISE, (PPM.size, N))
    for x0 in np.linspace(0.7, 1.8, 24):
        amounts = 30 * np.exp(rng.normal(0, 0.5, N))
        for j in range(N):
            X[:, j] += _line(x0, amounts[j])
    return X


def _detect(X):
    with contextlib.redirect_stdout(io.StringIO()):
        det = detect_peaks(PPM, X, noise_std=NOISE, min_snr=30.0, min_distance=0.003,
                           baseline_correct=False, global_align=False, samples_info=INFO,
                           per_sample_detection=True, detect_shoulders=False, min_group_presence=3,
                           excluded_regions=[])
    return [as_app_peak(p) for p in det]


def _study(seed=0, own=None, other=None, singlet=True):
    """Doublet (2 lines, 3 H) at 2.000 and singlet (3 H) at 3.020 of one
    molecule, and a singlet of another compound at 3.002."""
    rng = np.random.default_rng(seed)
    own = 40 * np.exp(rng.normal(0, 0.5, N)) if own is None else own
    other = 30 * np.exp(rng.normal(0, 0.5, N)) if other is None else other
    X = _background(rng)
    for j in range(N):
        X[:, j] += _line(1.994, 1.5 * own[j]) + _line(2.006, 1.5 * own[j]) + _line(3.002, 3 * other[j])
        if singlet:
            X[:, j] += _line(3.020, 3 * own[j])
    return X, _detect(X), own


def _rank(peaks, x):
    return min(peaks, key=lambda p: abs(p['ppm'] - x))['rank']


def _peak(peaks, rank):
    return next(p for p in peaks if p['rank'] == rank)


def _met(peaks, at=3.020, **singlet):
    """The analyte as annotated: its doublet and its singlet, listed at 3.015."""
    r = _rank(peaks, at)
    s = {'db_ppm': 3.015, 'peak_ppm': _peak(peaks, r)['ppm'], 'peak_rank': r, 'type': 's', 'protons': 3,
         'ppm_diff': 0.005, 'annotation_quality': None, 'predicted_lines': [{'ppm': _peak(peaks, r)['ppm']}]}
    s.update(singlet)
    d = {'db_ppm': 2.0, 'peak_ppm': 2.0, 'peak_rank': _rank(peaks, 1.994), 'type': 'd', 'protons': 3,
         'annotation_quality': 'good', 'delta_j_hz': 0.1, 'ppm_diff': 0.001,
         'predicted_lines': [{'ppm': 1.994}, {'ppm': 2.006}]}
    return {'id': 'M1', 'name': 'one', 'matched_peaks': [d, s]}


def _row(areas, **kw):
    """Measured on the doublet, the analyte's 3 protons at two dots."""
    r = {'hmdb_id': 'M1', 'metabolite': 'one', '_confidence': 2, '_identity_level': 2,
         '_areas_per_sample': list(3 * np.asarray(areas, float)), '_nh': 3.0,
         '_multiplet_window': [1.985, 2.015], '_identity_why': 'because'}
    r.update(kw)
    return r


def _run(X, peaks, met, row, worked=None, **kw):
    rows = [row]
    sl.apply_links(rows, [met], [met if worked is None else worked], peaks, PPM, X, kw.pop('info', INFO),
                   tolerance=TOL, noise_sigma=kw.pop('noise', NOISE), **kw)
    return row


def _singlet(met):
    return met['matched_peaks'][1]


def test_a_row_whose_singlet_moves_with_its_amount_reaches_three_dots():
    X, peaks, own = _study()
    met = _met(peaks)
    row = _run(X, peaks, met, _row(own))
    assert row['_confidence'] == 3 and row['_identity_level'] == 3
    assert row['status'] == '●●●', 'the table and the report print this field'
    link = row[sl.LINKS_KEY][0]
    assert link['rank'] == _rank(peaks, 3.020) and abs(link['gap']) <= sl.H_RATIO_GAP
    assert 'move with the amount' in row['_identity_why'] and row['_identity_why'].startswith('because')


def test_the_singlet_that_moves_with_the_amount_is_marked_good_on_its_peak():
    X, peaks, own = _study()
    met = _met(peaks)
    _run(X, peaks, met, _row(own))
    s = _singlet(met)
    assert s['annotation_quality'] == 'good' and s['peak_rank'] == _rank(peaks, 3.020)
    assert any('moves with the amount' in r for r in s['_status_reasons'])


def test_where_position_stands_elsewhere_the_row_rises_but_the_singlet_is_left_alone():
    """The peak that co-varies is not the one the position step stands on: over
    twelve seeds position was right more often."""
    X, peaks, own = _study()
    met = _met(peaks, at=3.002)
    before = copy.deepcopy(_singlet(met))
    row = _run(X, peaks, met, _row(own))
    assert row['_confidence'] == 3
    assert _singlet(met) == before


def test_the_annotation_is_moved_to_where_the_quantification_stands():
    """Annotation took the other compound's peak, the position step this one."""
    X, peaks, own = _study()
    met = _met(peaks, at=3.002)
    worked = _met(peaks)
    before = copy.deepcopy(_singlet(met))
    _run(X, peaks, met, _row(own), worked=worked)
    s = _singlet(met)
    assert s['peak_rank'] == _rank(peaks, 3.020) and s['annotation_quality'] == 'good'
    assert s['ppm_diff'] == pytest.approx(abs(_peak(peaks, s['peak_rank'])['ppm'] - 3.015))
    sl.forget_marks([met])
    assert _singlet(met) == before


def test_another_compounds_peak_does_not_link():
    X, peaks, own = _study(singlet=False)
    met = _met(peaks, at=3.002)
    before = copy.deepcopy(_singlet(met))
    row = _run(X, peaks, met, _row(own))
    assert sl.LINKS_KEY not in row and row['_confidence'] == 2
    assert _singlet(met) == before


def test_the_same_molecule_with_the_wrong_protons_does_not_link():
    X, peaks, own = _study()
    met = _met(peaks, protons=9)
    row = _run(X, peaks, met, _row(own))
    assert row['_confidence'] == 2 and _singlet(met)['annotation_quality'] is None


def test_the_signal_the_amount_was_measured_on_is_not_its_own_link():
    """Measured on the singlet, nothing else in the listing moves with it."""
    X, peaks, own = _study()
    met = _met(peaks)
    del met['matched_peaks'][0]
    row = _row(own, _multiplet_window=[3.010, 3.030])
    rows = [row]
    sl.apply_links(rows, [met], [met], peaks, PPM, X, INFO, tolerance=TOL, noise_sigma=NOISE)
    assert sl.LINKS_KEY not in row and row['_confidence'] == 2
    assert met['matched_peaks'][0]['annotation_quality'] is None


def test_the_singlet_measured_on_is_marked_when_another_signal_moves_with_it():
    X, peaks, own = _study()
    met = _met(peaks)
    row = _row(own, _multiplet_window=[3.010, 3.030])
    _run(X, peaks, met, row)
    assert row['_confidence'] == 3 and row[sl.LINKS_KEY][0]['type'] == 'd'
    assert _singlet(met)['annotation_quality'] == 'good'
    assert any('measured on' in r for r in _singlet(met)['_status_reasons'])


def test_a_difference_between_groups_is_not_co_variation():
    """Both compounds step between the groups and nothing else ties them."""
    rng = np.random.default_rng(3)
    step = np.array([30.0] * (N // 2) + [120.0] * (N // 2))
    X, peaks, _ = _study(seed=3, own=step * np.exp(rng.normal(0, 0.3, N)),
                         other=step * np.exp(rng.normal(0, 0.3, N)), singlet=False)
    met = _met(peaks, at=3.002)
    areas = step * np.exp(np.random.default_rng(4).normal(0, 0.3, N))
    row = _run(X, peaks, met, _row(areas))
    assert sl.LINKS_KEY not in row


def test_the_correlation_is_not_read_at_a_fixed_point():
    """A peak moving a fraction of its width between samples is still found."""
    rng = np.random.default_rng(5)
    own = 40 * np.exp(rng.normal(0, 0.5, N))
    other = 30 * np.exp(rng.normal(0, 0.5, N))
    X = _background(rng)
    for j in range(N):
        s = rng.normal(0, 0.0012)
        X[:, j] += (_line(1.994 + s, 1.5 * own[j]) + _line(2.006 + s, 1.5 * own[j])
                    + _line(3.002 + s, 3 * other[j]) + _line(3.020 + s, 3 * own[j]))
    peaks = _detect(X)
    met = _met(peaks)
    row = _run(X, peaks, met, _row(own))
    assert row[sl.LINKS_KEY][0]['rank'] == _rank(peaks, 3.020)


@pytest.mark.parametrize('change', [{'_shared_with': ['M2']}, {'_deconv': {'components': []}},
                                    {'_identity_level': 3}])
def test_two_dots_held_by_the_amount_stay_at_two(change):
    """A linked signal answers for the identity, not for a share of a window."""
    X, peaks, own = _study()
    row = _run(X, peaks, _met(peaks), _row(own, **change))
    assert row['_confidence'] == 2 and sl.LINKS_KEY in row


def test_a_shape_that_fell_short_stays_at_two_dots():
    X, peaks, own = _study()
    row = _run(X, peaks, _met(peaks), _row(own, _confidence_axes={'identity': 2, 'shape': 2}))
    assert row['_confidence'] == 2 and sl.LINKS_KEY in row


def test_co_variation_is_worth_one_dot():
    """A row the lift took from one dot to two does not go on to three."""
    X, peaks, own = _study()
    met = _met(peaks)
    row = _run(X, peaks, met, _row(own, _stocsy_lift={'pairs': 3, 'from': 1}))
    assert row['_confidence'] == 2 and sl.LINKS_KEY in row and '_signal_links_from' not in row
    assert _singlet(met)['annotation_quality'] == 'good', 'the singlet is still linked'


def test_the_cascade_is_raised_too():
    X, peaks, own = _study()
    row = _run(X, peaks, _met(peaks), _row(own, _by_cascade=True, _integ_mode='Cascade'))
    assert row['_confidence'] == 3


@pytest.mark.parametrize('level', [1, 3])
def test_only_two_dots_are_raised(level):
    X, peaks, own = _study()
    row = _run(X, peaks, _met(peaks), _row(own, _confidence=level, _identity_level=level))
    assert row['_confidence'] == level


@pytest.mark.parametrize('flag', ['_not_quantified', '_is_is'])
def test_rows_that_are_not_a_measured_analyte_are_not_asked(flag):
    X, peaks, own = _study()
    met = _met(peaks)
    row = _run(X, peaks, met, _row(own, **{flag: True}))
    assert sl.LINKS_KEY not in row and _singlet(met)['annotation_quality'] is None


def test_a_row_with_excluded_samples_is_read_on_the_samples_of_the_study():
    X, peaks, own = _study()
    info = [dict(i, excluded=k == 0) for k, i in enumerate(INFO)]
    row = _run(X, peaks, _met(peaks), _row(own), info=info)
    assert row['_confidence'] == 3


def test_the_analysts_mark_is_not_touched():
    X, peaks, own = _study()
    met = _met(peaks, annotation_quality='bad')
    before = copy.deepcopy(_singlet(met))
    marks = {('M1', before['peak_rank']): 'bad'}
    _run(X, peaks, met, _row(own), manual_marks=marks)
    assert _singlet(met) == before


def test_a_bad_verdict_is_not_raised():
    X, peaks, own = _study()
    met = _met(peaks, annotation_quality='bad')
    _run(X, peaks, met, _row(own))
    assert _singlet(met)['annotation_quality'] == 'bad'


def test_lines_the_analyst_chose_are_not_touched():
    X, peaks, own = _study()
    met = _met(peaks, manual_lines_ppm=[3.020])
    before = copy.deepcopy(_singlet(met))
    _run(X, peaks, met, _row(own))
    assert _singlet(met) == before


def test_too_few_samples_decide_nothing():
    X, peaks, own = _study()
    info = [dict(i, excluded=k >= 4) for k, i in enumerate(INFO)]
    row = _run(X, peaks, _met(peaks), _row(own), info=info)
    assert sl.LINKS_KEY not in row and row['_confidence'] == 2


@pytest.mark.parametrize('noise', [None, float('nan')])
def test_without_a_noise_level_nothing_is_decided(noise):
    X, peaks, own = _study()
    row = _run(X, peaks, _met(peaks), _row(own), noise=noise)
    assert sl.LINKS_KEY not in row


def test_forgetting_gives_back_the_annotation_exactly():
    X, peaks, own = _study()
    met = _met(peaks)
    before = copy.deepcopy(met)
    _run(X, peaks, met, _row(own))
    assert met != before
    sl.forget_marks([met])
    assert met == before


def test_a_signal_the_analyst_marked_since_keeps_the_analysts_mark():
    X, peaks, own = _study()
    met = _met(peaks)
    _run(X, peaks, met, _row(own))
    _singlet(met)['annotation_quality'] = 'bad'
    sl.forget_marks([met], {('M1', _singlet(met)['peak_rank']): 'bad'})
    assert _singlet(met)['annotation_quality'] == 'bad' and sl._MARK_KEY not in _singlet(met)


def test_deciding_again_is_the_same_answer():
    X, peaks, own = _study()
    met = _met(peaks)
    _run(X, peaks, met, _row(own))
    first = copy.deepcopy(met)
    sl.forget_marks([met])
    _run(X, peaks, met, _row(own))
    assert met == first


def test_promote_decides_on_the_annotations_own_verdict_after_forgetting():
    """The window's passes forget first, so a re-scoring starts from the
    annotation and not from the marks of the last quantification."""
    X, peaks, own = _study()
    met = _met(peaks)
    _run(X, peaks, met, _row(own))
    sl.forget_marks([met])
    promote_singlets([met], peaks, params=SCORING_DEFAULTS)
    assert sl._MARK_KEY not in _singlet(met)
