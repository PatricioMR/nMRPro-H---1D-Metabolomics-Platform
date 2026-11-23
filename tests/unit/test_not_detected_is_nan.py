# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A sample the engine found no amount in is NaN wherever the table is read.

A zero is the engine saying it found nothing, and a number written where it
found none puts a measurement where there is none (Helsel 2006, Chemosphere
65:2434). What the analyst is given -- the statistics, the CSV, the
MetaboAnalyst file -- reads every value through ``reported_values``, and
nothing in the statistics fills a NaN with the column's mean.
"""
import numpy as np
import pytest

from gui.stages._quant_adapter import (
    CEILING, MEASURED, NOT_DETECTED, NOT_MEASURED, identity_columns,
    reported_cells, reported_values, statistics_values, upper_bound_cells)
from processing.quant_engine import _ceilings
from gui.stages._quant_helpers import _pls_vip

NAN = float('nan')


def test_a_positive_amount_is_reported_and_a_zero_is_nothing_found():
    row = {'_concs_per_sample': [1.0, 2.0, NAN, 0.0]}
    vals, nd = reported_values(row, 4)
    assert vals[:2] == [1.0, 2.0] and nd[:2] == [False, False]
    assert np.isnan(vals[2]) and not nd[2], 'never measured: NA, not n.d.'
    assert np.isnan(vals[3]) and nd[3], 'a zero is nothing found, not a measurement'


def test_an_old_projects_detection_record_changes_nothing():
    """Rows saved before the limits were removed carry a per-sample detection
    verdict; a positive amount is the measurement whatever it says."""
    row = {'_concs_per_sample': [1.0, 2.0],
           '_detected_per_sample': [True, False]}
    assert reported_cells(row, 2) == [(1.0, MEASURED), (2.0, MEASURED)]


def test_a_short_vector_is_padded_with_na():
    vals, nd = reported_values({'_concs_per_sample': [1.0]}, 3)
    assert vals[0] == 1.0 and np.isnan(vals[1]) and np.isnan(vals[2])
    assert nd == [False, False, False]


def test_the_exports_carry_the_identity_of_every_row():
    assert identity_columns({'_by_cascade': True, '_confidence': 1,
                             '_identity_why': 'x'}) == (1, 'Cascade estimate', 'x')
    assert identity_columns({'_joint_fit': {'group': []}, '_confidence': 3}
                            )[:2] == (3, 'Joint fit')
    assert identity_columns({'_fit_refused': True, '_confidence': 1}
                            )[1] == 'Integral (joint fit refused)'
    assert identity_columns({'_integ_mode': 'Multiplet', '_confidence': 2}
                            )[:2] == (2, 'Multiplet')
    assert identity_columns({'_integ_mode': 'Multiplet', '_confidence': 2,
                             '_manual_regions': {1: (1.0, 2.0), 4: (1.0, 2.0)}}
                            )[1] == 'Multiplet + window set by hand in 2 sample(s)'
    assert identity_columns({})[0] == ''


# ── the VIP does not fill what was not detected ──────────────────────────────

def _vip_data(seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    X = rng.normal(size=(6, 4))
    X[:, 0] += 2.0 * y
    return X, y


def test_a_feature_with_a_not_detected_sample_gets_no_vip_and_does_not_bend_the_rest():
    X, y = _vip_data()
    full = _pls_vip(X, y, n_components=1)
    Xn = X.copy()
    Xn[2, 1] = np.nan
    got = _pls_vip(Xn, y, n_components=1)
    assert np.isnan(got[1])
    # the others are what they are on the three complete features alone: the
    # missing cell was not filled with its column mean to keep four in the model
    only = _pls_vip(X[:, [0, 2, 3]], y, n_components=1)
    assert got[[0, 2, 3]] == pytest.approx(only)
    assert np.all(np.isfinite(full))


def test_all_features_incomplete_gives_nothing():
    X, y = _vip_data()
    X[0, :] = np.nan
    assert np.all(np.isnan(_pls_vip(X, y, n_components=1)))


# ── the study proved the analyte: every sample has a number, of some kind ────

def _row(**kw):
    base = {'_concs_per_sample': [1.0, 2.0, 0.0, NAN, NAN, 0.0],
            '_ceiling_per_sample': [0.0, 0.0, 0.5, 0.1, 0.3, 0.0],
            '_included': [0, 1, 2, 3, 5]}
    base.update(kw)
    return base


def test_every_sample_of_the_study_has_a_number_and_the_kind_says_which():
    cells = reported_cells(_row(), 6)
    assert cells[0] == (1.0, MEASURED)
    assert cells[1] == (2.0, MEASURED)
    assert cells[2] == (0.5, CEILING), 'a zero becomes the most it can hold'
    assert cells[3] == (0.1, CEILING), 'no amplitude at all: the same'
    assert np.isnan(cells[4][0]) and cells[4][1] == NOT_MEASURED,         'a sample outside the study is not measured, whatever its ceiling'
    assert np.isnan(cells[5][0]) and cells[5][1] == NOT_DETECTED,         'nothing in the window either: n.d., not a number'
    assert upper_bound_cells(_row(), 6) == [False, False, True, True, False, False]


def test_a_row_without_a_ceiling_still_ends_in_nd_for_a_zero():
    cells = reported_cells(_row(_ceiling_per_sample=None), 6)
    assert cells[2][1] == NOT_DETECTED and np.isnan(cells[2][0])


def test_a_sample_of_the_study_without_a_number_is_nd_not_na():
    """NA says the sample is not in the study. A sample that is, and where
    the engine has no amount and no ceiling, is n.d. -- on the Mice study 47
    such cells read NA once the detection-limit floor stopped hiding them."""
    cells = reported_cells(_row(_ceiling_per_sample=None), 6)
    assert cells[3][1] == NOT_DETECTED, 'in the study, no amplitude'
    assert cells[4][1] == NOT_MEASURED, 'not in the study'


def test_the_statistics_read_measurements_only_unless_asked_otherwise():
    got = statistics_values(_row(), 6)
    assert got[:2] == [1.0, 2.0]
    assert np.isnan(got[2]) and np.isnan(got[3])
    kept = statistics_values(_row(), 6, censored=False)
    assert kept[:4] == [1.0, 2.0, 0.5, 0.1]


def test_the_ceiling_is_the_window_integral_and_nothing_below_it():
    got = _ceilings({0: 0.4, 1: 0.0, 2: NAN, 3: -1.0, 4: None})
    assert got == {0: 0.4}


def test_an_analyte_the_analyst_confirmed_is_exported_as_manual():
    """A signal marked Good by hand (Annotation Results, or lines saved in the
    STOCSY window) makes the analyte's row Manual in every export; marked
    Poor, or not marked, it stays Automatic, and a mark taken back is undone
    on the next run."""
    from gui.stages._quant_adapter import (
        ANNOTATION_AUTOMATIC, ANNOTATION_MANUAL, annotation_column, mark_analyst_confirmed)
    rows = [{'hmdb_id': 'HMDB1'}, {'hmdb_id': 'HMDB2'}, {'hmdb_id': 'HMDB3'}]
    marks = {('HMDB1', 12): 'good', ('HMDB2', 7): 'bad'}
    mark_analyst_confirmed(rows, marks)
    assert [annotation_column(r) for r in rows] == [
        ANNOTATION_MANUAL, ANNOTATION_AUTOMATIC, ANNOTATION_AUTOMATIC]
    mark_analyst_confirmed(rows, {})
    assert annotation_column(rows[0]) == ANNOTATION_AUTOMATIC


def test_a_good_mark_past_the_limits_is_taken_off_and_said():
    from gui.stages._quant_adapter import gate_manual_marks, manual_refused_reason
    mets = [{'id': 'OK', 'name': 'Fine', 'matched_peaks': [
                {'peak_rank': 1, 'type': 'd', 'ppm_diff': 0.01, 'delta_j_hz': 0.2,
                 'db_ppm': 1.33, 'annotation_quality': 'good'}]},
            {'id': 'NO', 'name': 'Far', 'matched_peaks': [
                {'peak_rank': 2, 'type': 'd', 'ppm_diff': 0.01, 'delta_j_hz': 0.9,
                 'db_ppm': 2.10, 'annotation_quality': 'good'}]}]
    marks = {('OK', 1): 'good', ('NO', 2): 'good', ('NO', 9): 'bad'}
    kept, refused = gate_manual_marks(mets, marks, {'dppm_gold': 0.05, 'dj_bad_hz': 0.5})
    assert kept == {('OK', 1): 'good', ('NO', 9): 'bad'}
    assert [r['name'] for r in refused] == ['Far']
    assert mets[1]['matched_peaks'][0]['annotation_quality'] is None
    assert 'at 2.100 ppm' in manual_refused_reason(refused[0])


def test_the_report_puts_one_blue_dot_left_of_the_confidence_dots():
    """Nothing written: one dot more, blue, then the engine's dots."""
    import re
    from gui.widgets.pdf_report import _quality_markup
    from gui.utils.verified_dot import VERIFIED_BLUE
    m = _quality_markup({'_annotation_manual': True}, 'CONF ●●○  ≥95%*', '#f9a825')
    text = re.sub(r'<[^>]+>', '', m)
    assert text == 'CONF ●●●○  ≥95%*'
    assert m.index(VERIFIED_BLUE) < m.index('●●○')
    assert 'verified' not in m
    plain = _quality_markup({'_confidence': 2}, 'CONF ●●○  ≥95%*', '#f9a825')
    assert re.sub(r'<[^>]+>', '', plain) == 'CONF ●●○  ≥95%*' and VERIFIED_BLUE not in plain
