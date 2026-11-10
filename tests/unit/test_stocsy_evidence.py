# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy_evidence.py
"""The lift: an analyte whose own signals co-vary rises one dot -- never
lowered, never above three, never from the cascade, never with few samples."""
import numpy as np
import pytest

from processing import stocsy_evidence as se

N = 12


def _study(seed=0, n=N):
    """Molecule A: three signals at one concentration per sample. Molecule B:
    three signals whose amounts are independent. A lone molecule C: two
    signals, one pair."""
    rng = np.random.default_rng(seed)
    a = 1000 * np.exp(rng.normal(0, 0.4, n))
    peaks, k = [], 0
    def add(vec, ppm):
        nonlocal k
        k += 1
        peaks.append({'rank': k, 'ppm': ppm, 'area_per_sample': np.asarray(vec, float)})
        return k
    A = [add(a * f * np.exp(rng.normal(0, 0.03, n)), p) for f, p in ((1, 1.3), (0.5, 4.1), (2, 2.2))]
    B = [add(800 * np.exp(rng.normal(0, 0.4, n)), p) for p in (3.0, 3.9, 6.5)]
    c = 900 * np.exp(rng.normal(0, 0.4, n))
    C = [add(c, 5.1), add(c * 0.7 * np.exp(rng.normal(0, 0.03, n)), 7.2)]
    mets = [{'id': 'A', 'matched_peaks': [{'peak_rank': r} for r in A]},
            {'id': 'B', 'matched_peaks': [{'peak_rank': r} for r in B]},
            {'id': 'C', 'matched_peaks': [{'peak_rank': r} for r in C]}]
    info = [{'index': i, 'label': 'G1' if i < n // 2 else 'G2', 'type': 'Sample'}
            for i in range(n)]
    return peaks, mets, info


def _rows(levels):
    return [{'hmdb_id': k, '_confidence': v, '_identity_level': v}
            for k, v in levels.items()]


def _lift(levels, **kw):
    peaks, mets, info = kw.pop('study', None) or _study()
    rows = _rows(levels)
    return se.lift_by_covariation(rows, mets, peaks, info, **kw)


def test_the_lift_does_not_lower_the_identity_it_read():
    """A row at one dot for another reason keeps the identity level it has."""
    peaks, mets, info = _study()
    rows = [{'hmdb_id': 'A', '_confidence': 1, '_identity_level': 3}]
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[0]['_confidence'] == 2 and rows[0]['_identity_level'] == 3


def test_the_lift_is_on_and_its_bars_are_the_validated_ones():
    assert se.STOCSY_LIFT is True
    assert (se.STOCSY_LIFT_MIN_PAIRS, se.STOCSY_LIFT_R, se.STOCSY_LIFT_ALPHA) == (2, 0.80, 0.05)


def test_an_analyte_whose_signals_move_together_rises_one_dot():
    rows = _lift({'A': 1})
    assert rows[0]['_confidence'] == 2 and rows[0]['_identity_level'] == 2
    assert rows[0]['status'] == '●●○', 'the table and the report print this field'
    assert rows[0]['_stocsy_lift'] == {'pairs': 3, 'from': 1}
    assert 'move together across the 12 samples' in rows[0]['_identity_why']


def test_two_dots_become_three_and_three_stays_three():
    assert _lift({'A': 2})[0]['_confidence'] == 3
    assert _lift({'A': 3})[0]['_confidence'] == 3
    assert '_stocsy_lift' not in _lift({'A': 3})[0]


def test_independent_signals_are_left_alone():
    assert _lift({'B': 1})[0]['_confidence'] == 1


def test_one_pair_is_not_enough_for_the_default_two():
    assert _lift({'C': 1})[0]['_confidence'] == 1
    assert _lift({'C': 1}, min_pairs=1)[0]['_confidence'] == 2


def test_a_low_correlation_never_lowers_anything():
    rows = _lift({'A': 2, 'B': 2})
    assert rows[1]['_confidence'] == 2 and '_stocsy_lift' not in rows[1]


def test_the_cascade_is_not_raised_and_neither_are_unquantified_rows():
    peaks, mets, info = _study()
    rows = [dict(r) for r in _rows({'A': 1})]
    rows[0]['_by_cascade'] = True
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[0]['_confidence'] == 1
    rows = _rows({'A': 1})
    rows[0]['_not_quantified'] = True
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[0]['_confidence'] == 1
    rows = _rows({'A': 1})
    rows[0]['_is_is'] = True
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[0]['_confidence'] == 1


def test_too_few_analytical_samples_change_nothing():
    study = _study(n=4)
    rows = _lift({'A': 1}, study=study)
    assert rows[0]['_confidence'] == 1


def test_blanks_do_not_count_and_do_not_manufacture_correlation():
    peaks, mets, info = _study()
    for p in peaks:                                   # four blanks, all near zero
        p['area_per_sample'] = np.append(p['area_per_sample'], [0.0] * 4)
    info = info + [{'index': N + i, 'label': 'BLK', 'type': 'BLK'} for i in range(4)]
    rows = _rows({'A': 1, 'B': 1})
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[0]['_confidence'] == 2 and rows[1]['_confidence'] == 1


def test_an_analyte_the_annotation_gave_no_signals_is_skipped():
    peaks, mets, info = _study()
    rows = _rows({'A': 1, 'ZZ': 1})
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[1]['_confidence'] == 1


def test_the_reason_is_appended_to_an_existing_one():
    peaks, mets, info = _study()
    rows = _rows({'A': 1})
    rows[0]['_identity_why'] = 'a singlet standing on the drift'
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[0]['_identity_why'].startswith('a singlet standing on the drift; its own')


def test_the_same_input_gives_the_same_output():
    a = _lift({'A': 1, 'B': 2, 'C': 1})
    b = _lift({'A': 1, 'B': 2, 'C': 1})
    assert [r['_confidence'] for r in a] == [r['_confidence'] for r in b]


def test_a_dilution_does_not_lift_independent_analytes():
    """Raw areas under a strong dilution would pair everything; PQN does not."""
    rng = np.random.default_rng(3)
    d = np.exp(rng.normal(0, 0.6, N))
    peaks, mets, info = _study(seed=3)
    for p in peaks:
        p['area_per_sample'] = p['area_per_sample'] * d
    rows = _rows({'B': 1})
    se.lift_by_covariation(rows, mets, peaks, info)
    assert rows[0]['_confidence'] == 1


def test_the_adapter_applies_it_only_when_the_flag_is_on():
    import inspect
    from gui.stages import _quant_adapter
    src = inspect.getsource(_quant_adapter.quantify_project)
    assert 'if stocsy_evidence.STOCSY_LIFT:' in src
