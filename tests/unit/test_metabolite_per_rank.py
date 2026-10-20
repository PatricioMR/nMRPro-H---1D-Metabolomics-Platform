# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Unit tests for the canonical per-peak metabolite selector.

Guards the Peak List ↔ Annotation Results agreement on which metabolite
owns a given peak rank (regression: overlap regions used to disagree).
"""
import pytest
from gui.stages._annotation_helpers import (
    _assignment_sort_key,
    select_metabolite_per_rank,
)


def _met(name, matched):
    return {'name': name, 'matched_peaks': matched}


def _mp(rank, score, mult='s', dj=None, quality=None, **extra):
    d = {'peak_rank': rank, 'composite_score': score,
         'type': mult, 'delta_j_hz': dj, 'annotation_quality': quality}
    d.update(extra)
    return d


def test_bad_quality_is_discarded():
    mets = [
        _met('X', [_mp(1, 0.9, quality='bad')]),
        _met('Y', [_mp(1, 0.3)]),
    ]
    assert select_metabolite_per_rank(mets) == {1: 'Y'}


def test_manual_good_beats_higher_score():
    mets = [
        _met('X', [_mp(1, 0.95)]),               # higher auto score
        _met('Y', [_mp(1, 0.40, quality='good')]),  # manually confirmed
    ]
    assert select_metabolite_per_rank(mets)[1] == 'Y'


def test_j_confirmed_multiplet_beats_higher_singlet():
    # Singlet scores higher on composite, but the J-confirmed doublet
    # (ΔJ < 0.5) gets the +0.20 anchor bonus → must win the rank.
    mets = [
        _met('Singlet', [_mp(1, 0.70, mult='s')]),
        _met('Doublet', [_mp(1, 0.60, mult='d', dj=0.2)]),
    ]
    assert select_metabolite_per_rank(mets)[1] == 'Doublet'


def test_two_singlets_apart_only_on_score_have_no_winner():
    # a singlet's score is its position again; the third decimal of it
    # does not settle a contest
    mets = [
        _met('Lo', [_mp(1, 0.50, mult='s')]),
        _met('Hi', [_mp(1, 0.80, mult='s')]),
    ]
    assert select_metabolite_per_rank(mets).get(1) is None


def test_high_delta_j_multiplet_not_excluded():
    # ΔJ = 0.8 (> 0.5, no anchor bonus) but not marked 'bad': still a valid
    # candidate. Old Peak List logic wrongly excluded it; canonical keeps it.
    mets = [_met('Wobbly', [_mp(1, 0.65, mult='d', dj=0.8)])]
    assert select_metabolite_per_rank(mets) == {1: 'Wobbly'}


def test_none_rank_ignored():
    mets = [_met('X', [_mp(None, 0.9), _mp(2, 0.5)])]
    assert select_metabolite_per_rank(mets) == {2: 'X'}


def test_companion_multiplet_no_longer_boosts_a_singlet():
    """Confirmation by a companion multiplet was removed: too variable.

    A singlet is ranked on its own score plus its own J evidence; the fact that
    the metabolite has a good multiplet elsewhere gives it nothing here.
    """
    base = _mp(1, 0.5, mult='s')
    boosted = _mp(1, 0.5, mult='s', j_met_confirmed=True,
                  anchor_status='clean', anchor_pos_confidence='high')
    assert _assignment_sort_key(0.5, boosted) == _assignment_sort_key(0.5, base)


def test_empty_input():
    assert select_metabolite_per_rank([]) == {}
    assert select_metabolite_per_rank(None) == {}
