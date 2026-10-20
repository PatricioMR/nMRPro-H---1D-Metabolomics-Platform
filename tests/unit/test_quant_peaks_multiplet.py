# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Quant Peaks window — multiplet-level selection helpers.

Quantification works per multiplicity: the user selects a whole multiplet
(header row) and applying it re-quantifies every analyte. These tests cover the
two pure helpers driving that flow without building the Tk window:

* _multiplet_anchor_rank — the override rank that forces the chosen multiplet.
* _multiplet_conflicts   — names of OTHER metabolites annotated on the same
                           signal (used to BLOCK the change).
"""
from types import SimpleNamespace

from gui.widgets.quant_peaks_window import QuantPeaksWindow


def _members(*pairs, anchor=None):
    """pairs = (line_rank,...). hit.peak_rank = anchor (multiplet anchor)."""
    hit = {'peak_rank': anchor}
    return [{'peak_rank': r, 'hit': hit} for r in pairs]


# ── _multiplet_anchor_rank ──────────────────────────────────────────────────

def test_anchor_uses_matched_peak_rank():
    # Doublet lines 49/51, matched-peak anchor 461 (the centre recorded by
    # annotation). Override must force the anchor so the whole multiplet wins.
    members = _members(49, 51, anchor=461)
    assert QuantPeaksWindow._multiplet_anchor_rank(members) == 461


def test_anchor_falls_back_to_first_line_when_no_anchor():
    members = _members(49, 51, anchor=None)
    assert QuantPeaksWindow._multiplet_anchor_rank(members) == 49


def test_anchor_none_for_empty():
    assert QuantPeaksWindow._multiplet_anchor_rank([]) is None


# ── _multiplet_conflicts ────────────────────────────────────────────────────

def _fake_self(my_id, iid_to_sc=None, metabolites=None):
    annot_win = SimpleNamespace(_iid_to_subchild=iid_to_sc or {})
    ann_stage = SimpleNamespace(_metabolites=metabolites or [])
    app = SimpleNamespace(_annot_win=annot_win)
    qs  = SimpleNamespace(app=app, annotation_stage=ann_stage)
    return SimpleNamespace(result={'hmdb_id': my_id}, qs=qs)


def _sc(met, rank):
    # (met, dp, mp_ref, line_dict, line_idx, nearest_pk)
    return (met, {}, {}, {}, 0, {'rank': rank})


def test_conflict_detected_via_subchild():
    members = _members(49, 51, anchor=461)
    other   = {'id': 'HMDB0000002', 'name': 'Lactate'}
    mine    = {'id': 'HMDB0000001', 'name': 'Alanine'}
    iid_to_sc = {
        'a': _sc(mine, 49),
        'b': _sc(other, 51),      # other metabolite claims line 51 → conflict
    }
    fs = _fake_self('HMDB0000001', iid_to_sc=iid_to_sc)
    assert QuantPeaksWindow._multiplet_conflicts(fs, members) == ['Lactate']


def test_no_conflict_when_only_own_metabolite():
    members = _members(49, 51, anchor=461)
    mine    = {'id': 'HMDB0000001', 'name': 'Alanine'}
    iid_to_sc = {'a': _sc(mine, 49), 'b': _sc(mine, 51)}
    fs = _fake_self('HMDB0000001', iid_to_sc=iid_to_sc)
    assert QuantPeaksWindow._multiplet_conflicts(fs, members) == []


def test_conflict_fallback_to_metabolites_when_window_closed():
    members = _members(49, 51, anchor=461)
    metabolites = [
        {'id': 'HMDB0000001', 'name': 'Alanine',
         'matched_peaks': [{'peak_rank': 49}]},
        {'id': 'HMDB0000003', 'name': 'Threonine',
         'matched_peaks': [{'peak_rank': 51}]},   # claims line 51 → conflict
    ]
    fs = _fake_self('HMDB0000001', iid_to_sc={}, metabolites=metabolites)
    assert QuantPeaksWindow._multiplet_conflicts(fs, members) == ['Threonine']


def test_conflict_ignores_ranks_outside_multiplet():
    members = _members(49, 51, anchor=461)
    other   = {'id': 'HMDB0000002', 'name': 'Lactate'}
    iid_to_sc = {'b': _sc(other, 900)}   # unrelated peak elsewhere
    fs = _fake_self('HMDB0000001', iid_to_sc=iid_to_sc)
    assert QuantPeaksWindow._multiplet_conflicts(fs, members) == []
