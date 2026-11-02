# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Quant Signals window — which multiplets it offers and what Quantify sends.

Quantification works per multiplicity: the analyst selects a whole multiplet
(header row) and Quantify measures that analyte alone on it. These tests
cover the pure helpers behind that, without building the Tk window:

* _taken_by          — the other quantified analytes whose window holds one
                       of the multiplet's peaks (the multiplet is hidden).
* _multiplet_quality — Good / Average / Poor of the whole multiplet (Poor is
                       hidden).
* _choice_of         — the multiplet as quantify_with_signal reads it.
"""
from types import SimpleNamespace

from gui.widgets.quant_peaks_window import QuantPeaksWindow


def _members(*ppms, dp_key=(3.006, 'd'), hit=None):
    return [{'peak_rank': i + 1, 'peak': {'ppm': p}, 'hit_ppm': p,
             'hit': dict(hit or {}), '_dp_key': dp_key,
             '_dp_ppm': dp_key[0] if dp_key else 0.0,
             '_dp_mult': dp_key[1] if dp_key else '?', '_dp': {},
             'is_current': False}
            for i, p in enumerate(ppms)]


def _fake_self(rows=(), marks=None, child_marks=None, my_id='A'):
    ann = SimpleNamespace(_quality_marks=marks or {},
                          _child_quality_marks=child_marks or {})
    qs = SimpleNamespace(_quant_results=list(rows), annotation_stage=ann)
    fs = SimpleNamespace(result={'hmdb_id': my_id}, qs=qs, _candidates=[])
    for name in ('_my_ids', '_effective_quality'):
        setattr(fs, name, getattr(QuantPeaksWindow, name).__get__(fs))
    return fs


def test_a_multiplet_inside_another_analytes_window_is_theirs():
    members = _members(3.000, 3.012)
    rows = [{'hmdb_id': 'B', 'metabolite': 'Lactate',
             '_integ_segments': [[3.008, 3.020]]},
            {'hmdb_id': 'C', 'metabolite': 'Ghost', '_not_quantified': True,
             '_multiplet_window': [2.99, 3.02]},
            {'hmdb_id': 'A', 'metabolite': 'Mine',
             '_multiplet_window': [2.99, 3.02]}]
    fs = _fake_self(rows)
    assert QuantPeaksWindow._taken_by(fs, members) == ['Lactate']
    assert QuantPeaksWindow._taken_by(fs, _members(4.0, 4.012)) == []


def test_poor_only_when_marked_or_every_line_is():
    members = _members(3.000, 3.012)
    q = QuantPeaksWindow._multiplet_quality
    assert q(_fake_self(), members) is None                       # Average
    assert q(_fake_self(marks={('A', 1): 'bad'}), members) is None
    assert q(_fake_self(marks={('A', 1): 'bad', ('A', 2): 'bad'}),
             members) == 'bad'
    assert q(_fake_self(marks={('A', 2): 'good'}), members) == 'good'
    assert q(_fake_self(child_marks={('A', 3.006): 'bad'}), members) == 'bad'


def test_quantify_sends_the_lines_and_the_pattern():
    hit = {'type': 'd', 'protons': 3,
           'predicted_lines': [{'ppm': 3.012}, {'ppm': 3.000}]}
    choice = QuantPeaksWindow._choice_of(_fake_self(),
                                         _members(3.000, 3.012, hit=hit))
    assert choice['line_ranks'] == [2, 1]          # high ppm first
    assert choice['protons'] == 3
    assert choice['predicted_ppms'] == [3.012, 3.000]
    assert choice['label'] == '3.006 ppm d'
