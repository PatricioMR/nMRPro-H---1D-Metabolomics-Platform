# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Why a metabolite has no number, and why the distinction is the whole point.

Every rejection used to read "lost the competition", including for analytes
that never competed. That sends the analyst to look at a rival who does not
exist. The reason is the actionable half: a withdrawn multiplet returns when
detection resolves its lines, an ambiguous rank when the analyst marks a
candidate, a beaten one only if the winner turns out to be wrong.
"""
import inspect

import numpy as np
import pytest

from gui.stages import _quant_computation as qc
from processing.quantification import (
    AMBIGUOUS_RANK, LOST_COMPETITION, NO_ELIGIBLE_SIGNAL,
    NO_MEASURABLE_WINDOW, REJECTION_TEXT, UNVERIFIED_MULTIPLET,
    Candidate, elect)


def _cand(mid, rank, lines=(), *, dj=0.0, n_found=2, n_exp=2, mult='d'):
    return Candidate(
        met_id=mid, peak_rank=rank, line_ranks=frozenset(lines),
        evidence={'multiplicity': mult, 'delta_j_hz': dj, 'ppm_diff': 0.0,
                  'n_found': n_found, 'n_expected': n_exp,
                  'quality': 'good', 'score': 1.0})


def test_a_withdrawn_multiplet_is_not_called_a_loser():
    """It never competed. Its lines were not all found, so its J was never
    measured — there was nothing to rank it by."""
    e = elect([_cand('partial', 1, [1], n_found=1, n_exp=3, mult='t')],
              dj_gold_hz=0.5)
    assert e.rejected['partial'] == UNVERIFIED_MULTIPLET


def test_only_a_metabolite_that_never_competed_gets_the_withdrawn_reason():
    e = elect([_cand('a', 1, [1], dj=0.0), _cand('b', 1, [1], dj=9.0)],
              dj_gold_hz=0.5)
    assert e.rejected['b'] == LOST_COMPETITION


def test_ambiguity_still_outranks_everything():
    """Candidates that tie on every criterion have no winner. Electing one by
    tie-break reports a decision the data does not support."""
    e = elect([_cand('a', 7, [7]), _cand('b', 7, [7])],
              dj_gold_hz=0.5, ambiguous_ranks=[7])
    assert e.rejected['a'] == e.rejected['b'] == AMBIGUOUS_RANK
    assert e.winners == {}


def test_a_winner_with_nothing_to_integrate_says_so():
    """It won and still has no number. Calling that "lost" would be false and
    dropping the row would hide it entirely."""
    from processing.quant_engine import quantify
    ppm = np.linspace(0.0, 2.0, 401)
    y = np.zeros((401, 1))
    rows = quantify(candidates=[_cand('ghost', None)], peaks_by_rank={},
                    ppm=ppm, intensities=y, samples_info=[{}], included=[0],
                    groups={}, dj_gold_hz=0.5)
    row = next(r for r in rows if r['hmdb_id'] == 'ghost')
    assert row['_rejection'] == NO_MEASURABLE_WINDOW


def test_every_reason_has_text_an_analyst_can_act_on():
    for reason in (LOST_COMPETITION, UNVERIFIED_MULTIPLET, AMBIGUOUS_RANK,
                   NO_ELIGIBLE_SIGNAL, NO_MEASURABLE_WINDOW):
        assert REJECTION_TEXT[reason].strip()


def test_the_reasons_are_distinct():
    """Two reasons sharing a wording is the defect this file exists for."""
    texts = [REJECTION_TEXT[r] for r in
             (LOST_COMPETITION, UNVERIFIED_MULTIPLET, AMBIGUOUS_RANK,
              NO_ELIGIBLE_SIGNAL, NO_MEASURABLE_WINDOW)]
    assert len(set(texts)) == len(texts)


def test_withdrawn_metabolites_still_get_a_row():
    """An analyte that disappears from the table cannot be questioned."""
    from processing.quant_engine import quantify
    ppm = np.linspace(0.0, 2.0, 401)
    y = np.zeros((401, 1))
    rows = quantify(
        candidates=[_cand('partial', 1, [1], n_found=1, n_exp=3, mult='t')],
        peaks_by_rank={}, ppm=ppm, intensities=y, samples_info=[{}],
        included=[0], groups={}, dj_gold_hz=0.5)
    assert [r['hmdb_id'] for r in rows] == ['partial']
    assert rows[0]['_not_quantified']


# ── The log ──────────────────────────────────────────────────────────────────

def test_every_unquantified_analyte_is_named_in_the_log_with_its_reason():
    """"34 were not quantified" is a fact; which 34 and why is what an analyst
    can act on. The count went to stdout and the names went nowhere."""
    src = inspect.getsource(qc)
    i = src.index("'── NOT QUANTIFIED")
    body = src[i:i + 900]
    assert "r.get('_status_reasons')" in body
    assert "r.get('metabolite'" in body


def test_they_are_grouped_by_reason():
    """Grouped, because the reason decides what to do next and the same action
    serves everyone under it."""
    src = inspect.getsource(qc)
    i = src.index("'── NOT QUANTIFIED")
    assert 'by_reason.setdefault' in src[i:i + 900]
