# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The analysis reads its inputs in one order defined by their content.

The same Mice ISS project, unchanged in content, quantified 37 or 40
analytes depending on the order its metabolite list was saved in; the same
99 peaks reversed gave another annotation. The result has to be a function
of the data alone.
"""
import random

import pytest

from processing.annotation_scoring import (
    SCORING_DEFAULTS, effective_annotation_params)
from processing.peak_detection import canonical_peak_order
from processing.quant_engine import canonical_order


def _mets():
    return [{'id': 'HMDB3', 'name': 'C', 'best_score': 0.5},
            {'id': 'HMDB1', 'name': 'A', 'best_score': 0.9},
            {'id': 'HMDB2', 'name': 'B', 'best_score': 0.5},
            {'id': 'HMDB4', 'name': 'D', 'best_score': None},
            {'id': 'HMDB5', 'name': 'E', 'best_score': float('nan')}]


class TestCanonicalOrder:
    def test_best_score_first_ties_by_id(self):
        assert [m['id'] for m in canonical_order(_mets())] == [
            'HMDB1', 'HMDB2', 'HMDB3', 'HMDB4', 'HMDB5']

    @pytest.mark.parametrize('seed', range(5))
    def test_any_permutation_gives_the_same_order(self, seed):
        a = _mets()
        b = _mets()
        random.Random(seed).shuffle(b)
        assert ([m['id'] for m in canonical_order(a)]
                == [m['id'] for m in canonical_order(b)])

    def test_the_dicts_are_not_copied_or_changed(self):
        a = _mets()
        out = canonical_order(a)
        assert all(any(o is m for m in a) for o in out)


class TestCanonicalPeakOrder:
    def test_by_rank_then_ppm_without_rank_last(self):
        peaks = [{'rank': 3, 'ppm': 1.0}, {'rank': None, 'ppm': 2.0},
                 {'rank': 1, 'ppm': 3.0}, {'rank': None, 'ppm': 5.0},
                 {'rank': 2, 'ppm': 0.5}]
        out = canonical_peak_order(peaks)
        assert [(p['rank'], p['ppm']) for p in out] == [
            (1, 3.0), (2, 0.5), (3, 1.0), (None, 5.0), (None, 2.0)]

    def test_reversed_input_gives_the_same_order(self):
        peaks = [{'rank': r, 'ppm': 10 - r} for r in range(1, 30)]
        assert canonical_peak_order(peaks) == canonical_peak_order(peaks[::-1])


class TestEffectiveAnnotationParams:
    def test_defaults_without_anything_set(self):
        assert effective_annotation_params() == dict(SCORING_DEFAULTS)

    def test_the_panel_tolerance_owns_dppm_gold(self):
        p = effective_annotation_params({'dppm_gold': 0.01}, 0.05)
        assert p['dppm_gold'] == 0.05

    def test_the_analysts_parameters_are_applied(self):
        p = effective_annotation_params({'dj_gold_hz': 0.7}, None)
        assert p['dj_gold_hz'] == 0.7
        assert p['dppm_gold'] == SCORING_DEFAULTS['dppm_gold']

    def test_a_blank_tolerance_keeps_the_default(self):
        assert (effective_annotation_params(None, '')['dppm_gold']
                == SCORING_DEFAULTS['dppm_gold'])
