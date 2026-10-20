# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Unit tests for the Box/Violin metabolite selection resolver.

_resolve_box_selection maps user-picked metabolite names to column indices,
caps them to the grid capacity, and falls back to highest-variance metabolites
when nothing is selected — without ever fabricating columns.
"""
import numpy as np

from gui.stages.statistics_stage import _resolve_box_selection, _fit_page


FEAT = ['Ala', 'Glu', 'Lac', 'Cit', 'Gly']


def _matrix():
    # Column variances increasing left→right so argsort order is predictable.
    return np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.1, 2.4, 3.9, 5.6, 7.5],
        [0.9, 1.6, 2.1, 2.4, 2.5],
    ])


def test_preserves_feat_order_not_selection_order():
    idx = _resolve_box_selection(['Lac', 'Ala'], FEAT, capacity=4)
    assert idx == [0, 2]                       # Ala(0) before Lac(2)


def test_caps_to_capacity():
    idx = _resolve_box_selection(FEAT, FEAT, capacity=2)
    assert idx == [0, 1]


def test_drops_unknown_names():
    idx = _resolve_box_selection(['Ala', 'Nonexistent'], FEAT, capacity=4)
    assert idx == [0]


def test_fallback_to_highest_variance_when_empty():
    X = _matrix()
    idx = _resolve_box_selection([], FEAT, capacity=3, x_raw=X)
    # Highest-variance columns are the rightmost ones (4,3,2).
    assert idx == [4, 3, 2]


def test_fallback_needs_matrix():
    assert _resolve_box_selection([], FEAT, capacity=3) == []


def test_zero_capacity_returns_empty():
    assert _resolve_box_selection(['Ala'], FEAT, capacity=0) == []


A4_LONG, A4_SHORT = 11.69, 8.27


def test_small_figure_not_scaled_up():
    assert _fit_page(4.0, 3.0) == (4.0, 3.0)


def test_landscape_capped_to_a4():
    w, h = _fit_page(40.0, 10.0)          # very wide -> landscape bound
    assert w <= A4_LONG + 1e-6
    assert h <= A4_SHORT + 1e-6


def test_portrait_capped_to_a4():
    w, h = _fit_page(10.0, 40.0)          # very tall -> portrait bound
    assert w <= A4_SHORT + 1e-6
    assert h <= A4_LONG + 1e-6


def test_aspect_ratio_preserved_when_scaled():
    w0, h0 = 40.0, 10.0
    w, h = _fit_page(w0, h0)
    assert abs((w / h) - (w0 / h0)) < 1e-6
