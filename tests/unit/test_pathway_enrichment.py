# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Tests for processing.pathway_enrichment — two-group ORA pipeline.

Covers:
  • per_metabolite_test  — fold-change sign + p ordering for clear effects
  • select_hits          — threshold selection
  • compute_ora          — contingency table → Fisher p, Enrichment Ratio,
                           monotone Benjamini-Hochberg FDR
"""
import numpy as np
from scipy.stats import fisher_exact

from processing.pathway_enrichment import (
    per_metabolite_test,
    select_hits,
    compute_ora,
)


def test_per_metabolite_foldchange_sign_and_pvalue():
    rng = np.random.default_rng(0)
    # M1 strongly up in A, M2 flat between groups.
    a = np.column_stack([rng.normal(10, 0.5, 8), rng.normal(5, 0.5, 8)])
    b = np.column_stack([rng.normal(5, 0.5, 8), rng.normal(5, 0.5, 8)])
    X = np.vstack([a, b])
    groups = ['A'] * 8 + ['B'] * 8
    res = per_metabolite_test(X, groups, 'A', 'B', ['M1', 'M2'])

    assert res['M1']['log2fc'] > 0.5        # A > B
    assert res['M1']['p'] < 0.01            # clearly separated
    assert res['M2']['p'] > 0.05            # no real difference
    assert abs(res['M2']['log2fc']) < 0.3


def test_select_hits_threshold():
    tr = {'A': {'p': 0.001}, 'B': {'p': 0.04}, 'C': {'p': 0.20}}
    assert select_hits(tr, p_thr=0.05) == {'A', 'B'}
    assert select_hits(tr, p_thr=0.01) == {'A'}


def test_compute_ora_matches_fisher_and_ratio():
    # Pathway P1: 4 mets, 3 are hits. Background N=100, total hits n=3.
    table = {'P1': {'mets': ['m1', 'm2', 'm3', 'm4'], 'smpdb_id': 'SMP1'}}
    bg = {'P1': 10}                       # K = 10 in background
    hits = {'m1', 'm2', 'm3'}
    res = compute_ora(table, hits, background_sizes=bg, n_background=100)
    assert len(res) == 1
    r = res[0]
    assert r['k'] == 3 and r['K'] == 10 and r['n'] == 3

    k, K, n, N = 3, 10, 3, 100
    expected_ratio = k / (n * K / N)
    assert np.isclose(r['enrichment_ratio'], expected_ratio)

    _, p_exp = fisher_exact([[k, n - k], [K - k, N - n - (K - k)]],
                            alternative='greater')
    assert np.isclose(r['p'], p_exp)


def test_compute_ora_skips_pathways_without_hits():
    table = {
        'P1': {'mets': ['m1', 'm2'], 'smpdb_id': ''},
        'P2': {'mets': ['x1', 'x2'], 'smpdb_id': ''},
    }
    res = compute_ora(table, {'m1'}, background_sizes={'P1': 5, 'P2': 5},
                      n_background=50)
    assert [r['name'] for r in res] == ['P1']   # P2 has zero hits → skipped


def test_compute_ora_fdr_monotone_and_sorted():
    # Three pathways with varying hit counts → varying p.
    table = {
        'Hi': {'mets': [f'h{i}' for i in range(5)], 'smpdb_id': ''},
        'Mid': {'mets': [f'm{i}' for i in range(5)], 'smpdb_id': ''},
        'Lo': {'mets': [f'l{i}' for i in range(5)], 'smpdb_id': ''},
    }
    bg = {'Hi': 5, 'Mid': 10, 'Lo': 30}
    hits = {'h0', 'h1', 'h2', 'h3', 'h4', 'm0', 'm1', 'l0'}
    res = compute_ora(table, hits, background_sizes=bg, n_background=200)

    ps = [r['p'] for r in res]
    assert ps == sorted(ps)                          # sorted by p ascending
    for r in res:
        assert 0.0 <= r['fdr'] <= 1.0
        assert r['fdr'] >= r['p'] - 1e-12            # BH-FDR >= raw p
