# tests/unit/test_j_resolve_per_sample.py
"""
Testes para _compute_per_sample_fractions de JResolveStage.

Cobre:
  • Achado A — frações variam por amostra (preservam diferenças de grupo).
  • Achado B — conservação de massa: Σ frações por pico ≤ 1 (renormaliza).
  • Fallback escalar quando a área observada da amostra é ≤ 0.
  • Replay da cascata por amostra (determinístico).
"""
import numpy as np
import pytest
from gui.stages.j_resolve_stage import JResolveStage


class _FakeStage:
    """Stub que expõe só o necessário para _compute_per_sample_fractions."""

    def __init__(self, n_samples):
        # all_int só serve para dimensionar n_s (áreas vêm de _areas)
        self._all_int = np.zeros((4, n_samples))
        self._ppm     = np.linspace(0, 1, 4)
        self._peak_overlap   = {}
        self._contributions  = {}
        self._anchor_plan    = {}
        self._cascade_plan   = []
        self._nnls_plan      = None
        self._contributions_per_sample = {}
        self._mets_map  = {}   # id → met dict
        self._peaks_map = {}   # rank → peak dict
        self._areas     = {}   # (rank, sample) → área controlada

    # ── stubs ────────────────────────────────────────────────────────────────
    def _met_by_id(self, mid):
        return self._mets_map.get(mid)

    def _met_by_name(self, name):
        return next((m for m in self._mets_map.values()
                     if m.get('name') == name), None)

    def _peak_by_rank(self, rank):
        return self._peaks_map.get(rank)

    def _get_area_sample(self, peak, s):
        return float(self._areas.get((peak['rank'], s), 0.0))

    def _effective_nh(self, met, mp):
        return float((mp or {}).get('_eff', 1.0))


_FakeStage._compute_per_sample_fractions = \
    JResolveStage._compute_per_sample_fractions


def _met(mid, name, rank, eff=1.0):
    return {'id': mid, 'name': name,
            'matched_peaks': [{'peak_rank': rank, '_eff': eff}]}


def _make_two_anchor_shared(areas):
    """A (âncora rank10) + B (âncora rank20), pico partilhado rank1.

    `areas` = dict {(rank, s): area}.
    """
    st = _FakeStage(n_samples=2)
    met_a = {'id': 'A', 'name': 'A',
             'matched_peaks': [{'peak_rank': 10, '_eff': 1.0},
                               {'peak_rank': 1,  '_eff': 1.0}]}
    met_b = {'id': 'B', 'name': 'B',
             'matched_peaks': [{'peak_rank': 20, '_eff': 1.0},
                               {'peak_rank': 1,  '_eff': 1.0}]}
    st._mets_map  = {'A': met_a, 'B': met_b}
    st._peaks_map = {1:  {'rank': 1,  'ppm': 2.0},
                     10: {'rank': 10, 'ppm': 1.0},
                     20: {'rank': 20, 'ppm': 3.0}}
    st._peak_overlap  = {1: ['A', 'B'], 10: ['A'], 20: ['B']}
    st._anchor_plan   = {'A': [10], 'B': [20]}
    st._contributions = {1: {'A': 50.0, 'B': 50.0,
                             '_observed': 10.0, '_residual_pct': 0.0}}
    st._areas = areas
    return st


# ───────────────────────────────────────────────────────────────────────────
# Achado A — frações variam por amostra
# ───────────────────────────────────────────────────────────────────────────

def test_fractions_differ_between_samples():
    # Amostra 0: A domina (conc_A=8, conc_B=2, obs=10)
    # Amostra 1: B domina (conc_A=2, conc_B=8, obs=10)
    areas = {
        (10, 0): 8.0, (20, 0): 2.0, (1, 0): 10.0,
        (10, 1): 2.0, (20, 1): 8.0, (1, 1): 10.0,
    }
    st = _make_two_anchor_shared(areas)
    st._compute_per_sample_fractions()

    fa = st._contributions_per_sample[1]['A']
    fb = st._contributions_per_sample[1]['B']

    assert fa[0] == pytest.approx(0.8, abs=1e-6)
    assert fb[0] == pytest.approx(0.2, abs=1e-6)
    assert fa[1] == pytest.approx(0.2, abs=1e-6)
    assert fb[1] == pytest.approx(0.8, abs=1e-6)
    # A diferença entre amostras é real (não congelada na média)
    assert fa[0] != pytest.approx(fa[1])


# ───────────────────────────────────────────────────────────────────────────
# Achado B — conservação de massa
# ───────────────────────────────────────────────────────────────────────────

def test_mass_conserved_when_overpredicted():
    # conc_A=8, conc_B=8, obs=10 → soma bruta = 1.6 > 1 → renormaliza para 1.0
    areas = {
        (10, 0): 8.0, (20, 0): 8.0, (1, 0): 10.0,
        (10, 1): 5.0, (20, 1): 3.0, (1, 1): 10.0,
    }
    st = _make_two_anchor_shared(areas)
    st._compute_per_sample_fractions()

    fa = st._contributions_per_sample[1]['A']
    fb = st._contributions_per_sample[1]['B']

    # Amostra 0: 0.8 + 0.8 = 1.6 → renormaliza para 0.5 + 0.5
    assert fa[0] + fb[0] == pytest.approx(1.0, abs=1e-6)
    assert fa[0] == pytest.approx(0.5, abs=1e-6)
    # Amostra 1: 0.5 + 0.3 = 0.8 ≤ 1 → mantém (resíduo = 0.2)
    assert fa[1] + fb[1] == pytest.approx(0.8, abs=1e-6)


# ───────────────────────────────────────────────────────────────────────────
# Fallback escalar quando obs ≤ 0
# ───────────────────────────────────────────────────────────────────────────

def test_degenerate_sample_falls_back_to_mean():
    # Amostra 1 tem obs=0 no pico partilhado → usa fração média (0.50)
    areas = {
        (10, 0): 6.0, (20, 0): 4.0, (1, 0): 10.0,
        (10, 1): 0.0, (20, 1): 0.0, (1, 1): 0.0,
    }
    st = _make_two_anchor_shared(areas)
    st._compute_per_sample_fractions()

    fa = st._contributions_per_sample[1]['A']
    fb = st._contributions_per_sample[1]['B']

    assert fa[1] == pytest.approx(0.5, abs=1e-6)   # média = 50%
    assert fb[1] == pytest.approx(0.5, abs=1e-6)
    assert not np.isnan(fa[1])


# ───────────────────────────────────────────────────────────────────────────
# Cascata por amostra (replay determinístico)
# ───────────────────────────────────────────────────────────────────────────

def test_cascade_replay_per_sample():
    # A âncora (rank10). Pico partilhado rank1: A conhecido, B resolvido por
    # cascata = (obs - conc_A·eff) / eff_B.
    st = _FakeStage(n_samples=2)
    met_a = {'id': 'A', 'name': 'A',
             'matched_peaks': [{'peak_rank': 10, '_eff': 1.0},
                               {'peak_rank': 1,  '_eff': 1.0}]}
    met_b = {'id': 'B', 'name': 'B',
             'matched_peaks': [{'peak_rank': 1, '_eff': 1.0}]}
    st._mets_map  = {'A': met_a, 'B': met_b}
    st._peaks_map = {1: {'rank': 1, 'ppm': 2.0}, 10: {'rank': 10, 'ppm': 1.0}}
    st._peak_overlap  = {1: ['A', 'B'], 10: ['A']}
    st._anchor_plan   = {'A': [10]}
    st._cascade_plan  = [(1, 'B', ['A'])]
    st._contributions = {1: {'A': 40.0, 'B': 60.0,
                             '_observed': 10.0, '_residual_pct': 60.0}}
    st._areas = {
        (10, 0): 4.0, (1, 0): 10.0,   # conc_A=4, B resid=6 → fB=0.6, fA=0.4
        (10, 1): 7.0, (1, 1): 10.0,   # conc_A=7, B resid=3 → fB=0.3, fA=0.7
    }
    st._compute_per_sample_fractions()

    fa = st._contributions_per_sample[1]['A']
    fb = st._contributions_per_sample[1]['B']
    assert fa[0] == pytest.approx(0.4, abs=1e-6)
    assert fb[0] == pytest.approx(0.6, abs=1e-6)
    assert fa[1] == pytest.approx(0.7, abs=1e-6)
    assert fb[1] == pytest.approx(0.3, abs=1e-6)
