# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_j_resolve_cascade.py
"""
Testes para a lógica de cascata de _phase3 em JResolveStage.
Testa o limite de iterações, o threshold de resíduo (2%), e a resolução de
um único desconhecido.
"""
import numpy as np
import pytest
from gui.stages.j_resolve_stage import JResolveStage


# ---------------------------------------------------------------------------
# Stub mínimo — evita instanciar JResolveStage com tkinter real
# ---------------------------------------------------------------------------

class _FakeStage:
    """Stub que expõe apenas o que _phase3 necessita."""

    def __init__(self):
        self._contributions  = {}
        self._peak_overlap   = {}
        self._conc_est       = {}
        self._anchor_peaks   = {}
        self._cascade_estimates = {}
        self._cascade_plan      = []
        self._log_lines      = []
        self._phase2_calls   = 0
        self._mets_map       = {}   # name → met dict
        self._peaks_map      = {}   # rank → peak dict

    def _log_line(self, msg=''):
        self._log_lines.append(str(msg))

    def _met_by_name(self, name):
        return self._mets_map.get(name)

    def _peak_by_rank(self, rank):
        return self._peaks_map.get(rank)

    def _get_area(self, peak):
        return float(peak.get('area', 0.0))

    def _effective_nh(self, met, mp):
        return float(mp.get('_eff_nh', 1.0))

    def _phase2(self, _cascade_iter=0):
        self._phase2_calls += 1


# Ligar _phase3 ao stub (método desvinculado → vinculado a instância fake)
_FakeStage._phase3 = JResolveStage._phase3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_met(name, mid=None):
    mid = mid or name
    return {'id': mid, 'name': name, 'matched_peaks': []}


def _make_stage_single_unknown(resid_pct=10.0):
    """Pico rank=1: A é âncora, B é o único desconhecido."""
    st = _FakeStage()
    met_a = _make_met('A')
    met_b = _make_met('B')
    mp_b  = {'peak_rank': 1, '_eff_nh': 1.0}
    met_b['matched_peaks'] = [mp_b]

    st._mets_map  = {'A': met_a, 'B': met_b}
    st._peaks_map = {1: {'rank': 1, 'ppm': 2.0, 'area': 10.0}}
    st._peak_overlap   = {1: ['A', 'B']}
    st._contributions  = {1: {'_residual_pct': resid_pct, 'A': 0.5, '_observed': 10.0}}
    st._conc_est       = {'A': 5.0}   # A já está ancorado
    return st


# ---------------------------------------------------------------------------
# Testes: limite de iterações
# ---------------------------------------------------------------------------

class TestCascadeMaxIterations:

    def _make_infinite_stage(self):
        """Stage onde _phase2 repõe B como desconhecido a cada iteração."""
        st = _FakeStage()
        met_a = _make_met('A')
        met_b = _make_met('B')
        met_b['matched_peaks'] = [{'peak_rank': 1, '_eff_nh': 1.0}]
        st._mets_map     = {'A': met_a, 'B': met_b}
        st._peaks_map    = {1: {'rank': 1, 'ppm': 2.0, 'area': 10.0}}
        st._peak_overlap = {1: ['A', 'B']}
        st._conc_est     = {'A': 5.0}
        # _contributions deve estar pré-populado — _phase3 itera sobre ele
        st._contributions = {1: {'_residual_pct': 10.0, 'A': 0.5, '_observed': 10.0}}

        def fake_phase2(_cascade_iter=0):
            st._phase2_calls += 1
            # Remove B → na iteração seguinte B volta a ser desconhecido
            st._conc_est.pop('B', None)
            st._contributions[1] = {'_residual_pct': 10.0, 'A': 0.5, '_observed': 10.0}

        st._phase2 = fake_phase2
        return st

    def test_limit_applied_on_infinite_work(self):
        """Loop para exactamente _MAX_CASCADE iterações quando há trabalho infinito."""
        st = self._make_infinite_stage()
        st._phase3()
        assert st._phase2_calls == 50, (
            f"Esperado 50 chamadas a _phase2, obtido {st._phase2_calls}")

    def test_warning_logged_when_limit_hit(self):
        """Aviso deve aparecer no log quando o limite é atingido."""
        st = self._make_infinite_stage()
        st._phase3()
        assert any('AVISO' in l for l in st._log_lines), (
            f"Esperado 'AVISO' no log. Log obtido: {st._log_lines}")

    def test_normal_convergence_no_warning(self):
        """Cascata que converge normalmente não deve gerar aviso."""
        st = _make_stage_single_unknown(resid_pct=10.0)
        st._phase3()

        assert not any('AVISO' in l for l in st._log_lines)
        assert st._phase2_calls == 1   # convergiu em 1 iteração


# ---------------------------------------------------------------------------
# Testes: threshold de resíduo 2%
# ---------------------------------------------------------------------------

class TestResidualThreshold:

    def test_peak_below_threshold_skipped(self):
        """Pico com resíduo < 2% deve ser ignorado pela cascata."""
        st = _make_stage_single_unknown(resid_pct=1.9)
        st._phase3()

        assert 'B' not in st._conc_est, (
            "B não devia ser resolvido (resíduo 1.9% < limiar 2%)")
        assert st._phase2_calls == 0

    def test_peak_at_threshold_processed(self):
        """Pico com resíduo ≥ 2% deve ser processado."""
        st = _make_stage_single_unknown(resid_pct=2.0)
        st._phase3()

        assert 'B' in st._conc_est, "B devia ser resolvido (resíduo = 2%)"

    def test_peak_above_threshold_processed(self):
        """Pico com resíduo claramente acima do limiar deve ser resolvido."""
        st = _make_stage_single_unknown(resid_pct=50.0)
        st._phase3()

        assert 'B' in st._conc_est

    def test_conc_est_plausible(self):
        """Estimativa de concentração deve ser positiva e plausível."""
        st = _make_stage_single_unknown(resid_pct=50.0)
        # area=10, A contribui 5 (conc_A=5, eff_A=1) → residual=5 → B_conc=5
        st._phase3()

        assert st._conc_est.get('B', 0) > 0

    def test_multiple_unknowns_not_resolved(self):
        """Pico com 2 desconhecidos não pode ser resolvido em cascata."""
        st = _FakeStage()
        met_a = _make_met('A')
        met_b = _make_met('B')
        met_c = _make_met('C')
        met_b['matched_peaks'] = [{'peak_rank': 1, '_eff_nh': 1.0}]
        met_c['matched_peaks'] = [{'peak_rank': 1, '_eff_nh': 1.0}]
        st._mets_map      = {'A': met_a, 'B': met_b, 'C': met_c}
        st._peaks_map     = {1: {'rank': 1, 'ppm': 2.0, 'area': 10.0}}
        st._peak_overlap  = {1: ['A', 'B', 'C']}
        st._contributions = {1: {'_residual_pct': 80.0, 'A': 0.2, '_observed': 10.0}}
        st._conc_est      = {'A': 2.0}   # só A está ancorado; B e C são desconhecidos

        st._phase3()

        assert 'B' not in st._conc_est
        assert 'C' not in st._conc_est
