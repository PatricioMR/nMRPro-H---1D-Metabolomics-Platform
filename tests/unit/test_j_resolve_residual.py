# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_j_resolve_residual.py
"""
Testes para _phase_residual (atribuição do residual gated-por-evidência).

Cobre:
  • E2 (J-pattern) → atribui o único candidato corroborado.
  • E3 (consistência multi-pico) → atribui.
  • IN1: ≥2 candidatos corroborados → NÃO atribui (não divide arbitrariamente).
  • IN3: residual < 2% → NÃO atribui (banda de ruído).
  • Sem evidência → NÃO atribui; diagnóstico registado (camada A).
"""
import numpy as np
import pytest
from gui.stages.j_resolve_stage import JResolveStage, _FOLDRANGE_MAX


class _DJVar:
    def __init__(self, v): self._v = v
    def get(self): return self._v


class _ResidualStage:
    def __init__(self, dj_tol=0.5):
        self._metabolites = []
        self._peak_overlap = {}
        self._peaks_map = {}
        self._conc_est = {}
        self._anchor_peaks = {}
        self._cascade_plan = []
        self._contributions = {}
        self._residual_diagnostic = {}
        self._residual_attributed = {}
        self._log_lines = []
        self._dj_tol = _DJVar(dj_tol)

    def _log_line(self, msg=''):
        self._log_lines.append(str(msg))

    def _met_by_name(self, name):
        return next((m for m in self._metabolites if m['name'] == name), None)

    def _met_by_id(self, mid):
        return next((m for m in self._metabolites if m['id'] == mid), None)

    def _peak_by_rank(self, rank):
        return self._peaks_map.get(rank)

    def _get_area(self, peak):
        return float(peak.get('area', 0.0))

    def _effective_nh(self, met, mp):
        return float((mp or {}).get('_eff', 1.0))

    def _phase2(self, _cascade_iter=None):
        # Minimal recompute: residual for each overlapped peak from conc_est
        for rank, names in self._peak_overlap.items():
            if len(names) <= 1:
                continue
            pk = self._peaks_map.get(rank)
            obs = float(pk.get('area', 0.0)) if pk else 0.0
            if obs <= 0:
                continue
            known = 0.0
            for n in names:
                m = self._met_by_name(n)
                if m is None or m['id'] not in self._conc_est:
                    continue
                mp = next((x for x in m['matched_peaks']
                           if x.get('peak_rank') == rank), None)
                known += self._conc_est[m['id']] * float((mp or {}).get('_eff', 1.0))
            resid = max(0.0, obs - known)
            self._contributions[rank] = {
                '_residual_pct': resid / obs * 100, '_observed': obs}


_ResidualStage._phase_residual = JResolveStage._phase_residual
_ResidualStage._residual_implied_concs = JResolveStage._residual_implied_concs


def _mp(rank, eff=1.0, dj=None):
    return {'peak_rank': rank, '_eff': eff, 'delta_j_hz': dj}


# ── E2: candidato J-confirmado único ─────────────────────────────────────────

def test_e2_attributes_single_j_confirmed():
    st = _ResidualStage(dj_tol=0.5)
    # A ancorado (conc=4). Pico 1 partilhado A+B, obs=10 → residual=6.
    # B tem ΔJ=0.2 ≤ 0.5 nesse pico (E2).
    met_a = {'id': 'A', 'name': 'A',
             'matched_peaks': [_mp(1, 1.0)]}
    met_b = {'id': 'B', 'name': 'B',
             'matched_peaks': [_mp(1, 1.0, dj=0.2)]}
    st._metabolites = [met_a, met_b]
    st._peaks_map = {1: {'rank': 1, 'ppm': 2.0, 'area': 10.0}}
    st._peak_overlap = {1: ['A', 'B']}
    st._conc_est = {'A': 4.0}
    st._contributions = {1: {'_residual_pct': 60.0, '_observed': 10.0}}

    st._phase_residual()

    assert 'B' in st._conc_est
    assert st._conc_est['B'] == pytest.approx(6.0)         # residual/eff
    assert 'E2' in st._residual_attributed['B']


# ── E3: consistência multi-pico ──────────────────────────────────────────────

def test_e3_attributes_consistent_multipeak():
    st = _ResidualStage(dj_tol=0.5)
    # B singleto (sem J) em 2 picos partilhados; residual-implied conc consistente.
    # Pico1: A(conc2,eff1) obs=5 → resid 3 → B eff1 → 3.  Pico2: C(conc1,eff1) obs=4 → resid 3 → B eff1 → 3.
    met_a = {'id': 'A', 'name': 'A', 'matched_peaks': [_mp(1, 1.0)]}
    met_c = {'id': 'C', 'name': 'C', 'matched_peaks': [_mp(2, 1.0)]}
    met_b = {'id': 'B', 'name': 'B',
             'matched_peaks': [_mp(1, 1.0), _mp(2, 1.0)]}
    st._metabolites = [met_a, met_c, met_b]
    st._peaks_map = {1: {'rank': 1, 'ppm': 2.0, 'area': 5.0},
                     2: {'rank': 2, 'ppm': 3.0, 'area': 4.0}}
    st._peak_overlap = {1: ['A', 'B'], 2: ['C', 'B']}
    st._conc_est = {'A': 2.0, 'C': 1.0}
    st._contributions = {1: {'_residual_pct': 60.0, '_observed': 5.0},
                         2: {'_residual_pct': 75.0, '_observed': 4.0}}

    st._phase_residual()

    assert 'B' in st._conc_est
    assert 'E3' in st._residual_attributed['B']


# ── IN1: dois candidatos corroborados → não divide ───────────────────────────

def test_in1_two_evidenced_not_split():
    st = _ResidualStage(dj_tol=0.5)
    # Pico 1 partilhado por B e C, ambos com ΔJ ok (E2). Nenhum ancorado.
    met_b = {'id': 'B', 'name': 'B', 'matched_peaks': [_mp(1, 1.0, dj=0.1)]}
    met_c = {'id': 'C', 'name': 'C', 'matched_peaks': [_mp(1, 1.0, dj=0.1)]}
    st._metabolites = [met_b, met_c]
    st._peaks_map = {1: {'rank': 1, 'ppm': 2.0, 'area': 10.0}}
    st._peak_overlap = {1: ['B', 'C']}
    st._contributions = {1: {'_residual_pct': 100.0, '_observed': 10.0}}

    st._phase_residual()

    assert 'B' not in st._conc_est
    assert 'C' not in st._conc_est
    assert any('not split' in l for l in st._log_lines)


# ── IN3: residual dentro do ruído (<2%) ──────────────────────────────────────

def test_in3_low_residual_not_attributed():
    st = _ResidualStage(dj_tol=0.5)
    met_a = {'id': 'A', 'name': 'A', 'matched_peaks': [_mp(1, 1.0)]}
    met_b = {'id': 'B', 'name': 'B', 'matched_peaks': [_mp(1, 1.0, dj=0.1)]}
    st._metabolites = [met_a, met_b]
    st._peaks_map = {1: {'rank': 1, 'ppm': 2.0, 'area': 10.0}}
    st._peak_overlap = {1: ['A', 'B']}
    st._conc_est = {'A': 9.9}
    st._contributions = {1: {'_residual_pct': 1.0, '_observed': 10.0}}  # <2%

    st._phase_residual()

    assert 'B' not in st._conc_est


# ── Sem evidência → diagnóstico mas sem atribuição (camada A) ─────────────────

def test_no_evidence_records_diagnostic_only():
    st = _ResidualStage(dj_tol=0.5)
    # B singleto sem J e só num pico (sem E2 nem E3).
    met_a = {'id': 'A', 'name': 'A', 'matched_peaks': [_mp(1, 1.0)]}
    met_b = {'id': 'B', 'name': 'B', 'matched_peaks': [_mp(1, 1.0, dj=None)]}
    st._metabolites = [met_a, met_b]
    st._peaks_map = {1: {'rank': 1, 'ppm': 2.0, 'area': 10.0}}
    st._peak_overlap = {1: ['A', 'B']}
    st._conc_est = {'A': 4.0}
    st._contributions = {1: {'_residual_pct': 60.0, '_observed': 10.0}}

    st._phase_residual()

    assert 'B' not in st._conc_est
    assert 1 in st._residual_diagnostic
    assert any(d['name'] == 'B' and not d['attributed']
               for d in st._residual_diagnostic[1])
