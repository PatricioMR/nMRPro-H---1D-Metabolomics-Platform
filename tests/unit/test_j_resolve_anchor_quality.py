# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_j_resolve_anchor_quality.py
"""
Testes para os gates de fiabilidade de âncora de JResolveStage._commit_anchor.

Cobre:
  • Rejeição de probe único com eff_nH baixo (linha fraca de multipleto).
  • Probe único forte → confiança 'single'.
  • ≥2 probes consistentes → 'high'; inconsistentes (CV alto) → 'low'.
  • Probe único de eff baixo é resgatado se houver ≥2 probes.
"""
import numpy as np
import pytest
from gui.stages.j_resolve_stage import (
    JResolveStage, _MIN_SINGLE_PROBE_EFF, _FOLDRANGE_MAX)


class _FakeStage:
    def __init__(self):
        self._conc_est          = {}
        self._anchor_peaks      = {}
        self._auto_anchored     = set()
        self._anchor_confidence = {}
        self._anchor_foldrange  = {}
        self._anchor_ci         = {}
        self._log_lines         = []

    def _log_line(self, msg=''):
        self._log_lines.append(str(msg))


_FakeStage._commit_anchor = JResolveStage._commit_anchor


def _stage():
    return _FakeStage()


# ── Probe único de eff baixo → rejeitado ─────────────────────────────────────

def test_single_low_eff_probe_rejected():
    st = _stage()
    eff = _MIN_SINGLE_PROBE_EFF - 0.05
    ok = st._commit_anchor('M', 'Met', [100.0], [1], [eff])
    assert ok is False
    assert 'M' not in st._conc_est
    assert any('REJECT' in l for l in st._log_lines)


def test_single_high_eff_probe_committed():
    st = _stage()
    ok = st._commit_anchor('M', 'Met', [100.0], [1], [1.0])
    assert ok is True
    assert st._conc_est['M'] == pytest.approx(100.0)
    assert st._anchor_confidence['M'] == 'single'


# ── Multi-probe: consistência via CV ─────────────────────────────────────────

def test_two_consistent_probes_high_confidence():
    st = _stage()
    ok = st._commit_anchor('M', 'Met', [100.0, 104.0], [1, 2], [1.0, 1.0])
    assert ok is True
    assert st._anchor_confidence['M'] == 'high'
    assert st._anchor_foldrange['M'] <= _FOLDRANGE_MAX   # 1.04 ≤ 2.5
    assert st._conc_est['M'] == pytest.approx(102.0)      # mediana


def test_two_inconsistent_probes_low_confidence():
    st = _stage()
    # 50 vs 200 → fold-range = 4.0 ≫ 2.5
    ok = st._commit_anchor('M', 'Met', [50.0, 200.0], [1, 2], [1.0, 1.0])
    assert ok is True
    assert st._anchor_confidence['M'] == 'low'
    assert st._anchor_foldrange['M'] > _FOLDRANGE_MAX
    assert any('LOW-CONFIDENCE' in l for l in st._log_lines)


def test_low_eff_rescued_by_multiple_probes():
    st = _stage()
    low = _MIN_SINGLE_PROBE_EFF - 0.1
    # Dois probes (um de eff baixo) → o gate de probe-único não se aplica
    ok = st._commit_anchor('M', 'Met', [100.0, 110.0], [1, 2], [low, 1.0])
    assert ok is True
    assert 'M' in st._conc_est


# ── auto flag ────────────────────────────────────────────────────────────────

def test_auto_flag_marks_auto_anchored():
    st = _stage()
    st._commit_anchor('M', 'Met', [100.0], [1], [1.0], auto=True)
    assert 'M' in st._auto_anchored


def test_empty_probes_returns_false():
    st = _stage()
    assert st._commit_anchor('M', 'Met', [], [], []) is False


# ── Fase 1: âncora só de picos únicos (bypass removido) ──────────────────────

class _Phase1Stage(_FakeStage):
    """Stub para exercitar _phase1 (anchoring só de picos únicos)."""

    def __init__(self):
        super().__init__()
        self._metabolites = []
        self._peak_overlap = {}
        self._peaks_map = {}

    def _peak_by_rank(self, rank):
        return self._peaks_map.get(rank)

    def _get_area(self, peak):
        return float(peak.get('area', 0.0))

    def _effective_nh(self, met, mp):
        return float(mp.get('_eff', 1.0))

    def _get_protons(self, met, mp):
        return int(mp.get('protons', 1) or 1)


_Phase1Stage._phase1 = JResolveStage._phase1


def test_phase1_does_not_anchor_from_overlapped_peak():
    """Pico Good sobreposto com anchor_expected_I NÃO deve ancorar (bypass removido)."""
    st = _Phase1Stage()
    # A: único pico Good está sobreposto com B; carrega os campos do antigo bypass
    met_a = {'id': 'A', 'name': 'A', 'matched_peaks': [
        {'peak_rank': 1, 'annotation_quality': 'good', '_eff': 1.0,
         'protons': 1, 'anchor_expected_I': 69.37,
         'anchor_status': 'clean', 'j_met_confirmed': True}]}
    met_b = {'id': 'B', 'name': 'B', 'matched_peaks': [
        {'peak_rank': 1, 'annotation_quality': 'good', '_eff': 1.0,
         'protons': 1}]}
    st._metabolites = [met_a, met_b]
    st._peaks_map   = {1: {'rank': 1, 'ppm': 8.46, 'area': 100.0}}
    st._peak_overlap = {1: ['A', 'B']}

    st._phase1()

    # Nenhum dos dois pode ancorar a partir do pico partilhado
    assert 'A' not in st._conc_est
    assert 'B' not in st._conc_est
    assert any('SKIP_OVL' in l for l in st._log_lines)


def test_phase1_anchors_from_unique_peak():
    """Pico Good único (sem competidores) ancora normalmente."""
    st = _Phase1Stage()
    met_a = {'id': 'A', 'name': 'A', 'matched_peaks': [
        {'peak_rank': 5, 'annotation_quality': 'good', '_eff': 1.0,
         'protons': 1}]}
    st._metabolites = [met_a]
    st._peaks_map   = {5: {'rank': 5, 'ppm': 1.33, 'area': 250.0}}
    st._peak_overlap = {5: ['A']}   # único

    st._phase1()

    assert st._conc_est['A'] == pytest.approx(250.0)
    assert st._anchor_peaks['A'] == [5]
