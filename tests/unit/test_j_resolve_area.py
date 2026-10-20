# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_j_resolve_area.py
"""
Testes para _get_area e _effective_nh em JResolveStage.
Valida o cálculo de área (baseline-corrected vs raw) e a integração eff_nH.
"""
import numpy as np
import pytest
from gui.stages.j_resolve_stage import JResolveStage


# ---------------------------------------------------------------------------
# Stub mínimo
# ---------------------------------------------------------------------------

class _FakeStage:
    def __init__(self, ppm, mean_int):
        self._ppm      = np.asarray(ppm, dtype=float)
        self._mean_int = np.asarray(mean_int, dtype=float)
        self.app = None

    def _get_protons(self, met, mp):
        return int(mp.get('protons', 1))


_FakeStage._get_area      = JResolveStage._get_area
_FakeStage._effective_nh  = JResolveStage._effective_nh
_FakeStage._get_mhz       = JResolveStage._get_mhz


# ---------------------------------------------------------------------------
# _get_area
# ---------------------------------------------------------------------------

class TestGetArea:

    def _make_ppm_int(self, lo=1.0, hi=2.0, n=100, baseline=0.0, peak_amp=10.0):
        ppm = np.linspace(hi, lo, n)   # descendente (convenção NMR)
        centre = (lo + hi) / 2
        sigma  = (hi - lo) * 0.1
        intensity = peak_amp * np.exp(-0.5 * ((ppm - centre) / sigma)**2) + baseline
        return ppm, intensity

    def test_positive_area(self):
        ppm, intensity = self._make_ppm_int(baseline=0.0)
        st   = _FakeStage(ppm, intensity)
        peak = {'ppm': 1.5, 'start_ppm': 1.2, 'end_ppm': 1.8}
        area = st._get_area(peak)
        assert area > 0

    def test_baseline_offset_inflates_raw_area(self):
        """Área com baseline elevada deve ser maior que sem baseline."""
        ppm, int_no_bl = self._make_ppm_int(baseline=0.0, peak_amp=10.0)
        ppm, int_bl    = self._make_ppm_int(baseline=5.0, peak_amp=10.0)
        peak = {'ppm': 1.5, 'start_ppm': 1.1, 'end_ppm': 1.9}

        area_no_bl = _FakeStage(ppm, int_no_bl)._get_area(peak)
        area_bl    = _FakeStage(ppm, int_bl)._get_area(peak)

        assert area_bl > area_no_bl, (
            "Área com baseline elevada deve ser maior — J-Resolve deve usar "
            "a média com baseline já corrigida (processor.mean_intensity)")

    def test_descending_ppm_handled(self):
        """ppm descendente (convenção NMR) não deve produzir área negativa."""
        ppm = np.linspace(3.0, 1.0, 200)   # descendente
        intensity = np.ones(200) * 5.0
        st   = _FakeStage(ppm, intensity)
        peak = {'ppm': 2.0, 'start_ppm': 1.8, 'end_ppm': 2.2}
        area = st._get_area(peak)
        assert area >= 0

    def test_narrow_window_fallback(self):
        """start_ppm == end_ppm deve usar janela ±0.01 ppm de fallback."""
        ppm = np.linspace(2.02, 1.98, 100)
        intensity = np.ones(100) * 3.0
        st   = _FakeStage(ppm, intensity)
        peak = {'ppm': 2.0, 'start_ppm': 2.0, 'end_ppm': 2.0}
        area = st._get_area(peak)
        assert area >= 0   # não deve lançar excepção

    def test_no_spectrum_returns_zero(self):
        """Sem espectro, _get_area deve retornar 0."""
        st = _FakeStage.__new__(_FakeStage)
        st._ppm      = None
        st._mean_int = None
        peak = {'ppm': 2.0, 'start_ppm': 1.9, 'end_ppm': 2.1}
        area = st._get_area(peak)
        assert area == 0.0


# ---------------------------------------------------------------------------
# _effective_nh — integração sobre janela real do pico
# ---------------------------------------------------------------------------

class TestEffectiveNH:

    def _make_app(self):
        import types
        app = types.SimpleNamespace(spectrometer_mhz=600.0)
        return app

    def _make_stage(self):
        st = _FakeStage.__new__(_FakeStage)
        st._ppm      = np.linspace(4.0, 0.0, 400)
        st._mean_int = np.zeros(400)
        st.app       = self._make_app()
        return st

    def test_singlet_returns_n_protons(self):
        """Singlet sem predicted_lines → eff_nH = n_protons."""
        st  = self._make_stage()
        met = {'id': 'A', 'name': 'A', 'matched_peaks': []}
        mp  = {'protons': 3, 'type': 's'}
        eff = st._effective_nh(met, mp)
        assert eff == pytest.approx(3.0)

    def test_doublet_point_estimate_fallback(self):
        """Doublet sem start/end_ppm usa ponto-estimativa (Pascal [1,1])."""
        st  = self._make_stage()
        met = {'id': 'A', 'name': 'A', 'matched_peaks': []}
        mp  = {
            'protons': 2, 'type': 'd',
            'j_coupling': '7.0',
            'ppm_center': 2.0,
            'peak_ppm': 2.003,   # próximo da linha superior
        }
        eff = st._effective_nh(met, mp)
        # Pascal [1,1]: uma linha → 2 * 1/2 = 1.0
        assert 0.5 <= eff <= 2.0, f"eff_nH={eff} fora do intervalo razoável para doublet 2H"

    def test_doublet_integration_with_window(self):
        """Com start/end_ppm, doublet usa integração → fracção entre 0 e n_protons."""
        st  = self._make_stage()
        met = {'id': 'A', 'name': 'A', 'matched_peaks': []}
        mp  = {
            'protons': 3,
            'predicted_lines': [
                {'ppm': 1.98, 'rel_intensity': 1.0},
                {'ppm': 2.02, 'rel_intensity': 1.0},
            ],
            'start_ppm': 1.97,   # janela cobre quase só a linha em 1.98
            'end_ppm':   2.00,
        }
        eff = st._effective_nh(met, mp)
        # Janela cobre ~metade do doublet → ~50% do sinal total → eff ≈ 1.5
        assert 0.5 < eff < 2.5, f"eff_nH={eff:.3f} inesperado para meia-janela do doublet"

    def test_integration_covers_full_doublet(self):
        """Janela que cobre o doublet inteiro deve retornar ~n_protons."""
        st  = self._make_stage()
        met = {'id': 'A', 'name': 'A', 'matched_peaks': []}
        mp  = {
            'protons': 2,
            'predicted_lines': [
                {'ppm': 1.96, 'rel_intensity': 1.0},
                {'ppm': 2.04, 'rel_intensity': 1.0},
            ],
            'start_ppm': 1.90,   # janela larga cobre ambas as linhas
            'end_ppm':   2.10,
        }
        eff = st._effective_nh(met, mp)
        # Janela cobre >95% do doublet → eff ≈ 2.0
        assert eff == pytest.approx(2.0, abs=0.2), f"eff_nH={eff:.3f} para doublet 2H completo"

    def test_integration_non_negative(self):
        """eff_nH nunca deve ser negativo."""
        st  = self._make_stage()
        met = {'id': 'A', 'name': 'A', 'matched_peaks': []}
        mp  = {
            'protons': 3,
            'predicted_lines': [
                {'ppm': 2.00, 'rel_intensity': 1.0},
                {'ppm': 2.03, 'rel_intensity': 2.0},
                {'ppm': 2.06, 'rel_intensity': 1.0},
            ],
            'start_ppm': 1.95,
            'end_ppm':   2.05,
        }
        eff = st._effective_nh(met, mp)
        assert eff >= 0, f"eff_nH negativo: {eff}"
