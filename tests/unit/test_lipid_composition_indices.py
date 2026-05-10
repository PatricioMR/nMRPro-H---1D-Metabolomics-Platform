# tests/unit/test_lipid_composition_indices.py
"""Unit tests for _compute_lipid_composition_indices (Li et al. 2017, Box 1)."""
import numpy as np
import pytest

from gui.stages._quant_computation import _QuantCompMixin


class _Stub(_QuantCompMixin):
    """Minimal stub — _QuantCompMixin requires no __init__ state for this method."""


STUB = _Stub()

# ── helpers ───────────────────────────────────────────────────────────────────

def _flat_spectrum(ppm, lo, hi, height=1.0):
    """Return intensity array with a rectangular peak between lo and hi."""
    y = np.zeros_like(ppm)
    y[(ppm >= lo) & (ppm <= hi)] = height
    return y


def _build_inputs(windows_on, n_s=3, height=1.0, ppm_range=(0.5, 6.0), n_pts=2000):
    """
    Build (ppm_arr, all_int, n_s, groups) where only windows listed in
    `windows_on` have signal (rectangular pulses of given height).
    """
    _W = {
        'a': (5.25, 5.45), 'b': (4.05, 4.40), 'c': (2.72, 2.88),
        'd': (2.22, 2.38), 'e': (1.97, 2.07), 'h': (0.84, 0.92),
        'i': (0.95, 1.01),
    }
    ppm = np.linspace(ppm_range[1], ppm_range[0], n_pts)  # descending (NMR convention)
    intensity = np.zeros((n_pts, n_s))
    for key in windows_on:
        lo, hi = _W[key]
        col = _flat_spectrum(ppm, lo, hi, height)
        for s in range(n_s):
            intensity[:, s] += col
    groups = {'A': list(range(n_s))}
    return ppm, intensity, n_s, groups


# ── output structure ──────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_returns_six_entries(self):
        ppm, intensity, n_s, groups = _build_inputs(['h', 'd', 'b'])
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        assert len(out) == 6

    def test_all_marked_lipid_index(self):
        ppm, intensity, n_s, groups = _build_inputs(['h', 'd', 'b'])
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        assert all(r['_is_lipid_index'] for r in out)

    def test_concs_per_sample_length(self):
        ppm, intensity, n_s, groups = _build_inputs(['h', 'd', 'b'], n_s=5)
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        assert all(len(r['_concs_per_sample']) == 5 for r in out)

    def test_status_is_dash(self):
        ppm, intensity, n_s, groups = _build_inputs(['h', 'd', 'b'])
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        assert all(r['status'] == '—' for r in out)

    def test_hmdb_id_is_sentinel(self):
        ppm, intensity, n_s, groups = _build_inputs(['h', 'd', 'b'])
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        assert all(r['hmdb_id'] == '__LIPID_INDEX__' for r in out)

    def test_integ_mode_label(self):
        ppm, intensity, n_s, groups = _build_inputs(['h', 'd', 'b'])
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        assert all(r['_integ_mode'] == 'LipidIndex' for r in out)


# ── zero-signal edge case ─────────────────────────────────────────────────────

class TestZeroSignal:
    def test_all_zeros_gives_zero_indices(self):
        # When U=0 (no detectable lipid), all composition indices must be 0
        # (sat_pct guarded by where(U>eps) to avoid spurious 100% saturated)
        ppm = np.linspace(6.0, 0.5, 1000)
        intensity = np.zeros((1000, 2))
        groups = {'A': [0, 1]}
        out = STUB._compute_lipid_composition_indices(ppm, intensity, 2, groups)
        idx = {r['metabolite']: r for r in out}
        pct_names = [
            'n-3 PUFA — ALA index (%)',
            'n-6 PUFA — bis-allylic index (%)',
            'MUFA — allylic index (%)',
            'Saturated FA (%)',
            'Degree of Unsaturation (C=C/chain)',
            'Unsaturation Index (UI)',
        ]
        for name in pct_names:
            for v in idx[name]['_concs_per_sample']:
                assert v == pytest.approx(0.0, abs=1e-9), f"{name} should be 0 on zero signal"

    def test_no_nan_in_output_on_zero_signal(self):
        ppm = np.linspace(6.0, 0.5, 1000)
        intensity = np.zeros((1000, 2))
        groups = {'A': [0, 1]}
        out = STUB._compute_lipid_composition_indices(ppm, intensity, 2, groups)
        for r in out:
            assert all(np.isfinite(v) for v in r['_concs_per_sample'])


# ── sat-only FA scenario ──────────────────────────────────────────────────────

class TestSaturatedOnly:
    """No olefinic, no bis-allylic, no allylic signal → 100 % saturated."""

    def setup_method(self):
        # Only h (terminal CH3), d (alpha-CH2), b (glycerol) present — no unsaturation
        ppm, intensity, n_s, groups = _build_inputs(['h', 'd', 'b'], height=10.0)
        self.out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        self._idx = {r['metabolite']: r for r in self.out}

    def test_n3_is_zero(self):
        vals = self._idx['n-3 PUFA — ALA index (%)']['_concs_per_sample']
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in vals)

    def test_n6_is_zero(self):
        vals = self._idx['n-6 PUFA — bis-allylic index (%)']['_concs_per_sample']
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in vals)

    def test_mufa_is_zero(self):
        vals = self._idx['MUFA — allylic index (%)']['_concs_per_sample']
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in vals)

    def test_sat_is_100(self):
        vals = self._idx['Saturated FA (%)']['_concs_per_sample']
        assert all(v == pytest.approx(100.0, abs=1e-3) for v in vals)

    def test_du_is_zero(self):
        vals = self._idx['Degree of Unsaturation (C=C/chain)']['_concs_per_sample']
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in vals)

    def test_ui_is_zero(self):
        vals = self._idx['Unsaturation Index (UI)']['_concs_per_sample']
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in vals)


# ── percentages sum to 100 ────────────────────────────────────────────────────

class TestPercentagesSum:
    """n3% + n6% + MUFA% + Sat% must equal 100 for any nonzero spectrum."""

    def _sum_pct(self, windows_on, height=5.0):
        ppm, intensity, n_s, groups = _build_inputs(windows_on, height=height)
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        idx = {r['metabolite']: r for r in out}
        n3  = np.array(idx['n-3 PUFA — ALA index (%)']['_concs_per_sample'])
        n6  = np.array(idx['n-6 PUFA — bis-allylic index (%)']['_concs_per_sample'])
        mu  = np.array(idx['MUFA — allylic index (%)']['_concs_per_sample'])
        sat = np.array(idx['Saturated FA (%)']['_concs_per_sample'])
        return n3 + n6 + mu + sat

    def test_sat_only(self):
        total = self._sum_pct(['h', 'd', 'b'])
        np.testing.assert_allclose(total, 100.0, atol=1e-3)

    def test_mixed_unsaturation_controlled(self):
        # Use b+d+h only (sat baseline) then add small allylic/bis-allylic
        # such that n3+n6+MUFA < 100 and sat fills the remainder.
        ppm = np.linspace(6.0, 0.5, 2000)
        col = np.zeros_like(ppm)
        # b: glycerol (large — establishes U)
        col[(ppm >= 4.05) & (ppm <= 4.40)] = 20.0
        # d: alpha-CH2
        col[(ppm >= 2.22) & (ppm <= 2.38)] = 12.0
        # h: terminal CH3
        col[(ppm >= 0.84) & (ppm <= 0.92)] = 10.0
        # small n3 (i) signal
        col[(ppm >= 0.95) & (ppm <= 1.01)] = 0.5
        # small bis-allylic (c) — less than 2*(i/9)*2 to stay sub-100%
        col[(ppm >= 2.72) & (ppm <= 2.88)] = 0.3
        # small allylic (e)
        col[(ppm >= 1.97) & (ppm <= 2.07)] = 0.4
        intensity = np.stack([col, col, col], axis=1)
        groups = {'A': [0, 1, 2]}
        out = STUB._compute_lipid_composition_indices(ppm, intensity, 3, groups)
        idx = {r['metabolite']: r for r in out}
        n3  = np.array(idx['n-3 PUFA — ALA index (%)']['_concs_per_sample'])
        n6  = np.array(idx['n-6 PUFA — bis-allylic index (%)']['_concs_per_sample'])
        mu  = np.array(idx['MUFA — allylic index (%)']['_concs_per_sample'])
        sat = np.array(idx['Saturated FA (%)']['_concs_per_sample'])
        np.testing.assert_allclose(n3 + n6 + mu + sat, 100.0, atol=1e-3)

    def test_only_n3(self):
        # i present (n3) but no n6 bis-allylic, no allylic → n3+sat=100
        total = self._sum_pct(['b', 'd', 'h', 'i'])
        np.testing.assert_allclose(total, 100.0, atol=1e-3)


# ── values clipped to valid range ─────────────────────────────────────────────

class TestClipping:
    def test_percentages_nonnegative(self):
        ppm, intensity, n_s, groups = _build_inputs(
            ['a', 'b', 'c', 'd', 'e', 'h', 'i'], height=100.0)
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        pct_names = [
            'n-3 PUFA — ALA index (%)',
            'n-6 PUFA — bis-allylic index (%)',
            'MUFA — allylic index (%)',
            'Saturated FA (%)',
        ]
        idx = {r['metabolite']: r for r in out}
        for name in pct_names:
            for v in idx[name]['_concs_per_sample']:
                assert v >= -1e-9, f"{name} went negative: {v}"

    def test_du_nonnegative(self):
        ppm, intensity, n_s, groups = _build_inputs(['a', 'b', 'd', 'h'])
        out = STUB._compute_lipid_composition_indices(ppm, intensity, n_s, groups)
        idx = {r['metabolite']: r for r in out}
        for v in idx['Degree of Unsaturation (C=C/chain)']['_concs_per_sample']:
            assert v >= 0.0


# ── sample-wise consistency ───────────────────────────────────────────────────

class TestSampleConsistency:
    def test_identical_samples_give_same_result(self):
        ppm = np.linspace(6.0, 0.5, 1000)
        col = _flat_spectrum(ppm, 0.84, 0.92, 5.0)  # only h
        col += _flat_spectrum(ppm, 2.22, 2.38, 3.0)  # d
        col += _flat_spectrum(ppm, 4.05, 4.40, 2.0)  # b
        intensity = np.stack([col, col, col], axis=1)
        groups = {'A': [0, 1, 2]}
        out = STUB._compute_lipid_composition_indices(ppm, intensity, 3, groups)
        for r in out:
            vals = r['_concs_per_sample']
            assert vals[0] == pytest.approx(vals[1], rel=1e-9)
            assert vals[0] == pytest.approx(vals[2], rel=1e-9)
