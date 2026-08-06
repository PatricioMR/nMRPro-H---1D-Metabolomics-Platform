# tests/unit/test_mcr_subpeak_quant.py
"""
Testes para MCRWindow._build_mcr_subpeak / _sync_peak_list_from_mcr /
_remap_peak_ranks — sem Tkinter (métodos puros sobre dicts/arrays,
instanciados via object.__new__ para não passar pelo __init__ que cria
janelas).

Cobre: (1) sub-peak aceito carrega start_idx/end_idx corretos e
area_per_sample = integral por amostra de S[:,k]*C[:,k] (o que a
quantificação vai consumir diretamente em vez de reintegrar o bruto);
(2) renumeração de rank ao aceitar um overlap remapeia _quality_marks/
_anchor_lines/_subline_quality_marks/_last_results_args/_quant_peak_overrides
de peaks NÃO envolvidos na divisão, que só tiveram o rank deslocado.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gui.widgets.mcr_window import MCRWindow


def _bare_window():
    """MCRWindow instance bypassing __init__ (no Tk widgets touched)."""
    return object.__new__(MCRWindow)


class TestBuildMCRSubpeak:

    def setup_method(self):
        self.ppm_reg = np.linspace(1.8, 2.2, 60)
        rng = np.random.default_rng(0)
        self.n_samples = 4
        # Component k=0: Gaussian bump around 1.95 ppm, distinct per-sample
        # concentrations so area_per_sample must vary sample to sample.
        s0 = np.exp(-0.5 * ((self.ppm_reg - 1.95) / 0.02) ** 2)
        s1 = np.exp(-0.5 * ((self.ppm_reg - 2.05) / 0.02) ** 2)
        self.S = np.column_stack([s0, s1])
        self.C = np.array([[1.0, 0.5],
                           [2.0, 0.5],
                           [0.5, 0.5],
                           [3.0, 0.5]])  # (n_samples, K)

        self.peak = {
            'rank': 5, 'ppm': 2.0, 'snr': 10.0, 'height': 1.0, 'status': None,
            'mcr_annotation': {
                'S': self.S, 'C': self.C, 'ppm_region': self.ppm_reg,
            },
        }
        self.win = _bare_window()
        # Full spectrum axis, wide enough to contain the region with margin.
        self.ppm_full = np.linspace(0.0, 5.0, 5001)
        self.win.app = SimpleNamespace(ppm=self.ppm_full)

    def test_start_end_idx_bracket_the_region(self):
        sp = self.win._build_mcr_subpeak(self.peak, 0, {'name': 'MetA'})
        assert sp is not None
        assert 0 <= sp['start_idx'] < sp['end_idx'] < len(self.ppm_full)
        # Indices must correspond to the nearest points to start_ppm/end_ppm.
        lo_expected = int(np.argmin(np.abs(self.ppm_full - min(sp['start_ppm'], sp['end_ppm']))))
        hi_expected = int(np.argmin(np.abs(self.ppm_full - max(sp['start_ppm'], sp['end_ppm']))))
        assert sp['start_idx'] == lo_expected
        assert sp['end_idx'] == hi_expected

    def test_area_per_sample_matches_manual_integration(self):
        sp = self.win._build_mcr_subpeak(self.peak, 0, {'name': 'MetA'})
        assert 'area_per_sample' in sp
        assert len(sp['area_per_sample']) == self.n_samples
        for j in range(self.n_samples):
            expected = float(np.trapezoid(self.S[:, 0] * self.C[j, 0], self.ppm_reg))
            assert sp['area_per_sample'][j] == pytest.approx(expected, rel=1e-9)

    def test_area_per_sample_isolates_component_from_neighbour(self):
        """Component k=0's area must track ITS OWN concentration column,
        not be diluted by the overlapping k=1 component (C[:,1] is constant
        while C[:,0] varies 0.5x-3x across samples — area_per_sample for k=0
        must reproduce that variation)."""
        sp0 = self.win._build_mcr_subpeak(self.peak, 0, {'name': 'MetA'})
        ratio = sp0['area_per_sample'] / sp0['area_per_sample'][2]  # sample 2 = C[:,0]=0.5 (baseline)
        expected_ratio = self.C[:, 0] / self.C[2, 0]
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-9)



class TestRemapPeakRanks:

    def test_remap_updates_all_rank_keyed_structures(self):
        win = _bare_window()
        win.app = SimpleNamespace(stages={
            'quantification': SimpleNamespace(_quant_peak_overrides={'HMDB001': 3}),
        })
        win.ann_stage = SimpleNamespace(
            _quality_marks={('M1', 3): 'good', ('M2', 9): 'bad'},
            _anchor_lines={('M1', 3): 1},
            _subline_quality_marks={('M1', 3, 0): 'good'},
            _last_results_args={'results': [{'peak_rank': 3}, {'peak_rank': 9}]},
        )

        win._remap_peak_ranks({3: 7})

        assert win.ann_stage._quality_marks == {('M1', 7): 'good', ('M2', 9): 'bad'}
        assert win.ann_stage._anchor_lines == {('M1', 7): 1}
        assert win.ann_stage._subline_quality_marks == {('M1', 7, 0): 'good'}
        ranks = [r['peak_rank'] for r in win.ann_stage._last_results_args['results']]
        assert ranks == [7, 9]
        assert win.app.stages['quantification']._quant_peak_overrides == {'HMDB001': 7}

    def test_remap_is_noop_when_nothing_shifted(self):
        win = _bare_window()
        win.app = SimpleNamespace(stages={})
        win.ann_stage = SimpleNamespace(
            _quality_marks={('M1', 3): 'good'},
            _anchor_lines={}, _subline_quality_marks={},
            _last_results_args={'results': [{'peak_rank': 3}]},
        )
        win._remap_peak_ranks({})
        assert win.ann_stage._quality_marks == {('M1', 3): 'good'}


class TestSyncPeakListFromMCR:

    def test_accept_overlap_renumbers_and_remaps_untouched_peak(self):
        """End-to-end: accepting a K=2 overlap splits the parent into 2
        sub-peaks; a peak downstream that had nothing to do with the split
        gets its rank shifted (3->4) — its quality mark must follow it."""
        ppm_reg = np.linspace(1.8, 2.2, 60)
        s0 = np.exp(-0.5 * ((ppm_reg - 1.95) / 0.02) ** 2)
        s1 = np.exp(-0.5 * ((ppm_reg - 2.05) / 0.02) ** 2)
        S = np.column_stack([s0, s1])
        C = np.array([[1.0, 0.8], [2.0, 0.4], [0.5, 0.6]])

        parent = {
            'rank': 2, 'ppm': 2.0,
            'mcr_annotation': {
                'K': 2,
                'confirmed_assignments': {
                    0: {'accepted': True}, 1: {'accepted': True},
                },
                'assignments': [
                    (0, 0.90, 0.0, None, None, None, 0.90),
                    (1, 0.85, 0.0, None, None, None, 0.85),
                ],
                'candidates': [{'name': 'MetA'}, {'name': 'MetB'}],
                'S': S, 'C': C, 'ppm_region': ppm_reg,
            },
        }
        peak_pre  = {'rank': 1, 'ppm': 1.0, 'label': 'pre'}
        peak_post = {'rank': 3, 'ppm': 3.5, 'label': 'post'}
        peaks = [peak_pre, parent, peak_post]

        peak_stage = SimpleNamespace(peaks=peaks, processor=SimpleNamespace())
        ann_stage = SimpleNamespace(
            peak_stage=peak_stage,
            _quality_marks={('MetPost', 3): 'good'},
            _anchor_lines={}, _subline_quality_marks={},
            _last_results_args={'results': [{'peak_rank': 3, 'name': 'MetPost'}]},
        )

        win = _bare_window()
        win.app = SimpleNamespace(
            ppm=np.linspace(0.0, 5.0, 5001),
            stages={'quantification': SimpleNamespace(_quant_peak_overrides={'HMDBPOST': 3})},
        )
        win.ann_stage = ann_stage

        win._sync_peak_list_from_mcr(parent)

        # Parent replaced by 2 sub-peaks; pre/post survive → 4 peaks total.
        assert len(peak_stage.peaks) == 4
        ranks = sorted(p['rank'] for p in peak_stage.peaks)
        assert ranks == [1, 2, 3, 4]

        post_now = next(p for p in peak_stage.peaks if p.get('label') == 'post')
        assert post_now['rank'] == 4   # shifted from 3 -> 4 (parent split added a peak)

        # Rank-keyed structures for the UNTOUCHED peak_post must follow it.
        assert ann_stage._quality_marks == {('MetPost', 4): 'good'}
        assert ann_stage._last_results_args['results'][0]['peak_rank'] == 4
        assert win.app.stages['quantification']._quant_peak_overrides == {'HMDBPOST': 4}

        subpeaks = [p for p in peak_stage.peaks if p.get('is_sub_peak')]
        assert len(subpeaks) == 2
        for sp in subpeaks:
            assert 'area_per_sample' in sp and len(sp['area_per_sample']) == 3
            assert 'start_idx' in sp and 'end_idx' in sp
