# tests/unit/test_project_serialization.py
"""Test project save/load roundtrip without a real GUI."""
import os
import pickle
import tempfile
import numpy as np
import pytest


def _make_state(with_quality_marks=True):
    """Build a minimal but realistic project state dict."""
    n_pts, n_s = 1000, 6
    ppm         = np.linspace(10.0, 0.0, n_pts)
    intensities = np.random.default_rng(42).random((n_pts, n_s)) * 1e6

    results = [
        {
            'hmdb_id': 'HMDB0000042', 'name': 'Creatine',
            'ppm_center': 3.03, 'ppm': 3.03, 'type': 's',
            'j_coupling': None, 'peak_ppm': 3.031, 'peak_rank': 1,
            'composite_score': 0.82, 'annotation_quality': 'good',
            'predicted_lines': [],
        },
        {
            'hmdb_id': 'HMDB0000161', 'name': 'Alanine',
            'ppm_center': 1.48, 'ppm': 1.48, 'type': 'd',
            'j_coupling': '7.2', 'peak_ppm': 1.481, 'peak_rank': 2,
            'composite_score': 0.71, 'annotation_quality': None,
            'predicted_lines': [],
        },
    ]

    state = {
        'version': 3,
        'ppm': ppm,
        'all_intensities': intensities,
        'samples_info': [{'index': i, 'name': f'S{i}', 'label': 'QC'}
                         for i in range(n_s)],
        'group_colors': {},
        'source_file': '/fake/path/data.csv',
        'active_stage': 'annotation',
        'axes_settings': {},
        'plot_settings': {},
        'show_legend': True,
        'ann': {
            'db_path': '/fake/plasma.db',
            'last_results_args': {'results': results},
            'annotations': {},
            'tolerance': 0.05,
            'align': True,
            'quality_marks':       {('HMDB0000042', 1): 'good',
                                    ('HMDB0000161', 2): 'bad'} if with_quality_marks else {},
            'child_quality_marks': {('HMDB0000042', round(3.03, 3)): 'good'} if with_quality_marks else {},
        },
        'qnt': {
            'quant_results': [],
            'integ': 'Area', 'norm': 'None',
            'proton_corr': True, 'mcr': False,
            'ref_ppm': '0.0', 'ref_nh': '9',
            'ref_conc': '1.0', 'ref_std': '',
            'ref_rank': '',
        },
    }
    return state


class TestPickleRoundtrip:
    def test_save_and_load(self, tmp_path):
        state = _make_state()
        path  = str(tmp_path / 'test.nmrproj')
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded['version'] == 3

    def test_ppm_array_preserved(self, tmp_path):
        state = _make_state()
        path  = str(tmp_path / 'test.nmrproj')
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        np.testing.assert_array_equal(loaded['ppm'], state['ppm'])

    def test_quality_marks_preserved(self, tmp_path):
        state = _make_state(with_quality_marks=True)
        path  = str(tmp_path / 'test.nmrproj')
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        qm = loaded['ann']['quality_marks']
        assert qm[('HMDB0000042', 1)] == 'good'
        assert qm[('HMDB0000161', 2)] == 'bad'

    def test_child_quality_marks_preserved(self, tmp_path):
        state = _make_state(with_quality_marks=True)
        path  = str(tmp_path / 'test.nmrproj')
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        cqm = loaded['ann']['child_quality_marks']
        assert cqm[('HMDB0000042', 3.03)] == 'good'

    def test_results_count_preserved(self, tmp_path):
        state = _make_state()
        path  = str(tmp_path / 'test.nmrproj')
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        results = loaded['ann']['last_results_args']['results']
        assert len(results) == 2

    def test_annotation_quality_reapplied(self, tmp_path):
        """Simulate what _restore_state does after loading."""
        state = _make_state(with_quality_marks=True)
        path  = str(tmp_path / 'test.nmrproj')
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)

        # Simulate restore: re-apply quality_marks to result dicts
        qm      = loaded['ann']['quality_marks']
        results = loaded['ann']['last_results_args']['results']
        for r in results:
            mid = r.get('hmdb_id', '')
            key = (mid, r.get('peak_rank'))
            if key in qm:
                r['annotation_quality'] = qm[key]

        creatine = next(r for r in results if r['name'] == 'Creatine')
        alanine  = next(r for r in results if r['name'] == 'Alanine')
        assert creatine['annotation_quality'] == 'good'
        assert alanine['annotation_quality']  == 'bad'
