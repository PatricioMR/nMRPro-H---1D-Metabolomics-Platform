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


class TestPerSampleReadingsSurviveReload:
    """A reading measured per sample, and the consensus the score used, are
    state the analyst cannot reproduce without re-running the search. Reloading
    a project must hand them back exactly, or the reopened window shows one set
    of numbers beside a verdict reached from another.
    """

    def _result(self):
        return {
            'hmdb_id': 'HMDB0000161', 'name': 'Alanine', 'type': 'd',
            'j_coupling': '7.2', 'db_ppm': 1.4670, 'peak_ppm': 1.4701,
            'peak_rank': 2, 'ppm_diff': 0.0031, 'delta_j_hz': 0.05,
            'observed_j_hz': 7.15, '_dj_source': 'per_sample',
            '_ppm_source': 'per_sample',
            'n_present': 3, 'n_included': 4, 'ppm_spread': 0.0021,
            'per_sample': [
                {'sample_idx': 0, 'present': True, 'partial': False,
                 'ppm_centre': 1.4701, 'ppm_diff': 0.0031, 'j_obs_hz': 7.14,
                 'delta_j_hz': 0.06, 'area': 1.2e6, 'n_lines': 2,
                 'n_found': 2, 'n_alternatives': 0, 'residual_hz': 0.04,
                 'align_shift_ppm': -0.0004},
                {'sample_idx': 1, 'present': False, 'partial': True,
                 'ppm_centre': 1.4712, 'ppm_diff': 0.0042, 'j_obs_hz': None,
                 'delta_j_hz': None, 'area': 0.4e6, 'n_lines': 2,
                 'n_found': 1, 'n_alternatives': 2, 'residual_hz': None,
                 'align_shift_ppm': None},
            ],
            'per_sample_consensus': {
                'ppm_centre': 1.4701, 'ppm_diff': 0.0031, 'ppm_n': 3,
                'ppm_sd': 0.0009, 'observed_j_hz': 7.15, 'delta_j_hz': 0.05,
                'j_n': 3, 'j_sd': 0.11},
        }

    def _roundtrip(self, tmp_path):
        state = _make_state()
        state['ann']['last_results_args']['results'].append(self._result())
        path = str(tmp_path / 'per_sample.nmrproj')
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        return next(r for r in loaded['ann']['last_results_args']['results']
                    if r.get('per_sample_consensus'))

    def test_the_consensus_comes_back_unchanged(self, tmp_path):
        r = self._roundtrip(tmp_path)
        assert r['per_sample_consensus'] == self._result()['per_sample_consensus']

    def test_every_per_sample_row_comes_back(self, tmp_path):
        r = self._roundtrip(tmp_path)
        assert r['per_sample'] == self._result()['per_sample']

    def test_the_uncertainty_is_not_dropped(self, tmp_path):
        """n, the scatter and the rival assignments are the reading's
        uncertainty; a value that survives without them says more than it
        knows."""
        r = self._roundtrip(tmp_path)
        con = r['per_sample_consensus']
        assert con['j_n'] == 3 and con['j_sd'] == pytest.approx(0.11)
        assert r['per_sample'][1]['n_alternatives'] == 2

    def test_the_score_can_still_say_where_its_numbers_came_from(self, tmp_path):
        r = self._roundtrip(tmp_path)
        assert r['_dj_source'] == 'per_sample'
        assert r['_ppm_source'] == 'per_sample'

    def test_reapplying_the_consensus_after_reload_is_a_no_op(self, tmp_path):
        """Load → rescore must not move the numbers: the same consensus applied
        to the same result has to land on the same ΔJ and Δppm."""
        from processing.annotation_scoring import apply_per_sample_consensus
        params = {'dj_gold_hz': 0.20, 'dj_bad_hz': 0.50, 'dj_neutral': 0.0}
        r = self._roundtrip(tmp_path)
        before = (r['delta_j_hz'], r['ppm_diff'])
        apply_per_sample_consensus(r, params=params)
        assert (r['delta_j_hz'], r['ppm_diff']) == pytest.approx(before)
