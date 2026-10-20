# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The project file: what goes in comes back, smaller, and an old one still
opens."""
import os
import pickle

import numpy as np
import pytest

from data import project_file as pf


def _state(n_pts=4096, n_s=5, seed=1):
    rng = np.random.default_rng(seed)
    ppm = np.linspace(10.0, -0.5, n_pts)
    X = rng.normal(0.0, 950.0, (n_pts, n_s))
    X[2000:2010] += 8.6e8                        # a tall peak beside the noise
    return {
        'version': 7, 'ppm': ppm, 'all_intensities': X,
        'samples_info': [{'index': i, 'name': f'S{i}'} for i in range(n_s)],
        'ann': {'quality_marks': {('HMDB0000042', 1): 'good'},
                'results': [{'name': 'Creatine', 'per_sample': [{'ppm': 3.03}] * n_s}]},
        'st': {'excluded_samples': {'S1', 'S3'}},
        'qnt': {'quant_results': [{'metabolite': 'x', '_concs_per_sample':
                                   [1.0, float('nan'), 0.0, 2.0, 3.0]}]},
    }


def test_everything_but_the_spectrum_comes_back_exactly(tmp_path):
    st = _state()
    path = str(tmp_path / 'p.nmrproj')
    pf.save(path, st)
    back = pf.read(path)
    assert set(back) == set(st)
    for k in st:
        if k != 'all_intensities':
            np.testing.assert_equal(back[k], st[k])


def test_the_spectrum_comes_back_as_float32_rounds_it(tmp_path):
    st = _state()
    X = st['all_intensities']
    path = str(tmp_path / 'p.nmrproj')
    pf.save(path, st)
    Y = pf.read(path)['all_intensities']
    assert Y.shape == X.shape and Y.dtype == np.float64 and Y.flags.c_contiguous
    np.testing.assert_array_equal(Y, X.astype(np.float32).astype(np.float64))
    # the rounding is a sliver of the noise even at the tallest peak
    assert np.abs(Y - X).max() < 0.05 * 950.0


def test_a_file_read_back_and_written_again_is_the_same(tmp_path):
    a, b = str(tmp_path / 'a.nmrproj'), str(tmp_path / 'b.nmrproj')
    pf.save(a, _state())
    pf.save(b, pf.read(a))
    assert open(a, 'rb').read() == open(b, 'rb').read()


def test_the_file_is_much_smaller_than_the_old_pickle(tmp_path):
    st = _state(n_pts=32768, n_s=8)
    old = len(pickle.dumps(st, protocol=5))
    path = str(tmp_path / 'p.nmrproj')
    new = pf.save(path, st)
    assert new == os.path.getsize(path)
    assert new < 0.6 * old


def test_a_project_in_the_old_format_is_read_as_it_was(tmp_path):
    st = _state()
    path = str(tmp_path / 'old.nmrproj')
    with open(path, 'wb') as f:
        pickle.dump(st, f, protocol=5)
    assert not pf.is_current_format(path)
    back = pf.read(path)
    np.testing.assert_array_equal(back['all_intensities'], st['all_intensities'])


def test_a_project_without_a_spectrum_is_saved(tmp_path):
    st = _state()
    st['all_intensities'] = None
    path = str(tmp_path / 'p.nmrproj')
    pf.save(path, st)
    assert pf.read(path)['all_intensities'] is None


def test_a_value_float32_cannot_hold_is_refused_not_stored_as_inf(tmp_path):
    st = _state()
    st['all_intensities'][5, 2] = 1e40
    with pytest.raises(pf.ProjectFileError, match='float32 range'):
        pf.save(str(tmp_path / 'p.nmrproj'), st)


def test_a_damaged_file_is_said_not_read(tmp_path):
    path = str(tmp_path / 'p.nmrproj')
    pf.save(path, _state())
    blob = bytearray(open(path, 'rb').read())
    blob[len(blob) // 2] ^= 0xFF
    open(path, 'wb').write(bytes(blob))
    with pytest.raises(Exception):
        pf.read(path)


def test_a_failed_write_leaves_the_old_file_and_no_tmp(tmp_path, monkeypatch):
    path = str(tmp_path / 'p.nmrproj')
    pf.save(path, _state(seed=1))
    before = open(path, 'rb').read()
    monkeypatch.setattr(pf, 'encode', lambda snap: b'PK\x03\x04half')
    with pytest.raises(Exception):
        pf.save(path, _state(seed=2))
    assert open(path, 'rb').read() == before
    assert not os.path.exists(path + '.tmp')


def test_the_digest_follows_the_content():
    a, b = _state(), _state()
    assert pf.snapshot(a).digest == pf.snapshot(b).digest
    b['qnt']['quant_results'][0]['_concs_per_sample'][0] = 1.5
    assert pf.snapshot(a).digest != pf.snapshot(b).digest
    c = _state()
    c['all_intensities'][100, 0] += 500.0
    assert pf.snapshot(a).digest != pf.snapshot(c).digest
