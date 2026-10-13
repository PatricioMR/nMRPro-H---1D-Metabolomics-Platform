"""Converting old projects: in place, verified, the date kept, and a file
that does not compare left alone."""
import os
import pickle

import numpy as np

from data import convert_projects as cp
from data import project_file as pf


def _old(path, seed=0):
    rng = np.random.default_rng(seed)
    st = {'version': 7, 'ppm': np.linspace(10, 0, 4096),
          'all_intensities': rng.normal(0, 1e3, (4096, 6)) + 5e5,
          'qnt': {'quant_results': [{'metabolite': 'x', 'conc_mm': 1.25}]},
          'st': {'excluded_samples': {'S1'}}}
    with open(path, 'wb') as f:
        pickle.dump(st, f, protocol=5)
    os.utime(path, (1_700_000_000, 1_700_000_000))
    return st


def test_an_old_project_is_converted_in_place_and_keeps_its_date(tmp_path):
    path = str(tmp_path / 'a.nmrproj')
    st = _old(path)
    old_size = os.path.getsize(path)
    assert cp.main([str(tmp_path)]) == 0
    assert pf.is_current_format(path)
    assert os.path.getsize(path) < old_size
    assert os.path.getmtime(path) == 1_700_000_000
    back = pf.read(path)
    assert back['qnt'] == st['qnt'] and back['st'] == st['st']
    np.testing.assert_array_equal(back['all_intensities'],
                                  pf.stored_spectrum(st['all_intensities']))
    assert not os.path.exists(path + '.tmp')


def test_autosaves_and_backups_are_converted_and_other_files_not(tmp_path):
    for name in ('a.nmrproj.autosave', 'a.nmrproj.bak'):
        _old(str(tmp_path / name))
    other = tmp_path / 'notes.pkl'
    other.write_bytes(pickle.dumps({'x': 1}))
    assert cp.main([str(tmp_path)]) == 0
    assert pf.is_current_format(str(tmp_path / 'a.nmrproj.autosave'))
    assert pf.is_current_format(str(tmp_path / 'a.nmrproj.bak'))
    assert not pf.is_current_format(str(other))


def test_a_file_already_converted_is_not_touched(tmp_path):
    path = str(tmp_path / 'a.nmrproj')
    _old(path)
    cp.main([path])
    before = open(path, 'rb').read()
    os.utime(path, (5, 5))
    cp.main([path])
    assert open(path, 'rb').read() == before and os.path.getmtime(path) == 5


def test_a_file_that_does_not_compare_is_left_as_it_was(tmp_path, monkeypatch):
    path = str(tmp_path / 'a.nmrproj')
    _old(path)
    before = open(path, 'rb').read()
    real_decode = pf.decode

    def wrong(blob):
        st = real_decode(blob)
        st['qnt']['quant_results'][0]['conc_mm'] = 9.0
        return st
    monkeypatch.setattr(pf, 'decode', wrong)
    assert cp.main([path]) == 1
    assert open(path, 'rb').read() == before
    assert not os.path.exists(path + '.tmp')


def test_a_dry_run_changes_nothing(tmp_path):
    path = str(tmp_path / 'a.nmrproj')
    _old(path)
    before = open(path, 'rb').read()
    assert cp.main([str(tmp_path), '--dry-run']) == 0
    assert open(path, 'rb').read() == before
