# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The application offers the BMRB_<matrix>_nMRPro-H databases and nothing
else, and a project saved against an older file reopens against the one
that replaced it.

Every .db in the folder used to be listed -- old versions, the unfiltered
build source, the statistics stage's pathway table -- and an installation
keeps every file it was ever seeded with, so old versions stayed on offer.
"""
import os

import app_paths
from app_paths import (REFERENCE_DATABASES, SUPPORT_DATABASES, reference_databases,
                       resolve_database)

OFFERED = ['BMRB_Mammalian_nMRPro-H.db', 'BMRB_General_nMRPro-H.db',
           'BMRB_Blood_nMRPro-H.db', 'BMRB_Urine_nMRPro-H.db',
           'BMRB_Feces_nMRPro-H.db', 'BMRB_Saliva_nMRPro-H.db',
           'BMRB_CSF_nMRPro-H.db', 'BMRB_Milk_nMRPro-H.db',
           'BMRB_Sweat_nMRPro-H.db']


def test_the_matrix_databases_are_offered_mammalian_first():
    assert [f for _l, f in REFERENCE_DATABASES] == OFFERED


def test_all_ship_with_the_repository():
    folder = app_paths.databases_dir()
    for f in [f for _l, f in REFERENCE_DATABASES] + list(SUPPORT_DATABASES):
        assert os.path.isfile(os.path.join(folder, f)), f


def test_an_old_version_left_in_the_folder_is_not_listed(tmp_path, monkeypatch):
    for f in ('BMRB_Metabolomics_Mammalian_v05.db', 'BMRB_Metabolomics_General_v05.db',
              'BMRB_Metabolomics_v06.db', 'BMRB_Mammalian_nMRPro-H.db',
              'BMRB_General_nMRPro-H.db', 'hmdb_pathways.db'):
        (tmp_path / f).write_bytes(b'')
    monkeypatch.setattr(app_paths, 'databases_dir', lambda: str(tmp_path))
    assert [os.path.basename(p) for _l, p in reference_databases()] == [
        'BMRB_Mammalian_nMRPro-H.db', 'BMRB_General_nMRPro-H.db']


def test_a_missing_one_is_not_listed(tmp_path, monkeypatch):
    (tmp_path / 'BMRB_General_nMRPro-H.db').write_bytes(b'')
    monkeypatch.setattr(app_paths, 'databases_dir', lambda: str(tmp_path))
    assert [l for l, _p in reference_databases()] == ['BMRB General']


def test_a_project_on_a_v05_file_reopens_on_its_replacement(tmp_path, monkeypatch):
    (tmp_path / 'BMRB_Mammalian_nMRPro-H.db').write_bytes(b'')
    monkeypatch.setattr(app_paths, 'databases_dir', lambda: str(tmp_path))
    gone = r'C:\other\machine\BMRB_Metabolomics_Mammalian_v05.db'
    assert resolve_database(gone) == str(tmp_path / 'BMRB_Mammalian_nMRPro-H.db')


def test_a_project_file_that_is_there_is_kept_as_it_is(tmp_path, monkeypatch):
    own = tmp_path / 'my_own.db'
    own.write_bytes(b'')
    monkeypatch.setattr(app_paths, 'databases_dir', lambda: str(tmp_path / 'x'))
    assert resolve_database(str(own)) == str(own)
    assert resolve_database(str(tmp_path / 'nowhere.db')) == str(tmp_path / 'nowhere.db')
