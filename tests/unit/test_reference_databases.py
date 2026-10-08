"""The application offers two reference databases and nothing else.

Every .db in the folder used to be listed -- old versions, the unfiltered
build source, the statistics stage's pathway table -- and an installation
keeps every file it was ever seeded with, so old versions stayed on offer.
"""
import os

import app_paths
from app_paths import REFERENCE_DATABASES, SUPPORT_DATABASES, reference_databases


def test_only_mammalian_and_general_are_offered():
    assert [f for _l, f in REFERENCE_DATABASES] == [
        'BMRB_Metabolomics_Mammalian_v05.db', 'BMRB_Metabolomics_General_v05.db']


def test_both_ship_with_the_repository():
    folder = app_paths.databases_dir()
    for f in [f for _l, f in REFERENCE_DATABASES] + list(SUPPORT_DATABASES):
        assert os.path.isfile(os.path.join(folder, f)), f


def test_an_old_version_left_in_the_folder_is_not_listed(tmp_path, monkeypatch):
    for f in ('BMRB_Metabolomics_Mammalian_v05.db', 'BMRB_Metabolomics_General_v05.db',
              'BMRB_Metabolomics_Mammalian_v04.db', 'BMRB_Metabolomics_v05.db',
              'hmdb_pathways.db'):
        (tmp_path / f).write_bytes(b'')
    monkeypatch.setattr(app_paths, 'databases_dir', lambda: str(tmp_path))
    assert [os.path.basename(p) for _l, p in reference_databases()] == [
        'BMRB_Metabolomics_Mammalian_v05.db', 'BMRB_Metabolomics_General_v05.db']


def test_a_missing_one_is_not_listed(tmp_path, monkeypatch):
    (tmp_path / 'BMRB_Metabolomics_General_v05.db').write_bytes(b'')
    monkeypatch.setattr(app_paths, 'databases_dir', lambda: str(tmp_path))
    assert [l for l, _p in reference_databases()] == ['BMRB General v05']
