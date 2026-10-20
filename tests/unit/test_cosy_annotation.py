# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_cosy_annotation.py
"""Testes do matching COSY por co-ocorrência de shifts."""
import sqlite3
import numpy as np
import pytest

from processing.cosy_annotation import (
    load_shift_library, annotate_cosy, _share_j, expected_crosspeaks)


@pytest.fixture
def tiny_db(tmp_path):
    """DB de teste com esquema shifts_1h e 2 metabolitos."""
    path = tmp_path / "test_cosy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE metabolites (hmdb_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE shifts_1h (
            id INTEGER PRIMARY KEY, hmdb_id TEXT, ppm REAL,
            multiplicity TEXT, j_coupling TEXT, protons INTEGER);
    """)
    conn.execute("INSERT INTO metabolites VALUES ('HMDB1','Alanine')")
    conn.execute("INSERT INTO metabolites VALUES ('HMDB2','Lactate')")
    # Alanine: protões acoplados a 1.48 e 3.78, partilham J=7.2
    conn.executemany(
        "INSERT INTO shifts_1h (hmdb_id,ppm,multiplicity,j_coupling,protons) "
        "VALUES (?,?,?,?,?)",
        [('HMDB1', 1.48, 'd', '7.2', 3),
         ('HMDB1', 3.78, 'q', '7.2', 1),
         ('HMDB2', 1.33, 'd', '6.9', 3),
         ('HMDB2', 4.11, 'q', '6.9', 1)])
    conn.commit(); conn.close()
    return str(path)


def test_load_library(tiny_db):
    lib = load_shift_library(tiny_db)
    assert set(lib) == {'HMDB1', 'HMDB2'}
    assert lib['HMDB1']['name'] == 'Alanine'
    assert len(lib['HMDB1']['shifts']) == 2


def test_share_j():
    a = [{'j': 7.2}]; b = [{'j': 7.3}]
    assert _share_j(a, b, j_tol=1.5)
    assert not _share_j([{'j': 7.2}], [{'j': 2.0}], j_tol=1.5)


def test_expected_crosspeaks_from_shared_j():
    # Alanine: 1.48 e 3.78 partilham J=7.2 → 1 cross-peak previsto, J-confirmado
    shifts = [{'ppm': 1.48, 'mult': 'd', 'j': 7.2, 'protons': 3},
              {'ppm': 3.78, 'mult': 'q', 'j': 7.2, 'protons': 1}]
    exp = expected_crosspeaks(shifts, j_tol=1.5)
    assert len(exp) == 1
    a, b, jc = exp[0]
    assert jc is True
    assert {round(a, 2), round(b, 2)} == {1.48, 3.78}


def test_expected_singlets_only_no_crosspeaks():
    # Dois singletos (sem J) → sem fallback, nenhum cross-peak previsto
    shifts = [{'ppm': 2.0, 'mult': 's', 'j': None, 'protons': 3},
              {'ppm': 7.0, 'mult': 's', 'j': None, 'protons': 1}]
    assert expected_crosspeaks(shifts, fallback=False) == []
    # Com fallback → todos os pares
    assert len(expected_crosspeaks(shifts, fallback=True)) == 1


def test_coverage_relative_to_metabolite(tiny_db):
    # Mistura com muitos cross-peaks; Alanine só explica o seu próprio par
    # → coverage do fingerprint dele = 1.0 (não 1/total)
    cps = [{'f2': 1.48, 'f1': 3.78, 'intensity': 1.0}]
    res = annotate_cosy(cps, tiny_db, tol_f2=0.03, tol_f1=0.03)
    ala = next(r for r in res if r['name'] == 'Alanine')
    assert ala['n_total'] == 1          # fingerprint do metabolito
    assert ala['n_matched'] == 1
    assert ala['coverage'] == 1.0       # totalmente presente
    assert ala['n_j_confirmed'] == 1


def test_no_match_outside_tolerance(tiny_db):
    cps = [{'f2': 5.0, 'f1': 6.0, 'intensity': 1.0}]
    res = annotate_cosy(cps, tiny_db, tol_f2=0.03, tol_f1=0.03)
    assert res == []


def test_partial_coverage(tiny_db):
    # Metabolito com 2 cross-peaks previstos mas só 1 observado → coverage 0.5
    import sqlite3
    conn = sqlite3.connect(tiny_db)
    conn.execute("INSERT INTO metabolites VALUES ('HMDB3','Tri')")
    # 3 protões em cadeia: 1.0-2.0 (J5), 2.0-3.0 (J8) → 2 cross-peaks previstos
    conn.executemany(
        "INSERT INTO shifts_1h (hmdb_id,ppm,multiplicity,j_coupling,protons) "
        "VALUES (?,?,?,?,?)",
        [('HMDB3', 1.0, 'd', '5.0', 1),
         ('HMDB3', 2.0, 'dd', '5.0', 1),
         ('HMDB3', 2.0, 'dd', '8.0', 1),
         ('HMDB3', 3.0, 'd', '8.0', 1)])
    conn.commit(); conn.close()
    # Observar só o cross-peak 1.0-2.0
    cps = [{'f2': 1.0, 'f1': 2.0, 'intensity': 1.0}]
    res = annotate_cosy(cps, tiny_db, tol_f2=0.03, tol_f1=0.03)
    tri = next(r for r in res if r['name'] == 'Tri')
    assert tri['n_total'] == 2
    assert tri['n_matched'] == 1
    assert tri['coverage'] == 0.5
