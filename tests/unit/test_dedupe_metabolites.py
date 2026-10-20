# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The duplicate rule must collapse only what 1H NMR cannot distinguish.

A false positive here silently deletes a real metabolite from a reference
database, so the negative cases matter more than the positive ones.
"""
import pytest

from data.dedupe_metabolites import (
    Entry,
    plan_collapses,
    stereo_normalized_name,
)


def _e(name, key, inchikey=None, formula='C3 H7 N O2', n_shifts=2,
       has_both_ids=False):
    return Entry(key=key, name=name, inchikey=inchikey, formula=formula,
                 n_shifts=n_shifts, has_both_ids=has_both_ids)


# ── name normalization ────────────────────────────────────────────────────────

@pytest.mark.parametrize('a,b', [
    ('Alanine',            'L-alanine'),
    ('Acetylcarnitine',    'Acetyl-L-carnitine'),
    ('Selenomethionine',   'L-Selenomethionine'),
    ('D-Methionine',       'L-methionine'),
    ('D-Aspartate',        'L-aspartic acid'),
    ('(S)-Lactate',        'Lactate'),
    ('DL-Serine',          'L-Serine'),
])
def test_same_species_share_a_key(a, b):
    assert stereo_normalized_name(a) == stereo_normalized_name(b)


@pytest.mark.parametrize('a,b', [
    ('Alanine',              'Beta-alanine'),        # constitutional isomer
    ('Alanine',              'N-Acetyl-L-Alanine'),  # different compound
    ('D-Glucose',            'D-Galactose'),         # diastereomers
    ('Myo-inositol',         'Muco-inositol'),       # diastereomers
    ('Sodium fumarate',      'Maleic Acid'),         # cis/trans
    ('D-Mannitol',           'D-sorbitol'),          # diastereomers
    ('Citraconic acid',      'Mesaconic acid'),      # cis/trans
    ('L-arabitol',           'Xylitol'),             # diastereomers
])
def test_different_species_keep_separate_keys(a, b):
    assert stereo_normalized_name(a) != stereo_normalized_name(b)


# ── collapse planning ─────────────────────────────────────────────────────────

def test_l_form_survives():
    plans = plan_collapses([
        _e('Alanine',   'bmse000282', 'QNAYBMKLOCPYGJ-UHFFFAOYSA-N'),
        _e('L-alanine', 'HMDB0000161', 'QNAYBMKLOCPYGJ-REOHCLBHSA-N'),
    ])
    assert len(plans) == 1
    assert plans[0].survivor.name == 'L-alanine'
    assert [e.key for e in plans[0].removed] == ['bmse000282']


def test_more_shifts_wins_when_neither_is_the_l_form():
    plans = plan_collapses([
        _e('Acetylcarnitine', 'a', 'RDHQFKQIGNGIED-UHFFFAOYSA-N', n_shifts=5),
        _e('Acetylcarnitine', 'b', 'RDHQFKQIGNGIED-UHFFFAOYSA-N', n_shifts=7),
    ])
    assert plans[0].survivor.key == 'b'


def test_diastereomers_are_never_collapsed():
    """Same skeleton, different spectrum — both must stay."""
    plans = plan_collapses([
        _e('D-Glucose',   'g', 'WQZGKKKJIJFFOK-GASJEMHNSA-N', n_shifts=9),
        _e('D-Galactose', 'l', 'WQZGKKKJIJFFOK-UHFFFAOYSA-N', n_shifts=9),
    ])
    assert plans == []


def test_name_match_without_structural_support_is_refused():
    """Same normalized name but a different skeleton — not the same compound."""
    plans = plan_collapses([
        _e('Lactate',   'x', 'JVTAAEKCZFNVCJ-UHFFFAOYSA-N'),
        _e('L-Lactate', 'y', 'AAAAAAAAAAAAAA-UHFFFAOYSA-N'),
    ])
    assert plans == []


def test_formula_is_the_fallback_when_no_inchikey_is_stored():
    plans = plan_collapses([
        _e('Alanine',   'a', None, formula='C3 H7 N O2'),
        _e('L-alanine', 'b', None, formula='C3 H7 N O2'),
    ])
    assert len(plans) == 1
    assert plans[0].survivor.name == 'L-alanine'


def test_entries_without_structure_are_left_alone():
    """No InChIKey and no formula — nothing justifies a deletion."""
    plans = plan_collapses([
        _e('Alanine',   'a', None, formula=None),
        _e('L-alanine', 'b', None, formula=None),
    ])
    assert plans == []


def test_singletons_produce_no_plan():
    assert plan_collapses([_e('L-alanine', 'a')]) == []


def test_three_way_group_keeps_exactly_one():
    plans = plan_collapses([
        _e('Methionine',   'a', 'FFEARJCKVFRZRR-UHFFFAOYSA-N', n_shifts=3),
        _e('D-Methionine', 'b', 'FFEARJCKVFRZRR-VKHMYHEASA-N', n_shifts=3),
        _e('L-methionine', 'c', 'FFEARJCKVFRZRR-BYPYZUCNSA-N', n_shifts=4),
    ])
    assert len(plans) == 1
    assert plans[0].survivor.key == 'c'
    assert sorted(e.key for e in plans[0].removed) == ['a', 'b']


# ── database round-trip ───────────────────────────────────────────────────────

import sqlite3  # noqa: E402

from data.dedupe_metabolites import apply_plans, plan_database  # noqa: E402


def _build_db(path):
    """Mimic HMDB_Metabolomics_v01: metabolites read by bmrb_id, but sticks
    and hsqc keyed only by hmdb_id."""
    con = sqlite3.connect(path)
    con.executescript('''
        CREATE TABLE metabolites (hmdb_id TEXT, bmrb_id TEXT, name TEXT,
                                  formula TEXT, inchikey TEXT);
        CREATE TABLE shifts_1h  (hmdb_id TEXT, bmrb_id TEXT, ppm REAL);
        CREATE TABLE sticks_1h  (hmdb_id TEXT, ppm REAL);
        CREATE TABLE hsqc       (hmdb_id TEXT, h_ppm REAL, c_ppm REAL);
    ''')
    con.executemany('INSERT INTO metabolites VALUES (?,?,?,?,?)', [
        ('HMDB0000161', 'HMDB0000161', 'L-alanine', 'C3H7NO2',
         'QNAYBMKLOCPYGJ-REOHCLBHSA-N'),
        ('HMDB0009999', 'HMDB0009999', 'Alanine', 'C3H7NO2',
         'QNAYBMKLOCPYGJ-UHFFFAOYSA-N'),
        ('HMDB0000056', 'HMDB0000056', 'Beta-alanine', 'C3H7NO2',
         'UCMIRNVEIXFBKS-UHFFFAOYSA-N'),
    ])
    con.executemany('INSERT INTO shifts_1h VALUES (?,?,?)', [
        ('HMDB0000161', 'HMDB0000161', 1.47),
        ('HMDB0000161', 'HMDB0000161', 3.77),
        ('HMDB0009999', 'HMDB0009999', 1.47),
        ('HMDB0000056', 'HMDB0000056', 2.55),
    ])
    con.executemany('INSERT INTO sticks_1h VALUES (?,?)',
                    [('HMDB0009999', 1.47), ('HMDB0000161', 1.47)])
    con.executemany('INSERT INTO hsqc VALUES (?,?,?)',
                    [('HMDB0009999', 1.47, 17.0)])
    con.commit()
    con.close()


def test_apply_removes_the_duplicate_and_leaves_no_orphans(tmp_path):
    db = str(tmp_path / 'ref.db')
    _build_db(db)

    key, children, plans = plan_database(db)
    assert 'sticks_1h' in children and 'hsqc' in children, \
        'tables keyed only by hmdb_id must be included, or their rows orphan'
    assert len(plans) == 1
    assert plans[0].survivor.name == 'L-alanine'

    apply_plans(db, key, children, plans, stamp='test')

    con = sqlite3.connect(db)
    names = sorted(n for (n,) in con.execute('SELECT name FROM metabolites'))
    assert names == ['Beta-alanine', 'L-alanine'], 'constitutional isomer kept'
    for table in ('shifts_1h', 'sticks_1h', 'hsqc'):
        orphans = con.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE hmdb_id NOT IN '
            '(SELECT hmdb_id FROM metabolites)').fetchone()[0]
        assert orphans == 0, f'{table} left orphaned rows'
    # the survivor's own data is untouched
    assert con.execute('SELECT COUNT(*) FROM shifts_1h '
                       "WHERE hmdb_id='HMDB0000161'").fetchone()[0] == 2
    con.close()


def test_apply_writes_a_backup(tmp_path):
    db = str(tmp_path / 'ref.db')
    _build_db(db)
    key, children, plans = plan_database(db)
    res = apply_plans(db, key, children, plans, stamp='test')
    import os
    assert os.path.exists(res['backup'])
    con = sqlite3.connect(res['backup'])
    assert con.execute('SELECT COUNT(*) FROM metabolites').fetchone()[0] == 3
    con.close()
