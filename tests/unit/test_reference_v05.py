# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""BMRB v05: the protons of each signal are the molecule's
(hmdb_api_download/build_reference_v05.py)."""
import os
import sqlite3

import pytest

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'databases', 'BMRB_Metabolomics_Mammalian_v05.db')

pytestmark = pytest.mark.skipif(not os.path.exists(DB), reason='v05 not built')


def _rows(con, name):
    return con.execute(
        'select protons, protons_v04, v05_status, v05_evidence from shifts_1h s '
        'join metabolites m on m.bmrb_id = s.bmrb_id where m.name = ?',
        (name,)).fetchall()


@pytest.mark.parametrize('name, c_h', [
    ('Creatinine', 5), ('Cysteamine', 4), ('Phenylacetylglycine', 9),
    ('Myo-inositol', 6), ('Meso-erythritol', 6), ('Sucrose', 14),
    ('Phenylacetic acid', 7), ('Phthalaldehydic acid', 5)])
def test_a_corrected_compound_lists_its_molecules_c_h_protons(name, c_h):
    con = sqlite3.connect(DB)
    assert sum(r[0] for r in _rows(con, name)) == pytest.approx(c_h)


def test_every_change_keeps_what_it_replaced_and_why():
    con = sqlite3.connect(DB)
    for nh, old, status, ev in con.execute(
            "select protons, protons_v04, v05_status, v05_evidence "
            "from shifts_1h where v05_status = 'corrected'"):
        assert old is not None and old != nh and ev


def test_myo_inositol_is_one_two_two_one():
    con = sqlite3.connect(DB)
    got = con.execute(
        'select protons from shifts_1h s join metabolites m on m.bmrb_id = '
        "s.bmrb_id where m.name = 'Myo-inositol' "
        'order by coalesce(s.ppm_meas, s.ppm) desc').fetchall()
    assert [r[0] for r in got] == [1, 2, 2, 1]
