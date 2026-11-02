# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""v06: the signals of a CASMDB compound, read from its spin system or its
peak list, and what may be built from (hmdb_api_download/build_reference_v06.py)."""
import os
import sqlite3
import sys

import pytest

from data.casmdb_nef import Entry, Spectrum

sys.path.insert(0, 'hmdb_api_download')
import build_reference_v06 as B  # noqa: E402

DBDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'databases')

# L-alanine as CASMDB holds it: GISSMO's matrix, HMDB's 500 MHz peak list
ALA_SPIN = Spectrum(origin='GISSMO', kind='spin_system', field_mhz=500.0,
                    similarity=0.95, source_file=None,
                    spin_ppm=[1.4662, 1.4662, 1.4662, 3.7674],
                    spin_j=[(0, 1, -12.5), (0, 2, -12.5), (1, 2, -12.5),
                            (0, 3, 7.2988), (1, 3, 7.2988), (2, 3, 7.2988)])
ALA_PEAKS = [(1.459, 0.9806), (1.474, 1.0), (3.751, 0.0631), (3.766, 0.2025),
             (3.78, 0.203), (3.795, 0.0649)]
ALA_LIST = Spectrum(origin='HMDB', kind='peak_list', field_mhz=500.0,
                    similarity=0.9765, source_file='SP-171.ndf5', peaks=ALA_PEAKS,
                    multiplets=[{'ppm': 1.4665, 'volume': None, 'height': 1.98,
                                 'peaks': ALA_PEAKS[:2]},
                                {'ppm': 3.773, 'volume': None, 'height': 0.53,
                                 'peaks': ALA_PEAKS[2:]}])
ALA_SMILES = 'C[C@H](N)C(O)=O'


def _by_ppm(signals):
    return sorted(signals, key=lambda s: s['ppm'])


def test_a_methyl_is_one_signal_of_three_spins_split_by_its_neighbour():
    ch3, ch = _by_ppm(B.spin_signals(ALA_SPIN))
    assert (ch3['protons'], ch3['multiplicity']) == (3, 'd')
    assert (ch['protons'], ch['multiplicity']) == (1, 'q')
    assert float(ch3['j_coupling']) == pytest.approx(7.30, abs=0.01)
    # lines from the solved system: the doublet's two, J apart (the slight
    # roof tilts the intensity-weighted centre off the middle)
    (lo, _a), (hi, _b) = ch3['lines']
    assert hi - lo == pytest.approx(7.30, abs=0.01)


def test_a_strongly_coupled_pair_is_not_given_a_letter():
    # AB: shift difference 0.02 ppm = 10 Hz at 500 MHz against J 15 Hz
    ab = Spectrum(origin='GISSMO', kind='spin_system', field_mhz=500.0,
                  similarity=0.95, source_file=None, spin_ppm=[2.52, 2.54],
                  spin_j=[(0, 1, 15.0)])
    assert {s['multiplicity'] for s in B.spin_signals(ab)} == {'m'}


def test_peak_list_protons_are_the_molecules_carbon_hydrogens():
    ch3, ch = _by_ppm(B.peak_signals(ALA_LIST, 4))
    assert (ch3['protons'], ch['protons']) == (3, 1)
    assert (ch3['multiplicity'], ch['multiplicity']) == ('d', 'q')
    assert float(ch3['j_coupling']) == pytest.approx(7.5, abs=0.1)
    assert 'estimated' in ch3['protons_source']


def test_apportion_keeps_the_total():
    assert B.apportion([1.98, 0.53], 4) == [3, 1]
    assert sum(B.apportion([1, 1, 1], 7)) == 7
    assert B.apportion([0, 0], 3) == [0, 0]


@pytest.mark.parametrize('sim, origin, ok', [
    (0.95, 'GISSMO', True), (0.89, 'BMRB', False),
    (1.0, 'HMDB', False),        # CASMDB's mark of a cleaned, unchecked spectrum
    (0.97, 'SDBS', False),       # not redistributable
    (None, 'BMRB', False)])
def test_only_a_verified_redistributable_spectrum_is_used(sim, origin, ok):
    s = Spectrum(origin=origin, kind='peak_list', field_mhz=500.0, similarity=sim,
                 source_file=None, peaks=ALA_PEAKS)
    assert B.usable(s) is ok


def test_the_spin_system_is_preferred_and_a_huge_one_is_not():
    e = Entry(name='x', inchi=None, smiles=ALA_SMILES, formula=None, hmdb_id=None,
              bmrb_id=None, cas=None, iupac=None, synonyms=[],
              spectra=[ALA_LIST, ALA_SPIN])
    assert B.best_spectrum(e) is ALA_SPIN
    chain = Spectrum(origin='GISSMO', kind='spin_system', field_mhz=500.0,
                     similarity=0.95, source_file=None,
                     spin_ppm=[1.0 + 0.1 * i for i in range(12)],
                     spin_j=[(i, i + 1, 7.0) for i in range(11)])
    assert B.max_fragment(12, chain.spin_j) == 12
    e.spectra = [chain, ALA_LIST]
    assert B.best_spectrum(e) is ALA_LIST


def test_quantified_in_people_is_the_mammalian_test():
    assert B.is_mammalian({'status': 'Detected and Quantified',
                           'biospecimens': ['Blood']})
    assert not B.is_mammalian({'status': 'Detected but not Quantified',
                               'biospecimens': ['Blood']})
    assert not B.is_mammalian({'status': 'Detected and Quantified',
                               'biospecimens': []})
    assert not B.is_mammalian(None)


V06 = os.path.join(DBDIR, 'BMRB_Metabolomics_v06.db')
V05 = os.path.join(DBDIR, 'BMRB_Metabolomics_v05.db')


@pytest.mark.skipif(not os.path.exists(V06), reason='v06 not built')
def test_every_v05_signal_travels_unchanged():
    # bmrb_id is remapped to the HMDB id where v06 found one; the rest stays
    cols = ('id', 'atom_name', 'ppm', 'multiplicity', 'j_coupling', 'protons',
            'ppm_meas', 'j_use', 'pattern_lines', 'v05_status')
    q = f'select {", ".join(cols)} from shifts_1h order by id'
    old = sqlite3.connect(V05).execute(q).fetchall()
    new = sqlite3.connect(V06).execute(q).fetchall()
    assert new[:len(old)] == old


@pytest.mark.skipif(not os.path.exists(V06), reason='v06 not built')
def test_a_remapped_compound_keeps_its_bmse_and_the_key_it_had():
    import json
    con = sqlite3.connect(V06)
    rows = con.execute("select bmrb_id, hmdb_id, bmse_id, v06_evidence from metabolites "
                       "where v06_status='id_filled'").fetchall()
    assert rows
    for bid, hid, bmse, ev in rows:
        was = json.loads(ev)['bmrb_id_was']
        assert bid == hid and bmse and was
        assert con.execute('select count(*) from shifts_1h where bmrb_id=?',
                           (was,)).fetchone()[0] == 0


@pytest.mark.skipif(not os.path.exists(V06), reason='v06 not built')
def test_every_added_compound_names_its_source_and_licence():
    con = sqlite3.connect(V06)
    added = con.execute("select bmrb_id from metabolites where v06_status='added'").fetchall()
    assert added
    missing = con.execute(
        "select count(*) from metabolites m where v06_status='added' and not exists "
        "(select 1 from ref_sources r where r.bmrb_id=m.bmrb_id and r.licence is not null "
        "and r.similarity >= 0.9)").fetchone()[0]
    assert missing == 0
    assert con.execute("select count(*) from ref_sources where origin='SDBS'").fetchone()[0] == 0


@pytest.mark.skipif(not os.path.exists(V06), reason='v06 not built')
def test_names_stay_unique():
    con = sqlite3.connect(V06)
    dup = con.execute('select name, count(*) from metabolites group by name '
                      'having count(*) > 1').fetchall()
    assert dup == []


STAR = """
   loop_
      _Sample_component.Sample_ID
      _Sample_component.Mol_common_name
      _Sample_component.Type

      1  quercetin      Solute
      1  'DMSO-d6'      Solvent
   stop_

   loop_
      _Sample_condition_variable.Type
      _Sample_condition_variable.Val
      _Sample_condition_variable.Val_units

     pH            7.4   pH
     temperature   298   K
   stop_
"""


def test_a_bmrb_entry_outside_d2o_is_known_by_its_sample(tmp_path):
    d = tmp_path / 'bmse001255'
    d.mkdir()
    (d / 'bmse001255.str').write_text(STAR)
    s = B.bmrb_sample('bmse001255', entries=str(tmp_path))
    assert s == {'solvent': ['DMSO-d6'], 'ph': '7.4', 'temp': '298 K'}
    assert not B.is_d2o(s['solvent'])
    assert B.bmrb_sample('bmse000000', entries=str(tmp_path)) is None


def test_one_multiplet_of_every_line_is_split_into_signals():
    lumped = Spectrum(origin='BMRB', kind='peak_list', field_mhz=500.0,
                      similarity=0.95, source_file=None, peaks=ALA_PEAKS,
                      multiplets=[{'ppm': 2.6, 'volume': None, 'height': 2.5,
                                   'peaks': ALA_PEAKS}])
    ch3, ch = _by_ppm(B.peak_signals(lumped, 4))
    assert (ch3['protons'], ch['protons']) == (3, 1)


@pytest.mark.parametrize('ph, out', [('10.5', True), ('7.4', False), ('6.98', False),
                                     (None, False), ('Not Applic', False), ('5.5', True)])
def test_only_a_known_ph_outside_6_to_8_is_refused(ph, out):
    assert B.ph_outside(ph) is out


def test_a_share_that_does_not_round_cleanly_refuses_the_list():
    # heights 1 : 1 over 3 protons -> 1.5 each: no honest count
    even = Spectrum(origin='HMDB', kind='peak_list', field_mhz=500.0,
                    similarity=0.95, source_file=None,
                    peaks=[(1.0, 1.0), (3.0, 1.0)], multiplets=[])
    assert B.peak_signals(even, 3) == []


def test_the_hdo_singlet_is_not_a_signal_of_the_compound():
    with_water = Spectrum(origin='BMRB', kind='peak_list', field_mhz=500.0,
                          similarity=0.95, source_file=None,
                          peaks=ALA_PEAKS + [(4.80, 5.0)], multiplets=[])
    sig = B.peak_signals(with_water, 4)
    assert sorted(s['protons'] for s in sig) == [1, 3]


def test_a_methyl_with_slightly_unequal_couplings_stays_one_signal():
    me = Spectrum(origin='GISSMO', kind='spin_system', field_mhz=500.0,
                  similarity=0.95, source_file=None,
                  spin_ppm=[1.23, 1.23, 1.23, 2.60, 2.60],
                  spin_j=[(0, 3, 7.3), (1, 3, 7.3), (2, 3, 7.1),
                          (0, 4, 7.3), (1, 4, 7.1), (2, 4, 7.3)])
    assert sorted(s['protons'] for s in B.spin_signals(me)) == [2, 3]


def test_a_ch2_split_by_gissmo_is_listed_as_one_multiplet():
    ch2 = Spectrum(origin='GISSMO', kind='spin_system', field_mhz=500.0,
                   similarity=0.95, source_file=None,
                   spin_ppm=[2.220, 2.222, 3.000],
                   spin_j=[(0, 1, -14.0), (0, 2, 7.0), (1, 2, 7.0)])
    sig = _by_ppm(B.spin_signals(ch2))
    assert [(s['protons'], s['multiplicity']) for s in sig] == [(2, 'm'), (1, 'm')]


def test_a_drug_quantified_in_patients_is_not_mammalian():
    assert not B.is_mammalian({'status': 'Detected and Quantified',
                               'biospecimens': ['Blood'], 'drugbank': 'DB00331'})


MAMMAL = os.path.join(DBDIR, 'BMRB_Mammalian_nMRPro-H.db')


@pytest.mark.skipif(not os.path.exists(MAMMAL), reason='v06 not split')
@pytest.mark.parametrize('matrix, biospecimen', B.MATRICES)
def test_a_matrix_is_the_mammalian_compounds_quantified_there(matrix, biospecimen):
    con = sqlite3.connect(os.path.join(DBDIR, B.output_name(matrix)))
    ids = {r[0] for r in con.execute('select bmrb_id from metabolites')}
    mam = {r[0] for r in sqlite3.connect(MAMMAL).execute('select bmrb_id from metabolites')}
    assert ids and ids <= mam
    there = {r[0] for r in con.execute(
        "select bmrb_id from biospecimens where kind='biospecimen' and biospecimen=?",
        (biospecimen,))}
    assert ids == there
    orphans = con.execute('select count(*) from shifts_1h where bmrb_id not in '
                          '(select bmrb_id from metabolites)').fetchone()[0]
    assert orphans == 0


def test_two_methyl_doublets_are_stored_as_four_lines_not_every_transition():
    # 2-isopropylmalic acid's isopropyl as GISSMO fits it: each methyl's
    # three spins coupled to the CH a few hundredths of a hertz apart. The
    # raw solution is hundreds of near-degenerate sticks; a doublet handed
    # them had 223 lines to fit in every sample (vault ADR-092)
    j = [(0, 1, -12.5), (0, 2, -12.5), (1, 2, -12.5), (0, 3, 6.98), (1, 3, 6.95),
         (2, 3, 7.01), (4, 5, -12.5), (4, 6, -12.5), (5, 6, -12.5), (4, 3, 6.97),
         (5, 3, 7.0), (6, 3, 6.96)]
    ipr = Spectrum(origin='GISSMO', kind='spin_system', field_mhz=500.0,
                   similarity=0.95, source_file=None,
                   spin_ppm=[0.8392, 0.8393, 0.8391, 1.8431, 0.8942, 0.8941, 0.8943],
                   spin_j=j)
    trans, field = B.spin_transitions(ipr)
    methyl = sorted(p for p, _i in trans if p < 1.0)
    assert field == B.PATTERN_FIELD_MHZ
    assert [round(p, 3) for p in methyl] == [0.832, 0.846, 0.887, 0.901]


@pytest.mark.skipif(not os.path.exists(V06), reason='v06 not built')
def test_no_added_spin_system_stores_two_transitions_closer_than_one_line():
    from processing.spin_patterns import MERGE_HZ
    con = sqlite3.connect(V06)
    rows = con.execute(
        "select t.bmrb_id, t.ppm, t.freq_mhz from transitions t join metabolites m "
        "on m.bmrb_id = t.bmrb_id where m.v06_status = 'added' and m.v06_evidence "
        "like '%GISSMO%' order by t.bmrb_id, t.ppm").fetchall()
    assert rows
    close = [(a[0], a[1], b[1]) for a, b in zip(rows, rows[1:])
             if a[0] == b[0] and (b[1] - a[1]) * a[2] <= MERGE_HZ]
    assert close == []
