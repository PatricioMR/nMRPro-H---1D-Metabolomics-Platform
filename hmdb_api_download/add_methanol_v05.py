# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Add methanol to BMRB_Metabolomics v05 (full, Mammalian, General).

Extraction spectra carry methanol as a strong singlet near 3.34 ppm
(SNR > 1000 in the Mice extract). The database did not know it: v01 dropped
bmse000294 because its BMRB entry has a peak list but no atom assignment, so
the peak was taken for a signal of no proton. With no owner the singlet was
read as "Formaldehyde" (bmse000256, one singlet at 3.342, the same line; the
hydrated formaldehyde stands near 4.8). That entry is left as it is.

The row is the BMRB entry's own: one singlet, 3 protons (the methyl; the
hydroxyl proton exchanges in D2O), D2O, DSS-referenced, pH 7.4
(Ulrich EL et al., Nucleic Acids Res 2008;36:D402; Hoch JC et al., Nucleic
Acids Res 2023;51:D368). Nothing is measured from a spectrum here --
``ppm_meas`` and the other measured columns stay NULL, as for the 104
signals of the database that were never measured -- and ``v05_status`` says
the row was added, with where it came from.

    python hmdb_api_download/add_methanol_v05.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from build_bmrb_db import loop  # noqa: E402

DBDIR = os.path.join(os.path.dirname(ROOT), 'databases')
BMRB_STR = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documentos',
                        'bmrb_entries', 'bmse000294', 'bmse000294.str')
TARGETS = tuple(os.path.join(DBDIR, f'BMRB_Metabolomics{s}_v05.db')
                for s in ('', '_Mammalian', '_General'))

ID = 'HMDB0001875'
BMSE = 'bmse000294'
ATOMS = 'H3,H4,H5'          # CH3; H6 (OH) exchanges in D2O
PROTONS = 3
FIELD_MHZ = 400.0
SOLVENT = 'D2O'


def _listed_ppm() -> float:
    text = open(BMRB_STR, encoding='utf-8', errors='ignore').read()
    peaks = loop(text, '_Peak_char')
    if len(peaks) != 1:
        raise SystemExit(f'{BMSE}: expected one peak, found {len(peaks)}')
    return float(peaks[0]['Chem_shift_val'])


def add(path: str, ppm: float) -> str:
    con = sqlite3.connect(path)
    try:
        if con.execute('select 1 from metabolites where bmrb_id = ?',
                       (ID,)).fetchone():
            return 'already there'
        why = json.dumps({'why': f'added: {BMSE} (BMRB) was dropped by v01 '
                          'for lack of an atom assignment; extraction '
                          'solvent seen in extract spectra'})
        with con:
            con.execute(
                'insert into metabolites(bmrb_id, hmdb_id, name, formula, '
                'inchi, inchikey, smiles, solvent, field_mhz, n_1h_signals, '
                'total_protons, chebi_id, keep_mammalian, keep_reason, '
                'bmse_id, name_full) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (ID, ID, 'Methanol', 'C H4 O', 'InChI=1S/CH4O/c1-2/h2H,1H3',
                 'OKKJLVBELUTLKV-UHFFFAOYSA-N', 'CO', SOLVENT, FIELD_MHZ, 1,
                 PROTONS, 'CHEBI:17790', '1',
                 'added: HMDB0001875 (human biofluids) and the extraction '
                 'solvent', BMSE, 'Methanol'))
            con.execute(
                'insert into shifts_1h(bmrb_id, hmdb_id, atom_name, ppm, '
                'ppm_min, ppm_max, multiplicity, protons, n_lines, '
                'intensity, field_mhz, solvent, v04_status, v05_status, '
                'v05_evidence) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (ID, ID, ATOMS, ppm, ppm, ppm, 's', PROTONS, 1, 1.0,
                 FIELD_MHZ, SOLVENT, 'unverified', 'added', why))
            con.execute('insert into transitions(bmrb_id, hmdb_id, ppm, '
                        'intensity, freq_mhz) values (?,?,?,?,?)',
                        (ID, ID, ppm, 1.0, FIELD_MHZ))
        return 'added'
    finally:
        con.close()


def main() -> None:
    ppm = _listed_ppm()
    for path in TARGETS:
        print(f'{os.path.basename(path)}: {add(path, ppm)} (ppm {ppm})')


if __name__ == '__main__':
    main()
