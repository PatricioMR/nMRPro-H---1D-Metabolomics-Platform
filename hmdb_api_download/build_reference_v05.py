"""Build BMRB_Metabolomics_v05: v04 with the protons of each signal checked
against the molecule.

Quantification divides a signal's area by its protons. BMRB lists a signal's
atoms, and where an entry's assignment is ambiguous the curated listing
spread a group of atoms evenly over the signals that share it: myo-inositol
1.5 protons on each of its four signals (the molecule has 1, 2, 2, 1), the
aromatic ring of phenylacetic acid 2.5 and 2.5 (2 and 3). Elsewhere an atom
list is short: creatinine's N-methyl listed with 2 protons, cysteamine's two
methylenes with 1 each.

Each correction here is the molecule's stoichiometry, and each was checked
two ways: the structure, and the compound's GISSMO spin system -- one spin
per proton, fitted to the same BMRB spectrum (Dashti H et al., Anal Chem
2017;89:12201-12208) -- whose spins were assigned to the signal nearest
them (``processing/spin_system``). A row changes only where the two agree
and the compound's total is unchanged or becomes its molecular count.

Nothing is overwritten without a trace: the listed count before the change
stays in ``protons_v04``; every row records ``v05_status`` (corrected /
review / unchanged) and its evidence (``v05_evidence``). The rows found
wrong but not corrected -- a signal of the molecule the database does not
list, a position that does not match the molecule, a compound in two forms
-- are marked review with what was found, and left as they were: adding a
signal is not a correction of a count. A CSV of both is written beside the
database.

Source data: Biological Magnetic Resonance Data Bank (BMRB) -- Ulrich, E.L.
et al. (2008). BioMagResBank. Nucleic Acids Res. 36:D402-D408; Hoch, J.C. et
al. (2023). Biological Magnetic Resonance Data Bank. Nucleic Acids Res.
51:D368-D376.

    python hmdb_api_download/build_reference_v05.py
    python hmdb_api_download/split_dbs.py --version v05
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(os.path.dirname(ROOT), 'databases')
V04 = os.path.join(DBDIR, 'BMRB_Metabolomics_v04.db')
V05 = os.path.join(DBDIR, 'BMRB_Metabolomics_v05.db')
REPORT = os.path.join(ROOT, '_v05_changes.csv')

# A row is found by its compound, its atom list and its position
# (COALESCE(ppm_meas, ppm)) to within this.
PPM_TOL = 0.005

# (compound, atom list, ppm, protons, why). GISSMO: the spins of the
# compound's spin system nearest this signal.
CORRECTIONS = (
    ('Creatinine', 'H11,H9', 3.034, 3,
     'N-methyl: 3 protons (GISSMO 3 spins); listed 2'),
    ('Cysteamine', 'H5,H6', 3.390, 2,
     'S-CH2: 2 protons (GISSMO 2 spins); listed 1'),
    ('Cysteamine', 'H5,H6', 3.024, 2,
     'N-CH2: 2 protons (GISSMO 2 spins); listed 1'),
    ('Phenylacetylglycine', 'H21', 3.657, 2,
     'phenyl-CH2: 2 protons (GISSMO 2 spins); listed 1'),
    ('Phenylacetylglycine', 'H15,H16,H17,H18,H19', 7.389, 2,
     'ring, meta: 2 of 5 (GISSMO 2 spins); listed 2.5'),
    ('Phenylacetylglycine', 'H15,H16,H17,H18,H19', 7.343, 3,
     'ring, ortho + para: 3 of 5 (GISSMO 3 spins); listed 2.5'),
    ('Phenylacetic acid', 'H11,H12,H13,H14,H15', 7.363, 2,
     'ring, meta: 2 of 5 (GISSMO 2 spins); listed 2.5'),
    ('Phenylacetic acid', 'H11,H12,H13,H14,H15', 7.291, 3,
     'ring, ortho + para: 3 of 5 (GISSMO 3 spins); listed 2.5'),
    ('2-phenylbutyric acid', 'H18,H19,H20,H21,H22', 7.365, 4,
     'ring, ortho + meta: 4 of 5 (GISSMO 4 spins); listed 2.5'),
    ('2-phenylbutyric acid', 'H18,H19,H20,H21,H22', 7.289, 1,
     'ring, para: 1 of 5 (GISSMO 1 spin); listed 2.5'),
    ('Phthalaldehydic acid', 'H12,H13,H14,H15', 7.871, 1,
     'ring: 1 of 4 (GISSMO 1 spin); listed 1.33'),
    ('Phthalaldehydic acid', 'H12,H13,H14,H15', 7.702, 1,
     'ring: 1 of 4 (GISSMO 1 spin); listed 1.33'),
    ('Phthalaldehydic acid', 'H12,H13,H14,H15', 7.569, 2,
     'ring: 2 of 4 (GISSMO 2 spins); listed 1.33'),
    ('Myo-inositol', 'H13,H14,H15,H16,H17,H18', 4.047, 1,
     'H2: 1 proton (GISSMO 1 spin); listed 1.5'),
    ('Myo-inositol', 'H13,H14,H15,H16,H17,H18', 3.606, 2,
     'H4/H6: 2 protons (GISSMO 2 spins); listed 1.5'),
    ('Myo-inositol', 'H13,H14,H15,H16,H17,H18', 3.522, 2,
     'H1/H3: 2 protons (GISSMO 2 spins); listed 1.5'),
    ('Myo-inositol', 'H13,H14,H15,H16,H17,H18', 3.264, 1,
     'H5: 1 proton (GISSMO 1 spin); listed 1.5'),
    ('Meso-erythritol', 'H10,H11,H12,H13,H14,H9', 3.644, 4,
     'C1/C4 methylenes: 4 protons (GISSMO 4 spins); listed 3.33'),
    ('Meso-erythritol', 'H10,H11,H12,H9', 3.751, 1,
     'C2/C3 methines, 2 protons over two listed signals; listed 1.33'),
    ('Meso-erythritol', 'H10,H11,H12,H9', 3.778, 1,
     'C2/C3 methines, 2 protons over two listed signals; listed 1.33'),
    ('Sucrose', 'H24,H25,H26,H27,H28,H29', 3.668, 2,
     'fructose H1: 2 protons (GISSMO 2 spins); listed 6'),
    ('Sucrose', 'H30,H32,H33,H34,H35,H36', 3.811, 5,
     'glucose H6a/b, fructose H6a/b and a ring methine overlap here '
     '(GISSMO 5 spins); listed 1'),
)

# Found wrong, not corrected (see the module).
REVIEW = (
    ('D-Glucose', None, 'GISSMO holds both anomers; the listing spreads '
     'the ring methines over overlapping signals (0.17-1.1 protons) and '
     'the anomeric doublets at 0.5 each against 0.36/0.64 at equilibrium '
     '(Angyal 1984)'),
    ('Gluconic acid', None, 'molecule has 6 C-H protons, the listing 3: '
     'H2 4.116 and H3 4.02 (GISSMO) not listed'),
    ('D-Glucono-1,5-lactone', None, 'molecule has 6 C-H protons, the '
     'listing 3: 4.03 and 4.14 (GISSMO) not listed'),
    ('Betaine aldehyde', None, 'in water the hydrate: N-CH2 4.30 (2H) and '
     'CH(OH)2 5.56 (1H) not listed'),
    ('Hexanoic acid', 'H18,H19', 'listed at 2.518; the alpha-methylene '
     'stands at 2.158 (GISSMO), and no other listed signal is there'),
    ('Thiamin', None, 'GISSMO holds a 1-proton line at 6.000 with no '
     'coupling, not in the listing and not assigned to the molecule here'),
    ('2,3-diphospho-D-glyceric acid', None, 'listed 4.469/3.974; GISSMO '
     '4.529/4.058 -- a phosphate ester, its shifts move with pH'),
    ('L-Glutathione reduced', 'H25,H26,H29', 'one row for the glycine '
     'methylene (AB) and the glutamate alpha-proton (t), labelled s: the '
     'protons (3) and the measured profile cover both, the label does not'),
)

NEW_COLUMNS = (('protons_v04', 'INTEGER'), ('v05_status', 'TEXT'),
               ('v05_evidence', 'TEXT'))


def _row(con, name, atoms, ppm):
    q = ('select s.id, s.protons, coalesce(s.ppm_meas, s.ppm) from shifts_1h s '
         'join metabolites m on m.bmrb_id = s.bmrb_id where m.name = ? '
         'and s.atom_name = ?')
    hits = [r for r in con.execute(q, (name, atoms))
            if abs(float(r[2]) - ppm) <= PPM_TOL]
    if len(hits) != 1:
        raise SystemExit(f'{name} {atoms} {ppm}: {len(hits)} rows match')
    return hits[0]


def build() -> None:
    shutil.copyfile(V04, V05)
    con = sqlite3.connect(V05)
    have = {r[1] for r in con.execute('pragma table_info(shifts_1h)')}
    for col, typ in NEW_COLUMNS:
        if col not in have:
            con.execute(f'alter table shifts_1h add column {col} {typ}')
    con.execute("update shifts_1h set v05_status = 'unchanged'")
    report = []
    for name, atoms, ppm, nh, why in CORRECTIONS:
        rid, old, at = _row(con, name, atoms, ppm)
        con.execute('update shifts_1h set protons_v04 = ?, protons = ?, '
                    "v05_status = 'corrected', v05_evidence = ? where id = ?",
                    (old, nh, json.dumps({'why': why}), rid))
        report.append({'name': name, 'atoms': atoms, 'ppm': round(at, 3),
                       'status': 'corrected', 'protons_v04': old,
                       'protons_v05': nh, 'why': why})
    for name, atoms, why in REVIEW:
        q = ('select s.id, coalesce(s.ppm_meas, s.ppm), s.atom_name, s.protons '
             'from shifts_1h s join metabolites m on m.bmrb_id = s.bmrb_id '
             'where m.name = ?' + (' and s.atom_name = ?' if atoms else ''))
        rows = con.execute(q, (name, atoms) if atoms else (name,)).fetchall()
        if not rows:
            raise SystemExit(f'{name}: not in the database')
        for rid, at, an, nh in rows:
            con.execute("update shifts_1h set v05_status = 'review', "
                        'v05_evidence = ? where id = ?',
                        (json.dumps({'why': why}), rid))
        report.append({'name': name, 'atoms': atoms or '(all)', 'ppm': '',
                       'status': 'review', 'protons_v04': '',
                       'protons_v05': '', 'why': why})
    con.commit()
    con.execute('VACUUM')
    con.close()
    with open(REPORT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    n_c = sum(r['status'] == 'corrected' for r in report)
    print(f'v05: {n_c} rows corrected, {len(REVIEW)} compounds for review '
          f'-> {V05}')
    print('changes and review items ->', REPORT)


if __name__ == '__main__':
    build()
