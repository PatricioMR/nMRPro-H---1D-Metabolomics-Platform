# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Insert HMDB IDs into the BMRB DBs — fully offline.

Two cross-reference sources (union, no API):
  1. ChEBI database_accession.tsv.gz: BMRB already carries chebi_id → HMDB xref.
  2. Local HMDB *.db: InChIKey → hmdb_id (for compounds ChEBI lacks an HMDB xref).

Writes metabolites.hmdb_id (and shifts_1h.hmdb_id / transitions.hmdb_id so the
columns are consistent) on v01 + General + Mammalian. Primary HMDB id = lowest
accession number when several map. Reports coverage; leaves NULL when unknown
(never fabricated).
"""
import glob
import gzip
import os
import sqlite3
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(os.path.dirname(ROOT), 'databases')
ACC = os.path.join(ROOT, 'chebi', 'database_accession.tsv.gz')
TARGETS = ('BMRB_Metabolomics_v01.db', 'BMRB_Metabolomics_General_v01.db',
           'BMRB_Metabolomics_Mammalian_v01.db')


def chebi2hmdb():
    m = defaultdict(list)
    with gzip.open(ACC, 'rt', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) >= 3 and c[2].startswith('HMDB'):
                m['CHEBI:' + c[1]].append(c[2])
    return {k: sorted(v)[0] for k, v in m.items()}      # primary = lowest id


def inchikey2hmdb():
    """EXACT InChIKey → hmdb_id only. No connectivity-block (first-14) fallback:
    it collapses epimers (D-glucose/galactose/mannose share the connectivity
    block) onto one wrong HMDB id."""
    full = {}
    for db in glob.glob(os.path.join(DBDIR, 'HMDB_*.db')):
        if 'Metabolomics_v01' in db:
            continue
        try:
            c = sqlite3.connect(db)
            cols = {r[1] for r in c.execute("PRAGMA table_info(metabolites)")}
            if 'inchikey' in cols and 'hmdb_id' in cols:
                for hid, ik in c.execute(
                        "SELECT hmdb_id,inchikey FROM metabolites "
                        "WHERE inchikey IS NOT NULL AND hmdb_id IS NOT NULL"):
                    if ik:
                        full.setdefault(ik, hid)
            c.close()
        except sqlite3.Error:
            pass
    return full


def main():
    c2h = chebi2hmdb()
    ikf = inchikey2hmdb()
    print(f"xref maps: chebi->hmdb={len(c2h)}  inchikey->hmdb(exact)={len(ikf)}")

    for name in TARGETS:
        path = os.path.join(DBDIR, name)
        if not os.path.exists(path):
            continue
        con = sqlite3.connect(path)
        cur = con.cursor()
        # ensure hmdb_id columns exist
        for tbl in ('shifts_1h', 'transitions'):
            cols = {r[1] for r in cur.execute(f"PRAGMA table_info({tbl})")}
            if 'hmdb_id' not in cols:
                cur.execute(f"ALTER TABLE {tbl} ADD COLUMN hmdb_id TEXT")
        # clear any prior (possibly connectivity-collapsed) hmdb_id, rebuild clean
        cur.execute("UPDATE metabolites SET hmdb_id=NULL")
        n_set = 0
        for bid, chebi, ik in cur.execute(
                "SELECT bmrb_id,chebi_id,inchikey FROM metabolites").fetchall():
            hid = None
            if chebi:
                hid = c2h.get(chebi)
            if not hid and ik:
                hid = ikf.get(ik)          # EXACT inchikey only
            if not hid:
                continue
            cur.execute("UPDATE metabolites SET hmdb_id=? WHERE bmrb_id=?", (hid, bid))
            cur.execute("UPDATE shifts_1h SET hmdb_id=? WHERE bmrb_id=?", (hid, bid))
            cur.execute("UPDATE transitions SET hmdb_id=? WHERE bmrb_id=?", (hid, bid))
            n_set += 1
        con.commit()
        tot = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
        have = cur.execute("SELECT COUNT(*) FROM metabolites WHERE hmdb_id IS NOT NULL").fetchone()[0]
        con.close()
        print(f"{name}: hmdb_id set this run={n_set}  total_with_hmdb={have}/{tot}")


if __name__ == '__main__':
    main()
