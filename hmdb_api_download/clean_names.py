"""
Simplify metabolite display names in the BMRB DBs.

The verbose BMRB names carry stereo/config noise ('D-(+)-Glucose',
'(1R,3R,4R,5R)-(-)-Quinic acid'). Replace the shown `name` with the BMRB name
with parenthetical stereo/config stripped (deterministic; keeps the standard name
and the short D-/L- prefix), keeping the original verbatim in `name_full`.

NOT used as a name source: BMRB `_Chem_comp_common_name` synonyms (historical/
commercial junk — 'Neuberg ester', 'Levulose' — and some wrong entries), and the
local HMDB name maps (hmdb_pathways.db has a corrupted hmdb_id→name column that
scrambled names). The deterministic cleaner is verifiable per row.
"""
import os
import re
import sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(os.path.dirname(ROOT), 'databases')
TARGETS = ('BMRB_Metabolomics_v01.db', 'BMRB_Metabolomics_General_v01.db',
           'BMRB_Metabolomics_Mammalian_v01.db')


def clean(n):
    if not n:
        return n
    s = re.sub(r'\((?:[0-9RSEZrsez,\s\-]+|\+/?\-?|±|RS)\)', '', n)  # (+),(R),(1R,3R..)
    s = re.sub(r'([DLdl])-\s+', r'\1-', s)        # 'L- lactic' -> 'L-lactic'
    s = re.sub(r'-{2,}', '-', s)
    s = re.sub(r'\s{2,}', ' ', s).strip(' -')
    return (s[:1].upper() + s[1:]) if s else n


def main():
    # HMDB-name override dropped: the only broad local source (hmdb_pathways.db)
    # has a corrupted hmdb_id→name column (HMDB0000017→'Alanine'), which scrambled
    # names. Use the deterministic BMRB-name cleaner only — verifiable per row.
    for name in TARGETS:
        path = os.path.join(DBDIR, name)
        if not os.path.exists(path):
            continue
        con = sqlite3.connect(path)
        cur = con.cursor()
        cols = {r[1] for r in cur.execute("PRAGMA table_info(metabolites)")}
        if 'name_full' not in cols:
            cur.execute("ALTER TABLE metabolites ADD COLUMN name_full TEXT")
            cur.execute("UPDATE metabolites SET name_full=name")
        # restore from name_full first (undo any prior bad rename), then clean
        for bid, full in cur.execute(
                "SELECT bmrb_id,name_full FROM metabolites").fetchall():
            cur.execute("UPDATE metabolites SET name=? WHERE bmrb_id=?",
                        (clean(full), bid))
        con.commit()
        n = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
        con.close()
        print(f"{name}: {n} names cleaned (deterministic, from name_full)")


if __name__ == '__main__':
    main()
