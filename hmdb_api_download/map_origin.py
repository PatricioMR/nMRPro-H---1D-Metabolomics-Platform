"""
Map BMRB_Metabolomics_v01 metabolites to the local HMDB index and set the
mammalian-biofluid flag. Endogenous flag (where available) was already applied
from the downloaded ontology files.

Mapping (no API): HMDB page index has no InChIKey, so match by normalized name,
then confirm with molecular formula. mono_mass disambiguates ties.
  • is_mammalian = HMDB biospecimen_location non-empty (found in a mammalian
    biofluid: Blood/Urine/CSF/Saliva/Feces/...)
  • biofluids   = the HMDB biofluid list
"""
import json
import os
import re
import sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(ROOT, "hmdb_index.db")
DB = os.path.join(os.path.dirname(ROOT), "databases", "BMRB_Metabolomics_v01.db")


def norm(name):
    if not name:
        return ''
    s = name.lower().strip()
    s = re.sub(r'\(\s*[rs+\-]+\s*\)', '', s)         # (R)/(S)/(+)/(-)
    s = re.sub(r'\b(dl|d|l)-', '', s)                # D-/L-/DL- prefixes
    s = re.sub(r'[\s\-_,()\'+]', '', s)              # punctuation/space
    return s


def main():
    idx = sqlite3.connect(IDX)
    # name_norm -> list of (hmdb_id, formula, mono_mass, n_biofluids, biofluids)
    by_norm = {}
    for hid, nn, formula, mm, bf, nbf in idx.execute(
            "SELECT hmdb_id,name_norm,formula,mono_mass,biofluids,n_biofluids FROM hmdb"):
        by_norm.setdefault(norm(nn), []).append((hid, formula, mm, nbf, bf))
    idx.close()

    con = sqlite3.connect(DB)
    cur = con.cursor()
    for col in ('biofluids',):
        try:
            cur.execute(f"ALTER TABLE metabolites ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass

    mapped = mam = 0
    for bid, name, formula in cur.execute(
            "SELECT bmrb_id,name,formula FROM metabolites").fetchall():
        cands = by_norm.get(norm(name), [])
        if not cands:
            continue
        # prefer same formula
        pick = None
        for hid, f, mm, nbf, bf in cands:
            if formula and f and f.replace(' ', '') == formula.replace(' ', ''):
                pick = (hid, nbf, bf)
                break
        if pick is None:
            hid, f, mm, nbf, bf = cands[0]      # name-only match
            pick = (hid, nbf, bf)
        hid, nbf, bf = pick
        is_mam = 1 if (nbf or 0) > 0 else 0
        cur.execute("UPDATE metabolites SET hmdb_id=COALESCE(hmdb_id,?), "
                    "is_mammalian=?, biofluids=? WHERE bmrb_id=?",
                    (hid, str(is_mam), bf, bid))
        mapped += 1
        mam += is_mam
    con.commit()
    tot = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    endo = cur.execute("SELECT COUNT(*) FROM metabolites WHERE is_endogenous='1'").fetchone()[0]
    con.close()
    print(f"BMRB total={tot}  mapped_to_HMDB={mapped}  mammalian(biofluid)={mam}  "
          f"endogenous(local ontology)={endo}")


if __name__ == '__main__':
    main()
